"""Two-level rollback to the last good release — ticket 22.

A rollback moves **both** levels or neither:

* the **data** level — the canonical ``active_release.json`` pointer, the source
  of truth (:func:`~src.infrastructure.canonical_store.publish_pointer`);
* the **retrieval** level — the LanceDB table restored to the release's tag, the
  running process rebound to the retriever that serves it, **and the knowledge
  graph swapped to the graph artifact the release pinned**.

The third artifact is not a decoration on the second. A release *serves* two
artifacts and pins both in its manifest: the LanceDB table (``lancedb_tag``) and
the graph (``graph_sha256``/``graph_path``/``graph_schema_version``). The graph
maps entity nodes to ``chunk_id``s, so a graph left on release R2 while the
table moves to R1 injects R2's chunk ids into R1's answers — a mixed-release
answer built from a *matching* pointer and table, which is why the LanceDB half
alone cannot detect it. Restoring one without the other is the defect this level
removes.

There is deliberately no ``level=data|retrieval`` switch. A half-rollback leaves
canonical and retrieval on different releases, which is precisely the mixed
state ticket 22's criterion 4 forbids; one route that moves both is the
contract.

Ordering is the whole design. Steps A and B are read-only, C moves canonical,
D moves retrieval, D2 moves the graph, and E compensates if D or D2 fails after
C:

* **A — validate first.** Resolve the pointer and the target, confirm the target
  manifest is readable, confirm the release's LanceDB tag exists, confirm the
  target is not already active, and confirm the target's pinned graph is present
  and still hashes to its pin. Any failure raises
  :class:`RollbackValidationError` (a 4xx) with nothing mutated.
* **B — snapshot.** The current release id, the retriever object currently
  serving and the graph object currently serving are the compensation inputs.
* **C — data.** Publish the pointer.
* **D — retrieval.** Restore the LanceDB tag, move ``prod`` to it, then rebind
  the runtime through the single swap seam (``src/main.py``'s
  ``swap_runtime_services``). That one rebind is the only point at which serving
  changes, which is what makes "no answer mixed between two releases" true.
* **D2 — graph.** Load the release's pinned graph, verified against the pin's
  digest, and hand it to the runtime through the existing
  ``graphrag.update_graph`` seam — the same call the incremental-update path
  uses to hot-swap a graph. A runtime that serves no graph reports
  ``"not-serving"`` and moves nothing; a runtime that serves one while the
  target pins none is refused in step A.
* **E — compensate.** If D or D2 fails, put the served artifacts back (reverse
  order: graph, retrieval, then the pointer) and raise
  :class:`RollbackAbortedError` naming **both** the failure and the compensation
  failure, if any. Nothing is swallowed.

What this is **not**: a single atomic transaction across the three artifacts.
There is no shared commit point — the pointer is a file rename, the LanceDB move
is a table version, and the graph is an in-memory rebind on the pipeline. What
the ordering buys is *consistency by construction*: the graph moves last, so
every failure before it leaves the graph matching the table it was serving, and
a failure inside it is compensated back. The residual window is the interval
between D's rebind and D2's, during which the runtime serves the target's table
with the previous graph — bounded by two in-process calls, and closed by the
drain gate the route holds (``request_tracker.gate``) so no query runs inside
it. Making the three one transaction would require a commit protocol none of the
three stores has; see ticket 22's notes for why the ordering was chosen instead.

The audit event is appended **after** every level has moved, so the ledger
never claims a rollback that did not happen. If that append fails, the mutation
stands and :class:`RollbackAuditError` says so loudly — the same philosophy as
``AuditRecordingError`` in :mod:`src.services.admin_lifecycle`.

The caller holds the swap lock (``routers/admin.py``'s ``_reload_lock``) around
:meth:`AdminRollbackService.rollback`, so a rollback and a reload cannot overlap;
this module knows nothing about HTTP, locks or requests.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from loguru import logger

from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.domain.release import ReleaseManifest
from src.infrastructure.audit_ledger import AuditEntry, AuditLedger
from src.infrastructure.canonical_store import publish_pointer, read_manifest
from src.infrastructure.graph_artifact import (
    GraphArtifactError,
    GraphArtifactNotPinnedError,
    load_pinned_graph,
    verify_pinned_graph,
)
from src.infrastructure.lancedb_versioning import promote_to_prod, rollback_to

__all__ = [
    "AdminRollbackService",
    "GraphRollback",
    "GraphRollbackError",
    "GraphSnapshot",
    "InvalidReleaseIdError",
    "LanceDBRetrievalRollback",
    "MANIFEST_FILENAME",
    "NoPreviousReleaseError",
    "POINTER_FILENAME",
    "RELEASES_DIRNAME",
    "ReleaseGraphRollback",
    "ReleaseManifestMismatchError",
    "ReleaseNotFoundError",
    "RetrievalRollback",
    "RetrievalRollbackError",
    "RetrievalSnapshot",
    "RetrievalUnavailableError",
    "RollbackAbortedError",
    "RollbackAuditError",
    "RollbackError",
    "RollbackOutcome",
    "RollbackValidationError",
    "TargetGraphNotPinnedError",
    "TargetIsActiveError",
    "TargetNotTaggedError",
    "release_tag",
]

RELEASES_DIRNAME = "releases"
POINTER_FILENAME = "active_release.json"
MANIFEST_FILENAME = "release_manifest.json"

# A release lineage is walked at most this far. A cycle cannot be constructed
# today (a parent is written before its child), but a bounded walk means a
# corrupt lineage degrades into "no previous release" instead of a hung request.
MAX_LINEAGE_HOPS = 32

_TAG_PREFIX = "release-"
# Characters Windows refuses in a path component, plus the separators of both
# worlds. ``target_release_id`` arrives from a request body, so the id is checked
# before it is ever joined onto the releases root.
_UNSAFE_ID_CHARS = frozenset('<>:"|?*')
_SEPARATORS = frozenset({"/", "\\", os.sep, os.altsep or ""})


def release_tag(release_id: str) -> str:
    """The LanceDB tag a release's chunks are tagged with.

    One convention, matching :func:`~src.infrastructure.lancedb_versioning.tag_release`:
    the tag a rollback restores is the tag the publish path wrote.
    """
    return f"{_TAG_PREFIX}{release_id}"


# ── failures ────────────────────────────────────────────────────────────────


class RollbackError(Exception):
    """Base class for rollback failures."""


class RollbackValidationError(RollbackError):
    """A rollback that cannot start.

    Every one of these is raised during step A or B — before anything has been
    mutated — so a refused rollback leaves the pointer byte-identical by
    construction rather than by cleanup.
    """


class ReleaseNotFoundError(RollbackValidationError):
    """No readable release manifest for the requested release."""


class NoPreviousReleaseError(RollbackValidationError):
    """The lineage offers no previous release to fall back to."""


class TargetIsActiveError(RollbackValidationError):
    """The requested target is the release already serving."""


class TargetNotTaggedError(RollbackValidationError):
    """The target release has no LanceDB tag — its chunks are not restorable."""


class ReleaseManifestMismatchError(RollbackValidationError):
    """The manifest disagrees with the release it is filed under."""


class InvalidReleaseIdError(RollbackValidationError):
    """A release id that cannot name a directory under the releases root."""


class RollbackAbortedError(RollbackError):
    """A rollback that failed mid-flight.

    Carries every failure of the attempt: the canonical write, the retrieval
    step, and the compensation, each when it happened. A caller must be able to
    see *both* halves — "the rollback failed" alone cannot tell an operator
    whether the process is serving one release or two.
    """

    def __init__(
        self,
        *,
        stage: str,
        canonical_error: BaseException | None = None,
        retrieval_error: BaseException | None = None,
        graph_error: BaseException | None = None,
        compensation_error: str | None = None,
    ) -> None:
        self.stage = stage
        self.canonical_error = canonical_error
        self.retrieval_error = retrieval_error
        self.graph_error = graph_error
        self.compensation_error = compensation_error
        parts = [f"rollback failed at stage '{stage}'"]
        if canonical_error is not None:
            parts.append(f"canonical pointer write failed: {canonical_error!r}")
        if retrieval_error is not None:
            parts.append(f"retrieval rollback failed: {retrieval_error!r}")
        if graph_error is not None:
            parts.append(f"graph rollback failed: {graph_error!r}")
        parts.append(
            "compensation succeeded: the served release is back to the pre-call one"
            if compensation_error is None
            else f"COMPENSATION ALSO FAILED: {compensation_error}"
        )
        super().__init__("; ".join(parts))


class RollbackAuditError(RollbackError):
    """Both levels rolled back, but the audit ledger does not know.

    Deliberately *not* raised as a bare I/O error: the bare error reads as "the
    rollback failed" while the truth is the opposite — the process is now
    serving the target release and the ledger says nothing happened. A caller
    that cannot tell those apart will re-run a rollback that already happened or
    report a rolled-back release as untouched.
    """

    def __init__(
        self,
        *,
        release_id: str,
        previous_release_id: str,
        actor_id: str,
        ledger_path: str | None,
        cause: BaseException,
    ) -> None:
        self.release_id = release_id
        self.previous_release_id = previous_release_id
        self.actor_id = actor_id
        self.ledger_path = ledger_path
        self.cause = cause
        super().__init__(
            f"rollback of {previous_release_id} -> {release_id} by {actor_id} "
            "SUCCEEDED but was NOT recorded in the audit trail"
            + (f" ({ledger_path})" if ledger_path else "")
            + f": {cause!r}. Both levels are on {release_id}; reconcile the ledger "
            "before doing anything else, and do not treat this rollback as "
            "unaudited-but-fine."
        )


class RetrievalRollbackError(RollbackError):
    """The retrieval half of a rollback failed."""


class RetrievalUnavailableError(RetrievalRollbackError):
    """There is no serving retriever to move — the runtime is not bound to one.

    Distinct from :class:`RetrievalRollbackError` because it is not the caller's
    fault and nothing has been mutated: the rollback cannot start because the
    retrieval level does not exist yet (warmup not finished, or a failed LanceDB
    init). The route answers 503 for it rather than 500 or a 4xx that would
    suggest the request itself was wrong.
    """


# ── the retrieval seam ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalSnapshot:
    """What the runtime was serving when the rollback started.

    ``retriever`` is the object itself rather than a copy: compensation exists
    to put *that* object back on the serving path.
    """

    retriever: Any
    release_id: str | None
    table_version: int | None


class RetrievalRollback(Protocol):
    """The retrieval half of a rollback, as the orchestrator uses it.

    Four methods, one per step it takes part in, so the orchestrator can keep
    "validate without mutating" (A), "snapshot" (B), "move" (D) and "undo" (E)
    apart. The production implementation is :class:`LanceDBRetrievalRollback`;
    tests substitute their own.
    """

    def validate(self, manifest: ReleaseManifest) -> int | None:
        """Confirm ``manifest``'s release is restorable. Must not mutate."""

    def snapshot(self, *, release_id: str | None) -> RetrievalSnapshot:
        """Capture the current serving state (the compensation input)."""

    def apply(self, manifest: ReleaseManifest) -> int | None:
        """Move retrieval onto ``manifest``'s release; return the new version."""

    def compensate(self, snapshot: RetrievalSnapshot) -> None:
        """Put retrieval back the way ``snapshot`` found it."""


class LanceDBRetrievalRollback:
    """Production retrieval half: the LanceDB tag plus the running runtime.

    ``retriever_provider`` returns the retriever the process is serving from
    (main.py's ``lancedb_retriever_service``) and ``swap`` rebinds the runtime to
    a retriever object (main.py's ``swap_runtime_services``). Both are called
    lazily, because the wiring runs at startup while the retriever it needs is
    built later by the warmup phase.
    """

    def __init__(
        self,
        *,
        retriever_provider: Callable[[], Any],
        swap: Callable[[Any], Any],
    ) -> None:
        self._retriever = retriever_provider
        self._swap = swap

    # -- step A -------------------------------------------------------------

    def validate(self, manifest: ReleaseManifest) -> int:
        """The version the release's tag points at; raises when it has none."""
        table = self._table()
        tag = release_tag(manifest.release_id)
        try:
            tags = table.tags.list()
        except Exception as exc:
            raise TargetNotTaggedError(
                f"cannot list LanceDB tags to verify release {manifest.release_id}: {exc!r}"
            ) from exc
        if tag not in tags:
            raise TargetNotTaggedError(
                f"LanceDB carries no tag '{tag}' for release {manifest.release_id} — "
                f"its chunks are not restorable (tags present: {sorted(tags)})"
            )
        return int(tags[tag]["version"])

    # -- step B -------------------------------------------------------------

    def snapshot(self, *, release_id: str | None = None) -> RetrievalSnapshot:
        table = self._table(required=False)
        return RetrievalSnapshot(
            retriever=self._retriever(),
            release_id=release_id,
            table_version=getattr(table, "version", None),
        )

    # -- step D -------------------------------------------------------------

    def apply(self, manifest: ReleaseManifest) -> int:
        retriever = self._retriever()
        if retriever is None:
            raise RetrievalUnavailableError(
                "no serving retriever: the runtime is not bound to any LanceDB "
                "retriever, so there is nothing to move to the target release"
            )
        table = self._table()
        try:
            version = rollback_to(table, manifest.release_id)
            promote_to_prod(table, manifest.release_id)
        except Exception as exc:
            raise RetrievalRollbackError(
                f"LanceDB restore to release {manifest.release_id} failed: {exc!r}"
            ) from exc
        try:
            self._swap(retriever)
        except Exception as exc:
            raise RetrievalRollbackError(
                f"runtime swap to release {manifest.release_id} failed after the "
                f"LanceDB table was restored: {exc!r}"
            ) from exc
        logger.info(
            f"Retrieval rollback: table restored to release {manifest.release_id} "
            f"(version {version}) and the runtime rebound to it"
        )
        return version

    # -- step E -------------------------------------------------------------

    def compensate(self, snapshot: RetrievalSnapshot) -> None:
        """Undo whatever :meth:`apply` managed to do, in reverse order."""
        problems: list[str] = []

        try:
            if snapshot.retriever is not None and self._retriever() is not snapshot.retriever:
                self._swap(snapshot.retriever)
        except Exception as exc:
            problems.append(f"runtime swap-back failed: {exc!r}")

        try:
            table = self._table(required=False)
            moved = (
                table is not None
                and snapshot.table_version is not None
                and getattr(table, "version", None) != snapshot.table_version
            )
            if moved:
                if snapshot.release_id:
                    rollback_to(table, snapshot.release_id)
                else:
                    problems.append(
                        "the LanceDB table moved but the previous release id is "
                        "unknown, so it cannot be restored by tag"
                    )
        except Exception as exc:
            problems.append(f"LanceDB restore-back failed: {exc!r}")

        if problems:
            raise RetrievalRollbackError("; ".join(problems))

    # -- internals ----------------------------------------------------------

    def _table(self, *, required: bool = True) -> Any:
        """The LanceDB table the serving retriever reads from.

        Taken from the *live* retriever rather than opened a second time: a
        restore is a new version on the table object, and a second handle would
        keep serving the version it was opened at.
        """
        retriever = self._retriever()
        adapter = getattr(retriever, "_adapter", None)
        if adapter is None:
            if not required:
                return None
            raise RetrievalUnavailableError(
                "the serving retriever exposes no LanceDB adapter, so its table "
                "cannot be tagged, restored or verified"
            )
        adapter.ensure_table()
        table = getattr(adapter, "_table", None)
        if table is None and required:
            raise RetrievalUnavailableError(
                "the serving retriever's adapter has no table open"
            )
        return table


# ── the graph seam ──────────────────────────────────────────────────────────

class GraphRollbackError(RollbackError):
    """The graph level could not be moved or restored."""

class TargetGraphNotPinnedError(RollbackValidationError):
    """The target release pins no graph, but the runtime is serving one.

    A validation failure (4xx), not a 500: nothing is wrong with the process,
    and nothing has been mutated. Serving the running graph over the target
    release's table is the mixed-release state this level exists to prevent —
    the graph maps entities to ``chunk_id``s, so the old graph would inject
    chunk ids from the release being rolled away from into the target's answers.
    """

@dataclass(frozen=True)
class GraphSnapshot:
    """The graph object the runtime was serving when the rollback started.

    The object itself, not a copy, for the same reason
    :class:`RetrievalSnapshot` holds the retriever itself: compensation exists
    to put *that* graph back.
    """

    graph: Any
    release_id: str | None

class GraphRollback(Protocol):
    """The graph half of a rollback, as the orchestrator uses it.

    Same four steps as :class:`RetrievalRollback`, and for the same reason: the
    orchestrator keeps "validate without mutating" (A), "snapshot" (B), "move"
    (D) and "undo" (E) apart. ``validate`` and ``apply`` take the release
    directory because the artifact to move is *inside* the release — the pin in
    the manifest is a relative path, and only the orchestrator resolves where a
    release directory is.
    """

    def validate(self, manifest: ReleaseManifest, release_dir: Path) -> None:
        """Confirm the graph level can be moved. Must not mutate."""

    def snapshot(self, *, release_id: str | None) -> GraphSnapshot:
        """Capture the graph currently serving (the compensation input)."""

    def apply(self, manifest: ReleaseManifest, release_dir: Path) -> str:
        """Move the served graph onto ``manifest``'s release; report how."""

    def compensate(self, snapshot: GraphSnapshot) -> None:
        """Put the graph back the way ``snapshot`` found it."""

class ReleaseGraphRollback:
    """Production graph half: the pipeline's graph, swapped to the release's pin.

    ``pipeline_provider`` returns the pipeline the process is serving from
    (main.py's ``_health_pipeline``), lazily — the wiring runs at startup while
    the pipeline it needs is built later by warmup, and a captured ``None``
    would make every rollback fail with a misleading "no pipeline".

    The move goes through ``UnifiedRAGPipeline``'s existing
    ``graphrag.update_graph`` seam: the same call the incremental-update path
    already uses to hot-swap a graph without a restart. This module does not
    open a second write path into the retriever; it reuses the one that exists.

    A runtime with no pipeline, or a pipeline with no GraphRAG retriever, serves
    no graph. That is reported as ``"not-serving"`` and moves nothing — it is
    not an error, because a deployment without GraphRAG must still be able to
    roll back, and it is not a mixed state, because no graph is served.
    """

    def __init__(self, *, pipeline_provider: Callable[[], Any]) -> None:
        self._pipeline = pipeline_provider

    # -- step A -------------------------------------------------------------

    def validate(self, manifest: ReleaseManifest, release_dir: Path) -> None:
        """Refuse a mixed pairing, then prove the pinned bytes are there.

        Read-only, and deliberately *before* anything moves: a rollback whose
        graph artifact is missing or has been rebuilt in place must fail with
        the pointer and the table untouched, not halfway through.
        """
        if self._serving_graph() is None:
            return
        if not manifest.pins_graph:
            raise TargetGraphNotPinnedError(
                f"release {manifest.release_id} pins no graph artifact, but the "
                "runtime is serving one — rolling back would answer from the "
                "target release's table with the previous release's graph, whose "
                "nodes name that release's chunk ids"
            )
        self._verify(manifest, release_dir)

    # -- step B -------------------------------------------------------------

    def snapshot(self, *, release_id: str | None = None) -> GraphSnapshot:
        return GraphSnapshot(graph=self._serving_graph(), release_id=release_id)

    # -- step D -------------------------------------------------------------

    def apply(self, manifest: ReleaseManifest, release_dir: Path) -> str:
        """Load the release's pinned graph and make the runtime serve it."""
        if self._serving_graph() is None:
            # No graph on the serving path: nothing to move, and loading one
            # into a runtime that was never built to serve it would invent a
            # serving path this deployment does not have. Recorded, not silent.
            return "not-serving"
        try:
            graph = load_pinned_graph(manifest, release_dir)
        except GraphArtifactError as exc:
            raise GraphRollbackError(
                f"release {manifest.release_id}'s pinned graph cannot be served: {exc}"
            ) from exc
        try:
            self._graphrag().update_graph(graph)
        except Exception as exc:
            raise GraphRollbackError(
                f"the runtime refused release {manifest.release_id}'s graph: {exc!r}"
            ) from exc
        logger.info(
            f"Graph rollback: runtime now serves release {manifest.release_id}'s "
            f"graph (sha256={manifest.graph_sha256[:16]}…, "
            f"schema_version={manifest.graph_schema_version})"
        )
        return "ok"

    # -- step E -------------------------------------------------------------

    def compensate(self, snapshot: GraphSnapshot) -> None:
        """Put the pre-rollback graph back, if the runtime still serves one."""
        current = self._serving_graph()
        if current is None or snapshot.graph is None or current is snapshot.graph:
            return
        try:
            self._graphrag().update_graph(snapshot.graph)
        except Exception as exc:
            raise GraphRollbackError(
                f"the runtime refused the graph it was serving before the rollback: "
                f"{exc!r}"
            ) from exc

    # -- internals ----------------------------------------------------------

    def _graphrag(self) -> Any:
        pipeline = self._pipeline()
        return getattr(pipeline, "_graphrag", None) if pipeline is not None else None

    def _serving_graph(self) -> Any:
        graphrag = self._graphrag()
        return getattr(graphrag, "_graph", None) if graphrag is not None else None

    def _verify(self, manifest: ReleaseManifest, release_dir: Path) -> None:
        try:
            verify_pinned_graph(manifest, release_dir)
        except GraphArtifactNotPinnedError:
            raise
        except GraphArtifactError as exc:
            raise TargetGraphNotPinnedError(
                f"release {manifest.release_id}'s pinned graph is not restorable: {exc}"
            ) from exc

# ── the orchestrator ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RollbackOutcome:
    """What a completed rollback did, step by step."""

    release_id: str
    previous_release_id: str
    lancedb_tag: str
    lancedb_tag_version: int | None
    lancedb_version: int | None
    #: The graph the runtime now serves, or ``None`` when it serves no graph.
    #: Carried on the outcome rather than only logged: "the rollback moved the
    #: graph too" is a claim about which bytes are being served, and a claim
    #: belongs in the returned record.
    graph_sha256: str | None
    graph_schema_version: str | None
    steps: Mapping[str, str]
    event_id: str | None


class AdminRollbackService:
    """Roll canonical + retrieval back to the last good release, together."""

    def __init__(
        self,
        *,
        releases_root: str | Path,
        retrieval: RetrievalRollback,
        graph: GraphRollback,
        ledger: AuditLedger,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._releases_root = Path(releases_root)
        self._retrieval = retrieval
        # Required, not optional — the same argument the ledger is required
        # under. An optional graph half would default to "the graph never
        # moves", which is the defect this level exists to remove, and it would
        # default there *silently*: the rollback would report success while
        # answering from the previous release's graph.
        self._graph = graph
        # Required, not optional: an unaudited rollback is the one thing this
        # ticket exists to make impossible, so there is no construction in which
        # one could happen.
        self._ledger = ledger
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # -- wiring surface ------------------------------------------------------

    @property
    def releases_root(self) -> Path:
        return self._releases_root

    @property
    def pointer_path(self) -> Path:
        return self._releases_root / POINTER_FILENAME

    # -- the rollback --------------------------------------------------------

    def rollback(
        self,
        *,
        reason: str,
        actor_id: str,
        role: str | None = None,
        target_release_id: str | None = None,
    ) -> RollbackOutcome:
        """Roll both levels back to ``target_release_id``, or to the lineage's
        previous release.

        The caller is the authenticated actor (``actor_id``/``role`` come from
        the credential, never from the body) and holds the swap lock.
        """
        reason = (reason or "").strip()
        if not reason:
            raise RollbackValidationError(
                "rollback requires a reason — a rollback without one is not accountable"
            )

        # ---- Step A: validate everything. Nothing below this block mutates. ----
        current_release_id = self._read_current_release_id()
        target = self._resolve_target(current_release_id, target_release_id)
        if target == current_release_id:
            raise TargetIsActiveError(
                f"release {target} is the one already active — a rollback to it "
                "would report a change that did not happen"
            )
        manifest = self._read_target_manifest(target)
        tag = release_tag(target)
        if manifest.lancedb_tag and manifest.lancedb_tag != tag:
            raise ReleaseManifestMismatchError(
                f"release {target}'s manifest names LanceDB tag "
                f"'{manifest.lancedb_tag}' but a rollback restores '{tag}' — the "
                "release was published against a different tagging convention"
            )
        tag_version = self._retrieval.validate(manifest)
        # The graph level is validated in the same read-only block, and for the
        # same reason: a release whose pinned graph is missing, rebuilt or
        # unreadable must fail here, with the pointer and the table untouched.
        target_dir = self._release_dir(target)
        self._graph.validate(manifest, target_dir)

        # ---- Step B: the compensation inputs. ----
        snapshot = self._retrieval.snapshot(release_id=current_release_id)
        graph_snapshot = self._graph.snapshot(release_id=current_release_id)
        steps: dict[str, str] = {"validate": "ok", "snapshot": "ok"}

        # ---- Step C: data. ----
        try:
            publish_pointer(self.pointer_path, target)
        except Exception as exc:
            # Retrieval has not moved and the pointer's own atomic write means it
            # is either the old value or the new one, never half of each.
            raise RollbackAbortedError(stage="canonical", canonical_error=exc) from exc
        steps["canonical"] = "ok"

        # ---- Step D: retrieval. ----
        try:
            lancedb_version = self._retrieval.apply(manifest)
        except Exception as retrieval_error:
            # ---- Step E: compensate, in reverse order. ----
            compensation_error = self._compensate(
                current_release_id, snapshot, graph_snapshot
            )
            raise RollbackAbortedError(
                stage="retrieval",
                retrieval_error=retrieval_error,
                compensation_error=compensation_error,
            ) from retrieval_error
        steps["retrieval"] = "ok"

        # ---- Step D2: the graph. Same unit as D — a release serves its table
        # and its graph together, and a graph left on the previous release
        # injects that release's chunk ids into the target's answers. It runs
        # after D because D's failure is the one that needs the graph still
        # untouched to compensate cleanly, and it compensates D in turn if it
        # fails.
        try:
            steps["graph"] = self._graph.apply(manifest, target_dir)
        except Exception as graph_error:
            compensation_error = self._compensate(
                current_release_id, snapshot, graph_snapshot
            )
            raise RollbackAbortedError(
                stage="graph",
                graph_error=graph_error,
                compensation_error=compensation_error,
            ) from graph_error

        # ---- Audit, once both levels have actually moved. ----
        event = AuditEvent(
            action=AuditAction.ROLLBACK,
            actor_id=actor_id,
            role=role,
            target=target,
            target_kind=TargetKind.RELEASE,
            before_version=current_release_id,
            after_version=target,
            reason=reason,
            release_id=target,
            timestamp=self._clock(),
        )
        entry = self._append_event(
            event,
            release_id=target,
            previous_release_id=current_release_id,
            actor_id=actor_id,
        )
        steps["audit"] = "ok"

        logger.info(
            f"Rollback complete: {current_release_id} -> {target} by {actor_id} "
            f"(lancedb tag {tag} @ version {lancedb_version})"
        )
        return RollbackOutcome(
            release_id=target,
            previous_release_id=current_release_id,
            lancedb_tag=tag,
            lancedb_tag_version=tag_version,
            lancedb_version=lancedb_version,
            graph_sha256=manifest.graph_sha256 if steps["graph"] == "ok" else None,
            graph_schema_version=(
                manifest.graph_schema_version if steps["graph"] == "ok" else None
            ),
            steps=steps,
            event_id=entry.event.event_id if entry is not None else None,
        )

    # -- steps ---------------------------------------------------------------

    def _compensate(
        self,
        previous_release_id: str,
        snapshot: RetrievalSnapshot,
        graph_snapshot: GraphSnapshot,
    ) -> str | None:
        """Undo a failed attempt; returns a description of what could not be undone.

        Reverse order, so the graph goes back before the table does: the
        graph is what maps entities to chunk ids, and a graph restored against
        a table that is still mid-move would be briefly paired with the wrong
        release. The pointer goes back last, after every served artifact has.
        """
        problems: list[str] = []
        try:
            self._graph.compensate(graph_snapshot)
        except Exception as exc:
            problems.append(f"graph: {exc!r}")
        try:
            self._retrieval.compensate(snapshot)
        except Exception as exc:
            problems.append(f"retrieval: {exc!r}")
        try:
            publish_pointer(self.pointer_path, previous_release_id)
        except Exception as exc:
            problems.append(f"canonical pointer: {exc!r}")
        return "; ".join(problems) if problems else None

    def _append_event(
        self,
        event: AuditEvent,
        *,
        release_id: str,
        previous_release_id: str,
        actor_id: str,
    ) -> AuditEntry:
        try:
            return self._ledger.append(event)
        except Exception as exc:
            raise RollbackAuditError(
                release_id=release_id,
                previous_release_id=previous_release_id,
                actor_id=actor_id,
                ledger_path=str(getattr(self._ledger, "path", "") or "") or None,
                cause=exc,
            ) from exc

    # -- reading the store ---------------------------------------------------

    def _read_current_release_id(self) -> str:
        pointer = self.pointer_path
        try:
            raw = pointer.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise NoPreviousReleaseError(
                f"no active-release pointer at {pointer} — there is no release to "
                "roll back from"
            ) from exc
        except OSError as exc:
            raise RollbackValidationError(
                f"active-release pointer {pointer} is unreadable: {exc!r}"
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise RollbackValidationError(
                f"active-release pointer {pointer} is not valid JSON: {exc!r}"
            ) from exc
        release_id = payload.get("release_id") if isinstance(payload, Mapping) else None
        if not isinstance(release_id, str) or not release_id.strip():
            raise RollbackValidationError(
                f"active-release pointer {pointer} names no release"
            )
        return release_id.strip()

    def _resolve_target(self, current_release_id: str, explicit: str | None) -> str:
        """The release to roll back to: the caller's, or the lineage's previous.

        ``None`` means the caller named no target, so the lineage decides. A
        string that is present but blank is a *supplied* target that cannot name
        a directory: it is refused rather than read as "no target", because a
        caller who asked for release ``"  "`` must not receive a rollback to
        some other release the lineage happened to offer. That fall-through
        returned 200 with a different release than the request named.
        """
        if explicit is not None:
            # Checked here, before it is ever joined onto the releases root.
            # ``_release_dir`` refuses the blank id this branch exists to catch.
            self._release_dir(explicit)
            return explicit.strip()

        candidate = self._parent_of(current_release_id)
        seen: list[str] = []
        hops = 0
        while candidate and hops < MAX_LINEAGE_HOPS and candidate not in seen:
            seen.append(candidate)
            if self._manifest_path(candidate).is_file():
                return candidate
            candidate = self._parent_of(candidate)
            hops += 1
        raise NoPreviousReleaseError(
            f"release {current_release_id} has no readable previous release in its "
            "lineage — pass target_release_id to roll back further than one release"
        )

    def _parent_of(self, release_id: str) -> str | None:
        try:
            manifest = read_manifest(self._release_dir(release_id))
        except Exception:
            return None
        parent = manifest.parent_release_id
        return parent.strip() if isinstance(parent, str) and parent.strip() else None

    def _read_target_manifest(self, release_id: str) -> ReleaseManifest:
        path = self._manifest_path(release_id)
        try:
            manifest = read_manifest(self._release_dir(release_id))
        except FileNotFoundError as exc:
            raise ReleaseNotFoundError(
                f"no release manifest at {path} — release {release_id} is not "
                "published here and cannot be rolled back to"
            ) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise RollbackValidationError(
                f"release manifest {path} is unreadable: {exc!r}"
            ) from exc
        if manifest.release_id != release_id:
            raise ReleaseManifestMismatchError(
                f"release manifest {path} names release '{manifest.release_id}' — a "
                "manifest that disagrees with its own directory describes a release "
                "nobody can reproduce"
            )
        return manifest

    def _manifest_path(self, release_id: str) -> Path:
        return self._release_dir(release_id) / MANIFEST_FILENAME

    def _release_dir(self, release_id: str) -> Path:
        """``<releases_root>/<release_id>``, refusing ids that could escape it.

        ``target_release_id`` crosses the HTTP boundary, so a release id is
        checked before it is joined: an id with a separator, a drive letter or a
        Windows-illegal character is refused rather than resolved.
        """
        candidate = (release_id or "").strip()
        unsafe = (
            not candidate
            or candidate in {".", ".."}
            or any(ch in _UNSAFE_ID_CHARS for ch in candidate)
            or any(sep and sep in candidate for sep in _SEPARATORS)
            or Path(candidate).name != candidate
        )
        if unsafe:
            raise InvalidReleaseIdError(
                f"'{release_id}' is not a usable release id — a release id names one "
                "directory under the releases root"
            )
        return self._releases_root / candidate
