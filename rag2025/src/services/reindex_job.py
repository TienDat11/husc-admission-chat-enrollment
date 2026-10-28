"""Incremental re-index job — ticket 15.

Two triggers, one job
---------------------
The outbox (``src/services/reindex_outbox.py``) names the canonical ids a
publication changed: a **record-level** trigger. It cannot see a parser or
chunker change — ticket 32/33 altered ``scripts/chunk_table_aware_v2.py`` so
that chunk *metadata* changes for records whose canonical content did not, and
those chunk ids do not move. So the job reads a second trigger off the corpus
itself: the stamp (ticket 06's ``version_stamp()``) the corpus was built with.

The rule, in one place (:func:`plan_reindex`)::

    built stamp is absent            -> FULL   (cannot prove the corpus is current)
    built stamp != current stamp     -> FULL   (the code changed under the corpus)
    built stamp == current stamp     -> INCREMENTAL over the affected ids

Absent counts as stale, never as "unknown, proceed". The served LanceDB table
carries no stamp at all today, and a table that cannot name the chunking logic
it was built by is exactly the corpus that must be rebuilt whole — reading that
as "no ids changed, do nothing" is the silent divergence this ticket closes.
The full-rebuild decision is therefore automatic: it is a comparison of two
stamps, not something an operator has to remember.

What the job writes
-------------------
A corpus directory — the artifacts the serving index is built from:

``chunks.jsonl``
    ticket 06's chunk records, one JSON object per line.
``vectors.jsonl``
    ``{"chunk_id", "content_hash", "vector"}`` — the embedded half.
``graph_edges.jsonl``
    the corpus graph's edges, one per line, each carrying the ``record_ids``
    that contributed it. That provenance is what makes a re-index scoped: the
    edges of affected records are rebuilt, and every other line is copied
    through byte-for-byte.
``chunk_manifest.json``
    the corpus stamp, written LAST so a reader that keys on the manifest never
    sees one describing chunks that are not in place yet.

The job builds into a staging directory and publishes by ``.tmp`` + ``os.replace``
per file, so a failure before publish leaves the serving corpus byte-identical.
Cross-file atomicity for a *reader* is ticket 16's mechanism (``publish_pointer``
+ the reload seam); this job never opens a second write path into the index — it
hands over a finished corpus.

Job state machine
-----------------
``pending`` → ``running`` → ``failed`` | ``active``. Nothing else.

``saved`` is deliberately **not** a state. A job whose artifacts are on disk in
the staging area has persisted a build; it is not serving it. Only ``active``
means the corpus in the serving directory is the one this job produced, and
:attr:`ReindexJob.serving` answers that question from the state alone.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from loguru import logger

from src.domain.canonical.record import CanonicalRecord
from src.domain.canonical.semantic_units import MA_NGANH, MOC_THOI_GIAN, TO_HOP, find_atomic_spans
from src.services.canonical_chunk_builder import (
    CHUNK_MANIFEST_FILENAME,
    CHUNKS_FILENAME,
    build_chunk_index,
)
from src.services.chunk_identity import version_stamp
from src.services.reindex_outbox import OutboxError, ReindexOutbox

VECTORS_FILENAME = "vectors.jsonl"
GRAPH_EDGES_FILENAME = "graph_edges.jsonl"

#: Files a corpus directory is made of. The manifest is last: it is the marker
#: that says "this corpus is complete, and this is the logic that built it".
CORPUS_FILENAMES = (CHUNKS_FILENAME, VECTORS_FILENAME, GRAPH_EDGES_FILENAME, CHUNK_MANIFEST_FILENAME)

#: Atom kinds the corpus graph's entity edges are derived from, mapped to the
#: ``EntityType`` value of the node they point at. Read off the domain's own
#: constants so a new atom shape shows up as a missing mapping, not as a silent
#: second vocabulary.
_ATOM_ENTITY_TYPES = {
    MA_NGANH: "NGANH",
    TO_HOP: "TO_HOP",
    MOC_THOI_GIAN: "THOI_GIAN",
}

RELATION_HAS_CHUNK = "HAS_CHUNK"
RELATION_HAS_ATOM = "HAS_ATOM"

class ReindexError(Exception):
    """Base class for re-index failures."""

class JobStateError(ReindexError):
    """A state transition the machine does not allow."""

class CorpusError(ReindexError):
    """The corpus directory is not readable as a corpus."""

class JobState(str, Enum):
    """Lifecycle of one re-index run. ``saved`` is not one of these, on purpose."""

    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    ACTIVE = "active"

    @property
    def is_serving(self) -> bool:
        """Whether this state means "the serving index is what this job built"."""
        return self is JobState.ACTIVE

#: The whole machine. ``failed`` and ``active`` are terminal; a job runs once,
#: so a retry is a new job rather than a resurrected one.
_ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PENDING: frozenset({JobState.RUNNING}),
    JobState.RUNNING: frozenset({JobState.FAILED, JobState.ACTIVE}),
    JobState.FAILED: frozenset(),
    JobState.ACTIVE: frozenset(),
}

class ReindexMode(str, Enum):
    """How much of the corpus a run rebuilds."""

    FULL = "full"
    INCREMENTAL = "incremental"

@dataclass(frozen=True)
class CorpusStamp:
    """The chunking logic a corpus was built by — ticket 06's ``version_stamp()``.

    Not a content hash and not a timestamp: a literal that is identical across
    runs on the same code and changes when the logic changes. Comparing two of
    these answers "is this corpus still what the code would produce?", which is
    the question a record-level trigger cannot ask.
    """

    chunker_version: str
    unit_rules_version: str
    chunk_manifest_version: int

    @classmethod
    def current(cls) -> "CorpusStamp":
        """The stamp the code running right now would produce."""
        return cls.from_mapping(version_stamp())  # type: ignore[return-value]

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "CorpusStamp | None":
        """The stamp ``data`` states, or ``None`` if it does not state one whole.

        A partial stamp is ``None``, never a defaulted one: filling in a missing
        ``unit_rules_version`` would make a corpus built by older rules compare
        equal to the current code, which is the failure mode the stamp exists to
        catch.
        """
        if not isinstance(data, Mapping):
            return None
        chunker = data.get("chunker_version")
        rules = data.get("unit_rules_version")
        manifest_version = data.get("chunk_manifest_version")
        if not isinstance(chunker, str) or not chunker:
            return None
        if not isinstance(rules, str) or not rules:
            return None
        if not isinstance(manifest_version, int) or isinstance(manifest_version, bool):
            return None
        return cls(
            chunker_version=chunker,
            unit_rules_version=rules,
            chunk_manifest_version=manifest_version,
        )

    @classmethod
    def from_manifest(cls, path: str | Path) -> "CorpusStamp | None":
        """The stamp a corpus manifest states, or ``None`` (absent/unreadable/partial)."""
        target = Path(path)
        if target.is_dir():
            target = target / CHUNK_MANIFEST_FILENAME
        if not target.is_file():
            return None
        try:
            return cls.from_mapping(json.loads(target.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, object]]) -> "CorpusStamp | None":
        """The stamp a set of stored rows agrees on, else ``None``.

        Rows carry it under ``metadata_json["corpus_stamp"]``. Every row must
        carry one and they must all be equal: a table where some rows predate
        the stamp cannot be described by a single stamp, and a table with no
        stamped row at all — the served table today, 1679 legacy rows — states
        nothing. Both are ``None``, and ``None`` is stale.
        """
        seen: CorpusStamp | None = None
        count = 0
        for row in rows:
            count += 1
            meta = row.get("metadata_json")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    return None
            if not isinstance(meta, Mapping):
                return None
            stamp = cls.from_mapping(meta.get("corpus_stamp"))  # type: ignore[arg-type]
            if stamp is None:
                return None
            if seen is None:
                seen = stamp
            elif seen != stamp:
                return None
        if count == 0:
            return None
        return seen

    def to_dict(self) -> dict:
        return {
            "chunker_version": self.chunker_version,
            "unit_rules_version": self.unit_rules_version,
            "chunk_manifest_version": self.chunk_manifest_version,
        }

    def matches(self, other: "CorpusStamp | None") -> bool:
        """Whether ``other`` is this same stamp. ``None`` never matches."""
        return other is not None and other == self

def read_corpus_stamp(corpus_dir: str | Path) -> CorpusStamp | None:
    """The stamp of the corpus at ``corpus_dir``, or ``None`` if it states none."""
    return CorpusStamp.from_manifest(Path(corpus_dir) / CHUNK_MANIFEST_FILENAME)

def stamp_is_stale(
    built: CorpusStamp | None, current: CorpusStamp | None = None
) -> bool:
    """Whether a corpus built under ``built`` is stale against ``current``.

    The one comparison the ticket is about, and it is deliberately asymmetric:
    a corpus that states no stamp is stale. There is no third answer — the
    alternative ("unknown, so assume current") is the silent divergence itself.

    "States no stamp" is not hypothetical. The corpus served today —
    ``husc_chunks``, 1679 rows — was produced by ``scripts/chunk_table_aware_v2.py``,
    which carries no version constant at all: every one of those rows has no
    ``corpus_stamp``, and none has a ``record_version_key``. A change to that
    script therefore cannot be seen by any record-level trigger, and the only
    thing that can see it is this rule. It fires on the real table today, which
    is what makes the ticket's gap closed rather than described.
    """
    if built is None:
        return True
    return not built.matches(current if current is not None else CorpusStamp.current())

@dataclass(frozen=True)
class ReindexPlan:
    """What a run will do, and why — the stale decision is a field, not a log line."""

    mode: ReindexMode
    affected_record_ids: tuple[str, ...]
    stale: bool
    reason: str

    @property
    def is_noop(self) -> bool:
        """An incremental run with no affected ids has nothing to re-index."""
        return self.mode is ReindexMode.INCREMENTAL and not self.affected_record_ids

    @property
    def forces_full(self) -> bool:
        return self.mode is ReindexMode.FULL

def plan_reindex(
    *,
    built_stamp: CorpusStamp | None,
    current_stamp: CorpusStamp | None = None,
    affected_record_ids: Iterable[str] = (),
) -> ReindexPlan:
    """Decide full vs incremental from the corpus stamp and the affected ids.

    The stale branch ignores ``affected_record_ids`` entirely — that is the
    point. A stamp bump with an EMPTY id set still forces a full re-index,
    because the ids are a record-level signal and the stamp is the only thing
    that can see a parser change.
    """
    affected = tuple(sorted({str(i) for i in affected_record_ids if str(i)}))
    current = current_stamp if current_stamp is not None else CorpusStamp.current()
    if built_stamp is None:
        return ReindexPlan(
            mode=ReindexMode.FULL,
            affected_record_ids=affected,
            stale=True,
            reason="corpus-unstamped",
        )
    if not built_stamp.matches(current):
        return ReindexPlan(
            mode=ReindexMode.FULL,
            affected_record_ids=affected,
            stale=True,
            reason=(
                f"stamp-mismatch: corpus built by chunker "
                f"{built_stamp.chunker_version}/rules {built_stamp.unit_rules_version}, "
                f"code now {current.chunker_version}/{current.unit_rules_version}"
            ),
        )
    return ReindexPlan(
        mode=ReindexMode.INCREMENTAL,
        affected_record_ids=affected,
        stale=False,
        reason="stamp-match",
    )

@dataclass(frozen=True)
class ReindexResult:
    """What one run did, counted in chunk ids rather than in "less than before"."""

    plan: ReindexPlan
    state: JobState
    staged_dir: Path | None
    embedded_chunk_ids: tuple[str, ...]
    reused_chunk_ids: tuple[str, ...]
    removed_chunk_ids: tuple[str, ...]
    changed_chunk_ids: tuple[str, ...]
    written_chunk_ids: tuple[str, ...]
    edge_keys_written: tuple[str, ...]
    edge_keys_removed: tuple[str, ...]
    #: Whether the outbox cursor moved. ``False`` means the corpus was published
    #: and is serving, but the entries it consumed stay pending for the next run.
    outbox_cursor_advanced: bool = True

    @property
    def embedded_count(self) -> int:
        return len(self.embedded_chunk_ids)

    @property
    def serving(self) -> bool:
        return self.state.is_serving

@dataclass
class _Built:
    """The staged artifacts of one build, before they are published."""

    staged_dir: Path
    chunks: list[dict] = field(default_factory=list)
    embedded: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    edges_written: list[str] = field(default_factory=list)
    edges_removed: list[str] = field(default_factory=list)

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]

class ReindexJob:
    """One run of the re-index: plan, build aside, publish on success.

    ``records`` is the published corpus to index — the materialized records of
    the release being served, as a sequence or as the path of a ticket-05
    ``materialized_records.jsonl``. ``corpus_dir`` is the serving corpus this
    job reads the built stamp from and publishes into.

    ``publish`` overrides where the finished corpus goes (``publish(staged_dir,
    corpus_dir)``); the default swaps it into ``corpus_dir``. It exists so a
    caller whose serving corpus lives elsewhere — a release directory, or an
    index built from the corpus — hands the staged artifacts over without this
    job acquiring a second write path.
    """

    def __init__(
        self,
        *,
        records: Iterable[CanonicalRecord] | str | Path,
        corpus_dir: str | Path,
        release_id: str,
        embedder: Embedder,
        outbox: ReindexOutbox | None = None,
        affected_record_ids: Iterable[str] | None = None,
        staging_root: str | Path | None = None,
        publish: Callable[[Path, Path], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._records = records
        self._corpus_dir = Path(corpus_dir)
        self._release_id = (release_id or "").strip()
        if not self._release_id:
            raise ValueError(
                "ReindexJob requires a release_id — chunks that cannot be traced "
                "to a release cannot be re-indexed or rolled back"
            )
        self._embedder = embedder
        self._outbox = outbox
        self._explicit_affected = (
            None if affected_record_ids is None else tuple(sorted(set(affected_record_ids)))
        )
        # Staging is a sibling of the serving corpus, never inside it: the
        # serving directory is the artifact set a reader enumerates.
        self._staging_root = (
            Path(staging_root)
            if staging_root
            else self._corpus_dir.parent / f"{self._corpus_dir.name}.staging"
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._publish_hook = publish

        self._state = JobState.PENDING
        self._saved = False
        self._staged_dir: Path | None = None
        self._error: BaseException | None = None

    # ── state ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def saved(self) -> bool:
        """Whether a build is persisted in the staging area.

        Deliberately not a state and deliberately not consulted by
        :attr:`serving`: a saved build is a build that exists, not one the
        serving index is answering from.
        """
        return self._saved

    @property
    def serving(self) -> bool:
        """Whether the serving corpus is the one this job built. True only when active."""
        return self._state.is_serving

    @property
    def staged_dir(self) -> Path | None:
        return self._staged_dir

    @property
    def error(self) -> BaseException | None:
        return self._error

    def _transition(self, to: JobState) -> None:
        allowed = _ALLOWED_TRANSITIONS[self._state]
        if to not in allowed:
            raise JobStateError(
                f"cannot move a re-index job from {self._state.value} to {to.value} "
                f"(allowed: {sorted(s.value for s in allowed) or 'none'})"
            )
        self._state = to

    # ── plan ───────────────────────────────────────────────────────────────

    def affected_record_ids(self) -> tuple[str, ...]:
        """The canonical ids this run re-indexes.

        An explicit set wins (a caller with its own trigger, or an admin asking
        for named ids); otherwise the outbox's undrained entries decide.
        """
        if self._explicit_affected is not None:
            return self._explicit_affected
        if self._outbox is not None:
            return self._outbox.pending_affected_record_ids()
        return ()

    def plan(self) -> ReindexPlan:
        """The plan for this run, computed fresh from the corpus on disk."""
        return plan_reindex(
            built_stamp=read_corpus_stamp(self._corpus_dir),
            current_stamp=CorpusStamp.current(),
            affected_record_ids=self.affected_record_ids(),
        )

    # ── run ────────────────────────────────────────────────────────────────

    def run(self) -> ReindexResult:
        """Build, then publish. A failure publishes nothing and leaves the job ``failed``.

        The outbox cursor advances only after a successful publish, so a failed
        run is retried with the same entries rather than losing them.
        """
        self._transition(JobState.RUNNING)
        try:
            plan = self.plan()
            built = (
                _Built(staged_dir=self._staging_dir())
                if plan.is_noop
                else self._build(plan)
            )
            if not plan.is_noop:
                self._hand_over(built)
        except BaseException as exc:  # noqa: BLE001 - state must reflect any failure
            self._error = exc
            self._transition(JobState.FAILED)
            raise
        self._transition(JobState.ACTIVE)
        drained = True
        if self._outbox is not None and not plan.is_noop:
            # The corpus is published and serving at this point. A cursor that
            # failed to advance is the benign half of the failure pair: the
            # entries stay pending, so the next run sees the same versions again
            # and re-indexing them is a no-op (chunk ids are content-addressed).
            # Raising here would tell the caller the re-index failed when the
            # serving corpus is the one this job built.
            try:
                self._outbox.drain()
            except OutboxError as exc:
                drained = False
                logger.warning(
                    f"Corpus published and serving, but the outbox cursor did not "
                    f"advance: {exc}"
                )
        return ReindexResult(
            plan=plan,
            state=self._state,
            staged_dir=self._staged_dir if not plan.is_noop else None,
            embedded_chunk_ids=tuple(built.embedded),
            reused_chunk_ids=tuple(built.reused),
            removed_chunk_ids=tuple(built.removed),
            changed_chunk_ids=tuple(built.changed),
            written_chunk_ids=tuple(c["id"] for c in built.chunks),
            edge_keys_written=tuple(built.edges_written),
            edge_keys_removed=tuple(built.edges_removed),
            outbox_cursor_advanced=drained,
        )

    def _staging_dir(self) -> Path:
        stamp = self._clock().strftime("%Y%m%dT%H%M%S%f")
        return self._staging_root / f"job-{stamp}"

    def _load_records(self) -> tuple[CanonicalRecord, ...]:
        from src.services.canonical_chunk_builder import load_materialized_records

        if isinstance(self._records, (str, Path)):
            return load_materialized_records(self._records)
        if isinstance(self._records, Mapping):
            return tuple(self._records.values())
        return tuple(self._records)

    def _build(self, plan: ReindexPlan) -> _Built:
        """Produce the staged corpus. Nothing outside the staging dir is touched."""
        staged = self._staging_dir()
        staged.mkdir(parents=True, exist_ok=True)
        built = _Built(staged_dir=staged)

        records = self._load_records()
        if plan.mode is ReindexMode.FULL:
            chunks = self._chunk_all(records, staged)
            if not chunks:
                raise ReindexError(
                    "full re-index produced no chunks — refusing to publish an "
                    "empty corpus over the serving one"
                )
        else:
            chunks = self._chunk_incremental(records, plan.affected_record_ids, staged)

        old_chunks, old_vectors, old_edge_lines = _read_serving_corpus(self._corpus_dir)
        old_vectors_by_id = {v["chunk_id"]: v for v in old_vectors}

        kept_ids = {c["id"] for c in chunks}
        built.removed = sorted(set(old_vectors_by_id) - kept_ids)

        vectors, embedded, reused, changed = self._vectors_for(
            chunks, old_chunks, old_vectors_by_id
        )
        built.chunks = chunks
        built.embedded, built.reused, built.changed = embedded, reused, changed

        edges, written, removed = _merge_edges(
            old_edge_lines, chunks, plan.affected_record_ids, mode=plan.mode
        )
        built.edges_written, built.edges_removed = written, removed

        _write_lines(staged / CHUNKS_FILENAME, (c for c in chunks))
        _write_lines(staged / VECTORS_FILENAME, vectors)
        # Carried edge lines are written as the exact bytes they were read as.
        _write_lines(staged / GRAPH_EDGES_FILENAME, (raw for _, raw in edges))
        _write_json(staged / CHUNK_MANIFEST_FILENAME, self._manifest(chunks, plan))
        self._staged_dir = staged
        self._saved = True
        logger.info(
            f"Re-index staged ({plan.mode.value}, {plan.reason}): "
            f"{len(chunks)} chunk(s), {len(embedded)} embedded, {len(reused)} reused, "
            f"{len(built.removed)} removed → {staged}"
        )
        return built

    def _chunk_all(self, records: Sequence[CanonicalRecord], staged: Path) -> list[dict]:
        """Every record's chunks — the full-rebuild half of a run."""
        if not records:
            return []
        return list(build_chunk_index(records, staged / "chunk_build", self._release_id).chunks)

    def _chunk_incremental(
        self,
        records: Sequence[CanonicalRecord],
        affected: Sequence[str],
        staged: Path,
    ) -> list[dict]:
        """Affected records re-cut; every other record's chunks carried over.

        A record that is affected but no longer among the published records
        (archived, superseded away, deleted) contributes nothing, so its chunks
        drop out of the corpus — which is the other half of "only what changed".
        """
        affected_set = set(affected)
        old_chunks, _, _ = _read_serving_corpus(self._corpus_dir)
        carried = [c for c in old_chunks if c["metadata"]["record_id"] not in affected_set]

        fresh_records = [r for r in records if r.record_id in affected_set]
        if not fresh_records:
            return sorted(carried, key=_chunk_sort_key)
        fresh = list(
            build_chunk_index(fresh_records, staged / "chunk_build", self._release_id).chunks
        )
        return sorted(carried + fresh, key=_chunk_sort_key)

    def _vectors_for(
        self,
        chunks: Sequence[dict],
        old_chunks: Sequence[dict],
        old_vectors_by_id: Mapping[str, dict],
    ) -> tuple[list[dict], list[str], list[str], list[str]]:
        """Embed exactly what changed; reuse what did not.

        Two reuse paths, both identity-preserving in the sense that matters: the
        vector handed to a chunk is the vector of *that chunk's text*.

        * by chunk id — the same id with the same ``content_hash``. This is the
          common case and costs one dict lookup.
        * by content hash — the text is byte-identical under
          ``normalize_for_hash``, so the vector is the same vector. Ticket 06's
          chunk id is ``sha256(record_version_key + unit)``, so a new record
          version moves every one of its chunk ids even when a given unit's text
          did not change; without this path, publishing a version that only
          corrects one field would re-embed the record's whole table too.

        The consequence is that the re-embed count is the number of chunks whose
        TEXT changed — never "the record's chunk count" and never "less than
        before". A metadata-only chunker change (ticket 32/33's rowspan
        attribution) therefore costs zero embeddings, and a version bump that
        changes nothing costs zero as well.
        """
        by_hash: dict[str, object] = {}
        for chunk in sorted(old_chunks, key=_chunk_sort_key):
            entry = old_vectors_by_id.get(chunk["id"])
            if entry is not None and entry["content_hash"] == chunk["content_hash"]:
                by_hash.setdefault(chunk["content_hash"], entry["vector"])

        vectors: list[dict] = []
        embedded: list[str] = []
        reused: list[str] = []
        changed: list[str] = []
        to_embed: list[dict] = []

        old_by_id = {c["id"]: c for c in old_chunks}
        for chunk in chunks:
            previous = old_by_id.get(chunk["id"])
            if previous is not None and previous != chunk:
                changed.append(chunk["id"])
            entry = old_vectors_by_id.get(chunk["id"])
            if entry is not None and entry["content_hash"] == chunk["content_hash"]:
                vectors.append(
                    {
                        "chunk_id": chunk["id"],
                        "content_hash": chunk["content_hash"],
                        "vector": entry["vector"],
                    }
                )
                reused.append(chunk["id"])
                continue
            reusable = by_hash.get(chunk["content_hash"])
            if reusable is not None:
                vectors.append(
                    {
                        "chunk_id": chunk["id"],
                        "content_hash": chunk["content_hash"],
                        "vector": reusable,
                    }
                )
                reused.append(chunk["id"])
                continue
            to_embed.append(chunk)

        if to_embed:
            produced = self._embedder([c["text"] for c in to_embed])
            produced = [list(map(float, v)) for v in produced]
            if len(produced) != len(to_embed):
                raise ReindexError(
                    f"embedder returned {len(produced)} vector(s) for "
                    f"{len(to_embed)} chunk(s) — refusing to write a corpus whose "
                    "vectors and chunks disagree"
                )
            for chunk, vector in zip(to_embed, produced):
                vectors.append(
                    {
                        "chunk_id": chunk["id"],
                        "content_hash": chunk["content_hash"],
                        "vector": vector,
                    }
                )
                embedded.append(chunk["id"])

        vectors.sort(key=lambda v: v["chunk_id"])
        return vectors, sorted(embedded), sorted(reused), sorted(changed)

    def _manifest(self, chunks: Sequence[dict], plan: ReindexPlan) -> dict:
        """The corpus manifest: what is in the corpus and the stamp that built it."""
        stamp = version_stamp()
        record_ids = sorted({c["metadata"]["record_id"] for c in chunks})
        return {
            "chunk_manifest_version": stamp["chunk_manifest_version"],
            "chunker_version": stamp["chunker_version"],
            "unit_rules_version": stamp["unit_rules_version"],
            "release_id": self._release_id,
            "mode": plan.mode.value,
            "reason": plan.reason,
            "affected_record_ids": list(plan.affected_record_ids),
            "chunk_count": len(chunks),
            "record_count": len(record_ids),
            "record_ids": record_ids,
            "built_at": self._clock().astimezone(timezone.utc).isoformat(),
            "chunks_filename": CHUNKS_FILENAME,
            "vectors_filename": VECTORS_FILENAME,
            "graph_edges_filename": GRAPH_EDGES_FILENAME,
        }

    def _hand_over(self, built: _Built) -> None:
        """Publish the staged corpus — the job's last step, and its only write.

        The default is the file swap below. A caller may supply its own
        ``publish(staged_dir, corpus_dir)`` when the serving corpus lives
        somewhere else (a release directory, an index built from the corpus):
        that hand-over is the seam ticket 16's swap/reload attaches to, and it
        keeps this job from growing a second write path of its own.
        """
        if self._publish_hook is not None:
            self._publish_hook(built.staged_dir, self._corpus_dir)
        else:
            self._publish(built)

    def _publish(self, built: _Built) -> None:
        """Swap the staged corpus into the serving directory, manifest last.

        Every file is written to a ``.tmp`` sibling first, so a failure while
        staging the swap leaves the serving corpus untouched. The manifest is
        replaced last: a reader that keys on the manifest never sees one
        describing chunks that are not in place.

        Replacing four files is not one atomic act, so a failure part-way
        through the replaces restores the files already replaced from their
        previous bytes. A run that fails therefore leaves the serving corpus
        byte-identical — the ticket's "a failed job does not damage the index
        being served" — instead of leaving it half new and half old. Making the
        swap atomic *and instant* is ticket 16's job; it attaches at
        :meth:`_hand_over`.
        """
        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        order = (CHUNKS_FILENAME, VECTORS_FILENAME, GRAPH_EDGES_FILENAME, CHUNK_MANIFEST_FILENAME)
        staged: list[tuple[Path, Path]] = []
        try:
            for name in order:
                source = built.staged_dir / name
                target = self._corpus_dir / name
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_bytes(source.read_bytes())
                staged.append((tmp, target))
        except OSError:
            for tmp, _ in staged:
                tmp.unlink(missing_ok=True)
            raise

        previous = {target: target.read_bytes() for _, target in staged if target.is_file()}
        replaced: list[Path] = []
        try:
            for tmp, target in staged:
                os.replace(tmp, target)
                replaced.append(target)
        except OSError:
            for target in replaced:
                target.write_bytes(previous[target])
            for tmp, target in staged:
                if target not in replaced:
                    tmp.unlink(missing_ok=True)
            raise

def _chunk_sort_key(chunk: Mapping) -> tuple[str, str]:
    meta = chunk["metadata"]
    return (str(meta.get("record_version_key", "")), str(meta.get("unit_key", "")))

def _read_serving_corpus(
    corpus_dir: Path,
) -> tuple[list[dict], list[dict], list[tuple[dict, str]]]:
    """``(chunks, vectors, (edge, raw line))`` from the serving corpus, if any."""
    chunks = _read_lines(corpus_dir / CHUNKS_FILENAME)
    vectors = _read_lines(corpus_dir / VECTORS_FILENAME)
    edges: list[tuple[dict, str]] = []
    path = corpus_dir / GRAPH_EDGES_FILENAME
    if path.is_file():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{lineno}: {exc}") from exc
            edges.append((payload, line))
    return chunks, vectors, edges

def _read_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{lineno}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CorpusError(f"{path}:{lineno}: not a JSON object")
        out.append(payload)
    return out

def edges_for_record(record_id: str, chunks: Sequence[Mapping]) -> list[dict]:
    """The corpus graph's edges contributed by one record's chunks.

    Two relations, both derived from the chunk records themselves — no model, no
    NER, and deterministic given the corpus:

    * ``RECORD:<record_id> --HAS_CHUNK--> CHUNK:<chunk_id>`` — corpus membership,
      what "this record is served by these chunks" means;
    * ``CHUNK:<chunk_id> --HAS_ATOM--> <EntityType>:<value>`` — the atomic shapes
      (major code, subject combination, timeline milestone) the chunk states,
      read with the domain's own :func:`find_atomic_spans`.

    Edge keys are ``head|relation|tail``, the convention
    ``src/domain/graph.py`` uses, so a corpus edge and a graph edge name the same
    thing the same way.
    """
    record_node = f"RECORD:{record_id}"
    edges: list[dict] = []
    for chunk in sorted(chunks, key=_chunk_sort_key):
        chunk_id = str(chunk["id"])
        chunk_node = f"CHUNK:{chunk_id}"
        edges.append(
            {
                "head": record_node,
                "relation": RELATION_HAS_CHUNK,
                "tail": chunk_node,
                "weight": 1.0,
                "chunk_ids": [chunk_id],
                "record_ids": [record_id],
            }
        )
        seen: set[tuple[str, str]] = set()
        for span in find_atomic_spans(str(chunk.get("text", ""))):
            entity_type = _ATOM_ENTITY_TYPES.get(span.kind)
            if entity_type is None:
                continue
            pair = (entity_type, span.value)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(
                {
                    "head": chunk_node,
                    "relation": RELATION_HAS_ATOM,
                    "tail": f"{entity_type}:{span.value}",
                    "weight": 1.0,
                    "chunk_ids": [chunk_id],
                    "record_ids": [record_id],
                }
            )
    for edge in edges:
        edge["edge_key"] = f"{edge['head']}|{edge['relation']}|{edge['tail']}"
    return edges

def _merge_edges(
    old_lines: Sequence[tuple[dict, str]],
    chunks: Sequence[Mapping],
    affected: Sequence[str],
    *,
    mode: ReindexMode,
) -> tuple[list[tuple[dict, str]], list[str], list[str]]:
    """Rebuild affected records' edges; carry every other line through verbatim.

    ``mode=FULL`` treats every record as affected. The carried lines keep their
    original bytes — they are returned as the raw strings that were read, not
    re-serialized — which is what makes "unrelated edges untouched" a byte
    comparison rather than a hope.
    """
    affected_set = set(affected)
    if mode is ReindexMode.FULL:
        affected_set = {str(c["metadata"]["record_id"]) for c in chunks} | affected_set

    carried: list[tuple[dict, str]] = []
    dropped: list[str] = []
    for edge, raw in old_lines:
        record_ids = {str(r) for r in edge.get("record_ids", [])}
        if record_ids & affected_set:
            dropped.append(str(edge.get("edge_key", "")))
        else:
            carried.append((edge, raw))

    by_record: dict[str, list[dict]] = {}
    for chunk in chunks:
        record_id = str(chunk["metadata"]["record_id"])
        if record_id in affected_set:
            by_record.setdefault(record_id, []).append(chunk)

    fresh: list[dict] = []
    for record_id in sorted(by_record):
        fresh.extend(edges_for_record(record_id, by_record[record_id]))

    merged = carried + [(edge, json.dumps(edge, ensure_ascii=False, sort_keys=True)) for edge in fresh]
    merged.sort(key=lambda item: str(item[0].get("edge_key", "")))
    return merged, sorted(str(e["edge_key"]) for e in fresh), sorted(k for k in dropped if k)

def _write_lines(path: Path, payloads: Iterable[object]) -> None:
    """Write JSONL: mappings are dumped canonically, strings (raw lines) verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for payload in payloads:
            if isinstance(payload, str):
                fh.write(payload + "\n")
            else:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
