"""Read model for "what is serving, and what is behind it" — ticket 26.

Ticket 26's first two criteria are questions an admissions officer asks out
loud: *which release is running?* and *what is the one before it?* The answers
already exist on disk — the pointer ``AdminRollbackService`` publishes
(``<releases_root>/active_release.json``) and the ``parent_release_id`` lineage
the release manifests carry — and this module only **reads** them. There is no
second registry here, deliberately: a parallel record of "the active release"
would be a copy that drifts from the one the rollback path actually moves.

Two rules shape everything below.

**Nothing is inferred.** Every field is a byte that is on disk right now, or
``None``. A pointer that cannot be read is not reported as "no release active";
a release whose manifest is unreadable is not reported as a release with zero
documents. Those cases land in :attr:`ReleaseStatus.problems`, because the one
thing this screen must never do is answer a question it could not actually
answer.

**Nothing is claimed that was not read.** The manifest's ``lancedb_tag`` is what
the release *records* about itself; whether that tag still exists in LanceDB is
a question for the retrieval half of a rollback
(:class:`~src.services.admin_rollback.LanceDBRetrievalRollback`), not for a read
model over files. So it is reported as recorded and not as verified, and the
docstring says so rather than leaving a caller to guess.

The lineage walk mirrors :meth:`AdminRollbackService._resolve_target` — the same
"parent, if its manifest is on disk" one step, the same
:data:`~src.services.admin_rollback.MAX_LINEAGE_HOPS` bound — so the release the
screen calls *previous* is the release a targetless ``POST /admin/rollback``
would move to. It is written out here rather than called because that method is
private and ticket 26 does not own that module; the constants it is built from
are imported, so a change to the bound or the filenames reaches both.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain.release import ReleaseManifest
from src.infrastructure.canonical_store import read_manifest
from src.services.admin_rollback import (
    MANIFEST_FILENAME,
    MAX_LINEAGE_HOPS,
    POINTER_FILENAME,
)
from src.services.reindex_job import (
    CHUNK_MANIFEST_FILENAME,
    CorpusStamp,
    JobState,
    read_corpus_stamp,
    stamp_is_stale,
)

__all__ = [
    "IndexJobView",
    "IndexJobsService",
    "IndexJobsStatus",
    "ReleaseStatus",
    "ReleaseStatusService",
    "ReleaseView",
]


@dataclass(frozen=True)
class ReleaseView:
    """One release, as its own manifest states it.

    ``lancedb_tag``/``lancedb_version``/``created_at`` are ``None`` when the
    manifest records nothing for them — the dataclass defaults (``""``/``0``)
    are the schema's "not recorded", not a value, and the screen has to be able
    to say "không ghi nhận" instead of printing a zero.
    """

    release_id: str
    created_at: str | None
    parent_release_id: str | None
    lancedb_tag: str | None
    lancedb_version: int | None
    document_count: int
    chunk_count: int

    @classmethod
    def from_manifest(cls, manifest: ReleaseManifest) -> "ReleaseView":
        parent = manifest.parent_release_id
        tag = manifest.lancedb_tag
        return cls(
            release_id=manifest.release_id,
            created_at=manifest.created_at.isoformat() if manifest.created_at else None,
            parent_release_id=parent.strip() if isinstance(parent, str) and parent.strip() else None,
            lancedb_tag=tag.strip() if isinstance(tag, str) and tag.strip() else None,
            lancedb_version=manifest.lancedb_version or None,
            document_count=manifest.document_count,
            chunk_count=manifest.chunk_count,
        )


@dataclass(frozen=True)
class ReleaseStatus:
    """What the release screen renders, straight off disk.

    ``active_release_id`` is the pointer's answer and is present even when the
    release it names cannot be read — the pointer is what the serving path
    reads, so "the pointer says R2" is true whether or not R2's manifest
    survives. ``active`` is the manifest-backed view, ``None`` when unreadable.
    """

    active_release_id: str | None
    previous_release_id: str | None
    active: ReleaseView | None
    previous: ReleaseView | None
    problems: tuple[str, ...]


class ReleaseStatusService:
    """Reads the release pointer and its lineage. Never writes, never mutates."""

    def __init__(self, *, releases_root: str | Path) -> None:
        self._releases_root = Path(releases_root)

    @property
    def releases_root(self) -> Path:
        return self._releases_root

    @property
    def pointer_path(self) -> Path:
        return self._releases_root / POINTER_FILENAME

    def status(self) -> ReleaseStatus:
        """The active release, the previous one, and anything that could not be read."""
        problems: list[str] = []

        active_release_id, problem = self._read_pointer()
        if problem is not None:
            problems.append(problem)

        active: ReleaseView | None = None
        previous_release_id: str | None = None
        previous: ReleaseView | None = None

        if active_release_id is not None:
            active_manifest, problem = self._read_release_manifest(active_release_id)
            if problem is not None:
                problems.append(problem)
            if active_manifest is not None:
                active = ReleaseView.from_manifest(active_manifest)
                previous_release_id, previous, walk_problems = self._previous_of(
                    active_manifest
                )
                problems.extend(walk_problems)

        return ReleaseStatus(
            active_release_id=active_release_id,
            previous_release_id=previous_release_id,
            active=active,
            previous=previous,
            problems=tuple(problems),
        )

    # ── the pointer ─────────────────────────────────────────────────────────

    def _read_pointer(self) -> tuple[str | None, str | None]:
        """``(release_id, problem)`` — the pointer's answer, or why there is none.

        A missing pointer is ``(None, None)``: nothing has been published here
        yet, which is a normal state, not a fault. An unreadable or malformed
        one is ``(None, reason)``: the screen must be able to say "the pointer
        is broken" rather than the far more dangerous "nothing is published".
        """
        pointer = self.pointer_path
        try:
            raw = pointer.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, None
        except OSError as exc:
            return None, f"active-release pointer {pointer} is unreadable: {exc!r}"

        import json

        try:
            payload = json.loads(raw)
        except ValueError as exc:
            return None, f"active-release pointer {pointer} is not valid JSON: {exc!r}"
        if not isinstance(payload, dict):
            return None, f"active-release pointer {pointer} is not a JSON object"
        release_id = payload.get("release_id")
        if not isinstance(release_id, str) or not release_id.strip():
            return None, f"active-release pointer {pointer} names no release"
        return release_id.strip(), None

    # ── the lineage ─────────────────────────────────────────────────────────

    def _previous_of(
        self, active: ReleaseManifest
    ) -> tuple[str | None, ReleaseView | None, list[str]]:
        """The release one step back, or ``None`` with the reason why not.

        Same walk as the rollback path's target resolution: the parent, if its
        manifest is on disk. A parent that is named but absent is reported —
        it is exactly the state in which "roll back one step" has nothing to
        move to, and silence about it would read as "there is nothing behind
        this release", which is a different statement.
        """
        problems: list[str] = []
        candidate = _clean_id(active.parent_release_id)
        if candidate is None:
            return None, None, problems

        seen: list[str] = []
        hops = 0
        while candidate and hops < MAX_LINEAGE_HOPS and candidate not in seen:
            seen.append(candidate)
            release_dir = self._release_dir(candidate)
            if release_dir is None:
                problems.append(
                    f"release {active.release_id} names parent {candidate!r}, which is "
                    "not a usable release id"
                )
                return None, None, problems
            if not (release_dir / MANIFEST_FILENAME).is_file():
                problems.append(
                    f"release {active.release_id} names parent {candidate}, but "
                    f"{release_dir / MANIFEST_FILENAME} does not exist — there is no "
                    "release one step back to roll back to"
                )
                return None, None, problems
            manifest, problem = self._read_release_manifest(candidate)
            if manifest is None:
                problems.append(problem or f"release {candidate} is unreadable")
                return None, None, problems
            return candidate, ReleaseView.from_manifest(manifest), problems

        return None, None, problems

    # ── reading one release ─────────────────────────────────────────────────

    def _read_release_manifest(
        self, release_id: str
    ) -> tuple[ReleaseManifest | None, str | None]:
        """``(manifest, problem)`` for one release directory."""
        release_dir = self._release_dir(release_id)
        if release_dir is None:
            return None, f"release id {release_id!r} is not a usable release id"
        try:
            manifest = read_manifest(release_dir)
        except FileNotFoundError:
            return None, (
                f"no release manifest at {release_dir / MANIFEST_FILENAME} — release "
                f"{release_id} is not readable here"
            )
        except (OSError, ValueError, TypeError) as exc:
            return None, (
                f"release manifest {release_dir / MANIFEST_FILENAME} is unreadable: {exc!r}"
            )
        if manifest.release_id != release_id:
            return None, (
                f"release manifest {release_dir / MANIFEST_FILENAME} names release "
                f"'{manifest.release_id}' — it disagrees with the directory it is filed under"
            )
        return manifest, None

    def _release_dir(self, release_id: str) -> Path | None:
        """``<releases_root>/<release_id>``, or ``None`` for an id that could escape it.

        This read model takes no request input — the ids it joins come from the
        pointer and the manifests. The guard is here anyway because a corrupted
        pointer must not become a way to read a file outside the releases root,
        and a read model that answered such a request would be a file-read
        oracle. Same rule as the rollback service's own id check: one path
        component, no drive, no ``.``/``..``.
        """
        candidate = _clean_id(release_id)
        if candidate is None:
            return None
        path = Path(candidate)
        if path.name != candidate or path.drive or candidate in {".", ".."}:
            return None
        return self._releases_root / candidate


def _clean_id(value: object) -> str | None:
    """A non-blank stripped id, or ``None``."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


# ═══════════════════════════════════════════════════════════════════════════
# Index jobs — "what has been built, what is serving, and what still needs one"
# ═══════════════════════════════════════════════════════════════════════════

#: Where the re-index job publishes the corpus the serving index is built from.
#: Anchored to ``rag2025/`` the way ``RAGSettings`` anchors its paths, and the
#: same directory ``scripts/ingest_lancedb.py`` calls ``CANONICAL_CORPUS_DIR`` —
#: one corpus, named the same way in both places.
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "canonical_corpus"

#: The staging suffix ``ReindexJob`` derives when no ``staging_root`` is given.
#: Written here so this read model finds the builds a default-constructed job
#: writes; the job's own default stays the single source of the layout.
STAGING_SUFFIX = ".staging"

#: The command that ingests an already-built corpus into the served table.
#: Real, and only half a retry: it ingests, it does not rebuild. Reported as
#: exactly that (see :func:`_retry_steps`) rather than as "retry the job".
INGEST_COMMAND = "python scripts/ingest_lancedb.py --canonical-corpus {corpus}"


#: The label every saved-but-not-active row carries. The ticket's third
#: criterion lives or dies on this string: ``saved`` is never called published.
SAVED_NOT_SERVING_LABEL = "đã build, chưa phục vụ"


@dataclass(frozen=True)
class RetryStep:
    """One step of getting a failed job running again, and whether it can run.

    ``runnable=False`` is a real answer, not a placeholder: the step exists, the
    process cannot perform it, and ``reason`` says why. Reporting the ingest half
    as runnable while the rebuild half is not is the honest split — the ingest
    command refuses a stale corpus itself, so it cannot be advertised as a full
    retry.
    """

    step: str
    runnable: bool
    command: str | None
    reason: str | None


@dataclass(frozen=True)
class IndexJobView:
    """One row of the index-job screen.

    ``state`` is a :class:`~src.services.reindex_job.JobState` value or ``None``.
    ``None`` means "this row is not a job": a build that exists in the staging
    area is an artifact, not a run, and giving it a state would be the exact
    saved-as-active confusion this ticket exists to prevent.

    ``saved`` and ``serving`` are separate booleans, and ``serving`` is true only
    for ``active``. They are derived from the state machine's own rule
    (:attr:`JobState.is_serving`), not restated here.
    """

    job_id: str
    state: str | None
    saved: bool
    serving: bool
    label: str
    release_id: str | None
    staged_dir: str | None
    reason: str | None
    retry: tuple[RetryStep, ...] = ()


@dataclass(frozen=True)
class IndexJobsStatus:
    """What the index-job screen renders, plus whether a re-index is owed.

    ``reindex_needed`` is the stamp comparison, computed from the serving corpus
    and the running code — the same call the job itself makes, so the screen
    cannot disagree with what a job would decide.
    """

    jobs: tuple[IndexJobView, ...]
    reindex_needed: bool
    reindex_reason: str
    pending_outbox_entries: int
    pending_affected_record_ids: tuple[str, ...]
    problems: tuple[str, ...]


def _retry_steps(
    *, corpus_dir: Path, staged: Path | None, runner_live: bool
) -> tuple[RetryStep, ...]:
    """The two real halves of a retry, each with its own truth.

    A re-index is rebuild-then-ingest. The rebuild is runnable when a live
    runner is wired into the process; the ingest step is only offered when
    there is a staged build to ingest *and* that build's own stamp matches the
    running code — the ingest command refuses a corpus built by other
    chunking logic, so offering it there would send an operator into a
    guaranteed failure.

    The staleness compared is the *staged build's*, not the serving corpus's:
    the build is what would be ingested, so the build is what must be current.
    It is computed here rather than by the caller so the two can never disagree.
    """
    if staged is None:
        ingest_reason = "không có build nào trong staging để ingest"
    elif stamp_is_stale(read_corpus_stamp(staged)):
        ingest_reason = (
            "build trong staging không khai stamp khớp code đang chạy — ingest sẽ "
            "từ chối phục vụ nó, phải rebuild trước"
        )
    else:
        ingest_reason = None
    return (
        RetryStep(
            step="rebuild",
            runnable=runner_live,
            command=None,
            reason=None,
        ),
        RetryStep(
            step="ingest",
            runnable=ingest_reason is None,
            command=(
                INGEST_COMMAND.format(corpus=corpus_dir) if ingest_reason is None else None
            ),
            reason=ingest_reason,
        ),
    )


class IndexJobsService:
    """Reads job state, staged builds and the outbox. Never writes, never runs.

    Three sources, each read for what it is:

    * **a live job** — an object the process hands in (``app.state.reindex_job``
      when a runner exists). This is the only source that can report one of the
      four ``JobState`` values, because a state is a property of a running job;
    * **the staging area** — directories a job left behind. A build on disk is
      ``saved`` and is reported as ``saved``, never as active: nothing here
      claims a staged build is the corpus the index serves;
    * **the outbox** — the publications queued for the next job, and the stamp
      comparison that says whether the serving corpus still matches the code.
    """

    def __init__(
        self,
        *,
        corpus_dir: str | Path | None = None,
        staging_root: str | Path | None = None,
        outbox: Any = None,
        job: Any = None,
        runner: Any = None,
    ) -> None:
        self._corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self._staging_root = (
            Path(staging_root)
            if staging_root
            else self._corpus_dir.parent / f"{self._corpus_dir.name}{STAGING_SUFFIX}"
        )
        self._outbox = outbox
        self._job = job
        self._runner = runner

    @property
    def corpus_dir(self) -> Path:
        return self._corpus_dir

    @property
    def staging_root(self) -> Path:
        return self._staging_root

    @property
    def _runner_live(self) -> bool:
        return bool(self._runner and getattr(self._runner, "started", False))


    def status(self) -> IndexJobsStatus:
        problems: list[str] = []
        jobs: list[IndexJobView] = []

        live = self._live_job(problems)
        if live is not None:
            jobs.append(live)

        jobs.extend(self._staged_builds(problems))

        pending = 0
        affected: tuple[str, ...] = ()
        if self._outbox is not None:
            try:
                entries = self._outbox.pending()
                pending = len(entries)
                affected = tuple(
                    sorted({i for e in entries for i in e.affected_record_ids})
                )
            except Exception as exc:  # noqa: BLE001 - a read model reports, never raises
                problems.append(f"không đọc được outbox: {exc!r}")

        stale = stamp_is_stale(read_corpus_stamp(self._corpus_dir))
        if stale:
            reason = (
                f"corpus đang phục vụ tại {self._corpus_dir} không khai stamp khớp code "
                "đang chạy — lần re-index tới sẽ rebuild toàn bộ"
            )
        else:
            reason = "corpus đang phục vụ khớp stamp của code đang chạy"

        return IndexJobsStatus(
            jobs=tuple(jobs),
            reindex_needed=stale,
            reindex_reason=reason,
            pending_outbox_entries=pending,
            pending_affected_record_ids=affected,
            problems=tuple(problems),
        )

    # ── the live job ────────────────────────────────────────────────────────

    def _live_job(self, problems: list[str]) -> IndexJobView | None:
        job = self._job
        if job is None:
            return None
        try:
            state = job.state
            saved = bool(job.saved)
            serving = bool(job.serving)
            staged = job.staged_dir
            error = job.error
        except Exception as exc:  # noqa: BLE001 - a broken job must not break the screen
            problems.append(f"không đọc được trạng thái job đang chạy: {exc!r}")
            return None

        state_value = state.value if isinstance(state, JobState) else str(state)
        staged_path = Path(staged) if staged is not None else None
        return IndexJobView(
            job_id=f"live:{state_value}",
            state=state_value,
            saved=saved,
            # Read off the state machine, not restated: `serving` is true for
            # `active` and for nothing else.
            serving=serving,
            label=_state_label(state_value, saved=saved, serving=serving),
            release_id=None,
            staged_dir=str(staged_path) if staged_path is not None else None,
            reason=str(error) if error is not None else None,
            retry=_retry_steps(
                corpus_dir=self._corpus_dir,
                staged=staged_path,
                runner_live=self._runner_live,
            ),
        )

    # ── builds left on disk ─────────────────────────────────────────────────

    def _staged_builds(self, problems: list[str]) -> list[IndexJobView]:
        """Every staged build, newest first. Always ``saved``, never ``serving``.

        A build is reported ``saved`` because its artifacts are on disk; it is
        reported ``serving=False`` because a staged build is by construction not
        the corpus the serving directory holds — the job publishes by moving the
        staged files into the serving corpus, so a build still in staging has not
        been published.
        """
        root = self._staging_root
        try:
            candidates = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
        except FileNotFoundError:
            return []
        except OSError as exc:
            problems.append(f"không đọc được staging {root}: {exc!r}")
            return []

        stale_serving = stamp_is_stale(read_corpus_stamp(self._corpus_dir))
        views: list[IndexJobView] = []
        for directory in candidates:
            stamp = read_corpus_stamp(directory)
            if stamp is None:
                # A directory with no manifest is a build that never finished —
                # the manifest is written last, on purpose. It is reported as an
                # unfinished build rather than silently dropped, because a
                # half-written staging directory is something an operator should
                # see and clear.
                views.append(
                    IndexJobView(
                        job_id=directory.name,
                        state=None,
                        saved=False,
                        serving=False,
                        label="build dở dang (không có manifest)",
                        release_id=None,
                        staged_dir=str(directory),
                        reason=(
                            f"{directory / CHUNK_MANIFEST_FILENAME} không tồn tại — "
                            "manifest được ghi cuối cùng, nên build này chưa hoàn tất"
                        ),
                    )
                )
                continue
            views.append(
                IndexJobView(
                    job_id=directory.name,
                    state=None,
                    saved=True,
                    serving=False,
                    label=SAVED_NOT_SERVING_LABEL,
                    release_id=None,
                    staged_dir=str(directory),
                    reason=(
                        None
                        if not stale_serving
                        else "corpus đang phục vụ đã cũ so với code đang chạy"
                    ),
                    retry=_retry_steps(
                        corpus_dir=self._corpus_dir,
                        staged=directory,
                        runner_live=self._runner_live,
                    ),
                )
            )
        return views


def _state_label(state: str, *, saved: bool, serving: bool) -> str:
    """The Vietnamese label for one job row, derived from state alone.

    ``serving`` is the only input that produces a "đang phục vụ" label, and it
    is true only for ``active`` — so a saved build cannot reach that wording.
    """
    if serving:
        return "đang phục vụ (active)"
    if state == JobState.FAILED.value:
        return "thất bại (failed)"
    if state == JobState.RUNNING.value:
        return "đang chạy (running)"
    if state == JobState.PENDING.value:
        return "chờ chạy (pending)"
    return SAVED_NOT_SERVING_LABEL if saved else "chưa rõ trạng thái"
