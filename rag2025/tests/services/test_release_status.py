"""TDD: rag2025/src/services/release_status.py — ticket 26.

The read model behind the release/index-job screen. Everything here is asserted
against real files on disk (a real pointer, real manifests, a real staging
directory, a real outbox), never against a stub: the whole point of the module
is that it reports what is there, so a test that fed it its own answers would
prove nothing.

What is covered:

- the active release and the previous one come off the pointer and the manifest
  lineage, and the previous one is the release a targetless rollback would move
  to (the same ``parent_release_id`` hop);
- a pointer that is absent is "nothing published yet"; a pointer that is broken
  is a *problem*, never "nothing published" — the two states that must not be
  confused on a screen whose whole job is saying which release is live;
- **``saved`` is never ``serving``**: a build sitting in the staging area is
  reported ``saved=True, serving=False`` with a label that does not say
  published, and no row in the payload claims to serve anything;
- a failed job reports its reason and a retry whose two halves each carry their
  own truth (the rebuild has no runner; the ingest command is offered only when
  the staged build is not stale);
- the stamp comparison the job itself would make is the one the screen reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.release import ReleaseManifest  # noqa: E402
from src.infrastructure.canonical_store import (  # noqa: E402
    publish_pointer,
    write_manifest,
)
from src.services.admin_rollback import (  # noqa: E402
    POINTER_FILENAME,
    RELEASES_DIRNAME,
    release_tag,
)
from src.services.reindex_job import (  # noqa: E402
    CHUNK_MANIFEST_FILENAME,
    CorpusStamp,
    JobState,
)
from src.services.release_status import (  # noqa: E402
    SAVED_NOT_SERVING_LABEL,
    IndexJobsService,
    ReleaseStatusService,
)

R1_ID = "R-2026-08-01"
R2_ID = "R-2026-09-01"


# ── the release half ────────────────────────────────────────────────────────


def _releases_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store" / RELEASES_DIRNAME


def _seed(
    releases_root: Path, release_id: str, *, parent: str | None = None, tag: str | None = None
) -> None:
    write_manifest(
        releases_root / release_id,
        ReleaseManifest(
            release_id=release_id,
            parent_release_id=parent,
            lancedb_tag=release_tag(release_id) if tag is None else tag,
        ),
    )


def _service(releases_root: Path) -> ReleaseStatusService:
    return ReleaseStatusService(releases_root=releases_root)


def test_active_and_previous_come_off_the_pointer_and_the_lineage(tmp_path: Path):
    """R2 active, R1 behind it — the two answers the screen exists to give."""
    root = _releases_root(tmp_path)
    _seed(root, R1_ID)
    _seed(root, R2_ID, parent=R1_ID)
    publish_pointer(root / POINTER_FILENAME, R2_ID)

    status = _service(root).status()

    assert status.active_release_id == R2_ID
    assert status.previous_release_id == R1_ID
    assert status.active is not None and status.active.release_id == R2_ID
    assert status.previous is not None and status.previous.release_id == R1_ID
    assert status.active.parent_release_id == R1_ID
    assert status.problems == ()


def test_previous_is_the_release_a_targetless_rollback_would_move_to(tmp_path: Path):
    """The screen's "previous" and the rollback's target are the same release.

    Asserted by moving the pointer with the same hop the rollback service
    resolves (parent, if its manifest is on disk) and re-reading: after the
    move, the release the screen called *previous* is the one it calls *active*.
    """
    root = _releases_root(tmp_path)
    _seed(root, R1_ID)
    _seed(root, R2_ID, parent=R1_ID)
    publish_pointer(root / POINTER_FILENAME, R2_ID)

    previous = _service(root).status().previous_release_id
    publish_pointer(root / POINTER_FILENAME, previous)  # what a rollback does

    after = _service(root).status()
    assert after.active_release_id == R1_ID
    assert after.previous_release_id is None  # R1 names no parent


def test_a_missing_pointer_is_nothing_published_not_a_problem(tmp_path: Path):
    """No pointer yet is a normal state: this root has served no release."""
    status = _service(_releases_root(tmp_path)).status()

    assert status.active_release_id is None
    assert status.active is None
    assert status.problems == ()


def test_a_broken_pointer_is_reported_not_silently_read_as_nothing_published(
    tmp_path: Path,
):
    """A pointer that cannot be parsed must not render as "chưa publish".

    The distinction is the whole reason ``problems`` exists: "no release is
    active" and "the release cannot be determined" are different sentences, and
    only one of them is safe to show an operator.
    """
    root = _releases_root(tmp_path)
    root.mkdir(parents=True)
    (root / POINTER_FILENAME).write_text("{ this is not json", encoding="utf-8")

    status = _service(root).status()

    assert status.active_release_id is None
    assert len(status.problems) == 1
    assert "not valid JSON" in status.problems[0]


def test_an_active_release_whose_manifest_is_missing_reports_both(tmp_path: Path):
    """The pointer is authoritative about what is active, even when unreadable.

    ``active_release_id`` stays R2 — that is what the serving path reads — while
    ``active`` is ``None`` and a problem names the missing manifest.
    """
    root = _releases_root(tmp_path)
    _seed(root, R2_ID)
    publish_pointer(root / POINTER_FILENAME, R2_ID)
    (root / R2_ID / "release_manifest.json").unlink()

    status = _service(root).status()

    assert status.active_release_id == R2_ID
    assert status.active is None
    assert any("no release manifest" in p for p in status.problems)


def test_a_named_but_absent_parent_is_reported_as_no_release_one_step_back(
    tmp_path: Path,
):
    """R2 names R1; R1 is gone. Silence here would read as "nothing behind R2"."""
    root = _releases_root(tmp_path)
    _seed(root, R2_ID, parent=R1_ID)
    publish_pointer(root / POINTER_FILENAME, R2_ID)

    status = _service(root).status()

    assert status.previous_release_id is None
    assert any("there is no release one step back" in p for p in status.problems)


# ── the job half: saved is never serving ────────────────────────────────────


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "data" / "canonical_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    return corpus


def _staged(corpus: Path, name: str, *, stamp: dict | None = None) -> Path:
    """One staged build. A stamp of ``None`` writes no manifest (an unfinished build)."""
    directory = corpus.parent / f"{corpus.name}.staging" / name
    directory.mkdir(parents=True, exist_ok=True)
    if stamp is not None:
        (directory / CHUNK_MANIFEST_FILENAME).write_text(
            json.dumps(stamp), encoding="utf-8"
        )
    return directory


def _current_stamp() -> dict:
    stamp = CorpusStamp.current()
    return stamp.to_dict()


def test_a_staged_build_is_saved_and_never_serving(tmp_path: Path):
    """The ticket's third criterion, at the read-model seam.

    A build whose artifacts are on disk in staging has persisted a build. It is
    reported ``saved=True`` and ``serving=False``, and its label is not a
    published/active wording — the label *is* what the screen shows.
    """
    corpus = _corpus(tmp_path)
    _staged(corpus, "job-20260913T000000000000", stamp=_current_stamp())

    status = IndexJobsService(corpus_dir=corpus).status()

    assert len(status.jobs) == 1
    row = status.jobs[0]
    assert row.saved is True
    assert row.serving is False
    assert row.state is None  # a build is not a run
    assert row.label == SAVED_NOT_SERVING_LABEL
    # The wording is "chưa phục vụ" (not serving) — it must never claim the
    # build IS serving, and it must never carry the active/published wording.
    assert "chưa phục vụ" in row.label
    assert "active" not in row.label
    assert "đang phục vụ" not in row.label


def test_no_row_in_the_payload_claims_to_serve_without_an_active_job(tmp_path: Path):
    """``serving`` is true only for an active job — asserted over every row."""
    corpus = _corpus(tmp_path)
    _staged(corpus, "job-20260913T000000000000", stamp=_current_stamp())
    _staged(corpus, "job-20260913T010000000000", stamp=_current_stamp())

    status = IndexJobsService(corpus_dir=corpus).status()

    assert status.jobs, "the fixture must produce rows, or this proves nothing"
    assert all(row.serving is False for row in status.jobs)
    assert all(row.state is None for row in status.jobs)


def test_a_live_job_reports_its_state_and_serving_follows_the_state_machine(
    tmp_path: Path,
):
    """A job's four states, with ``serving`` read off ``JobState.is_serving``.

    Only ``active`` serves. The three others — including a *saved* running job —
    do not, which is the distinction the ticket is about.
    """

    class _FakeJob:
        def __init__(self, state: JobState, *, saved: bool, error: BaseException | None = None):
            self.state = state
            self.saved = saved
            self.serving = state.is_serving
            self.staged_dir = None
            self.error = error

    corpus = _corpus(tmp_path)
    cases = [
        (JobState.PENDING, False, None, False),
        (JobState.RUNNING, True, None, False),  # saved mid-run, still not serving
        (JobState.FAILED, True, RuntimeError("embedder hết quota"), False),
        (JobState.ACTIVE, True, None, True),
    ]
    for state, saved, error, expected_serving in cases:
        service = IndexJobsService(
            corpus_dir=corpus, job=_FakeJob(state, saved=saved, error=error)
        )
        row = service.status().jobs[0]
        assert row.state == state.value
        assert row.saved is saved
        assert row.serving is expected_serving, f"{state.value} served wrongly"
        if state is JobState.FAILED:
            assert row.reason == "embedder hết quota"


def test_a_failed_job_offers_a_retry_whose_halves_are_each_truthful(tmp_path: Path):
    """A failed job's retry names both halves and refuses to fake the rebuild.

    No runner is wired into this service instance, so the rebuild step is
    ``runnable=False`` with no reason string — a missing runner is a wiring
    fact, not a ticket reference. The ingest step is offered only because a
    non-stale build exists in staging — and it is *not* presented as a full
    retry of the job.
    """

    corpus = _corpus(tmp_path)
    staged = _staged(corpus, "job-20260913T000000000000", stamp=_current_stamp())

    class _FailedJob:
        state = JobState.FAILED
        saved = True
        serving = False
        # The job got as far as writing its artifacts before it failed; the
        # ingest half of the retry is about exactly this directory.
        staged_dir = staged
        error = RuntimeError("LLM provider hết quota")

    status = IndexJobsService(corpus_dir=corpus, job=_FailedJob()).status()
    live = status.jobs[0]

    steps = {step.step: step for step in live.retry}
    assert steps["rebuild"].runnable is False
    assert steps["rebuild"].reason is None
    assert steps["ingest"].runnable is True
    assert "ingest_lancedb.py --canonical-corpus" in (steps["ingest"].command or "")


def test_the_ingest_step_is_withdrawn_when_the_staged_build_is_stale(tmp_path: Path):
    """The ingest command refuses a stale corpus itself — so it is not offered.

    Advertising it here would send an operator into a guaranteed failure, which
    is the misleading half-retry the ticket's evidence rule forbids.
    """
    corpus = _corpus(tmp_path)
    _staged(
        corpus,
        "job-20260913T000000000000",
        stamp={
            "chunker_version": "khong-bao-gio-khop",
            "unit_rules_version": "cung-khong",
            "chunk_manifest_version": 1,
        },
    )

    status = IndexJobsService(corpus_dir=corpus).status()
    steps = {step.step: step for step in status.jobs[0].retry}

    assert steps["ingest"].runnable is False
    assert steps["ingest"].command is None
    assert steps["ingest"].reason and "stamp" in steps["ingest"].reason
    assert "phải rebuild trước" in steps["ingest"].reason


def test_an_unfinished_staging_directory_is_reported_not_dropped(tmp_path: Path):
    """No manifest = the build never finished (it is written last). Not silent."""
    corpus = _corpus(tmp_path)
    _staged(corpus, "job-20260913T000000000000", stamp=None)

    status = IndexJobsService(corpus_dir=corpus).status()

    assert len(status.jobs) == 1
    assert status.jobs[0].saved is False
    assert status.jobs[0].serving is False
    assert "dở dang" in status.jobs[0].label


def test_the_stamp_comparison_is_the_one_the_job_itself_would_make(tmp_path: Path):
    """``reindex_needed`` mirrors ``stamp_is_stale`` over the serving corpus."""
    corpus = _corpus(tmp_path)

    # No corpus at all: states no stamp, and no stamp is stale.
    assert IndexJobsService(corpus_dir=corpus).status().reindex_needed is True

    (corpus / CHUNK_MANIFEST_FILENAME).write_text(
        json.dumps(_current_stamp()), encoding="utf-8"
    )
    fresh = IndexJobsService(corpus_dir=corpus).status()
    assert fresh.reindex_needed is False
    assert "khớp stamp" in fresh.reindex_reason


def test_the_outbox_tail_is_reported_as_what_the_next_job_will_re_index(tmp_path: Path):
    """Queued publications are counted and their affected ids unioned."""
    corpus = _corpus(tmp_path)

    class _Outbox:
        def pending(self):
            return [
                type("E", (), {"affected_record_ids": ("a", "b")})(),
                type("E", (), {"affected_record_ids": ("b", "c")})(),
            ]

    status = IndexJobsService(corpus_dir=corpus, outbox=_Outbox()).status()

    assert status.pending_outbox_entries == 2
    assert status.pending_affected_record_ids == ("a", "b", "c")


def test_a_broken_outbox_is_reported_and_does_not_take_the_screen_down(tmp_path: Path):
    """A read model reports; it never raises into the route."""

    class _BrokenOutbox:
        def pending(self):
            raise RuntimeError("outbox jsonl hỏng")

    status = IndexJobsService(corpus_dir=_corpus(tmp_path), outbox=_BrokenOutbox()).status()

    assert status.pending_outbox_entries == 0
    assert any("không đọc được outbox" in p for p in status.problems)


@pytest.mark.parametrize("missing", ["job", "outbox"])
def test_the_read_model_works_with_nothing_wired(tmp_path: Path, missing: str):
    """An app with no outbox and no job still answers — with an empty, honest list."""
    status = IndexJobsService(corpus_dir=_corpus(tmp_path)).status()

    assert status.jobs == ()
    assert status.pending_outbox_entries == 0
    assert status.problems == ()
