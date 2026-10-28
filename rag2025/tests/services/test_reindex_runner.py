"""TDD: rag2025/src/services/reindex_runner.py - ticket 15 (criterion 7, automatic path).

The runner is the thing that closes B15-1: before it, ``ReindexJob`` had zero
production construction sites and the outbox drained only when a human
remembered to run a script. These tests pin the runner's own contract - what
``due()`` considers work, what one tick does with it, and whether the
background worker actually self-activates - leaving the CLI's exit-code
mapping to ``tests/scripts/test_run_reindex_cli.py`` and the job's building to
``tests/services/test_reindex_job.py``.

The end-to-end criterion (7) is measured twice over here:
- ``test_the_worker_drains_a_publish_without_any_manual_call`` starts the
  thread with the autorun gate explicitly on (as production runs it), publishes
  through the real lifecycle service, and waits for the corpus to appear with
  nobody invoking ``run_once()``.
- ``test_a_stale_stamp_with_an_empty_queue_is_still_due`` pins that the stamp
  trigger is independent of the canonical-id trigger: an empty queue with a
  stale stamp is due, and the next tick rebuilds FULL.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.admin_lifecycle import AdminLifecycleService  # noqa: E402
from src.services.admin_rollback import RELEASES_DIRNAME  # noqa: E402
from src.services.reindex_job import CHUNK_MANIFEST_FILENAME  # noqa: E402
from src.services.reindex_outbox import OUTBOX_FILENAME, ReindexOutbox  # noqa: E402
from src.services.reindex_runner import AUTORUN_ENV, ReindexRunner  # noqa: E402

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
NOW = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)


class FakeEmbeddingService:
    """Deterministic stand-in for the real lazy model: same ``encode_documents``
    surface, no weights. Vectors are content-derived so two identical chunks
    embed identically and different chunks do not."""

    def __init__(self, settings) -> None:
        self.settings = settings

    def encode_documents(self, documents):
        return np.array(
            [[float(sum(b for b in doc.encode("utf-8")) % 977) / 977.0, 1.0] for doc in documents],
            dtype=np.float32,
        )


def _record(text: str = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.") -> CanonicalRecord:
    return CanonicalRecord(
        record_id=RECORD_ID,
        record_type=RecordType.DIEM_CHUAN,
        version=1,
        admission_year=2026,
        effective_from="2026-06-01",
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="de-an-tuyen-sinh-2026",
        text=text,
    )


def _publish(service: AdminLifecycleService) -> CanonicalRecord:
    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    return service.transition(RECORD_ID, LifecycleAction.PUBLISH, actor_id="cb.thu-ky")


@pytest.fixture
def wired(tmp_path: Path):
    """A store + outbox pair under one root, as the app wires them."""
    store_root = tmp_path / "canonical_store"
    outbox = ReindexOutbox(store_root / "outbox" / OUTBOX_FILENAME)
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(store, outbox=outbox, clock=lambda: NOW)
    corpus_dir = tmp_path / "canonical_corpus"
    return store_root, store, outbox, service, corpus_dir


def _runner(wired, **overrides) -> ReindexRunner:
    _store_root, store, outbox, _service, corpus_dir = wired
    kwargs: dict = dict(
        store=store,
        outbox=outbox,
        embedder_provider=lambda: FakeEmbeddingService(None),
        corpus_dir=corpus_dir,
        releases_root=_store_root / RELEASES_DIRNAME,
        interval_seconds=0.05,
        retry_seconds=0.02,
    )
    kwargs.update(overrides)
    return ReindexRunner(**kwargs)


def _wait_until(predicate, timeout: float = 15.0, message: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message or "condition not reached within timeout")


# ── the plan: what counts as work ───────────────────────────────────────────


def test_a_pending_publication_makes_the_runner_due(wired):
    """The record-level trigger: one published version is one unit of work."""
    _root, _store, outbox, service, _corpus = wired
    runner = _runner(wired)
    # A corpus that does not exist yet is due on the stamp trigger alone.
    assert runner.due() == (True, "corpus stamp absent; 0 pending outbox entries")

    _publish(service)
    assert len(outbox.pending()) == 1
    is_due, reason = runner.due()
    assert is_due
    assert "1 pending outbox entries" in reason


def test_a_stale_stamp_with_an_empty_queue_is_still_due(wired, monkeypatch):
    """Criterion 7's trigger, isolated from the id trigger.

    The queue is drained (so the canonical-id signal is silent) and only the
    corpus manifest names a chunker that no longer exists. A runner that
    watched the queue alone would report "nothing due" forever; this is the
    one place the stamp trigger is pinned at the runner level.
    """
    _root, _store, outbox, service, corpus_dir = wired
    runner = _runner(wired)
    _publish(service)
    assert runner.run_once().action == "ran"
    assert outbox.pending() == ()

    manifest_path = corpus_dir / CHUNK_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunker_version"] = "0.0-auto-stale"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    is_due, reason = runner.due()
    assert is_due, "a stale stamp with zero pending ids must still be due"
    assert "corpus stamp stale" in reason
    assert "0 pending outbox entries" in reason

    outcome = runner.run_once()
    assert outcome.action == "ran"
    assert outcome.result.plan.mode.value == "full"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["chunker_version"] != "0.0-auto-stale", (
        "the rebuild must re-stamp the corpus with the current logic"
    )


def test_run_once_on_a_quiet_system_builds_no_job(wired):
    """The no-op tick costs no job: the sink never sees a constructed one."""
    _root, _store, _outbox, service, _corpus = wired
    seen: list = []
    runner = _runner(wired, job_sink=seen.append)
    _publish(service)
    assert runner.run_once().action == "ran"
    seen.clear()

    outcome = runner.run_once()

    assert outcome.action == "noop"
    assert not outcome.due
    assert seen == [], "a tick with nothing due must not construct a ReindexJob"


# ── one tick: what a run leaves behind ──────────────────────────────────────


def test_a_tick_publishes_the_corpus_and_mirrors_the_job_state(wired):
    """The admin screen's contract: the sink gets the job BEFORE run() and the
    same object reports ACTIVE after a successful tick."""
    _root, _store, outbox, service, corpus_dir = wired
    seen: list = []
    runner = _runner(wired, job_sink=seen.append)
    _publish(service)

    outcome = runner.run_once()

    assert outcome.action == "ran"
    assert outcome.result.serving
    assert len(seen) == 1, "one tick, one fresh job"
    assert seen[0].state.value == "active"
    assert (corpus_dir / "chunks.jsonl").is_file()
    assert outbox.pending() == ()
    assert outbox.cursor() == len(outbox.entries())


def test_a_tick_without_an_embedder_skips_rather_than_crashes(wired):
    """The wiring runs before ``app.state.embedding_service`` is assigned; the
    first ticks legitimately abstain and must retry on the short interval."""
    _root, _store, outbox, service, corpus_dir = wired
    runner = _runner(wired, embedder_provider=lambda: None)
    _publish(service)

    outcome = runner.run_once()

    assert outcome.action == "skipped"
    assert "embedding service not available" in outcome.reason
    assert outcome.retry_soon
    assert not corpus_dir.exists(), "an abstaining tick must not touch the corpus"
    assert len(outbox.pending()) == 1, "and must not consume the trigger"


# ── the worker: self-activation, the gate, and survival ─────────────────────


def test_the_gate_keeps_pytest_processes_thread_free(wired, monkeypatch):
    """Under pytest, with the env unset, ``start()`` spawns nothing.

    Roughly ten router fixtures call the wiring seam directly; a thread per
    call would keep polling a torn-down ``tmp_path``. This is the safety that
    lets the runner be constructed inside that seam at all.
    """
    monkeypatch.delenv(AUTORUN_ENV, raising=False)
    runner = _runner(wired)

    assert runner.start() is False
    assert runner.disabled_reason and AUTORUN_ENV in runner.disabled_reason
    assert not any(t.name == "reindex-runner" for t in threading.enumerate())


def test_the_worker_drains_a_publish_without_any_manual_call(wired, monkeypatch):
    """Criterion 7 end-to-end: nobody invokes the runner.

    The gate is turned on explicitly (production's default), the worker is
    started before the publish, and the only thing the test does afterwards is
    wait. The corpus appearing on its own is the claim "self-activating without
    anyone remembering" measured, not read.
    """
    monkeypatch.setenv(AUTORUN_ENV, "1")
    _root, _store, outbox, service, corpus_dir = wired
    runner = _runner(wired)
    runner.start()
    try:
        _publish(service)
        _wait_until(
            lambda: (corpus_dir / "chunks.jsonl").is_file() and outbox.pending() == (),
            message="the worker never drained the publish without a manual call",
        )
        assert outbox.cursor() == len(outbox.entries())
    finally:
        runner.stop(timeout=5.0)
    assert not any(t.name == "reindex-runner" for t in threading.enumerate()), (
        "stop() must join the worker, not orphan it"
    )


def test_start_twice_does_not_stack_workers(wired, monkeypatch):
    """Re-configure must not double-poll: one runner object, one thread."""
    monkeypatch.setenv(AUTORUN_ENV, "1")
    runner = _runner(wired)
    try:
        assert runner.start() is True
        first = sum(t.name == "reindex-runner" for t in threading.enumerate())
        assert runner.start() is True
        assert sum(t.name == "reindex-runner" for t in threading.enumerate()) == first
    finally:
        runner.stop(timeout=5.0)


def test_a_corrupt_outbox_survives_in_the_worker_but_reaches_the_operator(
    wired, monkeypatch
):
    """One poison entry must not kill the daemon, and must not be swallowed.

    ``run_once()`` lets the format error propagate (a CLI operator sees it);
    the loop catches it, backs off, and keeps the thread alive. Without this,
    a single bad line would stop every future publish from ever draining.
    """
    _root, _store, outbox, _service, _corpus = wired
    outbox.path.parent.mkdir(parents=True, exist_ok=True)
    outbox.path.write_text("not json at all\n", encoding="utf-8")
    runner = _runner(wired)

    with pytest.raises(Exception):
        runner.run_once()

    monkeypatch.setenv(AUTORUN_ENV, "1")
    try:
        assert runner.start() is True
        _wait_until(
            lambda: any(
                t.name == "reindex-runner" and t.is_alive() for t in threading.enumerate()
            ),
            timeout=5.0,
            message="worker never started",
        )
        time.sleep(0.2)  # several backoff ticks against the poison entry
        assert any(
            t.name == "reindex-runner" and t.is_alive() for t in threading.enumerate()
        ), "the daemon must survive a poison tick"
    finally:
        runner.stop(timeout=5.0)
