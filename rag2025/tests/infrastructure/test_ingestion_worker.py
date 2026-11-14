"""Minimal ingestion worker loop: a claimed job completes or is scheduled for retry."""

from __future__ import annotations

import asyncio

import pytest

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from workers.ingestion_worker import run_worker_once


async def _seed_one(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM ingestion_jobs"))
        await conn.execute(
            text(
                "INSERT INTO ingestion_jobs (job_id, kind, payload, state, idempotency_key, created_by) "
                "VALUES (gen_random_uuid(), 'parse', '{}'::jsonb, 'queued', 'worker-1', 'test')"
            )
        )


async def _run(url: str, handler):
    engine = create_async_engine(url)
    try:
        await _seed_one(engine)
        outcome = await run_worker_once(engine, "w-test", handler)
        async with engine.connect() as conn:
            state = (await conn.execute(text("SELECT state FROM ingestion_jobs"))).scalar_one()
        return outcome, state
    finally:
        await engine.dispose()


def test_successful_handler_completes_job(migrated_database_url: str) -> None:
    async def ok(job):
        return None

    outcome, state = asyncio.run(_run(migrated_database_url, ok))
    assert outcome == "completed"
    assert state == "completed"


def test_failing_handler_schedules_retry(migrated_database_url: str) -> None:
    async def boom(job):
        raise RuntimeError("parser down")

    outcome, state = asyncio.run(_run(migrated_database_url, boom))
    assert outcome == "failed_retryable"
    assert state == "failed_retryable"


async def _run_and_read_last_error(url: str, handler):
    engine = create_async_engine(url)
    try:
        await _seed_one(engine)
        await run_worker_once(engine, "w-test", handler)
        async with engine.connect() as conn:
            return (await conn.execute(text("SELECT last_error FROM ingestion_jobs"))).scalar_one()
    finally:
        await engine.dispose()


def test_handler_error_text_is_not_stored_verbatim(migrated_database_url: str) -> None:
    # Exception text can carry secrets and paths; only the class name is persisted.
    async def leaky(job):
        raise RuntimeError("upstream token=sk-secret-123 failed at /srv/app/internal.py")

    last_error = asyncio.run(_run_and_read_last_error(migrated_database_url, leaky))
    serialised = str(last_error)
    assert "sk-secret-123" not in serialised
    assert "/srv/app" not in serialised
    assert last_error["code"] == "HANDLER_ERROR"
    assert "RuntimeError" in last_error["message"]


def test_lost_lease_on_complete_is_reported_not_completed(migrated_database_url: str, monkeypatch) -> None:
    # The lease expired and another worker took the job: this worker must not claim "completed".
    async def ok(job):
        return None

    async def lost(engine, job_id, worker_id):
        return False

    monkeypatch.setattr("workers.ingestion_worker.complete_job", lost)
    outcome, _ = asyncio.run(_run(migrated_database_url, ok))
    assert outcome == "lease_lost"


def test_lost_lease_on_fail_is_reported_not_retryable(migrated_database_url: str, monkeypatch) -> None:
    async def boom(job):
        raise RuntimeError("boom")

    async def lost(engine, job_id, worker_id, error):
        return None

    monkeypatch.setattr("workers.ingestion_worker.fail_job", lost)
    outcome, _ = asyncio.run(_run(migrated_database_url, boom))
    assert outcome == "lease_lost"


def test_lease_is_heartbeated_while_the_handler_runs(monkeypatch) -> None:
    # A handler that outlives the lease must keep renewing it, or the reaper re-runs the job.
    import workers.ingestion_worker as worker

    beats: list[str] = []

    async def claim(engine, worker_id):
        return {"job_id": "j-long"}

    async def beat(engine, job_id, worker_id, *a, **k):
        beats.append(job_id)
        return True

    async def complete(engine, job_id, worker_id):
        return True

    async def slow(job):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(worker, "claim_job", claim)
    monkeypatch.setattr(worker, "heartbeat", beat)
    monkeypatch.setattr(worker, "complete_job", complete)
    outcome = asyncio.run(worker.run_worker_once(None, "w-hb", slow, heartbeat_seconds=0.01))
    assert outcome == "completed"
    assert len(beats) >= 2


def test_run_forever_survives_a_transient_database_error(monkeypatch) -> None:
    # A transient error in the reaper must not kill the worker loop.
    import workers.ingestion_worker as worker

    calls = {"reap": 0}

    async def flaky_reap(engine):
        calls["reap"] += 1
        if calls["reap"] == 1:
            raise OSError("connection reset")
        raise asyncio.CancelledError  # stop the loop after the recovery iteration

    async def no_job(engine, worker_id):
        return None

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(worker, "reap_expired_leases", flaky_reap)
    monkeypatch.setattr(worker, "claim_job", no_job)
    monkeypatch.setattr(worker.asyncio, "sleep", no_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.run_forever(None, "w-loop", poll_seconds=0))
    assert calls["reap"] == 2
