"""Minimal ingestion worker loop: a claimed job completes or is scheduled for retry."""

from __future__ import annotations

import asyncio

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
