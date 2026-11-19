"""Job backoff, max-attempts terminal state, and the lease reaper (plan PL-5)."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.job_queue import claim_job, fail_job, reap_expired_leases


async def _with_engine(url: str, body):
    engine = create_async_engine(url)
    try:
        return await body(engine)
    finally:
        await engine.dispose()


async def _insert(engine, *, state: str = "queued", attempt: int = 0, max_attempts: int = 3) -> str:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM ingestion_jobs"))
        row = await conn.execute(
            text(
                "INSERT INTO ingestion_jobs (job_id, kind, payload, state, attempt, max_attempts, idempotency_key, created_by) "
                "VALUES (gen_random_uuid(), 'parse', '{}'::jsonb, :state, :attempt, :max_attempts, 'backoff-1', 'test') "
                "RETURNING job_id"
            ),
            {"state": state, "attempt": attempt, "max_attempts": max_attempts},
        )
        return str(row.scalar_one())


async def _state(engine, job_id: str) -> tuple[str, float]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT state, extract(epoch from (available_at - now())) AS delta "
                    "FROM ingestion_jobs WHERE job_id = CAST(:j AS uuid)"
                ),
                {"j": job_id},
            )
        ).one()
        return row.state, float(row.delta)


async def _make_due(engine, job_id: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE ingestion_jobs SET available_at = now() - interval '1 second' WHERE job_id = CAST(:j AS uuid)"),
            {"j": job_id},
        )


def test_failed_job_waits_for_available_at_then_is_claimed_again(migrated_database_url: str) -> None:
    async def body(engine):
        job_id = await _insert(engine)
        first = await claim_job(engine, "w-backoff")
        assert first is not None and str(first["job_id"]) == job_id and first["attempt"] == 1
        assert await fail_job(engine, job_id, "w-backoff", {"code": "BOOM", "message": "x"}) == "failed_retryable"

        state, delta = await _state(engine, job_id)
        assert state == "failed_retryable"
        assert 1.0 <= delta <= 3.0, delta  # attempt 1 -> 2^1 = 2 seconds

        assert await claim_job(engine, "w-backoff") is None  # not before available_at

        await _make_due(engine, job_id)
        second = await claim_job(engine, "w-backoff")
        assert second is not None and second["attempt"] == 2

    asyncio.run(_with_engine(migrated_database_url, body))


def test_backoff_grows_and_terminal_after_max_attempts(migrated_database_url: str) -> None:
    async def body(engine):
        job_id = await _insert(engine, max_attempts=2)
        await claim_job(engine, "w-b")  # attempt 1
        await fail_job(engine, job_id, "w-b", {"code": "BOOM", "message": "1"})
        _, delta_1 = await _state(engine, job_id)
        assert 1.0 <= delta_1 <= 3.0, delta_1  # 2 seconds

        await _make_due(engine, job_id)
        await claim_job(engine, "w-b")  # attempt 2 == max_attempts
        assert await fail_job(engine, job_id, "w-b", {"code": "BOOM", "message": "2"}) == "failed_terminal"
        state, _ = await _state(engine, job_id)
        assert state == "failed_terminal"

    asyncio.run(_with_engine(migrated_database_url, body))


def test_job_at_max_attempts_is_never_claimed(migrated_database_url: str) -> None:
    async def body(engine):
        await _insert(engine, state="failed_retryable", attempt=3, max_attempts=3)
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE ingestion_jobs SET available_at = now() - interval '1 second'"))
        assert await claim_job(engine, "w-max") is None

    asyncio.run(_with_engine(migrated_database_url, body))


def test_reaper_returns_dead_lease_to_failed_retryable(migrated_database_url: str) -> None:
    async def body(engine):
        job_id = await _insert(engine)
        await claim_job(engine, "dead-worker")
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE ingestion_jobs SET lease_expires_at = now() - interval '1 second' WHERE job_id = CAST(:j AS uuid)"),
                {"j": job_id},
            )
        reaped = await reap_expired_leases(engine)
        assert reaped >= 1
        state, _ = await _state(engine, job_id)
        assert state == "failed_retryable"
        async with engine.connect() as conn:
            owner = (
                await conn.execute(
                    text("SELECT lease_owner FROM ingestion_jobs WHERE job_id = CAST(:j AS uuid)"), {"j": job_id}
                )
            ).scalar_one()
        assert owner is None

    asyncio.run(_with_engine(migrated_database_url, body))


def test_reaper_marks_exhausted_dead_lease_terminal(migrated_database_url: str) -> None:
    async def body(engine):
        job_id = await _insert(engine, state="claimed", attempt=3, max_attempts=3)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE ingestion_jobs SET lease_owner = 'dead-worker', lease_expires_at = now() - interval '1 second' "
                    "WHERE job_id = CAST(:j AS uuid)"
                ),
                {"j": job_id},
            )
        await reap_expired_leases(engine)
        state, _ = await _state(engine, job_id)
        assert state == "failed_terminal"

    asyncio.run(_with_engine(migrated_database_url, body))
