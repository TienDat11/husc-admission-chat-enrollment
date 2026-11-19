"""Cancel acknowledgement by the worker (plan §PL-5).

A job in ``cancel_requested`` is leased to a worker. When that worker completes,
fails, or its lease is reaped, the job must end in ``cancelled`` rather than
``completed`` / ``failed_retryable`` / re-queued. A job that is not cancel-requested
keeps the ordinary transitions (control case).

Requires PostgreSQL; skipped when ``DATABASE_URL`` is unset.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.job_queue import complete_job, fail_job, reap_expired_leases  # noqa: E402

DATABASE_URL = os.environ.get("DATABASE_URL", "")
needs_pg = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set (needs migrated PostgreSQL)")


def _run(coro):
    return asyncio.run(coro)


async def _with_private_engine(work):
    # A private NullPool engine per helper call, so no pooled connection crosses
    # from one event loop to another.
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        return await work(engine)
    finally:
        await engine.dispose()


async def _insert_job(state: str, lease_owner: str | None = None, lease_offset_s: float = 60.0) -> str:
    """Insert a job; the lease expires ``lease_offset_s`` seconds from now (negative = past)."""
    job_id = str(uuid.uuid4())

    async def work(engine):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
INSERT INTO ingestion_jobs (job_id, kind, payload, state, idempotency_key, created_by,
                            lease_owner, lease_expires_at, heartbeat_at)
VALUES (CAST(:job AS uuid), 'parse', '{}'::jsonb, :state, :key, 'cb.test',
        :owner,
        CASE WHEN CAST(:owner AS text) IS NULL THEN NULL
             ELSE now() + make_interval(secs => CAST(:offset AS double precision)) END,
        CASE WHEN CAST(:owner AS text) IS NULL THEN NULL ELSE now() END)
"""
                ),
                {
                    "job": job_id,
                    "state": state,
                    "key": f"test-{job_id}",
                    "owner": lease_owner,
                    "offset": lease_offset_s,
                },
            )

    await _with_private_engine(work)
    return job_id


async def _delete_job(job_id: str) -> None:
    async def work(engine):
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM ingestion_jobs WHERE job_id = CAST(:job AS uuid)"), {"job": job_id}
            )

    await _with_private_engine(work)


async def _read_state(job_id: str) -> tuple[str, str | None]:
    async def work(engine):
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT state, lease_owner FROM ingestion_jobs WHERE job_id = CAST(:job AS uuid)"),
                    {"job": job_id},
                )
            ).one()
            return row[0], row[1]

    return await _with_private_engine(work)


@needs_pg
def test_complete_on_cancel_requested_acks_cancelled():
    job = _run(_insert_job("cancel_requested", lease_owner="w1"))
    try:
        completed = _run(_with_private_engine(lambda e: complete_job(e, job, "w1")))
        assert completed is True
        state, _owner = _run(_read_state(job))
        assert state == "cancelled", f"expected 'cancelled', got {state!r}"
    finally:
        _run(_delete_job(job))


@needs_pg
def test_fail_on_cancel_requested_acks_cancelled_not_retryable():
    job = _run(_insert_job("cancel_requested", lease_owner="w1"))
    try:
        new_state = _run(
            _with_private_engine(lambda e: fail_job(e, job, "w1", {"code": "X", "message": "m"}))
        )
        assert new_state == "cancelled", f"fail_job returned {new_state!r}"
        state, _owner = _run(_read_state(job))
        assert state == "cancelled", f"expected 'cancelled', got {state!r}"
    finally:
        _run(_delete_job(job))


@needs_pg
def test_reap_on_expired_cancel_requested_acks_cancelled_and_not_requeued():
    job = _run(_insert_job("cancel_requested", lease_owner="w1", lease_offset_s=-60.0))
    try:
        reaped = _run(_with_private_engine(lambda e: reap_expired_leases(e)))
        assert reaped >= 1, f"reaper touched {reaped} rows; expected at least our expired lease"
        state, owner = _run(_read_state(job))
        assert state == "cancelled", f"expected 'cancelled', got {state!r}"
        assert state not in ("queued", "failed_retryable", "failed_terminal")
        assert owner is None
    finally:
        _run(_delete_job(job))


@needs_pg
def test_control_complete_on_claimed_still_completes():
    job = _run(_insert_job("claimed", lease_owner="w1"))
    try:
        completed = _run(_with_private_engine(lambda e: complete_job(e, job, "w1")))
        assert completed is True
        state, _owner = _run(_read_state(job))
        assert state == "completed", f"expected 'completed', got {state!r}"
    finally:
        _run(_delete_job(job))
