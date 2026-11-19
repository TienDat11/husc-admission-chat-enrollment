"""PostgreSQL job queue for ingestion_jobs (plan PL-5).

- claim: one UPDATE ... FOR UPDATE SKIP LOCKED; a job is never handed to two workers.
- heartbeat: extends the lease of the worker that owns the job.
- reaper: a job whose lease expired goes to failed_retryable (or failed_terminal when
  attempt >= max_attempts).
- backoff: a retryable failure sets available_at = now() + least(2^attempt, 300) s.
- a job with attempt >= max_attempts is never claimed.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

LEASE_SECONDS = 60
MAX_BACKOFF_SECONDS = 300

CLAIM_SQL = text(
    """
UPDATE ingestion_jobs j
SET state = 'claimed', lease_owner = :worker,
    lease_expires_at = now() + make_interval(secs => CAST(:lease AS double precision)),
    heartbeat_at = now(), attempt = attempt + 1, updated_at = now()
WHERE job_id = (
    SELECT job_id FROM ingestion_jobs
    WHERE state IN ('queued', 'failed_retryable') AND available_at <= now() AND attempt < max_attempts
      AND (lease_expires_at IS NULL OR lease_expires_at < now())
    ORDER BY priority, created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1)
RETURNING job_id, kind, payload, state, attempt, max_attempts, lease_owner, lease_expires_at
"""
)

HEARTBEAT_SQL = text(
    """
UPDATE ingestion_jobs
SET heartbeat_at = now(),
    lease_expires_at = now() + make_interval(secs => CAST(:lease AS double precision)),
    updated_at = now()
WHERE job_id = CAST(:job AS uuid) AND lease_owner = :worker
RETURNING job_id
"""
)

FAIL_SQL = text(
    """
UPDATE ingestion_jobs
SET state = CASE WHEN state = 'cancel_requested' THEN 'cancelled' WHEN attempt >= max_attempts THEN 'failed_terminal' ELSE 'failed_retryable' END,
    available_at = now() + make_interval(secs => least(power(2, attempt), :cap)),
    lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
    last_error = CAST(:err AS jsonb), updated_at = now()
WHERE job_id = CAST(:job AS uuid) AND lease_owner = :worker
RETURNING state
"""
)

COMPLETE_SQL = text(
    """
UPDATE ingestion_jobs
SET state = CASE WHEN state = 'cancel_requested' THEN 'cancelled' ELSE 'completed' END, lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
WHERE job_id = CAST(:job AS uuid) AND lease_owner = :worker
RETURNING job_id
"""
)

REAP_SQL = text(
    """
UPDATE ingestion_jobs
SET state = CASE WHEN state = 'cancel_requested' THEN 'cancelled' WHEN attempt >= max_attempts THEN 'failed_terminal' ELSE 'failed_retryable' END,
    available_at = now() + make_interval(secs => least(power(2, attempt), :cap)),
    lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
    last_error = CAST(:err AS jsonb), updated_at = now()
WHERE lease_owner IS NOT NULL AND lease_expires_at < now()
RETURNING job_id
"""
)


def backoff_seconds(attempt: int) -> int:
    """Retry delay after the given attempt: min(2^attempt, 300) seconds."""
    return min(2**attempt, MAX_BACKOFF_SECONDS)


async def claim_job(
    engine: AsyncEngine, worker_id: str, lease_seconds: int = LEASE_SECONDS
) -> Optional[dict[str, Any]]:
    async with engine.begin() as conn:
        row = (
            await conn.execute(CLAIM_SQL, {"worker": worker_id, "lease": float(lease_seconds)})
        ).mappings().first()
    return dict(row) if row else None


async def heartbeat(
    engine: AsyncEngine, job_id: str, worker_id: str, lease_seconds: int = LEASE_SECONDS
) -> bool:
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                HEARTBEAT_SQL, {"job": str(job_id), "worker": worker_id, "lease": float(lease_seconds)}
            )
        ).first()
    return row is not None


async def fail_job(engine: AsyncEngine, job_id: str, worker_id: str, error: dict[str, Any]) -> Optional[str]:
    """Record a failure; returns the new state, or None if this worker no longer owns the job."""
    async with engine.begin() as conn:
        return (
            await conn.execute(
                FAIL_SQL,
                {"job": str(job_id), "worker": worker_id, "err": json.dumps(error), "cap": float(MAX_BACKOFF_SECONDS)},
            )
        ).scalar_one_or_none()


async def complete_job(engine: AsyncEngine, job_id: str, worker_id: str) -> bool:
    async with engine.begin() as conn:
        row = (await conn.execute(COMPLETE_SQL, {"job": str(job_id), "worker": worker_id})).first()
    return row is not None


async def reap_expired_leases(engine: AsyncEngine) -> int:
    """Return jobs whose lease expired to failed_retryable (or failed_terminal). Returns the count."""
    err = {"code": "LEASE_EXPIRED", "message": "worker lease expired"}
    async with engine.begin() as conn:
        rows = (
            await conn.execute(REAP_SQL, {"err": json.dumps(err), "cap": float(MAX_BACKOFF_SECONDS)})
        ).all()
    return len(rows)
