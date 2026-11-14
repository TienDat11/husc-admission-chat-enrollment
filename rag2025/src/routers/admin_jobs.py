"""Ingestion job and worker surface (plan §PL-5, §PL-7.1, unit U2).

Routes:

- ``GET  /admin/jobs``                 - job list, newest first, filter by state/kind (JOBS_READ)
- ``GET  /admin/jobs/{job_id}``        - one job with progress and last error (JOBS_READ)
- ``POST /admin/jobs/{job_id}/retry``  - ``failed_terminal`` -> ``queued`` (JOBS_MANAGE)
- ``POST /admin/jobs/{job_id}/cancel`` - non-terminal -> ``cancelled`` or ``cancel_requested`` (JOBS_MANAGE)
- ``GET  /admin/ops/workers``          - worker holders derived from live leases (JOBS_READ)

The job store is injected on ``app.state.db_engine``. Absent wiring is a 503,
not an empty list, for the same reason ``admin_audit`` uses: an empty list would
read as "no jobs ever ran".

Every SQL statement here is parameterised; the state transitions are enforced in
the ``WHERE`` clause so a concurrent worker claim cannot be overwritten.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import text

from src.domain.admin_identity import AdminAction, Identity
from src.routers.admin_auth import require_action
from src.routers.admin_rate_limit import admin_rate_limit

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(admin_rate_limit)],
)

DEFAULT_LIMIT = 50
MAX_LIMIT = 200

TERMINAL_STATES = ("completed", "failed_terminal", "cancelled")

_LIST_SQL = text(
    """
SELECT job_id, kind, state, priority, attempt, max_attempts, progress,
       lease_owner, heartbeat_at, available_at, last_error, created_by, created_at, updated_at
FROM ingestion_jobs
WHERE (CAST(:state AS text) IS NULL OR state = CAST(:state AS text))
  AND (CAST(:kind AS text) IS NULL OR kind = CAST(:kind AS text))
ORDER BY created_at DESC
LIMIT :limit
"""
)

_GET_SQL = text(
    """
SELECT job_id, kind, state, priority, attempt, max_attempts, progress,
       lease_owner, heartbeat_at, available_at, last_error, created_by, created_at, updated_at
FROM ingestion_jobs WHERE job_id = CAST(:job AS uuid)
"""
)

# Retry only from the terminal failure. Attempts reset so the job gets its full
# budget again; ``last_error`` is kept so the operator still sees why it failed.
_RETRY_SQL = text(
    """
UPDATE ingestion_jobs
SET state = 'queued', attempt = 0, available_at = now(),
    lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
WHERE job_id = CAST(:job AS uuid) AND state = 'failed_terminal'
RETURNING state
"""
)

# A job with no lease is idle and can be cancelled outright. A leased job is
# cancelled cooperatively: the worker acknowledges ``cancel_requested`` (§PL-5).
_CANCEL_SQL = text(
    """
UPDATE ingestion_jobs
SET state = CASE WHEN lease_owner IS NULL THEN 'cancelled' ELSE 'cancel_requested' END,
    updated_at = now()
WHERE job_id = CAST(:job AS uuid)
  AND state NOT IN ('completed', 'failed_terminal', 'cancelled', 'cancel_requested')
RETURNING state
"""
)

_WORKERS_SQL = text(
    """
SELECT lease_owner AS worker_id,
       count(*) AS leased_jobs,
       max(heartbeat_at) AS last_heartbeat_at,
       min(lease_expires_at) AS earliest_lease_expires_at
FROM ingestion_jobs
WHERE lease_owner IS NOT NULL
GROUP BY lease_owner
ORDER BY max(heartbeat_at) DESC NULLS LAST
"""
)


class JobSummary(BaseModel):
    job_id: uuid.UUID
    kind: str
    state: str
    priority: int
    attempt: int
    max_attempts: int
    progress: dict[str, Any]
    lease_owner: Optional[str] = None
    heartbeat_at: Optional[datetime] = None
    available_at: datetime
    last_error: Optional[dict[str, Any]] = None
    created_by: str
    created_at: datetime
    updated_at: datetime


class JobTransition(BaseModel):
    job_id: uuid.UUID
    state: str
    actor_id: str


class WorkerStatus(BaseModel):
    worker_id: str
    leased_jobs: int
    last_heartbeat_at: Optional[datetime] = None
    earliest_lease_expires_at: Optional[datetime] = None


def _engine(request: Request):
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="job store is not configured")
    return engine


def _redact_last_error(value: Any) -> Optional[dict[str, str]]:
    """Keep only string ``code`` and ``message``; drop stack, path, traceback, anything else."""
    if not isinstance(value, dict):
        return None
    kept = {key: value[key] for key in ("code", "message") if isinstance(value.get(key), str)}
    return kept or None


def _as_job(row: Any) -> JobSummary:
    data = dict(row._mapping)
    data["last_error"] = _redact_last_error(data.get("last_error"))
    return JobSummary(**data)


@router.get("/jobs", response_model=list[JobSummary])
async def list_jobs(
    request: Request,
    state: Optional[str] = Query(default=None, max_length=32),
    kind: Optional[str] = Query(default=None, max_length=32),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    _identity: Identity = Depends(require_action(AdminAction.JOBS_READ)),
) -> list[JobSummary]:
    async with _engine(request).connect() as conn:
        rows = (await conn.execute(_LIST_SQL, {"state": state, "kind": kind, "limit": limit})).all()
    return [_as_job(row) for row in rows]


@router.get("/jobs/{job_id}", response_model=JobSummary)
async def get_job(
    job_id: uuid.UUID,
    request: Request,
    _identity: Identity = Depends(require_action(AdminAction.JOBS_READ)),
) -> JobSummary:
    async with _engine(request).connect() as conn:
        row = (await conn.execute(_GET_SQL, {"job": str(job_id)})).first()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _as_job(row)


@router.post("/jobs/{job_id}/retry", response_model=JobTransition)
async def retry_job(
    job_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_action(AdminAction.JOBS_MANAGE)),
) -> JobTransition:
    engine = _engine(request)
    async with engine.begin() as conn:
        row = (await conn.execute(_RETRY_SQL, {"job": str(job_id)})).first()
        if row is None:
            exists = (await conn.execute(_GET_SQL, {"job": str(job_id)})).first()
            if exists is None:
                raise HTTPException(status_code=404, detail="job not found")
            raise HTTPException(
                status_code=409,
                detail=f"job is '{exists._mapping['state']}', only 'failed_terminal' can be retried",
            )
    return JobTransition(job_id=job_id, state=row._mapping["state"], actor_id=identity.actor_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobTransition)
async def cancel_job(
    job_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_action(AdminAction.JOBS_MANAGE)),
) -> JobTransition:
    engine = _engine(request)
    async with engine.begin() as conn:
        row = (await conn.execute(_CANCEL_SQL, {"job": str(job_id)})).first()
        if row is None:
            exists = (await conn.execute(_GET_SQL, {"job": str(job_id)})).first()
            if exists is None:
                raise HTTPException(status_code=404, detail="job not found")
            raise HTTPException(
                status_code=409,
                detail=f"job is '{exists._mapping['state']}' and cannot be cancelled",
            )
    return JobTransition(job_id=job_id, state=row._mapping["state"], actor_id=identity.actor_id)


@router.get("/ops/workers", response_model=list[WorkerStatus])
async def list_workers(
    request: Request,
    _identity: Identity = Depends(require_action(AdminAction.JOBS_READ)),
) -> list[WorkerStatus]:
    async with _engine(request).connect() as conn:
        rows = (await conn.execute(_WORKERS_SQL)).all()
    return [WorkerStatus(**dict(row._mapping)) for row in rows]
