"""Minimal ingestion worker loop on top of infrastructure.job_queue.

Skeleton only: the stage handlers (parse, preview, publish, index, ...) are not
implemented in U2. Callers pass a handler; the default handler does nothing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Any, Awaitable, Callable

from infrastructure.job_queue import (
    LEASE_SECONDS,
    claim_job,
    complete_job,
    fail_job,
    heartbeat,
    reap_expired_leases,
)

Handler = Callable[[dict[str, Any]], Awaitable[None]]

logger = logging.getLogger(__name__)


async def _noop_handler(job: dict[str, Any]) -> None:
    return None


async def _renew_lease(engine, job_id: str, worker_id: str, every: float) -> None:
    # Keeps the lease alive while a long handler runs; otherwise the reaper
    # hands the job to another worker and it runs twice.
    while True:
        await asyncio.sleep(every)
        await heartbeat(engine, job_id, worker_id)


async def run_worker_once(
    engine,
    worker_id: str,
    handler: Handler = _noop_handler,
    heartbeat_seconds: float = LEASE_SECONDS / 3,
) -> str | None:
    """Claim one job and run it. Returns the resulting state, or None if nothing was claimable."""
    job = await claim_job(engine, worker_id)
    if job is None:
        return None
    job_id = str(job["job_id"])
    renewer = asyncio.create_task(_renew_lease(engine, job_id, worker_id, heartbeat_seconds))
    try:
        await handler(job)
    except Exception as exc:  # noqa: BLE001 - any handler failure is a retryable job failure
        renewer.cancel()
        state = await fail_job(
            # Persist the exception class only: its text can carry secrets or paths.
            engine,
            job_id,
            worker_id,
            {"code": "HANDLER_ERROR", "message": f"handler raised {type(exc).__name__}"},
        )
        # None means this worker no longer owns the job (lease expired and was reaped).
        return state or "lease_lost"
    finally:
        renewer.cancel()
    # False means the same: another worker may already have re-claimed and run it.
    if not await complete_job(engine, job_id, worker_id):
        return "lease_lost"
    return "completed"


async def run_forever(engine, worker_id: str, handler: Handler = _noop_handler, poll_seconds: float = 1.0) -> None:
    while True:
        try:
            await reap_expired_leases(engine)
            if await run_worker_once(engine, worker_id, handler) is None:
                await asyncio.sleep(poll_seconds)
        except Exception:  # noqa: BLE001 - a transient DB error must not stop the worker
            logger.exception("ingestion worker loop error; retrying in %.1fs", poll_seconds)
            await asyncio.sleep(poll_seconds)


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


if __name__ == "__main__":
    from infrastructure.db import create_db_engine

    asyncio.run(run_forever(create_db_engine(), default_worker_id()))
