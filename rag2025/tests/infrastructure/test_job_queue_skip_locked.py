"""50 queued jobs + 4 concurrent workers: every job is claimed exactly once (SKIP LOCKED)."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.job_queue import claim_job

JOBS = 50
WORKERS = 4


async def _claim_all(url: str) -> list[tuple[str, int]]:
    engine = create_async_engine(url, pool_size=WORKERS, max_overflow=0)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM ingestion_jobs"))
            await conn.execute(
                text(
                    "INSERT INTO ingestion_jobs (job_id, kind, payload, state, idempotency_key, created_by) "
                    "SELECT gen_random_uuid(), 'parse', '{}'::jsonb, 'queued', 'skip-locked-' || g, 'test' "
                    "FROM generate_series(1, :n) g"
                ),
                {"n": JOBS},
            )

        claimed: list[tuple[str, int]] = []

        async def worker(worker_id: str) -> None:
            while True:
                job = await claim_job(engine, worker_id)
                if job is None:
                    return
                claimed.append((str(job["job_id"]), job["attempt"]))
                await asyncio.sleep(0.001)  # yield so the workers interleave

        await asyncio.gather(*(worker(f"w{i}") for i in range(WORKERS)))
        return claimed
    finally:
        await engine.dispose()


def test_fifty_jobs_four_workers_each_job_claimed_exactly_once(migrated_database_url: str) -> None:
    claimed = asyncio.run(_claim_all(migrated_database_url))
    ids = [job_id for job_id, _ in claimed]
    assert len(ids) == JOBS, len(ids)
    assert len(set(ids)) == JOBS, "a job was claimed by two workers"
    assert all(attempt == 1 for _, attempt in claimed), claimed
