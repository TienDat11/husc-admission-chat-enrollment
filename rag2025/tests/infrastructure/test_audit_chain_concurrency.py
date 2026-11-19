"""Audit hash chain: 20 concurrent appends verify; UPDATE/DELETE raise AUDIT_APPEND_ONLY."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.sql_audit_log import append_audit_event, verify_audit_chain


def _event(i: int) -> dict:
    return {
        "aggregate_type": "record",
        "aggregate_id": f"rec-{i}",
        "actor_id": "tester",
        "role": "admin",
        "action": "records.edit",
        "before": None,
        "after": {"n": i},
        "reason": None,
        "before_version": None,
        "after_version": 1,
        "release_id": None,
        "request_id": f"req-{i}",
    }


async def _concurrent_appends(url: str) -> dict:
    engine = create_async_engine(url, pool_size=20, max_overflow=0)
    try:
        before = await verify_audit_chain(engine)
        await asyncio.gather(*(append_audit_event(engine, _event(i)) for i in range(20)))
        after = await verify_audit_chain(engine)
        return {"before": before, "after": after}
    finally:
        await engine.dispose()


async def _tamper(url: str, statement: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(statement))
    finally:
        await engine.dispose()


def test_twenty_concurrent_appends_keep_chain_valid(migrated_database_url: str) -> None:
    result = asyncio.run(_concurrent_appends(migrated_database_url))
    assert result["after"]["ok"] is True, result
    assert result["after"]["count"] - result["before"]["count"] == 20, result


def test_update_on_audit_events_raises_append_only(migrated_database_url: str) -> None:
    asyncio.run(_concurrent_appends(migrated_database_url))
    with pytest.raises(Exception, match="AUDIT_APPEND_ONLY"):
        asyncio.run(
            _tamper(
                migrated_database_url,
                "UPDATE audit_events SET reason = 'tamper' WHERE seq = (SELECT max(seq) FROM audit_events)",
            )
        )


def test_delete_on_audit_events_raises_append_only(migrated_database_url: str) -> None:
    asyncio.run(_concurrent_appends(migrated_database_url))
    with pytest.raises(Exception, match="AUDIT_APPEND_ONLY"):
        asyncio.run(_tamper(migrated_database_url, "DELETE FROM audit_events"))
