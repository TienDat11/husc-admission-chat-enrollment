"""Append-only audit hash chain in PostgreSQL (plan PL-4 #12).

Each append runs inside one transaction under the advisory lock ``audit_chain``:
prev_hash = entry_hash of the newest row, entry_hash = sha256(prev_hash || canonical_json(row)).
The chain therefore never forks under parallel writers. UPDATE/DELETE/TRUNCATE on
audit_events are rejected by triggers installed in the baseline migration.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

HASH_FIELDS = (
    "event_id",
    "aggregate_type",
    "aggregate_id",
    "actor_id",
    "role",
    "action",
    "before",
    "after",
    "reason",
    "before_version",
    "after_version",
    "release_id",
    "request_id",
    "created_at",
)

INSERT_SQL = text(
    """
INSERT INTO audit_events (event_id, aggregate_type, aggregate_id, actor_id, role, action, before, after,
    reason, before_version, after_version, release_id, request_id, created_at, prev_hash, entry_hash)
VALUES (:event_id, :aggregate_type, :aggregate_id, :actor_id, :role, :action, CAST(:before AS jsonb),
    CAST(:after AS jsonb), :reason, :before_version, :after_version, :release_id, :request_id,
    :created_at, :prev_hash, :entry_hash)
"""
)


def _jsonish(value: Any) -> Any:
    """jsonb may come back from asyncpg as JSON text; normalise to the Python value."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def canonical_payload(row: dict[str, Any]) -> dict[str, Any]:
    created = row["created_at"]
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "event_id": str(row["event_id"]),
        "aggregate_type": row["aggregate_type"],
        "aggregate_id": row["aggregate_id"],
        "actor_id": row["actor_id"],
        "role": row["role"],
        "action": row["action"],
        "before": _jsonish(row.get("before")),
        "after": _jsonish(row.get("after")),
        "reason": row.get("reason"),
        "before_version": row.get("before_version"),
        "after_version": row.get("after_version"),
        "release_id": row.get("release_id"),
        "request_id": row.get("request_id"),
        "created_at": created.astimezone(timezone.utc).isoformat(),
    }


def compute_entry_hash(prev_hash: Optional[bytes], payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256((prev_hash or b"") + body).digest()


async def append_audit_event(engine: AsyncEngine, event: dict[str, Any]) -> dict[str, Any]:
    row = dict(event)
    row.setdefault("event_id", uuid.uuid4())
    row.setdefault("created_at", datetime.now(timezone.utc))
    row["event_id"] = uuid.UUID(str(row["event_id"]))
    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('audit_chain'))"))
        prev = (
            await conn.execute(text("SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1"))
        ).scalar_one_or_none()
        prev_hash = bytes(prev) if prev is not None else None
        payload = canonical_payload(row)
        entry_hash = compute_entry_hash(prev_hash, payload)
        await conn.execute(
            INSERT_SQL,
            {
                "event_id": row["event_id"],
                "aggregate_type": row["aggregate_type"],
                "aggregate_id": row["aggregate_id"],
                "actor_id": row["actor_id"],
                "role": row["role"],
                "action": row["action"],
                "before": json.dumps(row.get("before")),
                "after": json.dumps(row.get("after")),
                "reason": row.get("reason"),
                "before_version": row.get("before_version"),
                "after_version": row.get("after_version"),
                "release_id": row.get("release_id"),
                "request_id": row.get("request_id"),
                "created_at": row["created_at"],
                "prev_hash": prev_hash,
                "entry_hash": entry_hash,
            },
        )
    return {"event_id": str(row["event_id"]), "entry_hash": entry_hash}


async def verify_audit_chain(engine: AsyncEngine) -> dict[str, Any]:
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT seq, event_id, aggregate_type, aggregate_id, actor_id, role, action, before, after, "
                    "reason, before_version, after_version, release_id, request_id, created_at, prev_hash, entry_hash "
                    "FROM audit_events ORDER BY seq"
                )
            )
        ).mappings().all()
    prev: Optional[bytes] = None
    for row in rows:
        expected = compute_entry_hash(prev, canonical_payload(dict(row)))
        if bytes(row["prev_hash"] or b"") != (prev or b"") or bytes(row["entry_hash"]) != expected:
            return {"ok": False, "count": len(rows), "first_bad_seq": row["seq"]}
        prev = bytes(row["entry_hash"])
    return {"ok": True, "count": len(rows), "first_bad_seq": None}
