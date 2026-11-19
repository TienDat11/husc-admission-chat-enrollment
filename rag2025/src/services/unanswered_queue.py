"""Unanswered-question queue — ticket 7 (A12: SQL upgrade).

Supports two backends selected at construction:

* **File mode** (default): JSONL append-only queue, the ticket-7 behavior.
* **SQL mode** (``engine`` wired, or ``STORE_BACKEND=postgres``): tables
  ``23 unanswered_questions`` and ``20 chat_feedback``. All writes go through
  SQLAlchemy 2.0 async; reads are plain ``text()`` queries so the service has
  no ORM model coupling.

The raw query text is never persisted: ``redact_pii`` is applied at ingress
before writing to ``unanswered_questions.question`` and ``chat_feedback.reason``.

Open unanswered questions are deduplicated by ``normalized_hash``: asking the
same question again increments ``occurrence_count`` and refreshes ``asked_at``
(and therefore ``sla_due_at = asked_at + 48h``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from src.services.log_redaction import redact_pii as _log_redact_pii
from src.services.pii_redactor import redact_pii as _pii_redact_pii

__all__ = [
    "FEEDBACK_FILENAME",
    "Item",
    "ItemNotFoundError",
    "QUEUE_DIRNAME",
    "UnansweredQueueError",
    "UnansweredQueueService",
]

QUEUE_DIRNAME = "unanswered"
QUEUE_FILENAME = "queue.jsonl"
FEEDBACK_FILENAME = "feedback.jsonl"

_SLA_HOURS = 48
_WS = re.compile(r"\s+")


def _normalized_hash(question: str) -> str:
    """Stable content hash used to deduplicate the same question."""
    norm = _WS.sub(" ", question.strip()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sla_due(asked_at: datetime) -> datetime:
    return asked_at + timedelta(hours=_SLA_HOURS)


@dataclass
class Item:
    """One unanswered question plus its resolution state.

    ``question`` is already redacted at write time — never the raw user text.
    ``resolved`` is true only once an officer has resolved the item through
    ``UnansweredQueueService.resolve``.
    """

    question: str
    trace_id: str
    asked_at: str  # ISO-8601 UTC
    route: str
    status_code: str
    resolved_at: str | None = None
    resolved_by_record_id: str | None = None
    # K4 (T16): triage marker for the 48h-SLA denominator. False by default so
    # every older row and file-mode writer round-trips unchanged.
    out_of_scope: bool = False
    occurrence_count: int = 1
    normalized_hash: str | None = None
    sla_due_at: str | None = None

    @property
    def resolved(self) -> bool:
        return self.resolved_at is not None

    @property
    def is_overdue(self) -> bool:
        if self.resolved_at or self.out_of_scope or not self.sla_due_at:
            return False
        return datetime.fromisoformat(self.sla_due_at) < datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Item":
        return cls(
            question=str(row["question"]),
            trace_id=str(row["trace_id"]),
            asked_at=str(row["asked_at"]),
            route=str(row.get("route") or ""),
            status_code=str(row["status_code"]),
            resolved_at=(str(row["resolved_at"]) if row.get("resolved_at") else None),
            resolved_by_record_id=(
                str(row["resolved_by_record_id"]) if row.get("resolved_by_record_id") else None
            ),
            out_of_scope=bool(row.get("out_of_scope", False)),
            occurrence_count=int(row.get("occurrence_count") or 1),
            normalized_hash=row.get("normalized_hash"),
            sla_due_at=(str(row["sla_due_at"]) if row.get("sla_due_at") else None),
        )


class UnansweredQueueError(Exception):
    """Base class for unanswered-queue failures."""


class ItemNotFoundError(UnansweredQueueError):
    """No queued item (open or resolved) carries that ``trace_id``."""


class UnansweredQueueService:
    """Append-only queue of questions that could not be answered.

    All mutations happen under a single thread lock (file mode) or within one
    SQL transaction (SQL mode), so concurrent review batches and student
    feedback calls never tear a record.
    """

    def __init__(
        self,
        queue_path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        engine: Optional[AsyncEngine] = None,
    ) -> None:
        self._engine = engine
        self._session_factory = (
            async_sessionmaker(engine, expire_on_commit=False) if engine is not None else None
        )
        self._sql = engine is not None
        if queue_path is not None:
            self._queue_path = Path(queue_path)
            self._feedback_path = self._queue_path.with_name(FEEDBACK_FILENAME)
        elif data_dir is not None:
            self._queue_path = Path(data_dir) / QUEUE_DIRNAME / QUEUE_FILENAME
            self._feedback_path = self._queue_path.with_name(FEEDBACK_FILENAME)
        else:
            raise ValueError(
                "unanswered queue requires queue_path or data_dir — "
                "attach the service to app.state from a configured location"
            )
        self._lock = threading.Lock()

    @property
    def queue_path(self) -> Path:
        return self._queue_path

    @property
    def feedback_path(self) -> Path:
        return self._feedback_path

    # ------------------------------------------------------------------ SQL --
    async def _sql_fetch(self, conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = await conn.execute(text(sql), params)
        return [dict(row._mapping) for row in result]

    async def _item_from_row(self, row: dict[str, Any]) -> Item:
        return Item(
            question=str(row["question"]),
            trace_id=str(row["trace_id"]),
            asked_at=row["asked_at"].isoformat() if hasattr(row["asked_at"], "isoformat") else str(row["asked_at"]),
            route=str(row.get("route") or ""),
            status_code=str(row["status_code"]),
            resolved_at=(
                row["resolved_at"].isoformat()
                if hasattr(row.get("resolved_at"), "isoformat")
                else (str(row["resolved_at"]) if row.get("resolved_at") else None)
            ),
            resolved_by_record_id=(
                str(row["resolved_by_record_id"]) if row.get("resolved_by_record_id") else None
            ),
            out_of_scope=bool(row.get("out_of_scope", False)),
            occurrence_count=int(row.get("occurrence_count") or 1),
            normalized_hash=row.get("normalized_hash"),
            sla_due_at=(
                row["sla_due_at"].isoformat()
                if hasattr(row.get("sla_due_at"), "isoformat")
                else (str(row["sla_due_at"]) if row.get("sla_due_at") else None)
            ),
        )

    async def _sql_push(self, question: str, trace_id: str, route: str, status_code: str) -> Item:
        asked_at = datetime.now(timezone.utc)
        normalized_hash = _normalized_hash(question)
        redacted = _pii_redact_pii(question)
        sla_due_at = _sla_due(asked_at)
        async with self._session_factory() as conn:
            # Dedupe open questions by normalized_hash.
            existing = await self._sql_fetch(
                conn,
                """
                SELECT trace_id FROM unanswered_questions
                WHERE normalized_hash = :h AND resolved_at IS NULL AND out_of_scope = false
                ORDER BY asked_at DESC LIMIT 1
                """,
                {"h": normalized_hash},
            )
            if existing:
                found_trace = existing[0]["trace_id"]
                await conn.execute(
                    text(
                        """
                        UPDATE unanswered_questions
                        SET occurrence_count = occurrence_count + 1,
                            asked_at = :now,
                            sla_due_at = :sla
                        WHERE trace_id = :tid
                        """,
                    ),
                    {"now": asked_at, "sla": sla_due_at, "tid": found_trace},
                )
                await conn.commit()
                row = await self._sql_fetch(
                    conn,
                    """
                    SELECT * FROM unanswered_questions WHERE trace_id = :tid
                    """,
                    {"tid": found_trace},
                )
                return await self._item_from_row(row[0])
            await conn.execute(
                text(
                    """
                    INSERT INTO unanswered_questions
                        (trace_id, question, normalized_hash, occurrence_count,
                         asked_at, route, status_code, out_of_scope, sla_due_at)
                    VALUES
                        (:trace_id, :question, :normalized_hash, 1,
                         :asked_at, :route, :status_code, false, :sla_due_at)
                    """,
                ),
                {
                    "trace_id": trace_id,
                    "question": redacted,
                    "normalized_hash": normalized_hash,
                    "asked_at": asked_at,
                    "route": route,
                    "status_code": status_code,
                    "sla_due_at": sla_due_at,
                },
            )
            await conn.commit()
            row = await self._sql_fetch(
                conn,
                """
                SELECT * FROM unanswered_questions WHERE trace_id = :tid
                """,
                {"tid": trace_id},
            )
            return await self._item_from_row(row[0])

    async def _sql_items(self) -> list[Item]:
        async with self._session_factory() as conn:
            rows = await self._sql_fetch(
                conn,
                """
                SELECT * FROM unanswered_questions ORDER BY asked_at DESC
                """,
                {},
            )
        return [await self._item_from_row(r) for r in rows]

    async def _sql_unanswered(self) -> list[Item]:
        async with self._session_factory() as conn:
            rows = await self._sql_fetch(
                conn,
                """
                SELECT * FROM unanswered_questions
                WHERE resolved_at IS NULL AND out_of_scope = false
                ORDER BY asked_at DESC
                """,
                {},
            )
        return [await self._item_from_row(r) for r in rows]

    async def _sql_resolve(self, trace_id: str, resolved_by_record_id: str) -> Item:
        async with self._session_factory() as conn:
            cur = await conn.execute(
                text("SELECT trace_id FROM unanswered_questions WHERE trace_id = :tid"),
                {"tid": trace_id},
            )
            if cur.first() is None:
                raise ItemNotFoundError(f"no unanswered item with trace_id {trace_id!r}")
            await conn.execute(
                text(
                    """
                    UPDATE unanswered_questions
                    SET resolved_at = :now, resolved_by_record_id = :rid
                    WHERE trace_id = :tid
                    """,
                ),
                {"now": datetime.now(timezone.utc), "rid": resolved_by_record_id, "tid": trace_id},
            )
            await conn.commit()
            rows = await self._sql_fetch(
                conn,
                """
                SELECT * FROM unanswered_questions WHERE trace_id = :tid
                """,
                {"tid": trace_id},
            )
        return await self._item_from_row(rows[0])

    async def _sql_mark_out_of_scope(self, trace_id: str, out_of_scope: bool) -> Item:
        async with self._session_factory() as conn:
            cur = await conn.execute(
                text("SELECT trace_id FROM unanswered_questions WHERE trace_id = :tid"),
                {"tid": trace_id},
            )
            if cur.first() is None:
                raise ItemNotFoundError(f"no unanswered item with trace_id {trace_id!r}")
            await conn.execute(
                text(
                    """
                    UPDATE unanswered_questions
                    SET out_of_scope = :oos
                    WHERE trace_id = :tid
                    """,
                ),
                {"oos": out_of_scope, "tid": trace_id},
            )
            await conn.commit()
            rows = await self._sql_fetch(
                conn,
                """
                SELECT * FROM unanswered_questions WHERE trace_id = :tid
                """,
                {"tid": trace_id},
            )
        return await self._item_from_row(rows[0])

    async def _sql_feedback(
        self,
        trace_id: str,
        helpful: bool,
        reason: str | None,
        question: str | None,
        route: str,
        status_code: str,
    ) -> dict[str, Any]:
        clean_reason = (reason or "").strip()
        redacted_reason = _pii_redact_pii(clean_reason) if clean_reason else ""
        answered_at = datetime.now(timezone.utc)
        async with self._session_factory() as conn:
            cur = await conn.execute(
                text("SELECT feedback_id FROM chat_feedback WHERE trace_id = :tid"),
                {"tid": trace_id},
            )
            if cur.first() is not None:
                return {
                    "trace_id": trace_id,
                    "helpful": helpful,
                    "reason": redacted_reason,
                    "answered_at": str(answered_at),
                    "linked_item": False,
                    "recorded": True,
                }
            linked_item = False
            if question is not None:
                item = await self._sql_push(question, trace_id, route, status_code)
                linked_item = True
            await conn.execute(
                text(
                    """
                    INSERT INTO chat_feedback
                        (feedback_id, trace_id, helpful, reason, answered_at, linked_item, release_id, created_at)
                    VALUES
                        (:fid, :trace_id, :helpful, :reason, :answered_at, :linked_item, null, :created_at)
                    """,
                ),
                {
                    "fid": str(__import__("uuid").uuid4()),
                    "trace_id": trace_id,
                    "helpful": helpful,
                    "reason": redacted_reason,
                    "answered_at": answered_at,
                    "linked_item": linked_item,
                    "created_at": answered_at,
                },
            )
            await conn.commit()
        return {
            "trace_id": trace_id,
            "helpful": helpful,
            "reason": redacted_reason,
            "answered_at": str(answered_at),
            "linked_item": linked_item,
            "recorded": True,
        }

    # ----------------------------------------------------------------- file --
    def _ensure_dir(self) -> None:
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_lines(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _write_lines(self, path: Path, rows: list[dict[str, Any]]) -> None:
        self._ensure_dir()
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)

    # ------------------------------------------------------------- dispatch --
    async def push(self, question: str, trace_id: str, route: str, status_code: str) -> Item:
        """Queue an unanswered question.

        Redacts ``question`` through ``redact_pii`` before persisting. In SQL
        mode, open questions with the same ``normalized_hash`` are deduplicated:
        the existing row's ``occurrence_count`` increments and its ``asked_at``
        (and therefore ``sla_due_at``) refreshes to now().
        """
        if not trace_id.strip():
            raise ValueError("trace_id must not be empty")
        if self._sql:
            return await self._sql_push(question, trace_id, route, status_code)
        with self._lock:
            rows = self._read_lines(self._queue_path)
            for row in rows:
                if str(row["trace_id"]) == trace_id:
                    return Item.from_dict(row)
            item = Item(
                question=_log_redact_pii(question),
                trace_id=trace_id,
                asked_at=_now_iso(),
                route=str(route),
                status_code=str(status_code),
                normalized_hash=_normalized_hash(question),
                sla_due_at=_sla_due(datetime.now(timezone.utc)).isoformat(),
            )
            rows.append(item.to_dict())
            self._write_lines(self._queue_path, rows)
            return item

    async def resolve(self, trace_id: str, resolved_by_record_id: str) -> Item:
        """Mark an item as resolved. Raises ``ItemNotFoundError`` if unknown."""
        if self._sql:
            return await self._sql_resolve(trace_id, resolved_by_record_id)
        with self._lock:
            rows = self._read_lines(self._queue_path)
            for row in rows:
                if str(row["trace_id"]) == trace_id:
                    row["resolved_at"] = _now_iso()
                    row["resolved_by_record_id"] = resolved_by_record_id
                    self._write_lines(self._queue_path, rows)
                    return Item.from_dict(row)
            raise ItemNotFoundError(f"no unanswered item with trace_id {trace_id!r}")

    async def mark_out_of_scope(self, trace_id: str, out_of_scope: bool = True) -> Item:
        """Flag an item as out of scope. Raises ``ItemNotFoundError`` if unknown."""
        if self._sql:
            return await self._sql_mark_out_of_scope(trace_id, out_of_scope)
        with self._lock:
            rows = self._read_lines(self._queue_path)
            for row in rows:
                if str(row["trace_id"]) == trace_id:
                    row["out_of_scope"] = bool(out_of_scope)
                    self._write_lines(self._queue_path, rows)
                    return Item.from_dict(row)
            raise ItemNotFoundError(f"no unanswered item with trace_id {trace_id!r}")

    async def items(self) -> list[Item]:
        """All items ever queued, newest first."""
        if self._sql:
            return await self._sql_items()
        with self._lock:
            return [Item.from_dict(row) for row in self._read_lines(self._queue_path)]

    async def unanswered(self) -> list[Item]:
        """Open items still waiting for review, newest first."""
        if self._sql:
            return await self._sql_unanswered()
        return [item for item in await self.items() if not item.resolved]

    async def feedback(
        self,
        trace_id: str,
        helpful: bool,
        reason: str | None = None,
        *,
        question: str | None = None,
        route: str = "served",
        status_code: str = "SUCCESS",
    ) -> dict[str, Any]:
        """Record a student's feedback on a served answer.

        Redacts ``reason`` through ``redact_pii`` before persisting to
        ``chat_feedback``. Idempotent by ``trace_id``. A negative rating on a
        SUCCESS trace promotes the trace into the unanswered queue.
        """
        if not trace_id.strip():
            raise ValueError("trace_id must not be empty")
        if len(trace_id) > 128:
            raise ValueError(f"trace_id exceeds 128 characters (got {len(trace_id)})")
        clean_reason = (reason or "").strip()
        if len(clean_reason) > 4096:
            raise ValueError(f"reason exceeds 4096 characters (got {len(clean_reason)})")
        if self._sql:
            return await self._sql_feedback(
                trace_id, helpful, clean_reason, question, route, status_code
            )
        with self._lock:
            feedback_rows = self._read_lines(self._feedback_path)
            for row in feedback_rows:
                if str(row["trace_id"]) == trace_id:
                    return dict(row)
            queue_rows = self._read_lines(self._queue_path)
            matched = next(
                (row for row in queue_rows if str(row["trace_id"]) == trace_id),
                None,
            )
            if not helpful and matched is None:
                if question is not None:
                    item = Item(
                        question=_log_redact_pii(question),
                        trace_id=trace_id,
                        asked_at=_now_iso(),
                        route=route,
                        status_code=status_code,
                        normalized_hash=_normalized_hash(question),
                        sla_due_at=_sla_due(datetime.now(timezone.utc)).isoformat(),
                    )
                    queue_rows.append(item.to_dict())
                    self._write_lines(self._queue_path, queue_rows)
                    matched = item.to_dict()
                else:
                    row: dict[str, Any] = {
                        "question": "",
                        "trace_id": trace_id,
                        "asked_at": _now_iso(),
                        "route": route,
                        "status_code": status_code,
                        "resolved_at": None,
                        "resolved_by_record_id": None,
                        "out_of_scope": False,
                        "occurrence_count": 1,
                        "normalized_hash": None,
                        "sla_due_at": None,
                    }
                    queue_rows.append(row)
                    self._write_lines(self._queue_path, queue_rows)
                    matched = row
            record = {
                "trace_id": trace_id,
                "helpful": helpful,
                "reason": _log_redact_pii(clean_reason) if clean_reason else "",
                "answered_at": _now_iso(),
                "linked_item": matched is not None,
            }
            feedback_rows.append(record)
            self._write_lines(self._feedback_path, feedback_rows)
            return record
