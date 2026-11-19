"""SQL-backend tests for ``UnansweredQueueService`` (tables 20/23).

Runs against the isolated ``husc_u12`` database on port 5432 and covers the
A12 queue contract: ingress PII redaction, dedupe by ``normalized_hash``,
``occurrence_count`` bumping, ``sla_due_at = asked_at + 48h``, and PII
redaction of ``chat_feedback.reason``.

Every test performs all of its async work inside a single ``asyncio.run`` so
no SQLAlchemy engine/connection is ever shared across event loops.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.services.unanswered_queue import ItemNotFoundError, UnansweredQueueService

DATABASE_URL = "postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc_u12"

CLEAN_TABLES = ("unanswered_questions", "chat_feedback")

PHONE = "0905123456"      # -> [SĐT]
CCCD = "123456789012"     # 12-digit -> [CCCD]
EMAIL = "nv.van-a@hus.edu.vn"  # -> [EMAIL]


async def _new_engine() -> AsyncEngine:
    return create_async_engine(DATABASE_URL)


async def _clean(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for table in CLEAN_TABLES:
            await conn.execute(text(f"DELETE FROM {table}"))


async def _fetch_one(engine: AsyncEngine, sql: str, params: dict):
    async with engine.connect() as conn:
        result = await conn.execute(text(sql), params)
        return result.mappings().first()


async def _exec(body):
    """Create a fresh engine, run ``body(engine)``, dispose — one loop."""
    engine = await _new_engine()
    try:
        return await body(engine)
    finally:
        await engine.dispose()


def _parse(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(str(ts))


# ------------------------------------------------------------------ push / SLA

def test_push_redacts_phone_cccd_email():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        item = await svc.push(
            f"Học phí Dược gọi {PHONE} nhé, CCCD {CCCD}, email {EMAIL}",
            "t-pii", "guardrail", "NOT_IN_HUSC_SCOPE",
        )
        assert PHONE not in item.question
        assert CCCD not in item.question
        assert EMAIL not in item.question
        assert "[SĐT]" in item.question
        assert "[CCCD]" in item.question
        assert "[EMAIL]" in item.question
    asyncio.run(_exec(_run))


def test_sla_due_at_is_asked_at_plus_48h():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        item = await svc.push("câu hỏi thường", "t-sla", "guardrail", "NOT_IN_HUSC_SCOPE")
        asked = _parse(item.asked_at)
        sla = _parse(item.sla_due_at)
        assert abs((sla - asked) - timedelta(hours=48)) < timedelta(seconds=5)
        assert item.out_of_scope is False
        assert item.resolved is False
    asyncio.run(_exec(_run))


# ------------------------------------------------------------------ dedupe

def test_repeat_normalized_hash_increments_occurrence_count():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        await svc.push("Câu hỏi về điểm chuẩn CNTT?", "t-a", "guardrail", "NOT_IN_HUSC_SCOPE")
        second = await svc.push(
            "  CÂU HỎI VỀ ĐIỂM CHUẨN CNTT?  ", "t-b", "guardrail", "NOT_IN_HUSC_SCOPE"
        )
        assert second.trace_id == "t-a"
        assert second.occurrence_count == 2
        row = await _fetch_one(
            engine,
            "SELECT trace_id FROM unanswered_questions WHERE trace_id = :t",
            {"t": "t-b"},
        )
        assert row is None
    asyncio.run(_exec(_run))


def test_different_questions_get_separate_rows():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        await svc.push("điểm chuẩn CNTT?", "t-1", "guardrail", "NOT_IN_HUSC_SCOPE")
        await svc.push("học phí Dược?", "t-2", "guardrail", "NOT_IN_HUSC_SCOPE")
        items = await svc.unanswered()
        assert {it.trace_id for it in items} == {"t-1", "t-2"}
    asyncio.run(_exec(_run))


# ------------------------------------------------------------------ feedback

def test_feedback_reason_is_redacted():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        stored = await svc.feedback(
            "t-fb", True, f"liên hệ {PHONE} nhé",
            question="câu hỏi", route="guardrail", status_code="NOT_IN_HUSC_SCOPE",
        )
        assert PHONE not in stored["reason"]
        assert "[SĐT]" in stored["reason"]
        row = await _fetch_one(
            engine,
            "SELECT reason FROM chat_feedback WHERE trace_id = :t",
            {"t": "t-fb"},
        )
        assert row is not None
        assert PHONE not in row["reason"]
        assert "[SĐT]" in row["reason"]
    asyncio.run(_exec(_run))


# ------------------------------------------------------------------ resolve

def test_resolve_marks_resolved_by_record_id():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        await svc.push("câu hỏi", "t-res", "guardrail", "NOT_IN_HUSC_SCOPE")
        item = await svc.resolve("t-res", "HOI_DAP:QA:cntt:2026:faq-1")
        assert item.resolved is True
        assert item.resolved_by_record_id == "HOI_DAP:QA:cntt:2026:faq-1"
        assert item.resolved_at is not None
    asyncio.run(_exec(_run))


def test_resolve_unknown_trace_raises():
    async def _run(engine):
        await _clean(engine)
        svc = UnansweredQueueService(data_dir="__unused_sql_mode__", engine=engine)
        with pytest.raises(ItemNotFoundError):
            await svc.resolve("does-not-exist", "HOI_DAP:QA:cntt:2026:faq-1")
    asyncio.run(_exec(_run))
