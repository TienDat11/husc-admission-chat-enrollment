"""HTTP tests for the A12 unanswered-question admin surface (SQL mode).

Runs against the isolated ``husc_u12`` database on port 5432 with
``app.state.db_engine`` wired, exercising ``GET /admin/unanswered/``,
``POST .../out-of-scope``, ``POST .../promote-golden`` and
``GET /admin/overview``.

Each test runs entirely inside one ``asyncio.run`` so the SQLAlchemy engine
and the ASGI client share a single event loop.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.routers.admin import router as admin_router
from src.services.unanswered_queue import UnansweredQueueService

DATABASE_URL = "postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc_u12"

CLEAN_TABLES = (
    "golden_queries",
    "canonical_records",
    "chat_feedback",
    "unanswered_questions",
)

ADMIN_ACTOR = "cb.quan-tri"
VIEWER_ACTOR = "cb.chi-xem"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": f"tok-{role}"}


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


async def _insert_question(engine: AsyncEngine, trace_id: str, question: str,
                           *, out_of_scope: bool = False, resolved: bool = False,
                           hours_until_sla: float = 48.0) -> None:
    now = datetime.now(timezone.utc)
    asked_at = now
    sla_due_at = now + timedelta(hours=hours_until_sla)
    resolved_at = now if resolved else None
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO unanswered_questions
                    (trace_id, question, normalized_hash, occurrence_count,
                     asked_at, route, status_code, resolved_at,
                     resolved_by_record_id, out_of_scope, sla_due_at)
                VALUES
                    (:trace_id, :question, :normalized_hash, 1,
                     :asked_at, 'guardrail', 'NOT_IN_HUSC_SCOPE', :resolved_at,
                     NULL, :out_of_scope, :sla_due_at)
                """,
            ),
            {
                "trace_id": trace_id,
                "question": question,
                "normalized_hash": "hash:" + trace_id,
                "asked_at": asked_at,
                "resolved_at": resolved_at,
                "out_of_scope": out_of_scope,
                "sla_due_at": sla_due_at,
            },
        )


@pytest.fixture(autouse=True)
def identities(monkeypatch):
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": "tok-admin", "actor_id": ADMIN_ACTOR, "role": "admin"},
                {"token": "tok-viewer", "actor_id": VIEWER_ACTOR, "role": "viewer"},
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")


async def _client_and_clean():
    """Build a fresh app+engine, clean tables, return (client, engine) — all
    in the current event loop."""
    engine = await _new_engine()
    await _clean(engine)
    app = FastAPI()
    app.include_router(admin_router)
    app.state.db_engine = engine
    app.state.unanswered_queue = UnansweredQueueService(
        data_dir="__unused_sql_mode__", engine=engine
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return client, engine


# ------------------------------------------------------------------ list view

def test_default_list_excludes_out_of_scope_and_resolved():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            await _insert_question(engine, "t-open", "câu hỏi mở?", hours_until_sla=48)
            await _insert_question(engine, "t-oos", "câu hỏi ngoài phạm vi?",
                                   out_of_scope=True, hours_until_sla=48)
            await _insert_question(engine, "t-res", "câu hỏi đã giải quyết?",
                                   resolved=True, hours_until_sla=48)

            resp = await client.get("/admin/unanswered/", headers=_headers("admin"))
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["count"] == 1
            assert body["items"][0]["trace_id"] == "t-open"
            assert {it["trace_id"] for it in body["items"]} == {"t-open"}

            row = await _fetch_one(
                engine,
                "SELECT out_of_scope FROM unanswered_questions WHERE trace_id = :t",
                {"t": "t-oos"},
            )
            assert row is not None and row["out_of_scope"] is True
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())


def test_occurrence_count_sla_due_at_exposed():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            await _insert_question(engine, "t-cnt", "câu hỏi xảy ra nhiều lần?",
                                   hours_until_sla=48)
            resp = await client.get("/admin/unanswered/", headers=_headers("admin"))
            item = resp.json()["items"][0]
            assert item["occurrence_count"] == 1
            assert item["sla_due_at"] is not None
            assert item["is_overdue"] is False
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())


# ------------------------------------------------------------------ out-of-scope

def test_out_of_scope_hides_and_persists():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            await _insert_question(engine, "t-oos2", "câu hỏi này ngoài phạm vi?",
                                   hours_until_sla=48)

            before = (await client.get("/admin/unanswered/", headers=_headers("admin"))).json()
            assert before["count"] == 1

            resp = await client.post("/admin/unanswered/t-oos2/out-of-scope",
                                     headers=_headers("admin"))
            assert resp.status_code == 200, resp.text
            assert resp.json()["out_of_scope"] is True

            after = (await client.get("/admin/unanswered/", headers=_headers("admin"))).json()
            assert after["count"] == 0

            row = await _fetch_one(
                engine,
                "SELECT out_of_scope, resolved_at FROM unanswered_questions WHERE trace_id = :t",
                {"t": "t-oos2"},
            )
            assert row is not None
            assert row["out_of_scope"] is True
            assert row["resolved_at"] is None  # never deleted, never resolved
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())


def test_out_of_scope_twice_is_409():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            await _insert_question(engine, "t-oos3", "câu hỏi x?", hours_until_sla=48)
            r1 = await client.post("/admin/unanswered/t-oos3/out-of-scope",
                                   headers=_headers("admin"))
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/admin/unanswered/t-oos3/out-of-scope",
                                   headers=_headers("admin"))
            assert r2.status_code == 409, r2.text
            assert r2.json()["detail"]["code"] == "VERSION_CONFLICT"
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())


# ------------------------------------------------------------------ promote-golden

def test_viewer_cannot_promote_golden():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            await _insert_question(engine, "t-pg-viewer", "câu hỏi promo?", hours_until_sla=48)
            resp = await client.post(
                "/admin/unanswered/t-pg-viewer/promote-golden",
                headers=_headers("viewer"),
                json={"answer": "đáp án của eux", "coordinate_key": "GOLDEN:t-pg-viewer"},
            )
            assert resp.status_code == 403, resp.text
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())


def test_promote_golden_atomic_transaction():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            question = "điểm chuẩn ngành Dược, gọi 0905123456 nhé"
            await _insert_question(engine, "t-pg", question, hours_until_sla=48)

            resp = await client.post(
                "/admin/unanswered/t-pg/promote-golden",
                headers=_headers("admin"),
                json={
                    "answer": "Điểm chuẩn ngành Dược năm nay là 24.",
                    "coordinate_key": "GOLDEN:t-pg",
                    "subject": "cau_hoi_chua_tra_loi",
                    "scope": "tuyen_sinh",
                    "admission_year": 2026,
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["trace_id"] == "t-pg"
            assert body["record_id"] == "GOLDEN:t-pg"
            assert body["status"] == "draft"

            # 1. draft canonical record created (never published).
            rec = await _fetch_one(
                engine,
                "SELECT record_id, version, status, text, authored_by FROM canonical_records WHERE record_id = :r",
                {"r": "GOLDEN:t-pg"},
            )
            assert rec is not None
            assert rec["status"] == "draft"
            assert rec["text"] == "Điểm chuẩn ngành Dược năm nay là 24."
            assert rec["authored_by"] == ADMIN_ACTOR

            # 2. golden_queries row with origin='promoted' and redacted query text.
            gq = await _fetch_one(
                engine,
                "SELECT query_id, text, origin, expected_coordinate_keys FROM golden_queries WHERE source_trace_id = :t",
                {"t": "t-pg"},
            )
            assert gq is not None
            assert gq["origin"] == "promoted"
            assert "0905123456" not in gq["text"]
            assert "[SĐT]" in gq["text"]

            # 3. resolved_by_record_id + resolved_at stamped on the question.
            uq = await _fetch_one(
                engine,
                "SELECT resolved_by_record_id, resolved_at, out_of_scope FROM unanswered_questions WHERE trace_id = :t",
                {"t": "t-pg"},
            )
            assert uq is not None
            assert uq["resolved_by_record_id"] == "GOLDEN:t-pg"
            assert uq["resolved_at"] is not None
            assert uq["out_of_scope"] is False

            # 4. audit event appended.
            ev = await _fetch_one(
                engine,
                "SELECT action, aggregate_id, actor_id FROM audit_events WHERE aggregate_id = :t AND action = 'promote_golden'",
                {"t": "t-pg"},
            )
            assert ev is not None
            assert ev["actor_id"] == ADMIN_ACTOR
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())


# ------------------------------------------------------------------ overdue

def test_overdue_question_increments_overview_overdue_count():
    async def _run():
        client, engine = await _client_and_clean()
        try:
            # SLA already expired: sla_due_at in the past.
            await _insert_question(engine, "t-overdue", "câu hỏi quá hạn?",
                                   hours_until_sla=-24)

            resp = await client.get("/admin/overview", headers=_headers("admin"))
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["unanswered_overdue_count"] >= 1
        finally:
            await client.aclose()
            await engine.dispose()
    asyncio.run(_run())
