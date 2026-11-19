"""``GET /admin/overview`` — U6 (Frame 5): RBAC and accurate counts.

The file-mode tests attach the objects the running app does: the unanswered
queue over a real JSONL file (so the overdue rule reads real ``asked_at``
values), the release pointer over a real ``releases`` root, and small stand-ins
for the record store and conflict queue, which carry only what the overview
reads. The PostgreSQL test seeds a freshly migrated database and skips without
``DATABASE_URL``.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import RecordStatus  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402
from src.services.admin_rollback import POINTER_FILENAME  # noqa: E402
from src.services.conflict_queue import (  # noqa: E402
    STATUS_OPEN,
    STATUS_RESOLVED,
    ConflictQueueService,
)
from src.services.unanswered_queue import UnansweredQueueService  # noqa: E402
from tests.pg_fixtures import fresh_db  # noqa: F401  (fixture)

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure_identities(monkeypatch) -> None:
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)


class _FakeRecordStore:
    """Only what the overview reads: record ids and each record's latest status."""

    def __init__(self, statuses: dict[str, RecordStatus]) -> None:
        self._statuses = statuses

    def record_ids(self) -> tuple[str, ...]:
        return tuple(self._statuses)

    def get_record(self, record_id: str):
        return SimpleNamespace(latest=SimpleNamespace(status=self._statuses[record_id]))


class _FakeConflictQueue(ConflictQueueService):
    """A real queue type (the accessor requires one) with a fixed item list."""

    def __init__(self, statuses: list[str]) -> None:  # no store or file needed
        self._items = [SimpleNamespace(status=status) for status in statuses]

    def items(self):
        return list(self._items)


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _queue_row(trace_id: str, asked_hours_ago: float, **extra) -> dict:
    row = {
        "question": "câu hỏi đã che",
        "trace_id": trace_id,
        "asked_at": _iso(asked_hours_ago),
        "route": "guardrail",
        "status_code": "NOT_IN_HUSC_SCOPE",
        "resolved_at": None,
        "resolved_by_record_id": None,
        "out_of_scope": False,
    }
    row.update(extra)
    return row


@pytest.fixture
def queue_path(tmp_path: Path) -> Path:
    rows = [
        _queue_row("open-fresh", 1),  # open, within SLA
        _queue_row("open-overdue", 60),  # open, past 48h
        _queue_row("resolved-overdue", 60, resolved_at=_iso(2), resolved_by_record_id="r1"),
        _queue_row("out-of-scope-overdue", 60, out_of_scope=True),
    ]
    path = tmp_path / "queue.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


@pytest.fixture
def releases_root(tmp_path: Path) -> Path:
    root = tmp_path / "releases"
    root.mkdir()
    (root / POINTER_FILENAME).write_text(json.dumps({"release_id": "rel-active"}), encoding="utf-8")
    return root


@pytest.fixture
def app(queue_path: Path, releases_root: Path, monkeypatch) -> FastAPI:
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    application.state.canonical_record_store = _FakeRecordStore(
        {
            "rec-review-1": RecordStatus.REVIEW,
            "rec-review-2": RecordStatus.REVIEW,
            "rec-published": RecordStatus.PUBLISHED,
            "rec-draft": RecordStatus.DRAFT,
        }
    )
    application.state.conflict_queue_service = _FakeConflictQueue(
        [STATUS_OPEN, STATUS_OPEN, STATUS_RESOLVED]
    )
    application.state.unanswered_queue = UnansweredQueueService(queue_path=queue_path)
    application.state.admin_rollback_service = SimpleNamespace(releases_root=releases_root)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_unauthenticated_request_is_refused(client: TestClient) -> None:
    response = client.get("/admin/overview")
    assert response.status_code == 422


def test_unknown_credential_is_refused(client: TestClient) -> None:
    response = client.get("/admin/overview", headers={"X-Admin-Token": "not-a-token"})
    assert response.status_code == 403


def test_records_read_role_gets_the_overview(client: TestClient) -> None:
    response = client.get("/admin/overview", headers=_headers("viewer"))
    assert response.status_code == 200, response.text


def test_counts_are_accurate_in_file_mode(client: TestClient) -> None:
    body = client.get("/admin/overview", headers=_headers("viewer")).json()
    assert body == {
        "review_count": 2,
        "open_blocker_count": 2,
        "unanswered_open_count": 2,
        "unanswered_overdue_count": 1,
        "active_release_id": "rel-active",
    }


def test_no_active_release_is_null(client: TestClient, releases_root: Path) -> None:
    (releases_root / POINTER_FILENAME).unlink()
    body = client.get("/admin/overview", headers=_headers("admin")).json()
    assert body["active_release_id"] is None


def test_unconfigured_queue_is_503(app: FastAPI, client: TestClient) -> None:
    app.state.unanswered_queue = None
    response = client.get("/admin/overview", headers=_headers("admin"))
    assert response.status_code == 503


async def _seed_postgres(async_url: str) -> None:
    engine = create_async_engine(async_url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO unanswered_questions "
                    "(trace_id, question, normalized_hash, asked_at, route, status_code, sla_due_at, out_of_scope) "
                    "VALUES "
                    "('pg-open-fresh', 'q', 'h1', now(), 'guardrail', 'X', now() + interval '1 day', false), "
                    "('pg-open-overdue', 'q', 'h2', now(), 'guardrail', 'X', now() - interval '1 hour', false), "
                    "('pg-resolved', 'q', 'h3', now(), 'guardrail', 'X', now() - interval '1 hour', false)"
                )
            )
            await conn.execute(
                text("UPDATE unanswered_questions SET resolved_at = now() WHERE trace_id = 'pg-resolved'")
            )
            await conn.execute(
                text(
                    "INSERT INTO conflict_entries "
                    "(conflict_id, coordinate_key, record_ids, conflict_type, status) "
                    "VALUES ('c-open', 'k1', '[]'::jsonb, 'value', 'open'), "
                    "('c-done', 'k2', '[]'::jsonb, 'value', 'resolved')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO releases "
                    "(release_id, corpus_stamp, lancedb_table, manifest, status, created_by) "
                    "VALUES ('rel-pg', '{}'::jsonb, 'chunks_rel_pg', '{}'::jsonb, 'active', 'seed')"
                )
            )
    finally:
        await engine.dispose()


def test_postgres_branch_counts_match_file_definitions(fresh_db, monkeypatch) -> None:  # noqa: F811
    """The SQL branch over a migrated database, seeded with known rows."""
    from src.infrastructure.db import create_db_engine

    _configure_identities(monkeypatch)
    async_url = make_url(fresh_db).set(drivername="postgresql+asyncpg").render_as_string(
        hide_password=False
    )
    asyncio.run(_seed_postgres(async_url))

    application = FastAPI()
    application.include_router(admin_router)
    application.state.db_engine = create_db_engine(async_url)
    with TestClient(application) as client:
        response = client.get("/admin/overview", headers=_headers("viewer"))
    assert response.status_code == 200, response.text
    assert response.json() == {
        "review_count": 0,
        "open_blocker_count": 1,
        "unanswered_open_count": 2,
        "unanswered_overdue_count": 1,
        "active_release_id": "rel-pg",
    }
