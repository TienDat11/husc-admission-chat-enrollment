"""Tests for the U7 preview stale-transaction guard — PG-gated.

A concurrent edit that bumps ``record.version`` mid-preview must abort the
preview with HTTP 409 VERSION_CONFLICT rather than overwriting or silently
stale-committing.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.infrastructure.db import create_db_engine, make_session_factory  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402
from src.routers.admin_previews import router as admin_previews_router  # noqa: E402
from tests.pg_fixtures import fresh_db  # noqa: F401  (fixture)
from tests.routers.test_admin_audit import ACTORS, TOKENS  # noqa: E402

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


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
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")


@pytest.fixture
def pg_app(fresh_db, monkeypatch, tmp_path: Path):
    pg_root = tmp_path / "canonical_store"
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    _configure_identities(monkeypatch)
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(admin_previews_router)
    app.state.db_engine = create_db_engine(fresh_db)
    configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=pg_root, STORE_BACKEND="postgres")
    )
    with TestClient(app) as client:
        yield client


def _engine(pg_app):
    return pg_app.app.state.db_engine


def _seed_record(pg_app, record_id: str, version: int = 1) -> None:
    import asyncio

    from sqlalchemy import text

    session_factory = make_session_factory(_engine(pg_app))

    async def _insert() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO canonical_records (record_id, version, coordinate_key, record_type, scope, subject, admission_year, admission_round, methods_hash, status, archive_reason, authority_level, effective_from, effective_to, text, structured_payload, major_codes, admission_methods, source_document_id, source_url, source_page, supersedes, supersedes_version, authored_by, reviewed_by, approved_by, published_by, published_at, created_at, updated_at)
                    VALUES (:rid, :ver, :rid, 'THONG_BAO', 'all', 'm', 2026, 'all', 'all', 'draft', NULL, 'husc_canonical', '2026-01-01', NULL, 'Văn bản mẫu dùng kiểm tra giao dịch xem trước bị làm cũ.', '{}', '[]', '[]', NULL, NULL, NULL, NULL, NULL, 'cb.quan-tri', NULL, NULL, NULL, NULL, NOW(), NOW())
                    ON CONFLICT (record_id, version) DO NOTHING
                    """
                ),
                {"rid": record_id, "ver": version},
            )
            await session.commit()

    asyncio.run(_insert())


def test_preview_stale_after_concurrent_edit_409(pg_app):
    record_id = "CHI_TIEU:nganh:cntt:2026"
    _seed_record(pg_app, record_id, 1)

    # A concurrent editor bumps the record to version 2 (draft → new draft).
    edit_response = pg_app.post(
        f"/admin/records/{record_id}/edits",
        json={"expected_version": 1, "text": "Văn bản đã được sửa bởi tab khác."},
        headers=_headers("editor"),
    )
    assert edit_response.status_code in (200, 201), edit_response.text

    # The preview that still expects version 1 must be rejected as stale.
    response = pg_app.post(
        "/admin/previews",
        json={"record_id": record_id, "expected_version": 1},
        headers=_headers("editor"),
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "VERSION_CONFLICT"
