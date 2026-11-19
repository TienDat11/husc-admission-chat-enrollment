"""Tests for the U7 exceptions / blocker surface — PG-gated."""
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
from src.routers.admin_exceptions import router as admin_exceptions_router  # noqa: E402
from src.routers.admin_source_documents import router as source_documents_router  # noqa: E402
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
    app.include_router(admin_exceptions_router, prefix="/admin")
    app.include_router(source_documents_router, prefix="/admin")
    app.state.db_engine = create_db_engine(fresh_db)
    configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=pg_root, STORE_BACKEND="postgres")
    )
    with TestClient(app) as client:
        yield client


def _engine(pg_app):
    return pg_app.app.state.db_engine


def _seed(doc_id: str, pg_app) -> None:
    import asyncio

    from sqlalchemy import text

    session_factory = make_session_factory(_engine(pg_app))

    async def _insert() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO source_documents (document_id, sha256, kind, storage_uri, original_name, status, uploaded_by)
                    VALUES (:doc, '\\x00', 'pdf', :doc, 'file.pdf', 'uploaded', 'cb.quan-tri')
                    ON CONFLICT (document_id) DO NOTHING
                    """
                ),
                {"doc": doc_id},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO parse_runs (parse_run_id, document_id, parser, parser_version, options, page_routes, status)
                    VALUES ('00000000-0000-0000-0000-000000000000', :doc, 'pypdf', '1.0', '{}', '{}', 'succeeded')
                    ON CONFLICT (parse_run_id) DO NOTHING
                    """
                ),
                {"doc": doc_id},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO document_blocks (parse_run_id, block_id, document_id, page, ordinal, block_type, text, table_tsv, warnings, version)
                    VALUES ('00000000-0000-0000-0000-000000000000', 'b1', :doc, 1, 0, 'table', 'Hội đồng', 'c1\tc2', '[{"code":"TABLE_CELL_DROP_SUSPECTED","payload":{"missing_pypdf_rows":[[1,2],[3,4]]}}]', 1)
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"doc": doc_id},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO canonical_records (record_id, version, coordinate_key, record_type, scope, subject, admission_year, admission_round, methods_hash, status, archive_reason, authority_level, effective_from, effective_to, text, structured_payload, major_codes, admission_methods, source_document_id, source_url, source_page, supersedes, supersedes_version, authored_by, reviewed_by, approved_by, published_by, published_at, created_at, updated_at)
                    VALUES ('unanchored:X', 1, 'unanchored:X', 'THONG_BAO', 'all', 'm', 2026, 'all', 'all', 'archived', 'withdrawn', 'husc_canonical', '2026-01-01', NULL, 'draft text', '{}', '[]', '[]', :doc, NULL, 1, NULL, NULL, 'cb.quan-tri', NULL, NULL, NULL, NULL, NOW(), NOW())
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"doc": doc_id},
            )
            await session.commit()

    asyncio.run(_insert())


def test_list_exceptions_returns_blockers_and_unanchored(pg_app):
    doc_id = "THONG_TIN_TUYEN_SINH_2026.pdf"
    _seed(doc_id, pg_app)
    response = pg_app.get(f"/admin/source-documents/{doc_id}/exceptions", headers=_headers("editor"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == doc_id
    assert any(b["block_id"] == "b1" for b in body["blocks"])
    b1 = next(b for b in body["blocks"] if b["block_id"] == "b1")
    assert "TABLE_CELL_DROP_SUSPECTED" in b1["warnings"]
    assert any(u["record_id"] == "unanchored:X" for u in body["unanchored"])


def test_patch_block_assign_coordinate_creates_record_and_withdraws(pg_app):
    doc_id = "THONG_TIN_TUYEN_SINH_2026.pdf"
    _seed(doc_id, pg_app)
    response = pg_app.patch(
        f"/admin/source-documents/{doc_id}/blocks/b1",
        json={"coordinate_key": "CHI_TIEU:nganh:cntt:2026", "expected_version": 1},
        headers=_headers("editor"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "fixed"
    assert "UNANCHORED_DRAFT" in body["cleared_blockers"]
    assert body["record"]["record_id"] == "CHI_TIEU:nganh:cntt:2026"
    assert body["record"]["version"] == 1


def test_patch_block_stale_version_409(pg_app):
    doc_id = "THONG_TIN_TUYEN_SINH_2026.pdf"
    _seed(doc_id, pg_app)
    response = pg_app.patch(
        f"/admin/source-documents/{doc_id}/blocks/b1",
        json={"content": "x", "expected_version": 999},
        headers=_headers("editor"),
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "VERSION_CONFLICT"


def test_viewer_cannot_patch_block_403(pg_app):
    doc_id = "THONG_TIN_TUYEN_SINH_2026.pdf"
    _seed(doc_id, pg_app)
    response = pg_app.patch(
        f"/admin/source-documents/{doc_id}/blocks/b1",
        json={"content": "x"},
        headers=_headers("viewer"),
    )
    assert response.status_code == 403, response.text


def test_insert_missing_rows_clears_blocker(pg_app):
    doc_id = "THONG_TIN_TUYEN_SINH_2026.pdf"
    _seed(doc_id, pg_app)
    response = pg_app.post(
        f"/admin/source-documents/{doc_id}/blocks/b1/insert-missing-rows",
        json={"expected_version": 1},
        headers=_headers("editor"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "fixed"
    assert "TABLE_CELL_DROP_SUSPECTED" in body["cleared_blockers"]


def test_reocr_returns_202(pg_app):
    doc_id = "THONG_TIN_TUYEN_SINH_2026.pdf"
    _seed(doc_id, pg_app)
    response = pg_app.post(
        f"/admin/source-documents/{doc_id}/reocr",
        json={"page_numbers": [1, 2]},
        headers=_headers("editor"),
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["state"] == "queued"
    assert body["page_numbers"] == [1, 2]
    assert "job_id" in body
