"""Admin routes under ``STORE_BACKEND=postgres`` — PG-gated (skips without DATABASE_URL).

The app is wired the way startup wires it: ``configure_canonical_record_store``
over a fresh migrated database and a tmp root. The file-mode routes are covered
by their own suites; these tests prove the credentialed surface answers under
postgres and that nothing writes the pre-cutover file stores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.infrastructure.sql_record_store import _conninfo  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402
from src.routers.admin_audit import router as audit_router  # noqa: E402
from src.routers.admin_preview import router as preview_router  # noqa: E402
from src.routers.admin_source_documents import router as source_documents_router  # noqa: E402
from tests.pg_fixtures import fresh_db  # noqa: F401  (fixture)
from tests.routers.test_admin_audit import ACTORS, TOKENS, _record  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


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
def pg_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


def _build_pg_app(pg_root, fresh_db, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    _configure_identities(monkeypatch)
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(audit_router)
    app.include_router(source_documents_router, prefix="/admin")
    app.include_router(preview_router)
    configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=pg_root, STORE_BACKEND="postgres")
    )
    return app


@pytest.fixture
def pg_app(pg_root, fresh_db, monkeypatch):  # noqa: F811
    app = _build_pg_app(pg_root, fresh_db, monkeypatch)
    # The record's source document must be registered before a SQL write
    # (SqlRecordStore enforces the reference); seed the row the way the store tests do.
    with psycopg.connect(_conninfo(fresh_db), autocommit=True) as conn:
        conn.execute(
            "INSERT INTO source_documents (document_id, sha256, kind, storage_uri, original_name, uploaded_by) "
            "VALUES (%s, %s, 'pdf', 'uploads/x.pdf', 'x.pdf', 'alice')",
            (_record()["source_document_id"], bytes.fromhex("cd" * 32)),
        )
    yield app
    runner = getattr(app.state, "reindex_runner", None)
    if runner is not None:
        runner.stop()


@pytest.fixture
def pg_client(pg_app):
    with TestClient(pg_app) as client:
        yield client


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def test_verify_is_ok_with_legacy_fields_after_a_draft_create(pg_client, pg_app, pg_root):
    created = pg_client.post("/admin/records", json={"record": _record()}, headers=_headers("editor"))
    assert created.status_code in (200, 201), created.text

    assert len(pg_app.state.audit_ledger.entries()) >= 1
    body = pg_client.get("/admin/audit/verify", headers=_headers("viewer"))
    assert body.status_code == 200
    payload = body.json()
    assert payload["ok"] is True
    assert payload["checked"] >= 1
    assert payload["legacy_checked"] == 0
    assert payload["legacy_last_hash"] is None
    assert payload["skipped_legacy_rows"] == 0


def test_identity_credential_path_answers_under_postgres(pg_client):
    response = pg_client.get("/admin/audit/events", headers=_headers("viewer"))
    assert response.status_code == 200, response.text

    unknown = pg_client.get(
        "/admin/audit/events", headers={"X-Admin-Token": "not-a-registered-credential"}
    )
    assert unknown.status_code in (401, 403)


def test_source_document_list_is_answered_under_postgres(pg_client):
    response = pg_client.get("/admin/source-documents", headers=_headers("viewer"))
    assert response.status_code == 200, response.text


def test_preview_is_not_the_unexpected_type_refusal_under_postgres(pg_client):
    pg_client.post("/admin/records", json={"record": _record()}, headers=_headers("editor"))
    response = pg_client.post(
        "/admin/records/DIEM_CHUAN:NGANH:cntt:2026/review-preview",
        json={"query": "điểm chuẩn công nghệ thông tin"},
        headers=_headers("viewer"),
    )
    # No embedder is wired in this fixture, so the legitimate answer is the
    # embedder-missing 503, not the store-type refusal.
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "preview requires the startup embedding service"


def test_postgres_lifecycle_writes_never_create_the_file_stores(pg_client, pg_root):
    created = pg_client.post("/admin/records", json={"record": _record()}, headers=_headers("editor"))
    assert created.status_code in (200, 201), created.text
    assert not (pg_root / "records").exists()
    assert not (pg_root / "audit" / "ledger.jsonl").exists()


def test_create_submit_approve_publish_under_postgres_leaves_file_stores_alone(pg_client, pg_app, pg_root):
    record_id = _record()["record_id"]
    created = pg_client.post("/admin/records", json={"record": _record()}, headers=_headers("editor"))
    assert created.status_code in (200, 201), created.text

    for action, expected_version, role in (
        ("submit", 1, "editor"),
        ("approve", 2, "reviewer"),
        ("publish", 3, "publisher"),
    ):
        response = pg_client.post(
            f"/admin/records/{record_id}/transitions",
            json={"action": action, "expected_version": expected_version},
            headers=_headers(role),
        )
        assert response.status_code == 200, (action, response.text)

    assert len(pg_app.state.audit_ledger.entries()) >= 4
    assert pg_client.get("/admin/audit/verify", headers=_headers("viewer")).json()["ok"] is True
    assert not (pg_root / "records").exists()
    assert not (pg_root / "audit" / "ledger.jsonl").exists()


def test_a_source_document_only_in_the_file_registry_is_a_409_under_postgres(pg_root, fresh_db, monkeypatch):
    app = _build_pg_app(pg_root, fresh_db, monkeypatch)  # no source_documents row inserted
    with TestClient(app) as client:
        response = client.post("/admin/records", json={"record": _record()}, headers=_headers("editor"))
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        "source document is not registered in the SQL store; see cutover runbook"
    )
    assert "source_documents" not in response.text
