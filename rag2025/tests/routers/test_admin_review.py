"""Tests for the U8 review surface: supersession candidates and retrieval preview — PG-gated."""
from __future__ import annotations

import json
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.api.errors import install_error_handling  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.infrastructure.sql_record_store import _conninfo  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402
from tests.pg_fixtures import fresh_db  # noqa: F401  (fixture)

ADMIN_TOKEN = "tok-review-admin"
ADMIN_ACTOR = "cb.quan-tri"
OTHER_AUTHOR = "cb.bien-soan"
HEADERS = {"X-Admin-Token": ADMIN_TOKEN}

_INSERT_RECORD = """
INSERT INTO canonical_records (record_id, version, coordinate_key, record_type, scope, subject,
    admission_year, admission_round, methods_hash, status, archive_reason, authority_level,
    effective_from, effective_to, text, structured_payload, major_codes, admission_methods,
    source_document_id, source_url, source_page, supersedes, supersedes_version, authored_by,
    reviewed_by, approved_by, published_by, published_at, created_at, updated_at)
VALUES (%s, 1, %s, 'THONG_BAO', 'all', 'm', 2026, 'all', 'all', 'draft', NULL, 'husc_canonical',
    '2026-01-01', NULL, 'draft text', '{}'::jsonb, '[]'::jsonb, '[]'::jsonb,
    %s, NULL, 1, NULL, NULL, %s, NULL, NULL, NULL, NULL, NOW(), NOW())
"""


@pytest.fixture
def open_review_client(fresh_db, monkeypatch, tmp_path: Path):
    """Opens a TestClient on the admin routes over a fresh PostgreSQL database."""
    with ExitStack() as stack:

        def _open(*, four_eyes: bool) -> TestClient:
            monkeypatch.setenv("DATABASE_URL", fresh_db)
            monkeypatch.setenv(
                "ADMIN_IDENTITIES",
                json.dumps([{"token": ADMIN_TOKEN, "actor_id": ADMIN_ACTOR, "role": "admin"}]),
            )
            monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
            monkeypatch.setenv("ADMIN_FOUR_EYES", "1" if four_eyes else "0")
            app = FastAPI()
            install_error_handling(app)
            app.include_router(admin_router)
            configure_canonical_record_store(
                app,
                settings=RAGSettings(
                    CANONICAL_STORE_ROOT=tmp_path / "canonical_store", STORE_BACKEND="postgres"
                ),
            )
            return stack.enter_context(TestClient(app))

        yield _open


def _seed_candidate(database_url: str, *, author: str | None) -> str:
    """Seed a source document, a version-1 old record, a version-1 new draft by ``author``,
    and a pending supersession candidate between them."""
    token = uuid.uuid4().hex[:8]
    doc = f"doc-review-{token}"
    old_record = f"rec-old-{token}"
    new_record = f"rec-new-{token}"
    candidate_id = str(uuid.uuid4())
    with psycopg.connect(_conninfo(database_url), autocommit=True) as conn:
        conn.execute(
            "INSERT INTO source_documents (document_id, sha256, kind, storage_uri, original_name, status, uploaded_by) "
            "VALUES (%s, '\\x00', 'pdf', %s, 'file.pdf', 'uploaded', %s)",
            [doc, doc, ADMIN_ACTOR],
        )
        conn.execute(_INSERT_RECORD, [old_record, old_record, doc, ADMIN_ACTOR])
        conn.execute(_INSERT_RECORD, [new_record, new_record, doc, author])
        conn.execute(
            """
            INSERT INTO supersession_candidates (candidate_id, document_id, new_record_id, new_record_version,
                old_record_id, old_record_version, relation, tier)
            VALUES (%s, %s, %s, 1, %s, 1, 'REPLACES', 'explicit_ref')
            """,
            [candidate_id, doc, new_record, old_record],
        )
    return candidate_id


def _decide(client: TestClient, candidate_id: str, decision: str, expected_version: int = 1):
    return client.post(
        f"/admin/supersession-candidates/{candidate_id}/decision",
        json={"decision": decision, "expected_version": expected_version},
        headers=HEADERS,
    )


def test_list_returns_pending_candidates(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=False)

    response = client.get("/admin/supersession-candidates", params={"status": "pending_review"}, headers=HEADERS)

    assert response.status_code == 200, response.text
    rows = {row["candidate_id"]: row for row in response.json()}
    assert candidate_id in rows
    row = rows[candidate_id]
    assert row["status"] == "pending_review"
    assert row["decision"] is None
    assert row["new_record_version"] == 1


def test_confirm_accepts_candidate_and_records_supersede_verdict(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=False)

    response = _decide(client, candidate_id, "confirm")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate_id"] == candidate_id
    assert body["status"] == "accepted"
    assert body["decision"] == "supersede"


def test_reject_closes_candidate_without_verdict(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=False)

    response = _decide(client, candidate_id, "reject")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "rejected"
    assert response.json()["decision"] is None


def test_decided_candidate_cannot_be_decided_again(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=False)
    assert _decide(client, candidate_id, "confirm").status_code == 200

    second = _decide(client, candidate_id, "reject")

    assert second.status_code == 409, second.text
    assert second.json()["detail"]["code"] == "VERSION_CONFLICT"


def test_stale_expected_version_is_a_version_conflict(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=False)

    response = _decide(client, candidate_id, "confirm", expected_version=7)

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "VERSION_CONFLICT"
    assert response.json()["detail"]["current_version"] == 1


def test_four_eyes_refuses_confirming_own_draft(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=ADMIN_ACTOR)
    client = open_review_client(four_eyes=True)

    response = _decide(client, candidate_id, "confirm")

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "SELF_APPROVAL_FORBIDDEN"
    still_open = client.get(
        "/admin/supersession-candidates", params={"status": "pending_review"}, headers=HEADERS
    ).json()
    assert candidate_id in {row["candidate_id"] for row in still_open}


def test_four_eyes_allows_confirming_another_authors_draft(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=True)

    response = _decide(client, candidate_id, "confirm")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "accepted"


def test_unknown_decision_is_rejected_before_touching_the_store(open_review_client, fresh_db):
    candidate_id = _seed_candidate(fresh_db, author=OTHER_AUTHOR)
    client = open_review_client(four_eyes=False)

    response = _decide(client, candidate_id, "obliterate")

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "VALIDATION_FAILED"


def test_retrieval_preview_without_fixtures_reports_insufficient_fixture(open_review_client, fresh_db):
    client = open_review_client(four_eyes=False)

    response = client.post(
        "/admin/retrieval-preview",
        json={"record_ids": ["rec-no-fixture"], "k": 3},
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "insufficient_fixture"
    assert [q["query_id"] for q in body["queries"]] == ["fallback:rec-no-fixture"]
    assert body["elapsed_ms"] >= 0


def test_retrieval_preview_uses_golden_queries_linked_by_coordinate_key(open_review_client, fresh_db):
    record_id = f"rec-golden-{uuid.uuid4().hex[:8]}"
    query_id = f"gq-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(_conninfo(fresh_db), autocommit=True) as conn:
        conn.execute(
            "INSERT INTO golden_queries (query_id, text, expected_coordinate_keys, record_type, kind, origin, created_by) "
            "VALUES (%s, 'Học phí bao nhiêu?', %s::jsonb, 'THONG_BAO', 'in_scope', 'seed', %s)",
            [query_id, json.dumps([record_id]), ADMIN_ACTOR],
        )
    client = open_review_client(four_eyes=False)

    response = client.post(
        "/admin/retrieval-preview",
        json={"record_ids": [record_id], "k": 3},
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert [q["query_id"] for q in body["queries"]] == [query_id]
