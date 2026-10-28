"""TDD: the review-diff / review-preview endpoints — ticket 25.

Asserted at the HTTP seam (the spec's "mọi test admin chạy qua HTTP layer"
rule): the two routes exist with their paths and methods, the RECORDS_READ
grant guards both, the 404/503 mapping is the router's own behavior, and the
request body (query / version / top_k) reaches the service verbatim. The
service itself is a stub here — its real behavior is covered by
``tests/services/test_review_preview.py``; what this file owns is the wire
contract the review screen calls.

Identity is the ticket-13 credential registry: the viewer role holds
``records.read``; the shared ops credential holds nothing on the admissions
surface, which makes "the ops token cannot open the preview" a matrix fact
tested through HTTP rather than a comment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.diff import diff_records  # noqa: E402
from src.domain.canonical.record import CanonicalRecord  # noqa: E402
from src.services.review_preview import ReviewPreviewResult  # noqa: E402
from src.infrastructure.canonical_record_store import (  # noqa: E402
    CanonicalRecordStore,
    RecordNotFoundError,
)
from src.routers.admin_preview import router as admin_preview_router  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

DIFF_PATH = "/admin/records/DIEM_CHUAN:NGANH:2026/review-diff"
PREVIEW_PATH = "/admin/records/DIEM_CHUAN:NGANH:2026/review-preview"


def _configure(monkeypatch) -> None:
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


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _record() -> CanonicalRecord:
    return CanonicalRecord.model_validate(
        {
            "record_id": "DIEM_CHUAN:NGANH:2026",
            "record_type": "DIEM_CHUAN",
            "version": 2,
            "admission_year": 2026,
            "effective_from": "2026-01-01",
            "effective_to": "2026-12-31",
            "status": "draft",
            "authority_level": "husc_official",
            "source_document_id": "de-an-tuyen-sinh-2026",
            "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
            "source_page": 24,
            "text": "Diem chuan nganh Cong nghe thong tin nam 2026 la 24.0 diem.",
        }
    )


class _FakePreviewService:
    """Duck-typed stand-in for ReviewPreviewService: records the call, returns
    real RecordDiff/ReviewPreviewResult objects so the response shape asserted
    below is the production serializer's output, not a fixture's."""

    def __init__(self) -> None:
        self.diff_calls: list[tuple[str, int | None]] = []
        self.preview_calls: list[tuple[str, str, int | None, int]] = []

    def diff_for(self, record_id, version=None):
        self.diff_calls.append((record_id, version))
        if record_id == "khong-ton-tai":
            raise RecordNotFoundError(f"no versions stored for {record_id!r}")
        record = _record()
        return diff_records(None, record), record, (), record.status.value

    async def preview(self, record_id, query, version=None, top_k=5):
        self.preview_calls.append((record_id, query, version, top_k))
        if record_id == "khong-ton-tai":
            raise RecordNotFoundError(f"no versions stored for {record_id!r}")
        record = _record()
        diff = diff_records(None, record)
        return ReviewPreviewResult(
            record_id=record_id,
            draft_version=record.version,
            baseline_version=None,
            high_impact_version=diff.high_impact_version,
            diff=diff,
            validation_errors=(),
            anomaly_warnings=(),
            preview_status="ok",
            answer="Diem chuan nam 2026 la 24.0 diem.",
            answer_status="FakeModel(generation_ok)",
            model_answer=True,
            chunks=(
                {
                    "chunk_id": "DIEM_CHUAN:NGANH:2026@2#0",
                    "text": record.text,
                    "score": 0.9,
                    "source": record.source_url,
                    "metadata": {"record_id": record_id},
                },
            ),
            sources=(
                {
                    "source_document_id": record.source_document_id,
                    "source": record.source_url,
                    "page": record.source_page,
                },
            ),
        )


def _app(service=None, store=None) -> FastAPI:
    application = FastAPI()
    application.include_router(admin_preview_router)
    if service is not None:
        application.state.review_preview_service = service
    if store is not None:
        application.state.canonical_record_store = store
    return application


@pytest.fixture
def fake_service() -> _FakePreviewService:
    return _FakePreviewService()


@pytest.fixture
def client(monkeypatch, fake_service: _FakePreviewService) -> TestClient:
    _configure(monkeypatch)
    return TestClient(_app(service=fake_service))


# ---------------------------------------------------------------------------
# routes exist with their paths and methods
# ---------------------------------------------------------------------------


def test_both_preview_routes_are_registered():
    routes = {
        (route.path, tuple(sorted(route.methods)))
        for route in _app().routes
        if hasattr(route, "methods")
    }
    assert ("/admin/records/{record_id}/review-diff", ("GET",)) in routes
    assert ("/admin/records/{record_id}/review-preview", ("POST",)) in routes


def test_review_diff_is_get_only_and_preview_is_post_only(client: TestClient):
    # The method is part of the contract: a GET on the preview path must not
    # silently run a preview (it is the write-adjacent, expensive route).
    assert client.get(PREVIEW_PATH, headers=_headers("viewer")).status_code == 405
    assert client.post(DIFF_PATH, headers=_headers("viewer")).status_code == 405


# ---------------------------------------------------------------------------
# auth: RECORDS_READ guards both routes (criterion: reviewer permission only)
# ---------------------------------------------------------------------------

def test_missing_credential_is_rejected(client: TestClient):
    # The repo's convention (test_admin_records_api): the header is required,
    # so an anonymous call is a validation failure, never a served preview.
    assert client.get(DIFF_PATH).status_code == 422
    assert client.post(PREVIEW_PATH, json={"query": "?"}).status_code == 422


def test_unknown_credential_is_forbidden(monkeypatch):
    _configure(monkeypatch)
    anon = TestClient(_app(service=_FakePreviewService()))
    response = anon.get(
        DIFF_PATH, headers={"X-Admin-Token": "tok-khong-co-that"}
    )
    assert response.status_code == 403


@pytest.mark.parametrize("path,method", [(DIFF_PATH, "get"), (PREVIEW_PATH, "post")])
def test_ops_shared_credential_cannot_open_preview(
    client: TestClient, path, method
):
    # ops holds nothing on the admissions surface (ticket 13 matrix) — the
    # preview must not become a side door around that grant.
    response = getattr(client, method)(
        path, headers={"X-Admin-Token": OPS_TOKEN}
    )
    assert response.status_code == 403
    assert "records.read" in response.json()["detail"]


# ---------------------------------------------------------------------------
# happy path: response shape + body reaches the service
# ---------------------------------------------------------------------------


def test_review_diff_returns_serialized_diff_and_validation(
    client: TestClient, fake_service: _FakePreviewService
):
    response = client.get(DIFF_PATH, headers=_headers("viewer"))
    assert response.status_code == 200
    body = response.json()
    assert body["record_id"] == "DIEM_CHUAN:NGANH:2026"
    assert body["draft_version"] == 2
    assert body["high_impact_version"] == diff_records(None, _record()).high_impact_version
    assert body["diff"]["record_id"] == "DIEM_CHUAN:NGANH:2026"
    assert body["diff"]["changes"], "a new-record diff must list the draft's fields"
    assert all(
        {"field", "old", "new", "high_impact"} <= set(change)
        for change in body["diff"]["changes"]
    )
    assert body["validation_errors"] == []
    assert fake_service.diff_calls == [("DIEM_CHUAN:NGANH:2026", None)]


def test_review_diff_version_query_param_reaches_service(
    client: TestClient, fake_service: _FakePreviewService
):
    response = client.get(DIFF_PATH + "?version=1", headers=_headers("viewer"))
    assert response.status_code == 200
    assert fake_service.diff_calls == [("DIEM_CHUAN:NGANH:2026", 1)]


def test_review_preview_returns_full_result_payload(
    client: TestClient, fake_service: _FakePreviewService
):
    response = client.post(
        PREVIEW_PATH,
        headers=_headers("viewer"),
        json={"query": "Diem chuan CNTT 2026?", "version": 2, "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["preview_status"] == "ok"
    assert body["answer"] == "Diem chuan nam 2026 la 24.0 diem."
    assert body["model_answer"] is True
    assert [c["chunk_id"] for c in body["chunks"]] == ["DIEM_CHUAN:NGANH:2026@2#0"]
    assert body["sources"][0]["source_document_id"] == "de-an-tuyen-sinh-2026"
    # the request body reaches the service verbatim — dropping top_k/version
    # here would silently preview the wrong version with the wrong breadth.
    assert fake_service.preview_calls == [
        ("DIEM_CHUAN:NGANH:2026", "Diem chuan CNTT 2026?", 2, 3)
    ]


def test_preview_body_validation_rejects_empty_query(client: TestClient):
    response = client.post(
        PREVIEW_PATH, headers=_headers("viewer"), json={"query": ""}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 404 / 503 mapping
# ---------------------------------------------------------------------------


def test_unknown_record_is_404_on_both_routes(
    client: TestClient, monkeypatch
):
    _configure(monkeypatch)
    missing = "/admin/records/khong-ton-tai"
    assert (
        client.get(missing + "/review-diff", headers=_headers("viewer")).status_code
        == 404
    )
    response = client.post(
        missing + "/review-preview",
        headers=_headers("viewer"),
        json={"query": "Diem chuan?"},
    )
    assert response.status_code == 404
    assert "khong-ton-tai" in response.json()["detail"]


def test_service_without_store_is_503(monkeypatch):
    _configure(monkeypatch)
    client = TestClient(_app())  # no service, no store on state
    response = client.get(DIFF_PATH, headers=_headers("viewer"))
    assert response.status_code == 503
    assert "canonical record store is not configured" in response.json()["detail"]


def test_store_without_embedding_service_is_503(monkeypatch, tmp_path):
    _configure(monkeypatch)
    client = TestClient(_app(store=CanonicalRecordStore(tmp_path / "store")))
    response = client.post(
        PREVIEW_PATH, headers=_headers("viewer"), json={"query": "Diem chuan?"}
    )
    assert response.status_code == 503
    assert "embedding service" in response.json()["detail"]
