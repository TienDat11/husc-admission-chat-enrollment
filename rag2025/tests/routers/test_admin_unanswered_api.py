"""Unanswered-question queue HTTP surface — ticket 7.

Asserted at the HTTP seam, the same way the other admin surfaces are:

- criterion: GET lists queued items with their resolution markers;
- criterion: a missing credential answers 422 (the ticket-13 rule), a
  recognized credential answers 200;
- criterion: the served rows carry the redacted question, never the raw one.

The queue is attached to ``app.state`` the way the running app does it —
directly over a throwaway path — so the bytes the assertions read are the
bytes the service wrote, through the real admin router mount.
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

from src.routers.admin import router as admin_router  # noqa: E402
from src.services.unanswered_queue import UnansweredQueueService  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

PHONE = "0912345678"


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


@pytest.fixture
def svc(tmp_path: Path) -> UnansweredQueueService:
    return UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")


@pytest.fixture
def app(svc: UnansweredQueueService, monkeypatch) -> FastAPI:
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    application.state.unanswered_queue = svc
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_missing_credential_is_422(client: TestClient) -> None:
    response = client.get("/admin/unanswered/")
    assert response.status_code == 422


def test_credentialed_call_sees_queued_item(client: TestClient, svc: UnansweredQueueService) -> None:
    svc.push(f"Hỏi về điểm chuẩn, gọi {PHONE} nhé", "trace-admin-1", "guardrail", "NOT_IN_HUSC_SCOPE")

    response = client.get("/admin/unanswered/", headers=_headers("viewer"))

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["trace_id"] == "trace-admin-1"
    assert item["resolved"] is False
    assert PHONE not in item["question"]
    assert "[REDACTED]" in item["question"]


def test_resolved_items_carry_markers(client: TestClient, svc: UnansweredQueueService) -> None:
    svc.push("câu hỏi đã có đáp án bổ sung", "trace-admin-2", "padded_rag", "INSUFFICIENT_DATA")
    svc.resolve("trace-admin-2", "HOI_DAP:QA:cntt:2026:faq-1")

    default_view = client.get("/admin/unanswered/", headers=_headers("viewer")).json()
    assert default_view["count"] == 0

    full_view = client.get("/admin/unanswered/", headers=_headers("viewer"), params={"resolved": True}).json()
    assert full_view["count"] == 1
    item = full_view["items"][0]
    assert item["resolved"] is True
    assert item["resolved_at"] is not None
    assert item["resolved_by_record_id"] == "HOI_DAP:QA:cntt:2026:faq-1"


def test_no_wiring_is_503(monkeypatch) -> None:
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    with TestClient(application) as bare_client:
        response = bare_client.get("/admin/unanswered/", headers=_headers("viewer"))
    assert response.status_code == 503
