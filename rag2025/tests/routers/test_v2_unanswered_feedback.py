"""Unanswered hook + student feedback seam — ticket 7.

Asserted against the REAL ``src/main.py`` objects (``_is_unanswerable``,
``_queue_unanswered``, and the ``POST /v2/feedback`` route), without booting
the full pipeline:

- an out-of-scope payload is queued with the phone number redacted;
- a SUCCESS payload is never queued;
- a broken queue cannot disturb the answer (fail-soft);
- ``POST /v2/feedback`` records feedback and is idempotent by ``trace_id``.

The hook functions come from the loaded module itself, so a passing
assertion pins the production semantics — a local re-implementation of the
same logic would prove nothing.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAIN_PATH = ROOT / "src" / "main.py"

from src.services.unanswered_queue import UnansweredQueueService  # noqa: E402

PHONE = "0912345678"


def _load_main(monkeypatch):
    """Load src/main.py stand-alone with the provider keys blanked (only the
    hook functions and the feedback route are exercised, so startup never
    runs). Mirrors the FIX-05 harness in test_v2_status_contract.py."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")

    spec = importlib.util.spec_from_file_location("v2_unanswered_feedback_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module(monkeypatch):
    return _load_main(monkeypatch)


@pytest.fixture
def svc(tmp_path: Path) -> UnansweredQueueService:
    return UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")


def test_unanswerable_statuses(main_module) -> None:
    is_unanswerable = main_module._is_unanswerable
    assert is_unanswerable("NOT_IN_HUSC_SCOPE") is True
    assert is_unanswerable("INSUFFICIENT_DATA") is True
    assert is_unanswerable("HUSC_ENTITY_NOT_FOUND") is True
    assert is_unanswerable("ENTITY_PARSING_GAP") is True
    assert is_unanswerable("GENERATION_FAILED") is True
    assert is_unanswerable("SUCCESS") is False
    assert is_unanswerable("SENSITIVE_PII_DETECTED") is False


def test_out_of_scope_question_is_queued_with_phone_masked(main_module, svc: UnansweredQueueService) -> None:
    question = f"Học phí ngành Dược là bao nhiêu, gọi {PHONE} nhé"
    asyncio.run(main_module._queue_unanswered(svc, "trace-oos", question, "guardrail", "NOT_IN_HUSC_SCOPE"))

    raw = svc.queue_path.read_text(encoding="utf-8")
    assert PHONE not in raw
    items = svc.items()
    assert len(items) == 1
    assert items[0].trace_id == "trace-oos"
    assert "[REDACTED]" in items[0].question


def test_success_answer_is_never_queued(main_module, svc: UnansweredQueueService) -> None:
    asyncio.run(main_module._queue_unanswered(svc, "trace-ok", "Điểm chuẩn CNTT?", "padded_rag", "SUCCESS"))
    assert svc.items() == []


def test_broken_queue_is_fail_soft(main_module, svc: UnansweredQueueService) -> None:
    class BrokenQueue:
        def push(self, *args, **kwargs):
            raise RuntimeError("disk is gone")

    # Must not raise: the answer path keeps serving even when the queue is down.
    asyncio.run(
        main_module._queue_unanswered(BrokenQueue(), "trace-broken", "câu hỏi?", "guardrail", "NOT_IN_HUSC_SCOPE")
    )
    assert svc.items() == []


def test_feedback_route_is_public_and_idempotent(main_module, tmp_path: Path) -> None:
    service = UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")
    application = FastAPI()
    application.state.unanswered_queue = service
    # The real route handler from src/main.py, mounted on a bare app so no
    # pipeline or credential is involved — the endpoint is public by design.
    route = next(r for r in main_module.app.routes if getattr(r, "path", "") == "/v2/feedback")
    application.router.routes.append(route)

    with TestClient(application) as client:
        # No credential header anywhere: a student rates an answer.
        first = client.post("/v2/feedback", json={"trace_id": "trace-pub", "helpful": False})
        second = client.post("/v2/feedback", json={"trace_id": "trace-pub", "helpful": True})

    assert first.status_code == 200
    assert first.json()["helpful"] is False
    assert first.json()["recorded"] is True
    # Idempotent: the second call returns the first record, not a new one.
    assert second.json() == first.json()
    # Defect 4: the negative rating promotes the trace into the queue with a
    # truthful empty question (no raw query is retained for served SUCCESS
    # traces), so the T07 loop closes — the officer triage queue sees it.
    assert len(service.items()) == 1
    promoted = service.items()[0]
    assert promoted.trace_id == "trace-pub"
    assert promoted.question == ""
    assert not promoted.resolved

def test_feedback_route_rejects_empty_trace_id_with_400(main_module, tmp_path: Path) -> None:
    """A blank trace_id is malformed input — the route must return 400, not 500.

    T07 hardening: the queue service raises ValueError for bad input; the
    route catches it and converts to HTTPException(400). Before this change
    an empty trace_id caused an unhandled exception → 500.
    """
    service = UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")
    application = FastAPI()
    application.state.unanswered_queue = service
    route = next(r for r in main_module.app.routes if getattr(r, "path", "") == "/v2/feedback")
    application.router.routes.append(route)

    with TestClient(application) as client:
        resp = client.post("/v2/feedback", json={"trace_id": "", "helpful": True})
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"
    assert "trace_id" in resp.json()["detail"].lower()


def test_feedback_route_rejects_oversized_reason_with_400(main_module, tmp_path: Path) -> None:
    """A reason longer than 4096 chars must be 400, not 500 (T07 hardening)."""
    service = UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")
    application = FastAPI()
    application.state.unanswered_queue = service
    route = next(r for r in main_module.app.routes if getattr(r, "path", "") == "/v2/feedback")
    application.router.routes.append(route)

    with TestClient(application) as client:
        resp = client.post(
            "/v2/feedback",
            json={"trace_id": "trace-ok", "helpful": True, "reason": "x" * 4097},
        )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"
    assert "4096" in resp.json()["detail"]


def test_negative_feedback_with_question_lands_pii_redacted_item(main_module, tmp_path: Path) -> None:
    """T07: a negative rating with a question field promotes a PII-redacted
    item into the queue (the student's actual question, not an empty marker).
    """
    service = UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")
    application = FastAPI()
    application.state.unanswered_queue = service
    route = next(r for r in main_module.app.routes if getattr(r, "path", "") == "/v2/feedback")
    application.router.routes.append(route)

    raw_question = f"Học phí ngành Dược là bao nhiêu, gọi {PHONE} nhé"
    with TestClient(application) as client:
        resp = client.post(
            "/v2/feedback",
            json={
                "trace_id": "trace-neg",
                "helpful": False,
                "question": raw_question,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["recorded"] is True
    items = service.items()
    assert len(items) == 1
    assert items[0].trace_id == "trace-neg"
    assert not items[0].resolved
    # PII redacted: the raw phone number must NOT appear in the stored question
    assert PHONE not in items[0].question
    assert "[REDACTED]" in items[0].question
    # But the non-PII part of the question is preserved
    assert "Học phí" in items[0].question
