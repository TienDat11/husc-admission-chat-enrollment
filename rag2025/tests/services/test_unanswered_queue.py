"""Unanswered-question queue — ticket 7.

Asserted at the service seam, the way the conflict queue's own tests are:
PII must never survive into the queue file, resolution must fill
``resolved_at`` + ``resolved_by_record_id``, and feedback must be idempotent
by ``trace_id``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.unanswered_queue import (  # noqa: E402
    ItemNotFoundError,
    UnansweredQueueService,
)

PHONE = "0912345678"


@pytest.fixture
def svc(tmp_path: Path) -> UnansweredQueueService:
    return UnansweredQueueService(queue_path=tmp_path / "queue.jsonl")


def test_push_redacts_phone_before_writing(svc: UnansweredQueueService) -> None:
    question = f"Điểm chuẩn ngành CNTT là bao nhiêu, liên hệ {PHONE}?"
    item = svc.push(question, "trace-1", "guardrail", "NOT_IN_HUSC_SCOPE")

    raw = svc.queue_path.read_text(encoding="utf-8")
    assert PHONE not in raw
    assert "[REDACTED]" in raw
    assert PHONE not in item.question
    assert item.trace_id == "trace-1"
    assert item.status_code == "NOT_IN_HUSC_SCOPE"
    assert item.resolved_at is None
    assert item.resolved_by_record_id is None


def test_push_is_idempotent_by_trace_id(svc: UnansweredQueueService) -> None:
    svc.push("câu hỏi một", "trace-dup", "guardrail", "NOT_IN_HUSC_SCOPE")
    second = svc.push("câu hỏi khác cùng trace", "trace-dup", "guardrail", "NOT_IN_HUSC_SCOPE")

    assert second.question == "câu hỏi một"
    lines = [line for line in svc.queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_resolve_fills_resolved_at_and_record(svc: UnansweredQueueService) -> None:
    svc.push("câu hỏi cần giải quyết", "trace-resolve", "padded_rag", "INSUFFICIENT_DATA")

    resolved = svc.resolve("trace-resolve", "HOI_DAP:Q&A:cntt:2026:faq-1")

    assert resolved.resolved
    assert resolved.resolved_at is not None
    assert resolved.resolved_by_record_id == "HOI_DAP:Q&A:cntt:2026:faq-1"
    # The redacted question survives resolution untouched.
    assert resolved.question == "câu hỏi cần giải quyết"


def test_resolve_unknown_trace_id_raises(svc: UnansweredQueueService) -> None:
    with pytest.raises(ItemNotFoundError):
        svc.resolve("no-such-trace", "HOI_DAP:x")


def test_unanswered_lists_only_open_items(svc: UnansweredQueueService) -> None:
    svc.push("mở một", "t-open-1", "guardrail", "NOT_IN_HUSC_SCOPE")
    svc.push("mở hai", "t-open-2", "guardrail", "NOT_IN_HUSC_SCOPE")
    svc.push("sẽ đóng", "t-closed", "guardrail", "NOT_IN_HUSC_SCOPE")
    svc.resolve("t-closed", "HOI_DAP:x")

    assert {item.trace_id for item in svc.unanswered()} == {"t-open-1", "t-open-2"}
    assert len(svc.items()) == 3


def test_feedback_idempotent_by_trace_id(svc: UnansweredQueueService) -> None:
    svc.push("câu hỏi có feedback", "trace-fb", "guardrail", "NOT_IN_HUSC_SCOPE")

    first = svc.feedback("trace-fb", False, reason="chưa đủ chi tiết")
    second = svc.feedback("trace-fb", True, reason="đổi ý")

    assert first["helpful"] is False
    assert first["linked_item"] is True
    assert second == first
    feedback_lines = [
        line for line in svc.feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(feedback_lines) == 1


def test_feedback_without_queued_item_is_still_recorded(svc: UnansweredQueueService) -> None:
    record = svc.feedback("trace-unknown", True)

    assert record["helpful"] is True
    assert record["linked_item"] is False
    assert json.loads(svc.feedback_path.read_text(encoding="utf-8").splitlines()[0])["trace_id"] == "trace-unknown"


def test_feedback_redacts_phone_email_cccd_in_reason(svc: UnansweredQueueService) -> None:
    """Defect 1: PII in a feedback reason must be masked in the persisted row."""
    phone = "0912345678"
    cccd = "001234567890"
    email = "test@example.com"
    reason = f"call {phone}, email {email}, CCCD {cccd}"
    svc.push("câu hỏi", "trace-pii-fb", "guardrail", "NOT_IN_HUSC_SCOPE")

    record = svc.feedback("trace-pii-fb", False, reason=reason)

    raw = svc.feedback_path.read_text(encoding="utf-8")
    assert phone not in raw
    assert email not in raw
    assert cccd not in raw
    assert "[REDACTED]" in raw
    assert record["reason"] != reason  # the returned record is redacted too


def test_feedback_negative_rating_on_success_trace_promotes_to_queue(
    svc: UnansweredQueueService,
) -> None:
    """Defect 4: a negative rating on a served SUCCESS trace becomes visible
    in unanswered() so the T07 loop actually closes."""
    record = svc.feedback(
        "trace-success-neg",
        False,
        question="Điểm chuẩn CNTT?",
        route="padded_rag",
        status_code="SUCCESS",
    )
    assert record["linked_item"] is True
    open_items = svc.unanswered()
    assert len(open_items) == 1
    assert open_items[0].trace_id == "trace-success-neg"
    assert "Điểm chuẩn" in open_items[0].question


def test_feedback_negative_without_question_still_promotes(
    svc: UnansweredQueueService,
) -> None:
    """Defect 4 without a question string: the trace still appears in
    unanswered() with a truthful empty question — no invented text."""
    record = svc.feedback(
        "trace-success-neg2",
        False,
        route="padded_rag",
        status_code="SUCCESS",
    )
    assert record["linked_item"] is True
    open_items = svc.unanswered()
    assert len(open_items) == 1
    assert open_items[0].trace_id == "trace-success-neg2"
    assert open_items[0].question == ""


def test_feedback_positive_rating_does_not_promote(
    svc: UnansweredQueueService,
) -> None:
    """A helpful=True rating records feedback but does NOT create a queue
    item for a trace that was not already queued."""
    record = svc.feedback(
        "trace-success-pos",
        True,
        route="padded_rag",
        status_code="SUCCESS",
    )
    assert record["linked_item"] is False
    assert svc.unanswered() == []


def test_feedback_refuses_overlong_trace_id(svc: UnansweredQueueService) -> None:
    """Defect 5: a trace_id longer than 128 characters is refused."""
    with pytest.raises(ValueError, match="exceeds 128"):
        svc.feedback("x" * 129, True)


def test_feedback_refuses_overlong_reason(svc: UnansweredQueueService) -> None:
    """Defect 5: a reason longer than 4096 characters is refused."""
    with pytest.raises(ValueError, match="exceeds 4096"):
        svc.feedback("trace-ok", True, reason="x" * 4097)


def test_feedback_rejects_empty_trace_id(svc: UnansweredQueueService) -> None:
    """Defect 5: an empty or whitespace-only trace_id is refused."""
    with pytest.raises(ValueError, match="must not be empty"):
        svc.feedback("   ", True)


def test_feedback_within_bounds_records_normally(svc: UnansweredQueueService) -> None:
    """Defect 5: a normal rating at the boundary still records."""
    record = svc.feedback("trace-boundary", True, reason="x" * 4096)
    assert record["helpful"] is True
    assert len(record["reason"]) == 4096
