"""S-multi-turn: bounded conversation-history feature.

Hard contracts pinned by these tests (see /agents/architect verdict):

  1. `UnifiedQueryRequest` accepts an optional `history` field AND accepts
     requests with no `history` (backward-compatible byte-identical wire).
  2. `LLMGenerator.generate_answer` prepends a bounded Vietnamese
     "LỊCH SỬ HỘI THOẠI" prefix to the generation user-message WHEN
     history is non-empty; with empty/None history the prompt is
     byte-identical to today (no extra block).
  3. Guardrail: history is ONLY folded into the LLM-classifier user
     message. The deterministic regex/folded paths in `precheck` MUST
     see the raw `query` alone — folding prior turns corrupts major
     matching and re-opens denylist DoS holes.
  4. Regression guard: the denylist DoS cases still block with 0 LLM
     calls even when history is supplied.

Zero live LLM calls. AsyncGroq + UnifiedLLMClient are mocked.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# 1. UnifiedQueryRequest — accept history, accept absence
# ---------------------------------------------------------------------------
def test_unified_query_request_accepts_history():
    from main import UnifiedQueryRequest

    req = UnifiedQueryRequest(
        query="thế điểm chuẩn?",
        history=[
            {"role": "user", "content": "HUSC có những ngành nào?"},
            {"role": "assistant", "content": "HUSC có 28 ngành đào tạo."},
        ],
    )
    assert len(req.history) == 2
    assert req.history[0]["role"] == "user"
    assert req.history[1]["role"] == "assistant"


def test_unified_query_request_backward_compatible_omits_history():
    from main import UnifiedQueryRequest

    # No history kwarg → defaults to None (NOT a ValidationError).
    req = UnifiedQueryRequest(query="học phí ngành CNTT?")
    assert req.history is None
    # Serialization must omit the field (Pydantic excludes None by default).
    dumped = req.model_dump()
    assert "history" not in dumped or dumped.get("history") is None


def test_unified_query_request_accepts_empty_history():
    from main import UnifiedQueryRequest

    req = UnifiedQueryRequest(query="điểm chuẩn?", history=[])
    assert req.history == []


# ---------------------------------------------------------------------------
# Shared guardrail fixture (mock AsyncGroq, count LLM calls)
# ---------------------------------------------------------------------------
def _build_guardrail_with_mock(monkeypatch, is_in_scope: bool, call_counter=None):
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")

    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    class _FakeCompletions:
        def __init__(self, counter):
            self._counter = counter
            self.last_kwargs: Dict[str, Any] = {}

        async def create(self, *args, **kwargs):
            if self._counter is not None:
                self._counter["calls"] += 1
            self.last_kwargs = kwargs
            content_str = (
                '{"is_in_scope": %s, "reason": "mocked"}' % str(is_in_scope).lower()
            )
            resp = MagicMock()
            resp.choices[0].message.content = content_str
            return resp

    counter = call_counter or {"calls": 0}
    fake = _FakeCompletions(counter)

    class _FakeChat:
        completions = fake

    class _FakeAsyncGroq:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(guardrail_mod, "AsyncGroq", _FakeAsyncGroq)
    guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()
    settings = RAGSettings()
    svc = guardrail_mod.GuardrailService(settings)
    return svc, guardrail_mod, fake


# ---------------------------------------------------------------------------
# 2. LLMGenerator.generate_answer: prompt diff with/without history
# ---------------------------------------------------------------------------
class _CapturingUnified:
    """Fake UnifiedLLMClient that captures the user_message passed to chat()."""

    def __init__(self):
        self._providers = ["fake-provider"]
        self.last_user_message: str = ""
        self.last_system_message: str = ""
        self.response = MagicMock()
        self.response.content = "OK"
        self.response.model = "fake-model"
        self.response.provider = "fake-provider"

    async def chat(self, *, user_message, system_message, temperature, max_tokens):
        self.last_user_message = user_message
        self.last_system_message = system_message
        return self.response


class _NoOpFallbackUnified(_CapturingUnified):
    """When passed as fallback, never gets called."""


def _build_generator_with_capture(monkeypatch):
    """Build an LLMGenerator whose primary client captures the user-message."""
    from services import llm_generator as gen_mod

    primary = _CapturingUnified()
    fallback = _NoOpFallbackUnified()
    svc = gen_mod.LLMGenerator.__new__(gen_mod.LLMGenerator)
    svc.unified_client = primary
    svc.unified_fallback_client = fallback
    svc.groq_client = None
    svc.zai_client = None
    svc.gen_model = "fake-model"
    svc.gen_fallback_model = "fake-fallback"
    svc.generation_system_prompt = "SYS"
    return svc, primary, fallback


@pytest.mark.asyncio
async def test_generate_answer_no_history_keeps_prompt_identical(monkeypatch):
    svc, primary, _ = _build_generator_with_capture(monkeypatch)
    chunks = [{"text": "ctx", "summary": "", "metadata": {}, "score": 0.5}]
    await svc.generate_answer(query="học phí?", chunks=chunks, confidence=0.5)
    # No history → prefix must be absent (byte-identical to today).
    assert "LỊCH SỬ HỘI THOẠI" not in primary.last_user_message
    # The original prompt tail must still be present.
    assert "CÂU HỎI: học phí?" in primary.last_user_message
    # The CONTEXT block is present (build_context wraps it; just check the
    # marker + the chunk's text body are both present).
    assert "CONTEXT:" in primary.last_user_message
    assert "ctx" in primary.last_user_message


@pytest.mark.asyncio
async def test_generate_answer_with_history_prepends_bounded_block(monkeypatch):
    svc, primary, _ = _build_generator_with_capture(monkeypatch)
    chunks = [{"text": "ctx", "summary": "", "metadata": {}, "score": 0.5}]
    history: List[Dict[str, str]] = [
        {"role": "user", "content": "HUSC có những ngành nào?"},
        {"role": "assistant", "content": "HUSC có 28 ngành."},
    ]
    await svc.generate_answer(
        query="thế điểm chuẩn?",
        chunks=chunks,
        confidence=0.5,
        history=history,
    )
    prompt = primary.last_user_message
    # The bounded block must be present.
    assert "LỊCH SỬ HỘI THOẠI" in prompt
    assert "Người dùng: HUSC có những ngành nào?" in prompt
    assert "Trợ lý: HUSC có 28 ngành." in prompt
    # The current question still appears AFTER the block.
    assert "CÂU HỎI: thế điểm chuẩn?" in prompt
    # Order: history prefix must come BEFORE the CONTEXT block.
    assert prompt.index("LỊCH SỬ HỘI THOẠI") < prompt.index("CONTEXT:")


@pytest.mark.asyncio
async def test_generate_answer_history_capped_at_4_messages(monkeypatch):
    svc, primary, _ = _build_generator_with_capture(monkeypatch)
    chunks = [{"text": "ctx", "summary": "", "metadata": {}, "score": 0.5}]
    history: List[Dict[str, str]] = [
        {"role": "user", "content": f"old-{i}"} for i in range(10)
    ]
    await svc.generate_answer(
        query="q",
        chunks=chunks,
        confidence=0.5,
        history=history,
    )
    prompt = primary.last_user_message
    # Only the LAST 4 of the 10 user turns survive the slice.
    for i in range(6, 10):
        assert f"old-{i}" in prompt, f"recent turn old-{i} should survive"
    for i in range(0, 6):
        assert f"old-{i}" not in prompt, f"old turn old-{i} should be sliced off"


# ---------------------------------------------------------------------------
# 3. Guardrail: history ONLY goes into the LLM-classifier user-message.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guardrail_history_only_in_llm_user_message(monkeypatch):
    counter = {"calls": 0}
    svc, mod, fake = _build_guardrail_with_mock(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    history = [
        {"role": "user", "content": "Học phí ngành CNTT HUSC?"},
        {"role": "assistant", "content": "Khoảng 10-12 triệu/năm."},
    ]
    # A query that does NOT match the denylist and does NOT match the
    # in-scope ADMISSION_KEYWORDS fast-path verbatim — i.e. one that
    # forces the LLM-classifier path so we can observe the user-message.
    decision = await svc.precheck("trường có bao nhiêu cơ sở vật chất?", history=history)
    assert decision.is_in_scope is True
    assert counter["calls"] >= 1, (
        "expected the LLM classifier to fire for this ambiguous query"
    )
    # The captured user-message must contain the history block AND the
    # current question, and the system prompt must remain unchanged.
    user_msgs = fake.last_kwargs.get("messages", [])
    user_msg_content = next(
        (m["content"] for m in user_msgs if m.get("role") == "user"), ""
    )
    assert "Ngữ cảnh trước:" in user_msg_content
    assert "Học phí ngành CNTT HUSC?" in user_msg_content
    assert "trường có bao nhiêu cơ sở vật chất?" in user_msg_content
    # System prompt is the unchanged hardened scope prompt (HUSC anchor).
    system_content = next(
        (m["content"] for m in user_msgs if m.get("role") == "system"), ""
    )
    assert "HUSC" in system_content


# ---------------------------------------------------------------------------
# 4. REGRESSION GUARD — denylist DoS cases still block with 0 LLM calls
#    even when history is supplied. This is the load-bearing test that
#    proves history did NOT leak into the regex path.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_denylist_kinh_te_still_blocks_with_history_zero_llm(monkeypatch):
    counter = {"calls": 0}
    svc, mod, fake = _build_guardrail_with_mock(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    history = [
        {"role": "user", "content": "Học phí ngành CNTT HUSC?"},
        {"role": "assistant", "content": "Khoảng 10-12 triệu."},
    ]
    decision = await svc.precheck("Ngành Kinh Tế điểm chuẩn?", history=history)
    assert decision.is_in_scope is False
    assert decision.internal_code == "NOT_IN_HUSC_SCOPE"
    assert decision.reason == "major_not_offered"
    # Critical: zero LLM calls means history did NOT leak into the
    # LLM-classifier path either; the regex denylist handled it
    # deterministically.
    assert counter["calls"] == 0, (
        f"denylist must short-circuit BEFORE any LLM call; got {counter['calls']}"
    )


@pytest.mark.asyncio
async def test_denylist_luat_with_husc_alias_still_blocks_with_history(monkeypatch):
    counter = {"calls": 0}
    svc, mod, fake = _build_guardrail_with_mock(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    history = [
        {"role": "user", "content": "Học phí ngành CNTT HUSC?"},
        {"role": "assistant", "content": "Khoảng 10-12 triệu."},
    ]
    decision = await svc.precheck("Ngành Luật HUSC học phí?", history=history)
    assert decision.is_in_scope is False
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0


@pytest.mark.asyncio
async def test_denylist_y_da_khoa_still_blocks_with_history(monkeypatch):
    counter = {"calls": 0}
    svc, mod, fake = _build_guardrail_with_mock(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    history = [
        {"role": "user", "content": "Điểm chuẩn ngành Toán HUSC?"},
        {"role": "assistant", "content": "Khoảng 18-20."},
    ]
    decision = await svc.precheck("Ngành Y đa khoa điểm chuẩn?", history=history)
    assert decision.is_in_scope is False
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# 5. Backward compat: precheck with no history still works.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guardrail_no_history_keyword_fast_path(monkeypatch):
    counter = {"calls": 0}
    svc, mod, fake = _build_guardrail_with_mock(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = await svc.precheck("Học phí ngành Công nghệ thông tin?")
    assert decision.is_in_scope is True
    # Keyword fast-path → 0 LLM calls even without history.
    assert counter["calls"] == 0


@pytest.mark.asyncio
async def test_guardrail_history_does_not_pii_match(monkeypatch):
    """SENSITIVE_PII path uses the raw query, not the history. Folding
    history into the PII regex would over-match (e.g. an old turn
    containing a number would false-block)."""
    counter = {"calls": 0}
    svc, mod, fake = _build_guardrail_with_mock(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    # Prior turn contains a 12-digit number; current query does NOT.
    history = [
        {"role": "user", "content": "Số CCCD của tôi là 012345678901."},
        {"role": "assistant", "content": "Bạn nên ẩn thông tin cá nhân."},
    ]
    decision = await svc.precheck("Học phí ngành CNTT?", history=history)
    # Current query is clean and admission-related → in-scope.
    assert decision.is_in_scope is True
    assert decision.pii_detected is False
    assert counter["calls"] == 0  # keyword fast-path
