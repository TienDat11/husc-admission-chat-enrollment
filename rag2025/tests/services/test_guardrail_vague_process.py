"""Regression tests for the S19 guardrail false-block on vague process-questions.

The original S19 major-scope layer accidentally over-matched by including
interrogative tails (thế nào / ra sao / có gì / the nao / co gi) inside
``_MAJOR_TAIL_RX``. Pattern 2/4 of ``_extract_major_phrase`` then used
those tails to grab a verb-phrase like "Đăng ký vào trường" as if it were
a major name, missed the allowlist, and the layer wrongly blocked valid
admission process-questions. Compounded by Layer-4's ADMISSION_KEYWORDS
missing process-verbs (đăng ký / nhập học / thủ tục / nộp hồ sơ), which
forced colloquial process-questions to fall through to the LLM and get
rejected as out-of-scope.

These tests pin BOTH fixes:
  1. _MAJOR_TAIL_RX no longer contains interrogative tails.
  2. ADMISSION_KEYWORDS now contains process-verbs, so process-questions
     take the deterministic keyword fast-path (no LLM call needed).

CRITICAL — DoS hole must stay closed: denylist majors (Kinh tế / Luật /
Y đa khoa) must STILL be blocked. Those are caught by Pattern 1
("ngành <X>") + the denylist, which do NOT depend on the interrogative
tails we just removed.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_service_with_mock_groq(monkeypatch, is_in_scope: bool, call_counter=None):
    """Mirror the AsyncGroq mock fixture used by the existing guardrail tests.

    Returns ``(svc, guardrail_mod)``. The fake ``.create()`` returns
    ``is_in_scope`` and counts every invocation through ``call_counter``.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")

    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    class _FakeCompletions:
        def __init__(self, counter):
            self._counter = counter

        async def create(self, *args, **kwargs):
            if self._counter is not None:
                self._counter["calls"] += 1
            content_str = (
                '{"is_in_scope": %s, "reason": "mocked"}' % str(is_in_scope).lower()
            )
            resp = MagicMock()
            resp.choices[0].message.content = content_str
            return resp

    class _FakeChat:
        completions = _FakeCompletions(call_counter)

    class _FakeAsyncGroq:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(guardrail_mod, "AsyncGroq", _FakeAsyncGroq)
    guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()
    settings = RAGSettings()
    svc = guardrail_mod.GuardrailService(settings)
    return svc, guardrail_mod


def _llm_call_counter():
    return {"calls": 0}


# ---------------------------------------------------------------------------
# A. The two original user-reported false-blocks must now resolve to in_scope.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "Làm sao để đăng ký xét tuyển?",
    "Đăng ký vào trường thế nào?",
    "Thủ tục nhập học ra sao?",
    "Đăng ký vào trường như lào?",
    "Cách nộp hồ sơ vào trường?",
])
def test_vague_process_questions_in_scope(monkeypatch, q):
    """Process-questions must NOT be blocked. They should reach the
    keyword fast-path (in_scope_keyword) for the queries that contain
    one of the new ADMISSION_KEYWORDS (đăng ký, nhập học, thủ tục, nộp hồ sơ).
    Queries that still need the LLM are mocked to return in_scope=True.
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is True, (
        f"process-question should NOT be blocked: {q!r} → "
        f"reason={decision.reason!r} short={decision.short_answer!r}"
    )
    assert decision.reason != "major_not_offered", (
        f"process-question was wrongly classified as major_not_offered: {q!r}"
    )


# ---------------------------------------------------------------------------
# B. The DoS hole MUST stay closed: denylist majors still blocked.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "Ngành Kinh Tế điểm chuẩn?",
    "Ngành Luật HUSC?",
    "Ngành Y đa khoa?",
])
def test_denylist_still_blocked(monkeypatch, q):
    """HUSC has never offered Kinh tế / Luật / Y đa khoa — they MUST stay blocked.
    These are caught by Pattern 1 ('ngành <X>') + the MAJOR_DENYLIST, which
    do not depend on the interrogative tails we just removed.
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is False, (
        f"denylist major must be blocked: {q!r} → "
        f"reason={decision.reason!r}"
    )
    assert decision.reason == "major_not_offered", (
        f"denylist block must come from major-scope, not the LLM: {q!r} "
        f"(reason={decision.reason!r})"
    )
    # Denylist blocks are deterministic — NO LLM call should occur.
    assert counter["calls"] == 0, (
        f"denylist block must be deterministic (no LLM call): {q!r}"
    )


# ---------------------------------------------------------------------------
# C. Real HUSC majors must stay in_scope (zero regression).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "Học phí CNTT HUSC?",
    "Khoa học dữ liệu điểm chuẩn?",
])
def test_real_majors_still_in_scope(monkeypatch, q):
    """Real HUSC majors must keep working. They take the keyword fast-path."""
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=False, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is True, (
        f"real HUSC major should be in_scope: {q!r} → reason={decision.reason!r}"
    )
    assert decision.reason == "in_scope_keyword", (
        f"real majors with admission keyword should hit fast-path: {q!r} "
        f"(reason={decision.reason!r})"
    )
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# D. Direct structural assertions on the two fixes.
# ---------------------------------------------------------------------------

def test_major_tail_rx_no_interrogative_tails():
    """Pin the structural fix: _MAJOR_TAIL_RX must NOT contain
    thế nào | the nao | ra sao | có gì | co gi as anchors any more.
    """
    from services.guardrail import GuardrailService

    # Inspect the compiled regex pattern directly — interrogative tails
    # should not appear INSIDE the _MAJOR_TAIL_RX alternation.
    bad_tails = ["thế nào", "the nao", "ra sao", "có gì", "co gi"]
    pattern = GuardrailService._MAJOR_TAIL_RX.pattern
    for bad in bad_tails:
        assert bad not in pattern, (
            f"_MAJOR_TAIL_RX still contains interrogative tail {bad!r}; "
            f"this is the root cause of the false-block regression."
        )


def test_admission_keywords_has_process_verbs():
    """Pin the structural fix: ADMISSION_KEYWORDS must contain the
    process-verbs that short-circuit the fast-path.
    """
    from services.guardrail import GuardrailService

    for kw in ("đăng ký", "đăng kí", "nhập học", "thủ tục", "nộp hồ sơ"):
        assert kw in GuardrailService.ADMISSION_KEYWORDS, (
            f"ADMISSION_KEYWORDS missing process-verb {kw!r}; "
            f"process-questions will fall through to the LLM."
        )


# ---------------------------------------------------------------------------
# E. The full set of process-questions must NOT use the LLM.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "Làm sao để đăng ký xét tuyển?",
    "Đăng ký vào trường thế nào?",
    "Thủ tục nhập học ra sao?",
    "Cách nộp hồ sơ vào trường?",
])
def test_process_questions_reach_keyword_fast_path(monkeypatch, q):
    """Each of the process-questions that contains a new ADMISSION_KEYWORDS
    entry should short-circuit to the keyword fast-path. NO LLM call.
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.reason == "in_scope_keyword", (
        f"process-question should hit keyword fast-path: {q!r} "
        f"(reason={decision.reason!r})"
    )
    assert counter["calls"] == 0, (
        f"LLM must NOT be called for keyword-fast-path: {q!r}"
    )
