"""Tighten the scope guardrail so other-university queries (Bách Khoa / FPT / Y Hà Nội ...)
are blocked while HUSC-only AND HUSC-vs-other comparison queries remain in_scope.

No live GROQ calls — the AsyncGroq client is mocked the same way other guardrail
tests do it. We only need to verify the deterministic fast-path guard
(_mentions_other_school) and that the hardened prompt contains the HUSC-only
anchor + comparison-exception clauses.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_service_with_mock_groq(monkeypatch, is_in_scope: bool):
    """Construct a GuardrailService with a fake AsyncGroq client that returns
    the given `is_in_scope` verdict. Mirrors the AsyncGroq mock pattern used
    by the existing router/contract tests.
    """
    # Ensure GROQ_API_KEY is non-empty so _client is created (not None fallback).
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")

    # Lazy imports so monkeypatch is in effect.
    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    # Build the fake AsyncGroq that any client() call resolves to.
    class _FakeCompletions:
        async def create(self, *args, **kwargs):
            # NOTE: precheck() does:
            #   content = resp.choices[0].message.content.strip()
            #   data = json.loads(content[content.find("{"):content.rfind("}")+1])
            # So we need `content` to be a REAL str (not a MagicMock).
            content_str = (
                '{"is_in_scope": %s, "reason": "mocked"}' % str(is_in_scope).lower()
            )
            resp = MagicMock()
            # Drill down via attribute access so the str lives on the mock
            # without being re-mocked by auto-spec.
            resp.choices[0].message.content = content_str
            return resp

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeAsyncGroq:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    # Patch the AsyncGroq class so a fresh GuardrailService uses our fake.
    monkeypatch.setattr(guardrail_mod, "AsyncGroq", _FakeAsyncGroq)

    settings = RAGSettings()
    svc = guardrail_mod.GuardrailService(settings)
    return svc, guardrail_mod


# --- 5. Hardened prompt string assertions (no LLM call at all) ---

def test_hardened_prompt_anchors_husc_only():
    """The new system prompt must (a) state HUSC-only scope, (b) name several
    other schools so the LLM is aware, and (c) explicitly call out the
    HUSC-vs-other comparison exception.
    """
    svc, mod = _build_service_with_mock_groq_no_llm()
    # Pull the prompt directly from the source so we can assert substrings.
    src = Path(mod.__file__).read_text(encoding="utf-8")
    # Find the prompt literal assigned to `prompt = (` in precheck()
    marker = 'prompt = ('
    idx = src.find(marker)
    assert idx != -1, "could not find prompt literal in guardrail.py"
    snippet = src[idx:idx + 4000]
    # Anchors required
    assert "HUSC" in snippet
    assert "NGOÀI PHẠM VI" in snippet or "NGOÀI PHẠM VI" in snippet
    # Comparison exception
    assert "SO SÁNH" in snippet or "SO SÁNH" in snippet
    # Named other schools (at least 3 of the leak set)
    named = ["Bách Khoa", "FPT", "Y Hà Nội", "Ngoại thương", "Kinh tế Quốc dân", "RMIT", "UEH"]
    found = sum(1 for n in named if n in snippet)
    assert found >= 4, f"prompt must name several other schools; only found {found}"
    # Few-shot examples present
    assert "Vật lý học" in snippet and "bán dẫn" in snippet, "must include the 2026 major few-shot"
    assert "Đại học Huế khác HUSC" in snippet, "must include HUSC-vs-other comparison few-shot"


def _build_service_with_mock_groq_no_llm(monkeypatch=None):  # helper for prompt test
    """Light-weight helper: build service + return module, used only for prompt
    substring assertions (we never call .precheck() on the result here).
    """
    import os
    if monkeypatch is None:
        from _pytest.monkeypatch import MonkeyPatch
        monkeypatch = MonkeyPatch()
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    sys.path.insert(0, str(SRC))
    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod
    settings = RAGSettings()
    svc = guardrail_mod.GuardrailService(settings)
    return svc, guardrail_mod


# --- 1. Other-school query must NOT take keyword fast-path; LLM is consulted ---

def test_other_school_query_falls_through_to_llm_and_blocks(monkeypatch):
    """'Điểm chuẩn ĐH Bách Khoa Hà Nội 2026?' contains both an admission keyword
    AND a different-school marker. The fast-path guard must let it through to
    the LLM, and the LLM is mocked to return is_in_scope=False → blocked.
    """
    svc, mod = _build_service_with_mock_groq(monkeypatch, is_in_scope=False)
    decision = asyncio.run(svc.precheck("Điểm chuẩn ĐH Bách Khoa Hà Nội năm 2026?"))
    assert decision.is_in_scope is False
    assert decision.internal_code == "NOT_IN_HUSC_SCOPE"
    # Must NOT be the keyword short-circuit reason.
    assert decision.reason != "in_scope_keyword"
    # Direct proof the fast-path guard fired (it would have blocked the keyword
    # short-circuit). We assert it via the helper.
    assert svc._mentions_other_school("Điểm chuẩn ĐH Bách Khoa Hà Nội năm 2026?") is True


# --- 2. Pure HUSC query stays in_scope (zero regression) ---

def test_pure_husc_query_still_in_scope(monkeypatch):
    """'Học phí ngành CNTT HUSC?' contains only HUSC + admission keywords —
    keyword fast-path returns in_scope=True immediately.
    """
    svc, mod = _build_service_with_mock_groq(monkeypatch, is_in_scope=False)
    decision = asyncio.run(svc.precheck("Học phí ngành CNTT HUSC?"))
    assert decision.is_in_scope is True
    assert decision.reason == "in_scope_keyword"


# --- 3. HUSC-vs-other comparison must STAY in_scope ---

def test_husc_vs_other_comparison_stays_in_scope(monkeypatch):
    """Comparison queries that mention BOTH HUSC and another school are
    legitimate admission questions — the other-school guard must NOT
    force them out, and the LLM (mocked in_scope=true) confirms it.
    """
    svc, mod = _build_service_with_mock_groq(monkeypatch, is_in_scope=True)
    decision = asyncio.run(
        svc.precheck("HUSC so với Bách Khoa ngành CNTT cái nào hơn?")
    )
    assert decision.is_in_scope is True
    # The query contains BOTH a HUSC alias AND another-school marker.
    # _mentions_husc() should win, so the other-school guard is False.
    assert svc._mentions_husc("HUSC so với Bách Khoa ngành CNTT cái nào hơn?") is True
    assert svc._mentions_other_school("HUSC so với Bách Khoa ngành CNTT cái nào hơn?") is False


# --- 4. New 2026 HUSC major — no other school — must stay in_scope ---

def test_new_2026_husc_major_in_scope(monkeypatch):
    """'Vật lý học - Công nghệ bán dẫn' is a brand-new HUSC major. The original
    bug was the keyword fast-path missing it; with HUSC-anchored logic it stays in_scope.
    The LLM (mocked in_scope=True) is what rescues it because the query
    contains NO admission keyword (no "điểm chuẩn", "học phí" etc.).
    """
    svc, mod = _build_service_with_mock_groq(monkeypatch, is_in_scope=True)
    decision = asyncio.run(
        svc.precheck("Vật lý học - Chương trình Công nghệ bán dẫn có gì hay?")
    )
    assert decision.is_in_scope is True
    # Pure HUSC major, no other-school mention → fast-path guard must be False
    # and the LLM path must have been taken.
    assert svc._mentions_other_school(
        "Vật lý học - Chương trình Công nghệ bán dẫn có gì hay?"
    ) is False
    assert svc._looks_admission_related(
        "Vật lý học - Chương trình Công nghệ bán dẫn có gì hay?"
    ) is False


# --- 6. PII still wins first, even with a different-school token present ---

def test_pii_layer_wins_first_even_with_other_school(monkeypatch):
    """A query containing BOTH a CCCD (PII keyword) AND 'Bách Khoa' must hit the
    PII layer FIRST — the other-school guard never gets a chance.
    """
    svc, mod = _build_service_with_mock_groq(monkeypatch, is_in_scope=False)
    decision = asyncio.run(
        svc.precheck("Số CCCD 012345678901 của tôi, cho hỏi điểm chuẩn Bách Khoa")
    )
    assert decision.is_in_scope is False
    assert decision.internal_code == "SENSITIVE_PII_DETECTED"
    assert decision.pii_detected is True


# --- Direct unit checks on the deterministic guards (no LLM involved) ---

def test_mentions_other_school_true_for_pure_other_school():
    from services.guardrail import GuardrailService
    gs = GuardrailService.__new__(GuardrailService)  # bypass __init__ (no LLM client)
    assert gs._mentions_other_school("Học phí ĐH FPT là bao nhiêu?") is True
    assert gs._mentions_other_school("ĐH Y Hà Nội xét tuyển thế nào?") is True
    assert gs._mentions_other_school("Trường FPT có những ngành gì?") is True
    assert gs._mentions_other_school("Điểm chuẩn Đại học Huế Y Dược?") is True


def test_mentions_other_school_false_when_husc_mentioned():
    from services.guardrail import GuardrailService
    gs = GuardrailService.__new__(GuardrailService)
    # HUSC comparison — should be False
    assert gs._mentions_other_school("HUSC so với Bách Khoa ngành CNTT") is False
    # Pure HUSC — no other school, no HUSC-vs-other: still False
    assert gs._mentions_other_school("Học phí HUSC ngành CNTT") is False
    # ĐH Huế parent + HUSC alias — False (HUSC alias present)
    assert gs._mentions_other_school("Đại học Huế khác HUSC như thế nào?") is False
    # Pure HUSC alias
    assert gs._mentions_other_school("Trường Đại học Khoa học Huế tuyển sinh ngành gì?") is False
