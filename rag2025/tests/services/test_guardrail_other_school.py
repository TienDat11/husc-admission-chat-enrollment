"""Tighten the scope guardrail so other-university queries (Bách Khoa / FPT / Y Hà Nội ...)
are blocked while HUSC-only AND HUSC-vs-other comparison queries remain in_scope.

No live LLM calls — the ns24 guardrail client is replaced by a fake with
``chat_json`` (see ``_FakeLLM``). We only need to verify the deterministic fast-path guard
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


class _FakeLLM:
    """Stands in for the ns24 guardrail role client; records chat_json calls."""

    def __init__(self, is_in_scope: bool):
        self._is_in_scope = is_in_scope
        self.token_ledger = None
        self._providers = []
        self.chat_json = AsyncMock(
            side_effect=lambda **_kw: {"is_in_scope": is_in_scope, "reason": "mocked"}
        )


def _build_service_with_mock_llm(monkeypatch, is_in_scope: bool):
    """Build a GuardrailService whose ns24 guardrail client is a fake that
    returns the given `is_in_scope` verdict via chat_json."""
    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    svc = guardrail_mod.GuardrailService(RAGSettings(_env_file=None))
    svc._llm = _FakeLLM(is_in_scope)
    svc._model = "fake-guardrail-model"
    return svc, guardrail_mod


# --- 5. Hardened prompt string assertions (no LLM call at all) ---

def test_hardened_prompt_anchors_husc_only():
    """The new system prompt must (a) state HUSC-only scope, (b) name several
    other schools so the LLM is aware, and (c) explicitly call out the
    HUSC-vs-other comparison exception.
    """
    svc, mod = _build_service_with_mock_llm_no_llm()
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


def _build_service_with_mock_llm_no_llm(monkeypatch=None):  # helper for prompt test
    """Light-weight helper: build service + return module, used only for prompt
    substring assertions (we never call .precheck() on the result here).
    """
    if monkeypatch is None:
        from _pytest.monkeypatch import MonkeyPatch
        monkeypatch = MonkeyPatch()
    sys.path.insert(0, str(SRC))
    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod
    settings = RAGSettings()
    svc = guardrail_mod.GuardrailService(settings)
    return svc, guardrail_mod


# --- 1. Other-school query must NOT take keyword fast-path; LLM is consulted ---

def test_other_school_query_falls_through_to_llm_and_blocks(monkeypatch):
    """Other-school query is blocked deterministically before any LLM call."""
    svc, _ = _build_service_with_mock_llm(monkeypatch, is_in_scope=True)
    svc._llm.chat_json = AsyncMock()

    decision = asyncio.run(svc.precheck("ĐH Ngoại Thương tuyển sinh 2026 ngành nào?"))

    assert decision.is_in_scope is False
    assert decision.internal_code == "NOT_IN_HUSC_SCOPE"
    assert decision.reason == "other_school_detected"
    svc._llm.chat_json.assert_not_awaited()


# --- 2. Pure HUSC query stays in_scope (zero regression) ---

def test_pure_husc_query_still_in_scope(monkeypatch):
    """'Học phí ngành CNTT HUSC?' contains only HUSC + admission keywords —
    keyword fast-path returns in_scope=True immediately.
    """
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=False)
    decision = asyncio.run(svc.precheck("Học phí ngành CNTT HUSC?"))
    assert decision.is_in_scope is True
    assert decision.reason == "in_scope_keyword"


# --- 3. HUSC-vs-other comparison must STAY in_scope ---

def test_husc_vs_other_comparison_stays_in_scope(monkeypatch):
    """Comparison queries that mention BOTH HUSC and another school are
    legitimate admission questions — the other-school guard must NOT
    force them out, and the LLM (mocked in_scope=true) confirms it.
    """
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=True)
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
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=True)
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
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=False)
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
