"""Tests for S16.3 — Rewrite HyDE + step-back prompts + `_should_stepback` helper.

Contract (ADR-E, CMF nice-to-have — MANDATORY per the plan):
  - `_HYDE_SYSTEM` (≤40-word instruction): produces ONE short hypothetical
    passage (≤40 words), domain vocab, "plausible-but-wrong OK, NEVER
    invent URLs", output=passage only. For embedding use only.
  - `_STEP_BACK_SYSTEM`: PRESERVE entities (major code, year, tổ hợp,
    đối tượng); abstract only the intent.
  - `_should_stepback(query) -> bool`:
      False when query has BOTH a 4-digit year (20xx) AND a major signal,
      OR is a short single-fact lookup.
      True otherwise.
  - Wired in: when `_should_stepback` returns False, the step_back call is
    skipped and the raw query is used as `step_back_query`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.services.router_cache as router_cache_mod  # noqa: E402
from src.services.query_router import (  # noqa: E402
    _HYDE_SYSTEM,
    _STEP_BACK_SYSTEM,
    SmartQueryRouter,
    _should_stepback,
)


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# _HYDE_SYSTEM prompt contract
# ─────────────────────────────────────────────────────────────────────────────


def test_hyde_system_prompt_present_and_short():
    """The HyDE prompt must exist and instruct ≤40-word passage output."""
    assert _HYDE_SYSTEM, "HyDE prompt must be defined"
    # The prompt itself can be longer (it contains instructions); the
    # instruction TEXT should say "40" or "≤40" or "khoảng" to bound it.
    text = _HYDE_SYSTEM.lower()
    assert ("40" in text) or ("≤" in text) or ("khoảng" in text), (
        f"HyDE prompt must mention the ≤40 word cap; got:\n{_HYDE_SYSTEM}"
    )


def test_hyde_system_forbids_url_invention():
    """HyDE must explicitly tell the model NEVER to invent URLs."""
    text = _HYDE_SYSTEM.lower()
    assert ("không" in text and "url" in text) or ("never" in text and "url" in text), (
        f"HyDE prompt must forbid URL invention; got:\n{_HYDE_SYSTEM}"
    )


def test_hyde_system_says_output_is_passage_only():
    """The prompt must say the output is a passage / đoạn văn only — no
    markdown, no explanation, no labels."""
    text = _HYDE_SYSTEM.lower()
    # Either an explicit "passage only" or "đoạn văn" + no markdown rule.
    has_passage_only = "đoạn văn" in text or "passage" in text
    has_no_markdown = "không" in text and "markdown" in text
    assert has_passage_only and has_no_markdown, (
        f"HyDE prompt must say output=passage only, no markdown; got:\n{_HYDE_SYSTEM}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# _STEP_BACK_SYSTEM prompt contract
# ─────────────────────────────────────────────────────────────────────────────


def test_step_back_system_present_and_preserves_entities():
    """Step-back prompt must tell the model to PRESERVE entities (major code,
    year, tổ hợp, đối tượng) — abstract only the intent."""
    assert _STEP_BACK_SYSTEM, "Step-back prompt must be defined"
    text = _STEP_BACK_SYSTEM.lower()
    # Must mention "giữ" / "bảo toàn" / "preserve" / "giữ nguyên" + entities
    has_preserve = (
        "giữ" in text or "bảo toàn" in text or "preserve" in text
    )
    has_entities = (
        "mã ngành" in text or "năm" in text or "tổ hợp" in text
        or "đối tượng" in text or "major" in text or "year" in text
    )
    assert has_preserve and has_entities, (
        f"Step-back prompt must say PRESERVE entities; got:\n{_STEP_BACK_SYSTEM}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# _should_stepback — pure helper unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_should_stepback_year_and_major_returns_false():
    """Both 4-digit year (20xx) AND major signal → False (no step-back).

    The test query from the spec: 'điểm chuẩn ngành CNTT 2026'."""
    assert _should_stepback("điểm chuẩn ngành CNTT 2026") is False


def test_should_stepback_year_and_nghanh_returns_false():
    """Year + ngành keyword → False."""
    assert _should_stepback("học phí ngành Báo chí 2025") is False


def test_should_stepback_year_and_to_hop_returns_false():
    """Year + tổ hợp → False (tổ hợp is a strong entity, no abstraction)."""
    assert _should_stepback("tổ hợp A00 2026 cho ngành CNTT") is False


def test_should_stepback_short_single_fact_returns_false():
    """Short single-fact lookup (≤5 words with major) → False."""
    assert _should_stepback("học phí ngành CNTT") is False
    assert _should_stepback("điểm chuẩn CNTT 2026") is False


def test_should_stepback_abstract_concept_returns_true():
    """Abstract concept / generic procedure → True (step back helps)."""
    # The test query from the spec.
    assert _should_stepback("làm sao để xét tuyển đại học") is True


def test_should_stepback_procedure_query_returns_true():
    """Procedure / process query (no year, no major) → True."""
    assert _should_stepback("cách đăng ký xét tuyển online") is True
    assert _should_stepback("thời gian nộp hồ sơ") is True


def test_should_stepback_year_only_without_major_returns_true():
    """Year alone (no major signal) → True (entity-poor enough to step back)."""
    assert _should_stepback("tuyển sinh năm 2026 có gì mới") is True


def test_should_stepback_major_only_without_year_returns_true():
    """Major alone (no year) + long enough to be a concept question → True.
    (A very short major-only query like 'học phí ngành CNTT' is treated as a
    single-fact lookup and returns False — see test_short_single_fact.)"""
    assert _should_stepback("ngành CNTT tại HUSC học những môn gì và ra trường làm gì") is True


# ─────────────────────────────────────────────────────────────────────────────
# Wiring: _should_stepback(False) → step_back_query == raw query (no LLM call)
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_llm(chat_calls: list):
    """LLM mock that records `chat` calls (the step-back and HyDE channels)."""
    llm = MagicMock()

    async def _chat(**kwargs):
        chat_calls.append(kwargs)
        return MagicMock(content="stepback-or-hyde-placeholder")

    llm.chat = AsyncMock(side_effect=_chat)
    llm.chat_json = AsyncMock(
        return_value={
            "route": "padded",
            "complexity": 1,
            "intent": "diem_chuan",
            "reasoning": "mock",
        }
    )
    return llm


@pytest.mark.asyncio
async def test_step_back_skipped_when_should_stepback_false():
    """When _should_stepback returns False, `step_back_query` equals the raw
    query (no abstraction).

    L3 contract update: route() no longer makes separate `llm.chat` calls for
    step_back/HyDE — it makes ONE `chat_json` (_classify_combined) call and the
    `_should_stepback` gate then decides whether to echo the raw query or use
    the model's `step_back` field. So `llm.chat` is NEVER called; the
    load-bearing assertion is `step_back_query == raw query`.
    """
    chat_calls: list = []
    llm = _make_mock_llm(chat_calls)
    router = SmartQueryRouter(llm=llm)
    result = await router.route("điểm chuẩn ngành CNTT 2026")
    assert result.step_back_query == "điểm chuẩn ngành CNTT 2026", (
        f"step_back_query must equal raw query when _should_stepback=False; "
        f"got {result.step_back_query!r}"
    )
    # L3: the legacy step_back/HyDE `chat` channel is gone — route() uses a
    # single `chat_json` call. `chat` must never be invoked.
    assert len(chat_calls) == 0, (
        f"L3: route() must not call llm.chat anymore; got {chat_calls}"
    )
    assert llm.chat_json.await_count == 1, (
        f"route() must make exactly one chat_json call; "
        f"got {llm.chat_json.await_count}"
    )


@pytest.mark.asyncio
async def test_step_back_called_when_should_stepback_true():
    """When _should_stepback returns True, the router still routes via the
    single `chat_json` call (L3); the step_back text comes from that call's
    `step_back` field, NOT a separate `chat` round-trip."""
    chat_calls: list = []
    llm = _make_mock_llm(chat_calls)
    router = SmartQueryRouter(llm=llm)
    result = await router.route("làm sao để xét tuyển đại học")
    # L3: exactly one chat_json call, zero legacy `chat` calls.
    assert len(chat_calls) == 0, (
        f"L3: route() must not call llm.chat anymore; got {chat_calls}"
    )
    assert llm.chat_json.await_count == 1, (
        f"route() must make exactly one chat_json call; "
        f"got {llm.chat_json.await_count}"
    )
    # The result still carries a step_back_query (echoed raw or model-provided).
    assert result.step_back_query, "step_back_query must be non-empty"


@pytest.mark.asyncio
async def test_step_back_skipped_uses_raw_query_for_short_fact():
    """Short single-fact 'học phí ngành CNTT' → _should_stepback False →
    step_back_query == raw query. L3: no legacy `chat` call."""
    chat_calls: list = []
    llm = _make_mock_llm(chat_calls)
    router = SmartQueryRouter(llm=llm)
    result = await router.route("học phí ngành CNTT")
    assert result.step_back_query == "học phí ngành CNTT"
    # L3: zero legacy `chat` calls, exactly one chat_json call.
    assert len(chat_calls) == 0
    assert llm.chat_json.await_count == 1
