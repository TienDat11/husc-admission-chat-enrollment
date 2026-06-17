"""Tests for query_router auto-answer split (S14.5 / ADR-3).

Kills the structural hallucination: the `hyde_auto_answer` route used to return
a hardcoded block (HYDE_META_ANSWER_INTRO) listing 28 ngành 2026 + fabricated
codes + a wrong phone. ADR-3 splits the gate into:
  1. MAJORS questions ("ngành gì", "có những ngành", "tìm hiểu các ngành",
     "trường có ngành") → must flow through RETRIEVAL (skip_retrieval=False,
     auto_answer=None).
  2. Pure contact/intro (hotline, website, link, address, contact, fanpage,
     "giới thiệu về trường", "thông tin chung") → year-agnostic CONTACT_BLOCK
     (no majors, no codes, no tuition, no 4-digit year literal).

TDD: these tests pin the contract. The vague_reject gate is unchanged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.services.query_router import (  # noqa: E402
    CONTACT_BLOCK,
    HYDE_REJECT_VAGUE,
    SmartQueryRouter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_llm():
    """Build a mock LLM client that returns safe responses for the three router
    LLM calls (step-back, hyde doc, classify).

    The two pre-routing gates (vague_reject + auto_answer) run BEFORE any LLM
    call, so they don't need a mock. The major-question tests below DO need a
    mock because they intentionally fall through to the LLM-based routing.
    """
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=MagicMock(content="placeholder abstract or hypothetical doc")
    )
    llm.chat_json = AsyncMock(
        return_value={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "nganh_hoc",
            "reasoning": "test fallback",
            "hyde_variants": ["variant-1"],
        }
    )
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# ADR-3 #1: MAJORS questions must go through retrieval
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_truong_co_nganh_gi_routes_to_retrieval():
    """'Trường có ngành gì?' must NOT be answered by the static block - flow to retrieval."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Trường có ngành gì?")
    assert result.skip_retrieval is False
    assert result.auto_answer is None


@pytest.mark.asyncio
async def test_tim_hieu_cac_nganh_routes_to_retrieval():
    """'tìm hiểu các ngành của trường' must flow to retrieval (not auto-answer)."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("cho em tìm hiểu các ngành của trường")
    assert result.skip_retrieval is False
    assert result.auto_answer is None


@pytest.mark.asyncio
async def test_truong_co_nhung_nganh_routes_to_retrieval():
    """'trường có những ngành' variant must flow to retrieval."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Trường có những ngành nào ạ?")
    assert result.skip_retrieval is False
    assert result.auto_answer is None


# ─────────────────────────────────────────────────────────────────────────────
# ADR-3 #2: Pure contact/intro returns CONTACT_BLOCK
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hotline_returns_contact_block():
    """'Hotline trường là gì?' must return CONTACT_BLOCK (skip retrieval)."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Hotline trường là gì?")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


@pytest.mark.asyncio
async def test_website_link_returns_contact_block():
    """'cho em xin link website trường' must return CONTACT_BLOCK."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("cho em xin link website trường")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


@pytest.mark.asyncio
async def test_dia_chi_truong_returns_contact_block():
    """'địa chỉ trường' is pure contact."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Địa chỉ trường ở đâu vậy?")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


@pytest.mark.asyncio
async def test_lien_he_truong_returns_contact_block():
    """'liên hệ với trường' is pure contact."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Liên hệ với trường như thế nào?")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


@pytest.mark.asyncio
async def test_fanpage_returns_contact_block():
    """'fanpage' of the school is pure contact."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Trường có fanpage không ạ?")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


@pytest.mark.asyncio
async def test_gioi_thieu_truong_returns_contact_block():
    """'giới thiệu về trường' is pure intro/contact."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Cho em giới thiệu về trường")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT_BLOCK content invariants — year-agnostic, no fabricated facts
# ─────────────────────────────────────────────────────────────────────────────


def test_contact_block_has_no_major_code():
    """CONTACT_BLOCK must NOT contain a major-code pattern (7 digits + optional 2-letter suffix)."""
    assert not re.search(r"7\d{6}([A-Z]{2})?", CONTACT_BLOCK), (
        f"CONTACT_BLOCK must be year-agnostic and free of major codes, got:\n{CONTACT_BLOCK}"
    )


def test_contact_block_has_no_year_literal():
    """CONTACT_BLOCK must NOT contain any 4-digit year literal in factual claims."""
    assert not re.search(r"\b20\d{2}\b", CONTACT_BLOCK), (
        f"CONTACT_BLOCK must be year-agnostic (no 20xx literal), got:\n{CONTACT_BLOCK}"
    )


def test_contact_block_does_not_contain_hallucinated_phone():
    """The hallucinated phone 0385.887.111 (flagged in eval) must NOT appear in CONTACT_BLOCK."""
    assert "0385.887.111" not in CONTACT_BLOCK, (
        "CONTACT_BLOCK must not contain the hallucinated phone 0385.887.111"
    )


def test_contact_block_contains_official_hotline():
    """CONTACT_BLOCK must contain the official hotline 0234.3823290 (Phòng Đào tạo)."""
    assert "0234.3823290" in CONTACT_BLOCK, (
        f"CONTACT_BLOCK must reference the official hotline, got:\n{CONTACT_BLOCK}"
    )


def test_contact_block_contains_official_urls():
    """CONTACT_BLOCK must contain both the admission portal and the school homepage."""
    assert "https://tuyensinh.husc.edu.vn" in CONTACT_BLOCK
    assert "https://husc.edu.vn" in CONTACT_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Existing vague_reject gate must still work (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vague_question_mark_rejected():
    """Bare '?' must still trigger vague_reject (unchanged)."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("?")
    assert result.intent == "vague_reject"
    assert result.skip_retrieval is True
    assert result.auto_answer == HYDE_REJECT_VAGUE


@pytest.mark.asyncio
async def test_vague_em_hoi_rejected():
    """Bare 'em hỏi' must still trigger vague_reject (unchanged)."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("em hỏi")
    assert result.intent == "vague_reject"
    assert result.skip_retrieval is True


# ─────────────────────────────────────────────────────────────────────────────
# Specific-query exclusion must still flow through to retrieval
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_specific_query_still_routes_to_retrieval():
    """Even with contact keyword, a SPECIFIC question (e.g. điểm chuẩn ngành) must go to retrieval.

    The _SPECIFIC_QUERY exclusion must still apply to whatever auto-answer
    behaviour remains. This guards against accidental removal of the exclusion.
    """
    router = SmartQueryRouter(llm=_make_mock_llm())
    # "điểm chuẩn ngành" should be routed to retrieval, NOT the contact block.
    result = await router.route("Cho em hỏi hotline và điểm chuẩn ngành CNTT")
    # The "hotline" contact keyword would match, but "điểm chuẩn" is a specific
    # exclusion → must fall through to retrieval.
    assert result.skip_retrieval is False
    assert result.auto_answer is None
