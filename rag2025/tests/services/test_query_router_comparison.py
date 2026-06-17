"""Tests for query_router S15.3 — comparison patterns + threshold + default-graph.

ADR-C / Wave B Lane B. TDD: these tests pin the contract.

Three changes covered:
1. ``_COMPARISON_PATTERNS`` forces ``graph_rag`` for so sánh / khác nhau /
   so với / hơn / kém / chênh lệch / "A hay B" disjunction.
2. Threshold escalation changes from ``complexity > threshold`` to
   ``complexity >= threshold`` (so complexity-2 now escalates with the
   default threshold=2).
3. On the ``_classify`` exception fallback, the default route becomes
   ``graph_rag`` (fail toward MORE context, not less). Genuine upstream
   ``vague_reject`` is unaffected.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.services.router_cache as router_cache_mod  # noqa: E402
from src.services.query_router import (  # noqa: E402
    HYDE_REJECT_VAGUE,
    QueryRoute,
    SmartQueryRouter,
)


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    """Reset the global router cache between tests so cached results from
    other tests (or earlier runs) don't shadow our mocks."""
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield


def _make_mock_llm(classify_payload=None, classify_raises: bool = False):
    """Build a mock UnifiedLLMClient mirroring `test_query_router_autoanswer`.

    - ``chat`` (step-back, HyDE doc) returns harmless placeholder text.
    - ``chat_json`` (classify) returns ``classify_payload`` if given, else a
      ``padded_rag``/complexity=1 default. If ``classify_raises`` is True the
      classify call raises a RuntimeError to exercise the fallback path.
    """
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="placeholder doc"))
    if classify_raises:
        llm.chat_json = AsyncMock(side_effect=RuntimeError("classify boom"))
    else:
        llm.chat_json = AsyncMock(
            return_value=classify_payload
            or {
                "route": "padded_rag",
                "complexity": 1,
                "intent": "diem_chuan",
                "reasoning": "mock",
                "hyde_variants": ["v1"],
            }
        )
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# S15.3 — comparison patterns force graph_rag
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_so_sanh_pattern_forces_graph_rag():
    """'so sánh ngành A và ngành B' must route graph_rag even when the
    classifier said padded_rag."""
    router = SmartQueryRouter(llm=_make_mock_llm())  # classifier says padded_rag
    result = await router.route("so sánh ngành CNTT và ngành Toán năm 2026")
    assert result.route == QueryRoute.GRAPH_RAG
    assert result.complexity >= 3


@pytest.mark.asyncio
async def test_so_sanh_no_diacritic_forces_graph_rag():
    """'so sanh' (no Vietnamese diacritics) must also match — common typo."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("so sanh nganh CNTT va nganh Toan 2026")
    assert result.route == QueryRoute.GRAPH_RAG


@pytest.mark.asyncio
async def test_khac_nhau_pattern_forces_graph_rag():
    """'khác nhau' triggers comparison routing."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("ngành CNTT và Toán khác nhau ở điểm nào ạ?")
    assert result.route == QueryRoute.GRAPH_RAG


@pytest.mark.asyncio
async def test_so_voi_pattern_forces_graph_rag():
    """'so với' triggers comparison routing."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("học phí ngành CNTT so với ngành Toán cao hay thấp?")
    assert result.route == QueryRoute.GRAPH_RAG


@pytest.mark.asyncio
async def test_chenh_lech_pattern_forces_graph_rag():
    """'chênh lệch' triggers comparison routing."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("chênh lệch điểm chuẩn 2025 và 2026 là bao nhiêu?")
    assert result.route == QueryRoute.GRAPH_RAG


@pytest.mark.asyncio
async def test_hay_disjunction_forces_graph_rag():
    """'ngành A hay ngành B tốt hơn' → graph_rag (the 'A hay B' disjunction)."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("ngành CNTT hay ngành Toán phù hợp với em hơn?")
    assert result.route == QueryRoute.GRAPH_RAG


# ─────────────────────────────────────────────────────────────────────────────
# S15.3 — threshold escalation now uses >= (was >)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complexity_equal_threshold_bumps_padded_to_hybrid():
    """S16.2 / CMF-2: padded + complexity==threshold → HYBRID (not graph).
    The pre-S16.2 behavior was padded → graph (S15.3 widening); the new
    contract deletes the >=-to-graph gate and replaces it with a
    padded-to-hybrid bump. classify is authoritative, complexity only
    nudges padded → hybrid (never silent → graph)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 2,
            "intent": "diem_chuan",
            "reasoning": "two-things",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    # Use a benign query with NO comparison/enumeration keywords so the override
    # we are testing is the threshold one, not the comparison detector.
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.HYBRID
    assert result.complexity == 2


@pytest.mark.asyncio
async def test_complexity_below_threshold_stays_padded_rag():
    """complexity=1 < threshold=2 → stays padded_rag (sanity for the new >=)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "diem_chuan",
            "reasoning": "single",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.PADDED_RAG


# ─────────────────────────────────────────────────────────────────────────────
# S16.2 — _classify exception defaults to hybrid (CMF-1 (d) / fail toward
# safe middle, not graph). Replaces the S15.3 fail-toward-more-context default.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_exception_defaults_hybrid():
    """When the LLM classify call raises, the fallback is HYBRID
    (NOT graph_rag) so we lean toward the safe middle. S16.2 / CMF-1 (d)."""
    llm = _make_mock_llm(classify_raises=True)
    router = SmartQueryRouter(llm=llm)
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.HYBRID


# ─────────────────────────────────────────────────────────────────────────────
# Genuine upstream vague_reject still works (UNCHANGED — must NOT be affected)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bare_question_mark_still_vague_reject():
    """Bare '?' must still trigger vague_reject BEFORE the classify call so the
    default-graph fallback in S15.3 cannot leak into genuine vague inputs."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("?")
    assert result.intent == "vague_reject"
    assert result.skip_retrieval is True
    assert result.auto_answer == HYDE_REJECT_VAGUE
