"""Tests for S16.2 — HyDE out of classify + 3-way classify rewrite + precedence.

Contract (AMF-2, AMF-5, CMF-1, CMF-2, CMF-5):
  - `_classify(query)` takes ONLY the raw query (no hyde_doc concat).
  - Classify output JSON: {route: padded|hybrid|graph, complexity:1-5, intent,
    reasoning}.
  - route() precedence (terminal-marked, exact order):
      1. vague/contact auto-answer (unchanged, terminal).
      2. enum pattern → graph (terminal).
      3. comparison pattern → graph (terminal).
      4. classify route is AUTHORITATIVE (padded|hybrid|graph) — taken as-is.
      5. bump: padded+complexity >= threshold(2) → hybrid (NEVER → graph).
      6. safety demote: padded + ≥2 NGANH/TO_HOP entities detected → hybrid
         (only demotion; never demote graph or hybrid).
      7. classify-exception → hybrid (was graph).
  - DELETE the >= -to-graph gate (was at old :447-449). NO silent escalation
    from classify=graph to something else; classify is authoritative.
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
    _COMPARISON_PATTERNS,
    _ENUMERATION_PATTERNS,
)
from src.services.contact_block import CONTACT_BLOCK  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM factory
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_llm(
    classify_payload=None,
    classify_raises: bool = False,
    capture_calls: bool = False,
):
    """Mock UnifiedLLMClient.

    - ``chat`` (step-back, hyde) returns placeholder text.
    - ``chat_json`` (classify) returns ``classify_payload`` if given, else
      a default padded/complexity=1. If ``classify_raises`` is True, raises.
    - When ``capture_calls`` is True, records chat_json call args.
    """
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="placeholder"))
    if capture_calls:
        captured = []
        llm.chat_json = AsyncMock(
            side_effect=lambda **kw: captured.append(kw) or (
                classify_payload
                or {
                    "route": "padded_rag",
                    "complexity": 1,
                    "intent": "diem_chuan",
                    "reasoning": "default",
                    "hyde_variants": ["v1"],
                }
            )
        )
        llm._captured = captured
    if classify_raises:
        llm.chat_json = AsyncMock(side_effect=RuntimeError("classify boom"))
    elif not capture_calls:
        llm.chat_json = AsyncMock(
            return_value=classify_payload
            or {
                "route": "padded_rag",
                "complexity": 1,
                "intent": "diem_chuan",
                "reasoning": "default",
                "hyde_variants": ["v1"],
            }
        )
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# (a) Classify says padded but comparison pattern → graph (override terminal)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_padded_plus_comparison_pattern_forces_graph():
    """Even if the LLM classifier says padded, a comparison pattern in the
    query text still forces graph (override beats classify, terminal)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "looks like one-fact lookup",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    # Comparison pattern + SINGLE entity (no entity-pair demote) → must end graph.
    result = await router.route("học phí ngành CNTT so với ngành Toán")
    assert result.route == QueryRoute.GRAPH_RAG


@pytest.mark.asyncio
async def test_classify_padded_plus_enum_pattern_forces_graph():
    """Enumeration pattern beats classify=hybrid; forces graph (terminal)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "hybrid",
            "complexity": 2,
            "intent": "nganh_hoc",
            "reasoning": "lists majors",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = await router.route("trường có bao nhiêu ngành đào tạo?")
    assert result.route == QueryRoute.GRAPH_RAG


# ─────────────────────────────────────────────────────────────────────────────
# (b) Classify padded + complexity≥2 → bumped to hybrid
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_non_numeric_complexity_falls_back_to_one():
    """If the LLM returns a non-numeric string like 'high' for complexity,
    parsing must not crash and complexity must fall back to 1."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": "high",
            "intent": "hoc_phi",
            "reasoning": "model returned a label",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = await router.route("học phí ngành CNTT")
    assert result.complexity == 1
    assert result.route == QueryRoute.PADDED_RAG


@pytest.mark.asyncio
async def test_classify_padded_complexity_2_bumps_to_hybrid():
    """padded + complexity=2 (== threshold) → hybrid (CMF-2 bump, NO graph)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 2,
            "intent": "chinh_sach",
            "reasoning": "two-attribute lookup",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.HYBRID
    # Complexity preserved (the bump doesn't reset it).
    assert result.complexity == 2


@pytest.mark.asyncio
async def test_classify_padded_complexity_3_bumps_to_hybrid():
    """padded + complexity=3 → hybrid (must NEVER escalate to graph)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 3,
            "intent": "chinh_sach",
            "reasoning": "two-attribute",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.HYBRID


# ─────────────────────────────────────────────────────────────────────────────
# (c) Classify padded + complexity=1 → stays padded (no escalation)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_padded_complexity_1_stays_padded():
    """padded + complexity=1 (< threshold=2) → stays padded. No bump."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "single fact",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    result = await router.route("học phí ngành CNTT bao nhiêu")
    assert result.route == QueryRoute.PADDED_RAG


@pytest.mark.asyncio
async def test_classify_hybrid_complexity_1_stays_hybrid():
    """hybrid + complexity=1 → hybrid (no bump, no demote)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "hybrid",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "default safe-middle",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    result = await router.route("học phí ngành CNTT")
    assert result.route == QueryRoute.HYBRID


# ─────────────────────────────────────────────────────────────────────────────
# (d) Classify-exception → hybrid (the new safe-middle default)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_exception_defaults_hybrid():
    """When classify raises, the fallback is HYBRID (not graph).
    S16.2 / CMF-1 (d): the safe-middle default replaces the old graph default."""
    llm = _make_mock_llm(classify_raises=True)
    router = SmartQueryRouter(llm=llm)
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.HYBRID


# ─────────────────────────────────────────────────────────────────────────────
# (e) `_classify` called with query ONLY (no hyde_doc arg)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_invoked_with_query_only_no_hyde_doc():
    """The classifier user-message must NOT include the HyDE doc text.
    AMF-2: HyDE was the #1 structural cause of over-routing to graph."""
    llm = _make_mock_llm(capture_calls=True)
    # Patch the captured attribute to point to chat_json's calls.
    captured: list = []

    async def _capture(**kw):
        captured.append(kw)
        return {
            "route": "padded_rag",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "mock",
            "hyde_variants": ["v1"],
        }

    llm.chat_json = AsyncMock(side_effect=_capture)
    # Provide a distinctive HyDE doc so a regression that re-injects it
    # would be easy to spot.
    llm.chat = AsyncMock(return_value=MagicMock(content="HYDE-CONTENT-MARKER-XYZ"))

    router = SmartQueryRouter(llm=llm)
    await router.route("học phí ngành CNTT")

    # chat_json is called once (classify). Inspect the user_message.
    assert len(captured) == 1, f"expected 1 classify call, got {len(captured)}"
    user_msg = captured[0].get("user_message", "")
    assert "HYDE-CONTENT-MARKER-XYZ" not in user_msg, (
        f"HyDE doc leaked into classify user-message:\n{user_msg}"
    )
    # And the raw query must be present (sanity).
    assert "học phí ngành CNTT" in user_msg, (
        f"raw query missing from classify user-message:\n{user_msg}"
    )


@pytest.mark.asyncio
async def test_classify_signature_does_not_take_hyde_doc_kwarg():
    """`SmartQueryRouter._classify` must not accept a `hyde_doc` kwarg.

    This is a structural assertion — the function should take only `query`.
    """
    import inspect

    sig = inspect.signature(SmartQueryRouter._classify)
    params = list(sig.parameters.keys())
    # Self + query only.
    assert "query" in params
    assert "hyde_doc" not in params, (
        f"_classify must drop the hyde_doc arg; got params={params}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (f) Classify padded + ≥2 NGANH/TO_HOP entities → demoted to hybrid (CMF-5)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_padded_with_two_nganh_entities_demotes_to_hybrid():
    """Safety demote (CMF-5): classify says padded but ≥2 NGANH/TO_HOP entities
    in the query → demote to hybrid (NEVER graph, NEVER padded)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "diem_chuan",
            "reasoning": "looks 1-hop to me",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    # 2 NGANH entities (CNTT + Vật lý, both in NER keyword list), NO
    # comparison/enum keyword, NO year to confuse.
    result = await router.route("điểm chuẩn ngành CNTT và ngành Vật lý")
    assert result.route == QueryRoute.HYBRID


@pytest.mark.asyncio
async def test_enum_fast_path_with_two_tohop_entities_routes_graph():
    """Enumeration pattern is terminal before classify/entity demotion."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "to_hop",
            "reasoning": "single major",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = await router.route("tổ hợp A00 và tổ hợp B00 cho ngành CNTT")
    assert result.route == QueryRoute.GRAPH_RAG


@pytest.mark.asyncio
async def test_classify_padded_with_one_nganh_stays_padded():
    """Single NGANH entity (1) + classify=padded → no demote. Stays padded."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "single fact",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = await router.route("học phí ngành CNTT")
    assert result.route == QueryRoute.PADDED_RAG


# ─────────────────────────────────────────────────────────────────────────────
# (g) Classify graph → stays graph (no demotion)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_graph_stays_graph():
    """classify=graph + no comparison/enum pattern → graph (authoritative)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "graph_rag",
            "complexity": 4,
            "intent": "so_sanh",
            "reasoning": "multi-hop",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = await router.route("các ngành thuộc khối A")
    assert result.route == QueryRoute.GRAPH_RAG


# ─────────────────────────────────────────────────────────────────────────────
# Vague/contact pre-routing gates are UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vague_reject_still_short_circuits():
    """Bare '?' still triggers vague_reject BEFORE classify runs."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("?")
    assert result.intent == "vague_reject"
    assert result.skip_retrieval is True
    assert result.auto_answer == HYDE_REJECT_VAGUE


@pytest.mark.asyncio
async def test_contact_block_still_short_circuits():
    """'Hotline trường là gì?' still returns CONTACT_BLOCK before classify."""
    router = SmartQueryRouter(llm=_make_mock_llm())
    result = await router.route("Hotline trường là gì?")
    assert result.skip_retrieval is True
    assert result.auto_answer == CONTACT_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Re-validate: old >=-to-graph gate is GONE — complexity>=2 + classify=padded
# must now go to hybrid, NOT graph.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complexity_equal_threshold_does_NOT_silently_escalate_to_graph():
    """Old (pre-S16.2) behavior: padded+complexity==threshold → graph (>= gate).
    New (S16.2) behavior: padded+complexity>=threshold → hybrid (CMF-2 bump)."""
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 2,
            "intent": "diem_chuan",
            "reasoning": "test",
            "hyde_variants": ["v1"],
        }
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    # Plain query (no comparison/enum keywords) to isolate the threshold bump.
    result = await router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu")
    assert result.route == QueryRoute.HYBRID
    # Specifically NOT graph.
    assert result.route != QueryRoute.GRAPH_RAG
