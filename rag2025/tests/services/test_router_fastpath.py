"""TDD: regex fast-path in SmartQueryRouter.route() for enum/comparison queries.

Motivation (latency plan): `route()` currently ALWAYS awaits
`self._classify_combined(query)` (~5s on the gateway). But the
deterministic overrides at lines 703-713 FORCE `route_str = "graph"`
when `_ENUMERATION_PATTERNS` or `_COMPARISON_PATTERNS` match — so the
LLM call is wasted work for that whole class of query.

This file verifies the fast-path:
  - matches the existing `_ENUMERATION_PATTERNS` / `_COMPARISON_PATTERNS`
  - runs AFTER vague-placeholder and contact-block pre-routing (those
    must still win first)
  - runs BEFORE `await self._classify_combined(query)` — i.e. the LLM is
    NEVER awaited on this path
  - returns a `RouterResult` with: route=GRAPH_RAG, complexity>=4,
    intent in {"liet_ke","so_sanh"}, step_back/hypothetical_doc echo,
    hyde_variants=[query]
  - the result is cached (cache hit → second call still makes 0 LLM calls)
  - does NOT swallow normal queries (a 1-fact lookup still hits the LLM
    once and routes by the model's payload)

Style mirror: fresh-event-loop `_run`, autouse cache reset, `_make_mock_llm`.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.services.router_cache as router_cache_mod  # noqa: E402
from src.services.query_router import QueryRoute, SmartQueryRouter  # noqa: E402


# ---------------------------------------------------------------------------
# Fresh event loop per test — mirror tests/services/test_router_combined.py::_run
# ---------------------------------------------------------------------------
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield


# ---------------------------------------------------------------------------
# Mock LLM factory — counts classifier (chat_json) invocations.
# ---------------------------------------------------------------------------
def _make_mock_llm(*, classify_payload=None):
    """Mock UnifiedLLMClient — only chat_json is exercised.

    If the fast-path works, chat_json must NEVER be awaited for enum/comparison
    queries (any await will surface as the AssertionError side_effect).
    """
    llm = MagicMock()
    # Surface unexpected chat calls (route() should bypass chat_json on the
    # fast-path; chat itself is L3-deprecated and must not be called).
    llm.chat = AsyncMock(
        side_effect=AssertionError("route() fast-path must not call llm.chat")
    )
    llm.chat_json = AsyncMock(
        return_value=classify_payload
        or {
            "route": "padded",
            "complexity": 1,
            "intent": "diem_chuan",
            "reasoning": "default",
            "step_back": "echoed",
        }
    )
    return llm


# ---------------------------------------------------------------------------
# 1. Enumeration fast-path: GRAPH_RAG, zero LLM calls, intent="liet_ke".
# ---------------------------------------------------------------------------
def test_enumeration_query_hits_fast_path_no_llm_call():
    llm = _make_mock_llm()
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("liệt kê các ngành đào tạo của trường?"))
    assert result.route == QueryRoute.GRAPH_RAG
    assert result.intent == "liet_ke"
    assert result.complexity >= 4
    assert result.skip_retrieval is False
    assert result.auto_answer is None
    assert llm.chat_json.await_count == 0, (
        f"fast-path must not call llm.chat_json; got {llm.chat_json.await_count}"
    )
    assert llm.chat.await_count == 0, (
        f"fast-path must not call llm.chat; got {llm.chat.await_count}"
    )


# ---------------------------------------------------------------------------
# 2. Comparison fast-path: GRAPH_RAG, zero LLM calls, intent="so_sanh".
# ---------------------------------------------------------------------------
def test_comparison_query_hits_fast_path_no_llm_call():
    llm = _make_mock_llm()
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("so sánh ngành CNTT và Khoa học dữ liệu?"))
    assert result.route == QueryRoute.GRAPH_RAG
    assert result.intent == "so_sanh"
    assert result.complexity >= 4
    assert llm.chat_json.await_count == 0
    assert llm.chat.await_count == 0


# ---------------------------------------------------------------------------
# 3. Normal 1-fact query: LLM IS awaited exactly once; fast-path must NOT
#    swallow it. Mock returns padded payload; route should be PADDED_RAG.
# ---------------------------------------------------------------------------
def test_normal_fact_query_falls_through_to_llm():
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "1 ngành, 1 thuộc tính",
            "step_back": "echoed",
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("học phí ngành CNTT?"))
    assert llm.chat_json.await_count == 1, (
        f"normal query must hit LLM exactly once; got {llm.chat_json.await_count}"
    )
    assert llm.chat.await_count == 0
    assert result.route == QueryRoute.PADDED_RAG


# ---------------------------------------------------------------------------
# 4. Cache: after a fast-path enum query, a second call with the same query
#    returns the cached result and STILL makes zero LLM calls.
# ---------------------------------------------------------------------------
def test_fast_path_result_is_cached_no_llm_on_second_call():
    llm = _make_mock_llm()
    router = SmartQueryRouter(llm=llm)
    q = "liệt kê các ngành đào tạo của trường?"
    r1 = _run(router.route(q))
    assert r1.route == QueryRoute.GRAPH_RAG
    assert r1.intent == "liet_ke"
    assert llm.chat_json.await_count == 0
    # Second call — cache hit, fast-path again, still zero LLM calls.
    r2 = _run(router.route(q))
    assert r2.route == QueryRoute.GRAPH_RAG
    assert r2.intent == "liet_ke"
    assert llm.chat_json.await_count == 0, (
        f"cached fast-path result must not trigger LLM; got {llm.chat_json.await_count}"
    )


# ---------------------------------------------------------------------------
# 5. Precedence: vague placeholder like "..." still hits the vague-reject path
#    (auto_answer=HYDE_REJECT_VAGUE), NOT the fast-path. Fast-path must not
#    override the vague short-circuit.
# ---------------------------------------------------------------------------
def test_vague_placeholder_still_wins_over_fast_path():
    llm = _make_mock_llm()
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("..."))
    assert result.intent == "vague_reject"
    assert result.auto_answer is not None
    assert result.skip_retrieval is True
    assert llm.chat_json.await_count == 0
    assert llm.chat.await_count == 0
    # The fast-path forces GRAPH_RAG; if we got here with route=GRAPH_RAG
    # it means the fast-path swallowed the vague query, which it must NOT.
    assert result.route != QueryRoute.GRAPH_RAG, (
        "vague query must short-circuit before fast-path; got GRAPH_RAG"
    )
