"""TDD: L3 latency plan — DROP HyDE + MERGE step-back into classify (single LLM call).

PART 2: route() must now make EXACTLY ONE chat_json call (the old
step_back→hyde chain is removed; step_back is folded into the classify prompt;
HyDE doc is dropped to a no-op echo of the query).

All LLM calls are MOCKED — gateway is down (503s), no live model traffic.

Contract (L3):
  - Single `await self._classify_combined(query)` replaces the
    `asyncio.gather(chain_task, classify_task)` block.
  - Deterministic overrides (enum→graph, comparison→graph, padded+complexity
    ≥ threshold → hybrid, padded+≥2 NGANH/TO_HOP → hybrid) UNCHANGED.
  - `hypothetical_doc == query` (HyDE dropped) and `_generate_hyde_doc` is
    NEVER invoked.
  - chat_json raises → safe-middle hybrid fallback, step_back=query.
  - `_should_stepback` gate still applies: when False, step_back=query even
    if the model returns an abstracted step_back.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.services.router_cache as router_cache_mod  # noqa: E402
from src.services import query_router as qr_mod  # noqa: E402
from src.services.query_router import (  # noqa: E402
    QueryRoute,
    SmartQueryRouter,
    _should_stepback,
)


# ---------------------------------------------------------------------------
# Fresh event loop per test — mirror tests/services/test_ner_fallback.py::_run
# to dodge the suite-wide loop pollution that bit earlier async tests.
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
# Mock LLM factory — single chat_json call only (chain is gone).
# ---------------------------------------------------------------------------
def _make_mock_llm(*, classify_payload=None, classify_raises: bool = False):
    """Mock UnifiedLLMClient — single chat_json, NO chain."""
    llm = MagicMock()
    # Pre-L3 used a separate `chat` call for step_back/hyde. The new
    # pipeline does not invoke it; if any test triggers a call we want
    # it to surface as a test failure, not a silent side-effect.
    llm.chat = AsyncMock(
        side_effect=AssertionError("route() must not call llm.chat anymore (L3)")
    )
    if classify_raises:
        llm.chat_json = AsyncMock(side_effect=RuntimeError("classify boom"))
    else:
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


def _good_payload(route: str = "padded", complexity: int = 1, step_back: str = "abstracted query") -> Dict[str, Any]:
    return {
        "route": route,
        "complexity": complexity,
        "intent": "diem_chuan",
        "reasoning": "mock",
        "step_back": step_back,
    }


# ---------------------------------------------------------------------------
# 1. The L3 contract — single chat_json call per route().
# ---------------------------------------------------------------------------
def test_route_makes_exactly_one_chat_json_call():
    """route() must invoke llm.chat_json EXACTLY ONCE (no chain)."""
    llm = _make_mock_llm(classify_payload=_good_payload())
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu"))
    assert llm.chat_json.await_count == 1, (
        f"expected exactly 1 chat_json call, got {llm.chat_json.await_count}"
    )
    # And llm.chat must NEVER be called.
    assert llm.chat.await_count == 0, (
        f"route() must not call llm.chat anymore; got {llm.chat.await_count} calls"
    )
    assert result.route == QueryRoute.PADDED_RAG


# ---------------------------------------------------------------------------
# 2. HyDE dropped — hypothetical_doc == query, _generate_hyde_doc never invoked.
# ---------------------------------------------------------------------------
def test_hyde_dropped_hypothetical_doc_echoes_query():
    """HyDE is dropped: hypothetical_doc just echoes the query (no-op)."""
    llm = _make_mock_llm(classify_payload=_good_payload())
    router = SmartQueryRouter(llm=llm)
    query = "điểm chuẩn ngành CNTT năm 2026 là bao nhiêu"
    # Patch _generate_hyde_doc to raise if invoked — proves the call site is gone.
    router._generate_hyde_doc = AsyncMock(
        side_effect=AssertionError("_generate_hyde_doc must not be called (HyDE dropped)")
    )
    result = _run(router.route(query))
    assert result.hypothetical_doc == query
    assert result.hyde_variants == [query]
    router._generate_hyde_doc.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. step_back is sourced from the model output.
# ---------------------------------------------------------------------------
def test_step_back_uses_model_output_when_should_stepback_true():
    llm = _make_mock_llm(
        classify_payload=_good_payload(
            route="padded", complexity=1, step_back="tổng quát về điểm chuẩn ngành CNTT"
        )
    )
    router = SmartQueryRouter(llm=llm)
    query = "điểm chuẩn ngành CNTT năm 2026 là bao nhiêu"
    # The query has year+major → _should_stepback is False. We need a query
    # that triggers step-back so the model's step_back survives.
    query_sb = "cách xét học bạ và thời gian xét tuyển?"
    result = _run(router.route(query_sb))
    assert result.step_back_query == "tổng quát về điểm chuẩn ngành CNTT"


def test_step_back_falls_back_to_query_when_should_stepback_false():
    """`_should_stepback` gate wins over the model's step_back field."""
    llm = _make_mock_llm(
        classify_payload=_good_payload(
            route="padded", complexity=1, step_back="WRONG-ABSTRACTED"
        )
    )
    router = SmartQueryRouter(llm=llm)
    # year + major → _should_stepback is False → step_back must be the raw query
    query = "điểm chuẩn ngành CNTT năm 2026"
    assert _should_stepback(query) is False
    result = _run(router.route(query))
    assert result.step_back_query == query


# ---------------------------------------------------------------------------
# 4. Deterministic overrides still fire on the combined classify.
# ---------------------------------------------------------------------------
def test_enum_pattern_forces_graph():
    llm = _make_mock_llm(
        classify_payload=_good_payload(route="padded", complexity=1)
    )
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("liệt kê các ngành đào tạo của trường"))
    assert result.route == QueryRoute.GRAPH_RAG
    assert result.complexity >= 4


def test_comparison_pattern_forces_graph():
    llm = _make_mock_llm(
        classify_payload=_good_payload(route="padded", complexity=1)
    )
    router = SmartQueryRouter(llm=llm)
    # Comparison pattern + 1 NGANH/TO_HOP entity (no demote trigger).
    result = _run(router.route("học phí ngành CNTT so với ngành Toán"))
    assert result.route == QueryRoute.GRAPH_RAG


def test_padded_complexity_at_threshold_bumps_to_hybrid():
    llm = _make_mock_llm(
        classify_payload=_good_payload(route="padded", complexity=2)
    )
    router = SmartQueryRouter(llm=llm, simple_complexity_threshold=2)
    result = _run(router.route("điểm chuẩn ngành CNTT năm 2026 là bao nhiêu"))
    assert result.route == QueryRoute.HYBRID


def test_padded_with_two_nganh_entities_demotes_to_hybrid():
    llm = _make_mock_llm(
        classify_payload=_good_payload(route="padded", complexity=1)
    )
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("điểm chuẩn ngành CNTT và ngành Vật lý"))
    assert result.route == QueryRoute.HYBRID


# ---------------------------------------------------------------------------
# 5. Bad-JSON / chat_json raises → safe-middle hybrid fallback, step_back=query.
# ---------------------------------------------------------------------------
def test_classify_raises_falls_back_to_hybrid_with_step_back_query():
    llm = _make_mock_llm(classify_raises=True)
    router = SmartQueryRouter(llm=llm)
    query = "điểm chuẩn ngành CNTT năm 2026 là bao nhiêu"
    result = _run(router.route(query))
    assert result.route == QueryRoute.HYBRID
    assert result.complexity == 2
    assert result.intent == "general"
    assert result.step_back_query == query
    assert result.hypothetical_doc == query
    assert result.hyde_variants == [query]


# ---------------------------------------------------------------------------
# 6. Legacy route tokens (padded_rag / graph_rag) still map cleanly.
# ---------------------------------------------------------------------------
def test_legacy_padded_rag_token_accepted():
    llm = _make_mock_llm(
        classify_payload={
            "route": "padded_rag",
            "complexity": 1,
            "intent": "hoc_phi",
            "reasoning": "legacy",
            "step_back": "echoed",
        }
    )
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("học phí ngành CNTT"))
    assert result.route == QueryRoute.PADDED_RAG


# ---------------------------------------------------------------------------
# 7. Structural assertion — `_classify_combined` exists, `_chain_stepback_hyde`
# is no longer called by route() (the helper is retained for back-compat).
# ---------------------------------------------------------------------------
def test_classify_combined_method_exists():
    assert hasattr(SmartQueryRouter, "_classify_combined"), (
        "SmartQueryRouter must expose _classify_combined for L3 single-call pipeline"
    )
    sig = inspect.signature(SmartQueryRouter._classify_combined)
    params = list(sig.parameters.keys())
    # self + query only.
    assert "query" in params
    assert "hyde_doc" not in params and "step_back" not in params, (
        f"_classify_combined must take only (self, query); got {params}"
    )


def test_classify_combined_prompt_does_not_ask_for_hyde():
    """The combined prompt must not request a hypothetical_doc / hyde field."""
    # The new constant is required for the single-call pipeline.
    assert hasattr(qr_mod, "_CLASSIFY_STEPBACK_SYSTEM"), (
        "query_router must define _CLASSIFY_STEPBACK_SYSTEM prompt"
    )
    prompt = qr_mod._CLASSIFY_STEPBACK_SYSTEM
    # Should ask for the 5 fields and not request hyde/hypothetical doc.
    for must_have in ("route", "complexity", "intent", "reasoning", "step_back"):
        assert must_have in prompt, (
            f"_CLASSIFY_STEPBACK_SYSTEM must mention '{must_have}' in schema"
        )
    # Don't ask for hyde / hypothetical_doc fields.
    for must_not_have in ("hypothetical_doc", '"hyde"', "hyde_variants"):
        assert must_not_have not in prompt, (
            f"_CLASSIFY_STEPBACK_SYSTEM must not ask for '{must_not_have}' (HyDE dropped)"
        )


# ---------------------------------------------------------------------------
# 8. Vague/contact pre-routing gates STILL short-circuit (no LLM call at all).
# ---------------------------------------------------------------------------
def test_vague_reject_does_not_call_llm():
    llm = _make_mock_llm()
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("?"))
    assert result.skip_retrieval is True
    assert result.intent == "vague_reject"
    assert llm.chat_json.await_count == 0
    assert llm.chat.await_count == 0


def test_contact_block_does_not_call_llm():
    llm = _make_mock_llm()
    router = SmartQueryRouter(llm=llm)
    result = _run(router.route("Hotline trường là gì?"))
    assert result.skip_retrieval is True
    assert llm.chat_json.await_count == 0
    assert llm.chat.await_count == 0
