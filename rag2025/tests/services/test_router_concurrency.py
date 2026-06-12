"""@spec(L3) — Router single-call contract (supersedes L1/L2 chain-concurrency).

HISTORY: L1/L2 of `latency_abstain_respec_plan.md` ran `_classify` CONCURRENTLY
with a `step_back → HyDE` chain via `asyncio.gather`, and this file proved that
concurrency. **L3 (latency_v4_tokenstream_routermerge.md) supersedes that design**:
HyDE is DROPPED entirely and step_back is FOLDED into the classify call, so
`route()` now makes exactly ONE LLM call (`_classify_combined`). There is no
longer a chain to run concurrently — the concurrency the L1/L2 tests asserted is
obsolete by construction.

This file is rewritten to pin the L3 contract:
  - route() makes EXACTLY ONE `chat_json` call and ZERO `chat` calls.
  - The single call's wall time bounds route() (no hidden extra round-trips).
  - The RouterResult is still well-formed; HyDE fields degrade to the raw query.

The mock LLM uses `asyncio.sleep` so the test never touches the live gateway.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.services.router_cache as router_cache_mod  # noqa: E402
from src.services.query_router import (  # noqa: E402
    QueryRoute,
    SmartQueryRouter,
)


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)


def _make_l3_llm(*, classify_sleep: float = 0.05, payload: dict | None = None,
                 chat_log: list | None = None, json_log: list | None = None):
    """Mock UnifiedLLMClient for the L3 single-call router.

    `chat_json` (the merged classify) sleeps `classify_sleep` and records its
    call. `chat` (the dropped step-back/HyDE chain) records any invocation —
    the L3 contract is that `chat` is NEVER called by route().
    """
    chat_log = chat_log if chat_log is not None else []
    json_log = json_log if json_log is not None else []
    payload = payload or {
        "route": "padded",
        "complexity": 1,
        "intent": "diem_chuan",
        "reasoning": "stub",
        "step_back": "stub-step-back",
    }

    async def _chat(**kw):
        chat_log.append(kw)
        return MagicMock(content="should-not-be-called")

    async def _classify(**kw):
        json_log.append(kw)
        await asyncio.sleep(classify_sleep)
        return payload

    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=_chat)
    llm.chat_json = AsyncMock(side_effect=_classify)
    return llm, chat_log, json_log


# ─────────────────────────────────────────────────────────────────────────────
# (a) L3 contract: route() makes EXACTLY ONE chat_json call, ZERO chat calls.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l3_route_makes_single_classify_call_no_chat():
    chat_log: list = []
    json_log: list = []
    llm, chat_log, json_log = _make_l3_llm(chat_log=chat_log, json_log=json_log)
    router = SmartQueryRouter(llm=llm)

    result = await router.route("quy trình xét tuyển đại học hoạt động như thế nào")

    assert len(json_log) == 1, (
        f"L3: route() must make EXACTLY ONE merged classify call; got {len(json_log)}"
    )
    assert len(chat_log) == 0, (
        f"L3: route() must make ZERO chat calls (HyDE/step-back chain dropped); "
        f"got {len(chat_log)}"
    )
    assert result is not None
    assert result.intent in ("diem_chuan", "stub")


# ─────────────────────────────────────────────────────────────────────────────
# (b) L3 wall time is bounded by the single call (no hidden extra round-trips).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l3_wall_time_bounded_by_single_call():
    SLEEP_S = 0.05
    llm, _chat_log, _json_log = _make_l3_llm(classify_sleep=SLEEP_S)
    router = SmartQueryRouter(llm=llm)

    # Warm up to avoid first-call scheduling noise.
    await router.route("quy trình xét tuyển hoạt động như thế nào")

    t0 = time.perf_counter()
    result = await router.route("quy trình xét tuyển đại học hoạt động như thế nào v2")
    elapsed = time.perf_counter() - t0

    # A single ~50ms call. The old 3-call sum was 150ms; assert we're well
    # under 2× the single-call sleep (no second/third hidden round-trip).
    assert elapsed < SLEEP_S * 2.0, (
        f"L3: route() wall time {elapsed:.4f}s should be ~1 call ({SLEEP_S:.3f}s), "
        f"not the old 3-call sum. Hidden extra round-trip?"
    )
    assert result is not None
    print(
        f"\n[L3 single-call bench] one_call={SLEEP_S*1000:.0f}ms "
        f"route_wall={elapsed*1000:.1f}ms (old 3-call sum was {3*SLEEP_S*1000:.0f}ms)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (c) L3: short factual query (year+major) — still one call, HyDE dropped.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l3_short_factual_query_single_call_hyde_dropped():
    chat_log: list = []
    json_log: list = []
    llm, chat_log, json_log = _make_l3_llm(chat_log=chat_log, json_log=json_log)
    router = SmartQueryRouter(llm=llm)

    # "học phí ngành CNTT 2026" — year + major, ≤6 word tokens.
    result = await router.route("học phí ngành CNTT 2026")

    assert len(json_log) == 1, f"expected 1 classify call; got {len(json_log)}"
    assert len(chat_log) == 0, f"expected 0 chat calls (HyDE dropped); got {len(chat_log)}"
    assert result is not None
    # HyDE dropped → hypothetical_doc echoes the raw query (documented no-op).
    assert result.hypothetical_doc == "học phí ngành CNTT 2026"


# ─────────────────────────────────────────────────────────────────────────────
# (d) L3: RouterResult is well-formed from the merged payload.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l3_router_result_wellformed_from_merged_payload():
    llm, _chat_log, _json_log = _make_l3_llm(
        payload={
            "route": "padded",
            "complexity": 1,
            "intent": "diem_chuan",
            "reasoning": "stub",
            "step_back": "nguyên lý xét tuyển đại học",
        }
    )
    router = SmartQueryRouter(llm=llm)

    res = await router.route("quy trình xét tuyển đại học hoạt động như thế nào")

    assert res.route == QueryRoute.PADDED_RAG
    assert res.complexity == 1
    assert res.intent == "diem_chuan"
    assert res.reasoning == "stub"
    assert res.auto_answer is None
    assert res.skip_retrieval is False
    # step_back from the merged payload (procedural query → _should_stepback True).
    assert res.step_back_query == "nguyên lý xét tuyển đại học"
