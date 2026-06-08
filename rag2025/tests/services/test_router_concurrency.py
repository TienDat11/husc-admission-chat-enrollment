"""@spec(L1) — Router pipeline concurrency: classify ‖ (step-back → HyDE).

L1 / L2 / L3 of plan `latency_abstain_respec_plan.md`, section "#2 — Router latency".

Root cause (verified): `SmartQueryRouter.route()` performs 3 sequential LLM
calls (`_step_back`, `_generate_hyde_doc`, `_classify`). `_classify(query)` only
consumes the RAW query, so it is INDEPENDENT of the step_back→hyde chain. The
fix is to kick off `_classify` concurrently with the step_back+hyde chain via
`asyncio.gather`, cutting ~1 call of wall-time without changing output.

This file is the test-first proof. The "red" phase asserts that, given the
current sequential implementation, the timings DO NOT prove concurrency;
once L2 flips the implementation to `asyncio.gather`, these tests must turn
green (proving the structural change is real and the output stays
byte-identical).

The mock LLM is patched per-instance to use `asyncio.sleep` so the test never
touches the live gateway.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mirror the path/import pattern from tests/services/test_query_router_3way.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.services.router_cache as router_cache_mod  # noqa: E402
from src.services.query_router import (  # noqa: E402
    QueryRoute,
    SmartQueryRouter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-reset the global cache between tests so state doesn't leak.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM factory — stubbed async sleeps + a shared call-order log.
# ─────────────────────────────────────────────────────────────────────────────


def _make_concurrency_llm(
    *,
    step_back_sleep: float = 0.05,
    hyde_sleep: float = 0.05,
    classify_sleep: float = 0.05,
    step_back_text: str = "step-back-output",
    hyde_text: str = "hyde-doc-output",
    classify_payload: dict | None = None,
    call_log: list | None = None,
):
    """Mock UnifiedLLMClient with timed stubbed LLM calls.

    All three LLM helpers (`_step_back` uses `chat`, `_generate_hyde_doc` uses
    `chat`, `_classify` uses `chat_json`) record their start/end timestamps in
    a shared `call_log` (a list of dicts). This lets the tests assert
    CONCURRENCY (classify starts before step_back+hyde chain finishes).

    No real network calls; sleeps are `asyncio.sleep` only.
    """
    call_log = call_log if call_log is not None else []
    payload = classify_payload or {
        "route": "padded",
        "complexity": 1,
        "intent": "diem_chuan",
        "reasoning": "stub",
        "hyde_variants": ["v1"],
    }

    async def _step_back(**_kw):
        call_log.append({"name": "step_back_start", "t": time.perf_counter()})
        await asyncio.sleep(step_back_sleep)
        call_log.append({"name": "step_back_end", "t": time.perf_counter()})
        return MagicMock(content=step_back_text)

    async def _hyde(**_kw):
        call_log.append({"name": "hyde_start", "t": time.perf_counter()})
        await asyncio.sleep(hyde_sleep)
        call_log.append({"name": "hyde_end", "t": time.perf_counter()})
        return MagicMock(content=hyde_text)

    async def _classify(**_kw):
        call_log.append({"name": "classify_start", "t": time.perf_counter()})
        await asyncio.sleep(classify_sleep)
        call_log.append({"name": "classify_end", "t": time.perf_counter()})
        return payload

    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=_step_back)  # default — overridden per-test below
    # Provide both endpoints so tests can choose; default to "the right one"
    # for each helper by patching on the instance after construction.
    llm.chat_json = AsyncMock(side_effect=_classify)
    return llm, call_log, _step_back, _hyde, _classify


def _patch_helpers(router: SmartQueryRouter, _step_back, _hyde, _classify):
    """Patch the router's three LLM helpers to use the timed stubs.

    Each helper still routes through `self._llm`; we replace the `chat` and
    `chat_json` mocks on the SAME llm instance with a tagged dispatcher so
    that the call-order log distinguishes the three helpers.
    """
    llm = router._llm

    async def _chat_router(**kw):
        # step-back and hyde both use chat(). Distinguish by which kwarg the
        # caller set: step_back passes `user_message="Câu hỏi cụ thể: ..."`,
        # hyde passes `user_message="Câu hỏi: ...\nNguyên lý: ..."`.
        if kw.get("user_message", "").startswith("Câu hỏi cụ thể:"):
            return await _step_back(**kw)
        return await _hyde(**kw)

    llm.chat = AsyncMock(side_effect=_chat_router)
    llm.chat_json = AsyncMock(side_effect=_classify)


# ─────────────────────────────────────────────────────────────────────────────
# (a) Concurrency proof: classify runs in parallel with the step_back+hyde chain.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l1_classify_runs_concurrently_with_step_back_then_hyde():
    """Total wall time should be ≈ max(step_back, hyde, classify), NOT the sum.

    This is the load-bearing L1 assertion. With 50ms sleeps on each helper,
    the SUM is 150ms. CONCURRENT execution caps wall time at ~100ms (the
    step_back→hyde chain) running in parallel with classify.

    The call-order log must also show that `classify_start` happens BEFORE
    `hyde_end` (proving classify was kicked off before the chain finished).

    Query choice: "quy trình xét tuyển đại học hoạt động như thế nào" — an
    abstract / procedural query that triggers `_should_stepback`=True (no
    year+major pair), so the full step_back→hyde chain executes.
    """
    call_log: list = []
    llm, _, _sb, _hyde, _clf = _make_concurrency_llm(
        step_back_sleep=0.05,
        hyde_sleep=0.05,
        classify_sleep=0.05,
        call_log=call_log,
    )
    router = SmartQueryRouter(llm=llm)
    _patch_helpers(router, _sb, _hyde, _clf)

    t0 = time.perf_counter()
    result = await router.route("quy trình xét tuyển đại học hoạt động như thế nào")
    elapsed = time.perf_counter() - t0

    # Structural call-order checks.
    names = [e["name"] for e in call_log]
    assert "step_back_start" in names
    assert "step_back_end" in names
    assert "hyde_start" in names
    assert "hyde_end" in names
    assert "classify_start" in names
    assert "classify_end" in names

    # The classify task must START before the step_back→hyde chain ENDS.
    # That is the literal definition of concurrency in this pipeline.
    sbe = next(e["t"] for e in call_log if e["name"] == "step_back_start")
    cs = next(e["t"] for e in call_log if e["name"] == "classify_start")
    he = next(e["t"] for e in call_log if e["name"] == "hyde_end")
    ce = next(e["t"] for e in call_log if e["name"] == "classify_end")

    assert cs < he, (
        f"classify must START before hyde ENDS (concurrency). "
        f"classify_start={cs:.4f} hyde_end={he:.4f} delta={cs-he:+.4f}s"
    )

    # Wall time bound: sum=150ms; concurrent max≈100ms. Allow generous slack
    # for asyncio scheduling on slow CI but demand a clear margin.
    sum_s = 0.05 + 0.05 + 0.05
    assert elapsed < sum_s * 0.85, (
        f"expected wall time ≪ sum({sum_s:.3f}s); got {elapsed:.4f}s. "
        f"Concurrency not achieved — calls ran sequentially."
    )
    # And a lower bound to catch a future "skip the chain entirely" regression:
    assert elapsed >= max(0.05, 0.05) * 0.7  # at least the longest branch

    # And a result is still produced.
    assert result is not None
    assert result.intent in ("diem_chuan", "stub")


# ─────────────────────────────────────────────────────────────────────────────
# (b) Output identity: the returned RouterResult is byte-identical to a
# sequential baseline for the same mocked outputs.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l1_router_result_is_identical_to_sequential_baseline():
    """Concurrent execution must NOT change the RouterResult shape.

    We construct TWO routers with IDENTICAL mocked helpers. We then drive
    one of them through the concurrent path (after L2 lands) and the other
    through a manually-sequential equivalent (step_back → hyde → classify
    awaited in that order). Their RouterResult fields must match.

    Query choice: an abstract / procedural query so `_should_stepback`=True
    and the full step_back→hyde chain executes (otherwise step_back is
    skipped and the chain returns a degenerate `(query, hyde)` tuple).
    """
    call_log_a: list = []
    call_log_b: list = []
    llm_a, _, _sb_a, _hyde_a, _clf_a = _make_concurrency_llm(call_log=call_log_a)
    llm_b, _, _sb_b, _hyde_b, _clf_b = _make_concurrency_llm(call_log=call_log_b)

    router_a = SmartQueryRouter(llm=llm_a)
    router_b = SmartQueryRouter(llm=llm_b)
    _patch_helpers(router_a, _sb_a, _hyde_a, _clf_a)
    _patch_helpers(router_b, _sb_b, _hyde_b, _clf_b)

    q = "quy trình xét tuyển đại học hoạt động như thế nào"

    # A: concurrent path (calls router.route — same code path the L2 fix uses).
    res_a = await router_a.route(q)

    # B: sequential baseline — invoke the three helpers in the original
    # order without overlap. The router exposes `_step_back`, `_classify`,
    # `_generate_hyde_doc` as the same coroutines the route() pipeline uses.
    step_back_b = await router_b._step_back(q)
    hyde_b = await router_b._generate_hyde_doc(q, step_back_b)
    clf_b = await router_b._classify(q)

    # ─── Field-by-field equality on the RouterResult — only timing differs. ──
    assert res_a.original_query == q == res_a.original_query
    assert res_a.step_back_query == step_back_b, (
        f"step_back_query differs: concurrent={res_a.step_back_query!r} "
        f"vs sequential={step_back_b!r}"
    )
    assert res_a.hypothetical_doc == hyde_b, (
        f"hypothetical_doc differs: concurrent={res_a.hypothetical_doc!r} "
        f"vs sequential={hyde_b!r}"
    )
    assert res_a.hyde_variants == clf_b.get("hyde_variants", [q]), (
        f"hyde_variants differs: concurrent={res_a.hyde_variants!r} "
        f"vs sequential={clf_b.get('hyde_variants', [q])!r}"
    )
    assert res_a.route == QueryRoute.PADDED_RAG
    assert res_a.complexity == 1
    assert res_a.intent == "diem_chuan"
    assert res_a.reasoning == "stub"
    assert res_a.auto_answer is None
    assert res_a.skip_retrieval is False


# ─────────────────────────────────────────────────────────────────────────────
# (c) `_should_stepback`=False branch: short factual query → step_back LLM
# call is SKIPPED, but classify still runs concurrently with the hyde call.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l1_short_factual_query_skips_step_back_but_classify_still_concurrent():
    """A query that trips `_should_stepback`=False (year+major, ≤6 word tokens)
    must NOT invoke the step_back LLM helper, yet classify must still run
    concurrently with `_generate_hyde_doc`.

    This is the L1 part-(c) assertion from the spec.
    """
    call_log: list = []
    llm, _, _sb, _hyde, _clf = _make_concurrency_llm(
        step_back_sleep=0.10,  # would be very visible in the log if invoked
        hyde_sleep=0.05,
        classify_sleep=0.05,
        call_log=call_log,
    )
    router = SmartQueryRouter(llm=llm)
    _patch_helpers(router, _sb, _hyde, _clf)

    # "học phí ngành CNTT 2026" — has year (2026) + major (CNTT, học phí)
    # and is ≤6 word tokens → _should_stepback returns False.
    result = await router.route("học phí ngành CNTT 2026")

    # step_back LLM helper must NOT have been invoked.
    names = [e["name"] for e in call_log]
    assert "step_back_start" not in names, (
        f"step_back must be SKIPPED for short factual year+major query; "
        f"log={names}"
    )
    # But hyde + classify BOTH ran.
    assert "hyde_start" in names
    assert "hyde_end" in names
    assert "classify_start" in names
    assert "classify_end" in names

    # And they ran concurrently: classify_start < hyde_end.
    cs = next(e["t"] for e in call_log if e["name"] == "classify_start")
    he = next(e["t"] for e in call_log if e["name"] == "hyde_end")
    assert cs < he, (
        f"classify must START before hyde ENDS even when step_back is "
        f"skipped (concurrency). classify_start={cs:.4f} hyde_end={he:.4f}"
    )

    # The result is still well-formed and the step_back_query falls back to
    # the raw query (per the existing `_should_stepback`=False branch).
    assert result is not None
    assert result.step_back_query == "học phí ngành CNTT 2026"
    assert result.hypothetical_doc == "hyde-doc-output"
    assert result.intent == "diem_chuan"


# ─────────────────────────────────────────────────────────────────────────────
# (d) L3 — offline micro-bench: sequential vs concurrent wall time.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l3_offline_micro_bench_concurrent_wall_lt_sum():
    """L3 micro-bench (offline, stubbed async sleeps).

    Stubbed LLM helpers each sleep 50ms. With the pre-L2 sequential layout,
    the wall time is 3 × 50ms = 150ms. With the L2 concurrent layout it is
    bounded by max(step_back+hyde, classify) = 100ms (the step_back→hyde
    chain is 100ms, classify is 50ms — chain dominates).

    The bench prints the wall time so a human can grep it from pytest -s;
    the load-bearing assertion is `concurrent < sum * 0.85` and
    `concurrent >= chain * 0.7` to catch future regressions.
    """
    SLEEP_S = 0.05
    call_log: list = []
    llm, _, _sb, _hyde, _clf = _make_concurrency_llm(
        step_back_sleep=SLEEP_S,
        hyde_sleep=SLEEP_S,
        classify_sleep=SLEEP_S,
        call_log=call_log,
    )
    router = SmartQueryRouter(llm=llm)
    _patch_helpers(router, _sb, _hyde, _clf)

    # Warm up: ensure no first-call scheduling noise skews the measurement.
    await router.route("quy trình xét tuyển hoạt động như thế nào")
    call_log.clear()

    # Measure: a representative abstract query that triggers the full chain.
    t0 = time.perf_counter()
    result = await router.route("quy trình xét tuyển đại học hoạt động như thế nào")
    elapsed = time.perf_counter() - t0

    # Mathematical bounds.
    sum_s = 3 * SLEEP_S           # 0.150s — the pre-L2 sequential total
    chain_s = 2 * SLEEP_S         # 0.100s — the step_back→hyde chain
    classify_s = SLEEP_S          # 0.050s — the independent classify task
    expected_concurrent_s = max(chain_s, classify_s)  # 0.100s

    # Load-bearing assertion: wall time is bounded by the concurrent max,
    # not the sum. Allow generous slack for asyncio scheduling on CI.
    assert elapsed < sum_s * 0.85, (
        f"concurrent wall time {elapsed:.4f}s should be ≪ sequential sum "
        f"{sum_s:.3f}s. Concurrency not achieved."
    )
    # Lower bound: at least the longest branch (catches "skip the chain"
    # regressions that would silently make the test pass).
    assert elapsed >= expected_concurrent_s * 0.7, (
        f"concurrent wall time {elapsed:.4f}s should be ≥ max-branch "
        f"{expected_concurrent_s:.3f}s × 0.7. Did the chain actually run?"
    )

    # Sanity: the result is well-formed.
    assert result is not None
    assert result.intent == "diem_chuan"

    # Reported numbers — grep-able from `pytest -s` output.
    speedup = sum_s / elapsed if elapsed > 0 else float("inf")
    print(
        f"\n[L3 micro-bench] sequential_sum={sum_s*1000:.0f}ms "
        f"concurrent_wall={elapsed*1000:.1f}ms "
        f"expected_max={expected_concurrent_s*1000:.0f}ms "
        f"speedup_vs_sum={speedup:.2f}x"
    )


@pytest.mark.asyncio
async def test_l3_offline_micro_bench_classify_only_no_step_back():
    """L3 micro-bench for the `_should_stepback`=False branch.

    When step_back is suppressed, the chain reduces to a single
    `_generate_hyde_doc` call (50ms) and the classify task runs
    concurrently with it. Total wall time should be ≈ max(hyde, classify)
    = 50ms, NOT 100ms (sum).
    """
    SLEEP_S = 0.05
    call_log: list = []
    llm, _, _sb, _hyde, _clf = _make_concurrency_llm(
        step_back_sleep=SLEEP_S,
        hyde_sleep=SLEEP_S,
        classify_sleep=SLEEP_S,
        call_log=call_log,
    )
    router = SmartQueryRouter(llm=llm)
    _patch_helpers(router, _sb, _hyde, _clf)

    # Year + major + ≤6 word tokens → _should_stepback=False.
    t0 = time.perf_counter()
    result = await router.route("học phí ngành CNTT 2026")
    elapsed = time.perf_counter() - t0

    sum_s = 2 * SLEEP_S   # 0.100s — sequential hyde + classify
    expected_concurrent_s = SLEEP_S  # 0.050s — both run together

    assert elapsed < sum_s * 0.85, (
        f"concurrent wall time {elapsed:.4f}s should be ≪ sequential sum "
        f"{sum_s:.3f}s. Concurrency not achieved on the no-step-back path."
    )
    assert elapsed >= expected_concurrent_s * 0.7, (
        f"concurrent wall time {elapsed:.4f}s should be ≥ "
        f"max-branch {expected_concurrent_s:.3f}s × 0.7."
    )
    assert result is not None
    # step_back was suppressed; the result's step_back_query is the raw query.
    assert result.step_back_query == "học phí ngành CNTT 2026"

    print(
        f"\n[L3 micro-bench no-stepback] sequential_sum={sum_s*1000:.0f}ms "
        f"concurrent_wall={elapsed*1000:.1f}ms "
        f"expected_max={expected_concurrent_s*1000:.0f}ms "
        f"speedup_vs_sum={(sum_s/elapsed):.2f}x"
    )
