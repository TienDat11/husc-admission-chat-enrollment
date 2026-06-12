"""@spec(G4) — Router-decision cache tests.

Group 4: cut router latency by short-circuiting duplicate queries.

Contract (G4-T1, G4-T2):
  - Two IDENTICAL queries → underlying classifier invoked ONCE, second served
    from cache with the SAME route (proves the cache returns the cached
    RouterResult, not a fresh LLM call).
  - A genuinely different query → cache miss → classifier invoked again.
  - NEGATIVE / collision test: "điểm chuẩn 2025" vs "điểm chuẩn 2026" → TWO
    distinct cache entries / two classifier invocations. Years (digits) must
    NOT collide. This is the load-bearing assertion for the cache key design.

G4-T3: offline micro-bench on a repeat set with a stubbed classifier that
sleeps; the measured delta is the cache's contribution to route_ms.
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
# Mock LLM factory — counts classifier (chat_json) invocations.
# ─────────────────────────────────────────────────────────────────────────────


def _make_counting_llm(
    classify_payload=None,
    call_log: list | None = None,
    sleep_s: float = 0.0,
):
    """Return a mock UnifiedLLMClient whose ``chat_json`` is counted.

    - ``chat`` (step-back, hyde) returns placeholder text.
    - ``chat_json`` (classify) returns ``classify_payload`` and appends
      (user_message,) to ``call_log`` so the test can count real invocations.
    - If ``sleep_s`` > 0 the classifier sleeps that long, simulating the
      real ~5.8s LLM latency. Used by the micro-bench.
    """
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="placeholder"))

    payload = classify_payload or {
        "route": "padded",
        "complexity": 1,
        "intent": "diem_chuan",
        "reasoning": "default",
        "hyde_variants": ["v1"],
    }
    log = call_log if call_log is not None else []

    if sleep_s > 0:

        async def _slow(**_kw):
            await asyncio.sleep(sleep_s)
            log.append(_kw.get("user_message", ""))
            return payload

        llm.chat_json = AsyncMock(side_effect=_slow)
    else:

        async def _fast(**_kw):
            log.append(_kw.get("user_message", ""))
            return payload

        llm.chat_json = AsyncMock(side_effect=_fast)

    return llm


# ─────────────────────────────────────────────────────────────────────────────
# (a) G4-T1 — identical queries hit the cache.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identical_queries_invoke_classifier_once():
    """Two identical queries → underlying chat_json called EXACTLY once.

    The second call is served from the router cache and returns the same
    RouterResult.
    """
    call_log: list = []
    llm = _make_counting_llm(call_log=call_log)
    router = SmartQueryRouter(llm=llm)

    q = "điểm chuẩn ngành CNTT năm 2026 là bao nhiêu"

    r1 = await router.route(q)
    r2 = await router.route(q)

    # Classifier was invoked exactly once across both route() calls.
    assert len(call_log) == 1, (
        f"expected 1 classifier call across 2 identical queries, got {len(call_log)}"
    )
    # Same route, same intent — the cache returned the same RouterResult.
    assert r1.route == r2.route
    assert r1.intent == r2.intent
    assert r1.route == QueryRoute.PADDED_RAG


@pytest.mark.asyncio
async def test_whitespace_and_case_normalized_for_cache_hit():
    """Different surface forms of the SAME query must hit the cache.

    Normalization is: strip + lowercase + collapse-whitespace. Diacritics
    and digits are NOT folded (see collision test below).
    """
    call_log: list = []
    llm = _make_counting_llm(call_log=call_log)
    router = SmartQueryRouter(llm=llm)

    # Three surface variants that should normalize to one key.
    r1 = await router.route("điểm chuẩn ngành CNTT 2026")
    r2 = await router.route("  ĐIỂM CHUẨN NGÀNH CNTT 2026  ")
    r3 = await router.route("điểm  chuẩn   ngành   CNTT   2026")

    assert len(call_log) == 1, (
        f"expected 1 classifier call (whitespace+case normalized), got {len(call_log)}"
    )
    assert r1.route == r2.route == r3.route


# ─────────────────────────────────────────────────────────────────────────────
# (b) G4-T1 — a genuinely different query misses the cache.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_different_query_invokes_classifier_again():
    """A genuinely different query (different year) → cache miss → new call.

    Note: this is the SAME expected behaviour as the collision test below,
    but expressed at the route-level — the cache must distinguish them.
    """
    call_log: list = []
    llm = _make_counting_llm(call_log=call_log)
    router = SmartQueryRouter(llm=llm)

    await router.route("điểm chuẩn ngành CNTT 2026")
    await router.route("học phí ngành CNTT 2026")  # different attribute

    assert len(call_log) == 2, (
        f"expected 2 classifier calls for different queries, got {len(call_log)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (c) G4-T1 — NEGATIVE / collision test (load-bearing for the cache key).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_year_digits_do_not_collide():
    """'"điểm chuẩn 2025"' and '"điểm chuẩn 2026"' MUST produce 2 distinct
    cache entries / 2 classifier invocations.

    The cache key MUST preserve year digits (and major codes, etc.) — folding
    digits or the entire query to a prefix would silently alias semantically
    different queries to the wrong cached route. This is the spec's
    "collision test".
    """
    call_log: list = []
    llm = _make_counting_llm(call_log=call_log)
    router = SmartQueryRouter(llm=llm)

    r1 = await router.route("điểm chuẩn 2025")
    r2 = await router.route("điểm chuẩn 2026")

    assert len(call_log) == 2, (
        f"expected 2 classifier calls (years must NOT collide), got {len(call_log)}"
    )
    # Sanity: each result is independently built (separate step-back/HyDE
    # upstream of classify) — both came through, so they should be valid
    # RouterResult objects even if the route happens to be the same.
    assert r1 is not None
    assert r2 is not None


@pytest.mark.asyncio
async def test_diacritics_preserved_in_cache_key():
    """Diacritics must be preserved — '"CNTT"' (no diacritics) and a
    diacritic-bearing variant must NOT collide.

    E.g. Vietnamese major names like '"toán"' vs '"Toán"' are case-folded
    (we already lowercase) but a hypothetical 'toán' (with full diacritic)
    vs 'toan' (no diacritic) MUST produce two cache entries.
    """
    call_log: list = []
    llm = _make_counting_llm(call_log=call_log)
    router = SmartQueryRouter(llm=llm)

    await router.route("ngành toán")
    await router.route("ngành toan")  # no diacritic

    assert len(call_log) == 2, (
        f"expected 2 classifier calls (diacritics must be preserved), got {len(call_log)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (d) Pure-read invariant — the cache hit path must not call the LLM.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_makes_no_llm_calls():
    """Cache hit → ZERO LLM invocations of any kind (chat, chat_json).

    This proves the hot read is pure (no breaker, no retry, no per-hit
    refresh) — the spec's "hot_path_purity" requirement.

    L3 update: route() now makes a SINGLE merged classify call (chat_json)
    and NO `chat` calls (step-back/HyDE chain dropped — HyDE removed,
    step_back folded into the merged classify). So on the cold path we
    assert exactly 1 chat_json and 0 chat; on the warm hit, 0 of both.
    """
    call_log_json: list = []
    call_log_text: list = []
    llm = MagicMock()
    llm.chat = AsyncMock(
        side_effect=lambda **kw: call_log_text.append(kw) or MagicMock(content="x")
    )
    llm.chat_json = AsyncMock(
        side_effect=lambda **kw: call_log_json.append(kw)
        or {
            "route": "padded",
            "complexity": 1,
            "intent": "diem_chuan",
            "reasoning": "x",
            "step_back": "x",
        }
    )
    router = SmartQueryRouter(llm=llm)

    q = "học phí ngành CNTT 2026"
    await router.route(q)
    # After the first call: the single merged classify (chat_json) fired once;
    # the step-back/HyDE `chat` chain is gone (L3) → zero `chat` calls.
    assert len(call_log_json) == 1
    assert len(call_log_text) == 0  # L3: no step-back/HyDE chat calls

    # Reset the counters, then a repeat hit MUST NOT touch the LLM at all.
    call_log_json.clear()
    call_log_text.clear()
    await router.route(q)

    assert len(call_log_json) == 0, (
        f"chat_json called {len(call_log_json)} time(s) on cache hit (must be 0)"
    )
    assert len(call_log_text) == 0, (
        f"chat called {len(call_log_text)} time(s) on cache hit (must be 0)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (e) G4-T3 — offline micro-bench with a stubbed classifier that sleeps.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_g4_t3_micro_bench_repeats_skip_classify_cost():
    """Micro-benchmark: 10-query set with 50% repeats.

    Stubbed classifier sleeps ``SLEEP_S`` per call. The cached path
    short-circuits the sleep, so total wall time ≈ (#unique) × SLEEP_S,
    not (#calls) × SLEEP_S.

    Reported in the test docstring output for the spec to pick up.
    """
    # 10 queries, 5 of which are repeats of an earlier one → 5 unique
    # UNIQUE LLM-CALLING queries. The "so sánh ..." line is served by the
    # enum/comparison fast-path (route() does NOT await chat_json for that
    # class of query — see test_router_fastpath.py), so it never enters
    # call_log. The benchmark measures LLM-call savings, so its expected
    # count must reflect the fast-path skipping the LLM entirely.
    queries = [
        "điểm chuẩn ngành CNTT 2026",
        "học phí ngành CNTT 2026",
        "điểm chuẩn ngành CNTT 2026",          # repeat of #0
        "so sánh ngành CNTT và ngành Toán",    # served by fast-path (no LLM)
        "học phí ngành CNTT 2026",              # repeat of #1
        "điểm chuẩn ngành Vật lý 2026",
        "so sánh ngành CNTT và ngành Toán",     # repeat of #3 (fast-path)
        "điểm chuẩn ngành CNTT 2026",          # repeat of #0
        "học phí ngành Vật lý 2026",
        "điểm chuẩn ngành CNTT 2026",          # repeat of #0
    ]
    unique = {q for q in queries}
    # Only non-fast-path queries ever reach the LLM. Subtract the fast-path
    # ones so the structural assertion matches the new behavior.
    fast_path_queries = {q for q in unique
                         if q.startswith("so sánh ngành CNTT và ngành Toán")}
    expected_unique_calls = len(unique) - len(fast_path_queries)  # 6 - 1 = 5
    expected_total_calls = len(queries)  # 10
    skipped = expected_total_calls - expected_unique_calls

    SLEEP_S = 0.05  # 50ms — keeps the test under ~1s
    call_log: list = []

    # Cold path (no cache) — measure the upper bound.
    cold_llm = _make_counting_llm(call_log=[], sleep_s=SLEEP_S)
    cold_router = SmartQueryRouter(llm=cold_llm)
    # Bypass cache for the cold measurement: monkey-patch the cache factory
    # so a no-op cache is returned.
    class _NoopCache:
        hits = misses = 0

        def get(self, _q):  # always miss
            self.misses += 1
            return None

        def put(self, _q, _r):  # noqa: ARG002
            return None

    import src.services.router_cache as cache_mod
    orig_cache_get = cache_mod.get_router_cache
    cache_mod.get_router_cache = lambda: _NoopCache()  # type: ignore[assignment]

    try:
        t0 = time.perf_counter()
        for q in queries:
            await cold_router.route(q)
        cold_elapsed = time.perf_counter() - t0
    finally:
        cache_mod.get_router_cache = orig_cache_get  # type: ignore[assignment]

    # Warm path (cache enabled) — measure the cached bound.
    # _reset_router_cache fixture already gave us a fresh global cache.
    warm_llm = _make_counting_llm(call_log=call_log, sleep_s=SLEEP_S)
    warm_router = SmartQueryRouter(llm=warm_llm)
    t0 = time.perf_counter()
    for q in queries:
        await warm_router.route(q)
    warm_elapsed = time.perf_counter() - t0

    # ─── Structural assertions ───
    assert len(call_log) == expected_unique_calls, (
        f"expected {expected_unique_calls} unique calls (cache dedup), got {len(call_log)}"
    )
    assert skipped == expected_total_calls - expected_unique_calls

    # ─── Wall-clock delta ───
    delta_s = cold_elapsed - warm_elapsed
    speedup = cold_elapsed / warm_elapsed if warm_elapsed > 0 else float("inf")
    # Both must have run, and cold must be measurably slower.
    assert cold_elapsed > 0
    assert warm_elapsed >= 0
    # We expect the warm path to be at least 30% faster on a 50% repeat set.
    assert speedup > 1.3, (
        f"cache should yield >1.3x speedup; got {speedup:.2f}x "
        f"(cold={cold_elapsed*1000:.1f}ms warm={warm_elapsed*1000:.1f}ms)"
    )

    # Print the delta so the spec/operator can grep it from pytest -s output.
    print(
        f"\n[G4-T3 micro-bench] queries={expected_total_calls} unique={expected_unique_calls} "
        f"sleep/call={SLEEP_S*1000:.0f}ms "
        f"cold={cold_elapsed*1000:.1f}ms warm={warm_elapsed*1000:.1f}ms "
        f"delta={delta_s*1000:.1f}ms speedup={speedup:.2f}x"
    )
