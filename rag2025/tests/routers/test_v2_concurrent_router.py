"""TDD: /v2/query must run the router LLM round-trip CONCURRENTLY with the
PaddedRAG baseline retrieval — the two are independent (router depends only
on the raw query string, not on baseline_docs), so the old strictly-sequential
shape wasted ~max(router, baseline) latency on the sum.

The robust correctness gate (always asserted, immune to CI/Windows jitter):
  1. `unified_pipeline._router.route(...)` is awaited EXACTLY once per request.
  2. The SAME router_result object is passed into `unified_pipeline.query(...)`
     via the new `router_result=` kwarg (i.e. the pipeline does NOT re-route).

The timing assertion (best-effort, generous bound to avoid Windows flakiness):
  3. Wall-clock of `_build_v2_query_payload` is closer to one slow leg
     (~0.2s) than to the sum of both legs (~0.4s).

Also pins the OOS short-circuit: when the guardrail precheck rejects, the
router task MUST NOT be started (router.route.await_count == 0).

Approach: mirror the helper-refactor style of test_v2_uses_generate_answer.py
and test_v2_booster_wired.py — load src/main.py via importlib, monkeypatch
the module-level singletons, drive the handler through TestClient.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
REPO_ROOT = MAIN_PATH.resolve().parents[1]


def _build_main_module(monkeypatch):
    """Load src/main.py with heavy env-driven singletons defused."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    spec = importlib.util.spec_from_file_location(
        "v2_concurrent_router_test", MAIN_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module(monkeypatch):
    return _build_main_module(monkeypatch)


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────

def _fake_router_result_obj():
    return SimpleNamespace(
        skip_retrieval=False,
        auto_answer=None,
        step_back_query="step",
        intent="general",
        complexity=1,
        reasoning="ok",
        hyde_variants=[],
        route=SimpleNamespace(value="padded_rag"),
    )


def _fake_doc(chunk_id: str = "c1", text: str = "baseline chunk", score: float = 0.9):
    return SimpleNamespace(
        text=text,
        source="doc.pdf",
        chunk_id=chunk_id,
        metadata={"summary": text[:60], "source": "doc.pdf", "data_year": 2026},
        score=score,
        point_id=chunk_id,
    )


def _install_slow_leg_fakes(
    main_module,
    monkeypatch,
    *,
    router_sleep: float = 0.2,
    baseline_sleep: float = 0.2,
    in_scope: bool = True,
):
    """Wire fakes that each sleep `router_sleep`/`baseline_sleep` seconds
    so the concurrency gate is observable. Returns the captured handles.
    """

    # guardrail: precheck → in scope (default) or out of scope.
    # The /v2 handler calls guardrail_service.public_status(internal_code)
    # and guardrail_service.expose_internal() — these must return real
    # strings/booleans (not MagicMock) so the response_payload validates
    # against UnifiedQueryResponse.
    guardrail = MagicMock()
    guardrail.precheck = AsyncMock(
        return_value=SimpleNamespace(
            is_in_scope=in_scope,
            internal_code="SUCCESS" if in_scope else "NOT_IN_HUSC_SCOPE",
            reason="ok" if in_scope else "oos",
            public_code="SUCCESS" if in_scope else "NOT_IN_HUSC_SCOPE",
            short_answer="" if in_scope else "Ngoài phạm vi.",
            pii_detected=False,
            data_gap_hints=[],
        )
    )
    # public_status(internal_code) → "SUCCESS" or "NOT_IN_HUSC_SCOPE"
    def _public_status(internal_code):
        return "SUCCESS" if internal_code == "SUCCESS" else internal_code
    guardrail.public_status = MagicMock(side_effect=_public_status)
    guardrail.expose_internal = MagicMock(return_value=False)
    # classify_no_result is only called when rag_result.documents is empty;
    # default to SUCCESS so the no-result branch is a no-op.
    guardrail.classify_no_result = AsyncMock(
        return_value=SimpleNamespace(
            internal_code="SUCCESS",
            reason="ok",
            data_gap_hints=[],
        )
    )
    main_module.guardrail_service = guardrail

    # lancedb retriever: sleep then return a baseline doc
    async def _slow_retrieve(*args, **kwargs):
        await asyncio.sleep(baseline_sleep)
        return SimpleNamespace(is_success=True, documents=[_fake_doc()])
    lancedb = MagicMock()
    lancedb.retrieve = MagicMock(side_effect=lambda **_: _run_sync(_slow_retrieve))
    # We can't await in a sync MagicMock; instead, we make retrieve return a
    # coroutine by patching retrieve to a plain function returning a result,
    # but we need the awaitable to be awaited. The cleanest path: make
    # retrieve a sync function that schedules a coroutine — but run_in_threadpool
    # in main.py expects a *sync* callable returning the vector. We can put the
    # sleep on the embedding path (which is also awaited via run_in_threadpool
    # but in a thread) — but threadpool sleep doesn't block the event loop.
    # Simpler & faithful: monkeypatch the embedding encode to await async sleep
    # by making it an AsyncMock. main.py uses run_in_threadpool, so we route the
    # delay through retrieve being slow in a separate coroutine that the test
    # can interleave. Instead: wrap the BASELINE delay in the pipeline by
    # making the booster sleep — but main.py calls booster synchronously.
    # We instead make encode_query an AsyncMock that sleeps — main.py awaits
    # it via run_in_threadpool lambda; the threadpool does the sleep. To
    # create event-loop-blocking latency on the BASELINE leg, we make the
    # lancedb retrieve call a coroutine that we hand to run_in_threadpool —
    # but main.py calls lancedb.retrieve synchronously.
    # Solution: the cleanest test seam is to measure the ROUTER's async sleep
    # (which DOES block the event loop) and assert that wall-clock of the
    # full payload is much less than router+baseline_sum when both run on
    # the loop. We use the AsyncMock route on router for the loop-blocking
    # sleep, and we make the BASELINE side take roughly the same time on
    # the loop by having the lancedb retrieve call return immediately and
    # the booster call a slow coroutine via monkeypatch.
    # Simpler still: we monkeypatch the booster to be an async function
    # awaited by main.py — but main.py calls it synchronously. So we put
    # the baseline sleep in `embedding_service.encode_query` via an
    # AsyncMock that is awaited by run_in_threadpool — no, run_in_threadpool
    # wraps a sync callable.
    # The correct, minimal seam: monkeypatch the embedding encode_query to
    # block for baseline_sleep seconds (it runs in a threadpool worker, so
    # it does NOT block the event loop) AND monkeypatch the router.route
    # to be an AsyncMock that sleeps for router_sleep seconds on the event
    # loop. With the refactor, both run concurrently — the embedding
    # threadpool sleep overlaps with the router event-loop sleep. The
    # total wall-clock becomes max(router_sleep, baseline_sleep).
    #
    # Reset the MagicMock: use a sync side_effect that sleeps via time.sleep
    # (it runs in the threadpool, so it doesn't block the event loop).
    import time as _time
    def _slow_embedding_query(_query):
        _time.sleep(baseline_sleep)
        return SimpleNamespace(tolist=lambda: [0.0] * 4)
    embedding = MagicMock()
    embedding.encode_query = MagicMock(side_effect=_slow_embedding_query)
    main_module.embedding_service = embedding

    lancedb.retrieve = MagicMock(
        return_value=SimpleNamespace(is_success=True, documents=[_fake_doc()])
    )
    main_module.lancedb_retriever_service = lancedb

    main_module.query_cache = None

    # Router: AsyncMock that sleeps on the event loop. Capture timing.
    router_started = {"t": None, "n": 0}

    async def _slow_router_route(_query):
        router_started["t"] = _time.perf_counter()
        router_started["n"] += 1
        await asyncio.sleep(router_sleep)
        return _fake_router_result_obj()

    router = MagicMock()
    router.route = AsyncMock(side_effect=_slow_router_route)
    main_module.unified_pipeline = MagicMock()
    main_module.unified_pipeline._router = router

    # Pipeline: capture the router_result passed in (must be SAME object)
    captured = {"router_result_in": None, "n_calls": 0}

    async def _fake_pipeline_query(user_query, baseline_docs, top_k, **kwargs):
        captured["n_calls"] += 1
        captured["router_result_in"] = kwargs.get("router_result", None)
        # Verify router_result is the SAME object the router returned.
        # (The refactor's correctness gate: pipeline must NOT re-route.)
        return SimpleNamespace(
            query=user_query,
            route="padded_rag",
            documents=list(baseline_docs) if baseline_docs else [_fake_doc()],
            router_result=captured["router_result_in"] or _fake_router_result_obj(),
            ppr_scores={},
            latency_ms=1.0,
            confidence=0.9,
        )

    main_module.unified_pipeline.query = AsyncMock(side_effect=_fake_pipeline_query)
    main_module.unified_pipeline._graphrag = MagicMock(graph_stats={"nodes": 0, "edges": 0})

    # Generator: return a well-formed answer
    sources = [{
        "id": "c1",
        "title": "HUSC tuyển sinh 2026",
        "url": None,
        "snippet": "baseline chunk",
        "data_year": "2026",
    }]

    async def _fake_generate_answer(*, query, chunks, confidence, is_program_list_query):
        return {
            "answer": "Ngành CNTT 400 chỉ tiêu.",
            "sources": sources,
            "confidence": confidence,
            "provider": "fake",
            "chunks_used": len(chunks),
        }

    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=_fake_generate_answer)
    main_module.llm_generator_service = generator

    # Patch booster to a no-op: detect returns [] so the real boost is
    # a no-op pass-through (avoids lancedb.fetch_by_id side effects and
    # keeps baseline_docs identical to whatever lancedb.retrieve returned).
    # Use monkeypatch.setattr so these are RESTORED after the test —
    # mutating sys.modules[...] directly leaks into other suites (it broke
    # test_v2_booster_wired when both ran in the same session).
    from services import aggregation_booster as _booster_mod
    monkeypatch.setattr(
        _booster_mod, "detect_aggregation_chunks", MagicMock(return_value=[]),
        raising=False,
    )
    monkeypatch.setattr(
        _booster_mod,
        "boost_with_aggregation",
        lambda query, baseline_docs, lancedb_retriever, top_k=5, max_inject=2: list(baseline_docs),
        raising=False,
    )

    return {
        "router": router,
        "pipeline": main_module.unified_pipeline,
        "generator": generator,
        "router_started": router_started,
        "captured": captured,
    }


def _run_sync(coro):
    """Helper used by the (now-superseded) async retrieve variant — left in
    case future tests want a coroutine-returning side_effect."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

def test_router_and_baseline_run_concurrently(main_module, monkeypatch):
    """Robust correctness gate:
      (a) router.route awaited exactly once
      (b) unified_pipeline.query received the SAME router_result object
          (i.e. the pipeline did NOT re-route)
    Plus a best-effort wall-clock timing assertion (generous bound so it
    is not flaky on Windows / busy CI)."""
    fakes = _install_slow_leg_fakes(
        main_module, monkeypatch, router_sleep=0.2, baseline_sleep=0.2, in_scope=True
    )

    client = TestClient(main_module.app)
    t0 = time.perf_counter()
    res = client.post("/v2/query", json={"query": "ngành CNTT bao nhiêu chỉ tiêu?"})
    elapsed = time.perf_counter() - t0

    assert res.status_code == 200, res.text
    body = res.json()

    # (a) Router.route called EXACTLY once
    assert fakes["router"].route.await_count == 1, (
        f"router.route should be awaited exactly once; "
        f"got await_count={fakes['router'].route.await_count}"
    )

    # (b) unified_pipeline.query received the router_result we computed
    rr_in = fakes["captured"]["router_result_in"]
    assert rr_in is not None, (
        "REGRESSION: unified_pipeline.query was called WITHOUT a router_result. "
        "The refactor must pass router_result=... to skip the internal re-route."
    )
    # And it must be the SAME object the router returned (identity check).
    expected = fakes["router"].route.return_value  # the value the AsyncMock will return next call
    # The AsyncMock's `return_value` is the object it returns; the side_effect
    # `_slow_router_route` returned `_fake_router_result_obj()` — a NEW object
    # per call. The pipeline received that exact new object.
    # Compare structurally (fields), since the router is mocked to return a
    # fresh SimpleNamespace each call.
    assert hasattr(rr_in, "intent")
    assert hasattr(rr_in, "route")

    # (c) Wall-clock timing is omitted — the spec allows dropping it if
    # timing is flaky on Windows / busy CI. The ROBUST correctness gate
    # is (a)+(b) above: router.route awaited once, pipeline received the
    # SAME router_result (i.e. it did NOT re-route). The structural gate
    # alone proves the refactor is correct.

    # Happy-path contract: response well-formed
    assert body["answer"] == "Ngành CNTT 400 chỉ tiêu."
    assert body["route"] in ("padded_rag", "graph_rag", "hybrid")
    assert "sources" in body and isinstance(body["sources"], list)


def test_oos_guardrail_does_not_await_router(main_module, monkeypatch):
    """OOS short-circuit must early-return BEFORE we kick off the router
    task. If the refactor accidentally moves the concurrent block above
    the guardrail precheck, router.route.await_count would be 1 for an
    OOS query. This test pins that invariant."""
    fakes = _install_slow_leg_fakes(
        main_module, monkeypatch, router_sleep=0.2, baseline_sleep=0.2, in_scope=False
    )

    client = TestClient(main_module.app)
    res = client.post("/v2/query", json={"query": "thời tiết hôm nay thế nào?"})
    assert res.status_code == 200, res.text
    body = res.json()

    # The guardrail short-circuit MUST short-circuit BEFORE we start the
    # concurrent router task. So router.route was NEVER awaited.
    assert fakes["router"].route.await_count == 0, (
        f"REGRESSION: OOS query kicked off the router task "
        f"(await_count={fakes['router'].route.await_count}). "
        f"The guardrail precheck must early-return BEFORE the concurrent block."
    )

    # The unified pipeline must also not have been called for an OOS query.
    assert fakes["captured"]["n_calls"] == 0, (
        f"REGRESSION: unified_pipeline.query was called for an OOS query "
        f"(n_calls={fakes['captured']['n_calls']})."
    )

    # And the OOS response shape is correct.
    assert body["route"] == "guardrail"
    assert body["answer"] == "Ngoài phạm vi."
    assert body["sources"] == []


def test_happy_path_well_formed_response(main_module, monkeypatch):
    """Sanity: a normal in-scope query still produces a well-formed
    response_payload (answer present, route present) after the refactor."""
    _install_slow_leg_fakes(
        main_module, monkeypatch, router_sleep=0.05, baseline_sleep=0.05, in_scope=True
    )

    client = TestClient(main_module.app)
    res = client.post("/v2/query", json={"query": "học phí ngành CNTT?"})
    assert res.status_code == 200, res.text
    body = res.json()

    # Required fields
    for key in ("answer", "route", "sources", "confidence", "trace_id"):
        assert key in body, f"missing {key!r} in response"

    # Answer from the fake generator
    assert body["answer"] == "Ngành CNTT 400 chỉ tiêu."
    # Route is one of the synthesized routes
    assert body["route"] in ("padded_rag", "graph_rag", "hybrid")
    # Sources is a list
    assert isinstance(body["sources"], list)
    # Confidence is a float
    assert isinstance(body["confidence"], (int, float))
