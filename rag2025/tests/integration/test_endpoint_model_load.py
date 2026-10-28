"""Ticket 46, criteria 2 and 3 — measured through the REAL HTTP endpoints.

`test_groundedness_model_cache.py` proves the cache at the `faithfulness_score`
boundary. It does NOT prove the two ticket claims, which are stated in terms of
the serving surface:

* C2 - ``/query`` and ``/v2/query`` no longer load the model *inside a request*,
  and no "Loading weights" line appears between two consecutive requests.
* C3 - under concurrent load (>=2 requests at once) the wall time is NOT the sum
  of the individual request times. That is the check that separates "cached for
  speed" from "off the event loop": an ``lru_cache`` alone would still block the
  loop during the first load.

Both routes call ``_safe_groundedness_score`` through ``asyncio.to_thread``
(``src/main.py:1119`` and ``src/main.py:1905``). These tests drive the real
routes on the real app so the claim is about the endpoint, not about a helper.

Only the LLM generator and the embedding are stubbed (the provider is down), and
C3 additionally replaces the model with a slow stand-in so the overlap window is
deterministic instead of depending on how fast this machine loads weights.

Run:  python -m pytest tests/integration/test_endpoint_model_load.py -q
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # rag2025/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "chaos_publish_rollback.py"

ENUM_QUERY = "Liệt kê các ngành tuyển sinh năm 2026"

# `/query` builds a metadata filter from the query TEXT
# (`src/main.py:925-951`): "các ngành" selects only rows whose ``faq_type`` is
# one of the liet_ke types. The drill's seeded rows carry none of those, so a
# filtered search returns "No matching documents found", `/query` yields zero
# chunks, and `_safe_groundedness_score` early-returns - the model is never
# loaded and every criterion below would pass vacuously. This text hits none of
# the overview / tuition / program-list keyword branches, so retrieval is
# unfiltered and returns rows (the drill's embedding pins all seeded rows at
# distance 0 from the query anchor).
PLAIN_QUERY = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là bao nhiêu?"


def _text_for(path: str) -> str:
    """ENUM_QUERY exercises the /v2 booster path; /query needs an unfiltered one."""
    return PLAIN_QUERY if path == "/query" else ENUM_QUERY


def _load_drill():
    """Import the drill script as a module (registers sys.modules for dataclasses)."""
    spec = importlib.util.spec_from_file_location("chaos_drill_script_t46", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["chaos_drill_script_t46"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("chaos_drill_script_t46", None)
        raise
    return module


class LoadCountingEmbedding:
    """The drill's deterministic embedding plus the ``encode_batch`` ``/query``
    needs. Delegates everything else, so retrieval behaviour is unchanged.

    ``/query`` embeds its variants through ``encode_batch``; ``/v2/query`` goes
    through the retriever's ``encode_query``.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._inner.encode_query(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._inner.encode_query(text)


class StubEnhancer:
    """Minimal HYDE stand-in: the query itself as the single variant.

    The real enhancer builds a ``UnifiedLLMClient`` and would call the (dead)
    provider. Enhancement quality is not what ticket 46 measures.
    """

    async def enhance_query(self, user_query: str, force_rag_only: bool = False) -> Dict[str, Any]:
        return {
            "original_query": user_query,
            "detected_intent": "liet_ke",
            "query_type": "program_list",
            "variants": [user_query],
            "top_k": 5,
        }


class _SlowModel:
    """Stand-in model whose ``encode`` is slow and GIL-releasing.

    Used by the C3 tests: a deterministic overlap window that does not depend on
    this machine's disk speed. ``time.sleep`` releases the GIL, so two threads
    running it genuinely overlap - which is exactly the property under test.
    """

    def __init__(self, stall_s: float) -> None:
        self.stall_s = stall_s

    def encode(self, texts, normalize_embeddings: bool = True):
        time.sleep(self.stall_s)
        return np.ones((len(texts), 8), dtype=np.float32)


@pytest.fixture(scope="module")
def drill():
    return _load_drill()


@pytest.fixture
def client_and_main(drill, tmp_path_factory):
    """Build the drill app and yield (TestClient, main module).

    ``TestClient`` is ENTERED, not used bare: ``__enter__`` starts the single
    blocking portal every request shares, so all requests run on one event loop.
    A bare client would start a fresh loop per request and the concurrency
    question would be meaningless (``scripts/chaos_publish_rollback.py:480-491``).
    """
    from fastapi.testclient import TestClient

    workdir = tmp_path_factory.mktemp("t46-endpoint")
    scenario = {
        "workers": 1,
        "duration_s": 0.0,
        "rate_rps": 0.0,
        "publish_at_s": 1e9,
        "rollback_at_s": 1e9,
        "query_delay_s": 0.0,
        "seed": 29,
    }

    # `_configure_env` is what the drill's CLI calls before building the app
    # (`scripts/chaos_publish_rollback.py:1317`). It is not optional: it pins
    # EMBEDDING_DIM to the 640 the seeded table uses. Skipping it leaves the
    # production 1024 in place, retrieval fails with a dimension mismatch,
    # every response comes back with zero chunks, and `_safe_groundedness_score`
    # early-returns - so the model is never loaded and these tests would be
    # vacuously green.
    #
    # Order matters: `_configure_env` MUTATES os.environ, so the snapshot has to
    # be taken BEFORE it runs. Capturing afterwards (as this fixture originally
    # did) saves the already-overridden values and "restores" them on the way
    # out, which pins EMBEDDING_DIM=640 for the rest of the process. Measured:
    # `pytest tests/integration/test_endpoint_model_load.py tests/test_embedding.py`
    # -> 8 failed with "expected 640, got 1024"; `tests/test_embedding.py` alone
    # -> 10 passed.
    overrides = {
        "APP_ENV", "LOG_LEVEL", "LANCEDB_URI", "LANCEDB_TABLE", "EMBEDDING_DIM",
        "CANONICAL_STORE_ROOT", "USE_HYBRID_RETRIEVAL", "RATE_LIMIT_PER_MINUTE",
        "RERANKER_ENABLED", "RERANKER_ADAPTIVE_CONFIDENCE",
        "CURRENT_ADMISSION_YEAR", "RAG_CONF_THRESHOLD",
    }
    saved_env = {k: os.environ.get(k) for k in overrides}

    async def _prepare():
        await drill._build_quiet_app(scenario, workdir)
        main = drill._MAIN_BOX["main"]
        # /query reads these globals; the drill only wires the /v2 path.
        main.query_enhancer_service = StubEnhancer()
        main.embedding_service = LoadCountingEmbedding(main.embedding_service)
        return main

    # Everything that can fail lives INSIDE the try - including setup.
    # `_configure_env` mutates os.environ and `_prepare()` can raise (model
    # download failure, missing seeded artifact, a bad workdir), so guarding
    # only the yield would still leak the drill's env whenever setup dies
    # before the test body ever runs. Restoring on the setup path is the whole
    # point: a setup failure must not change what later tests see.
    try:
        drill._configure_env(workdir)
        main = asyncio.run(_prepare())
        with TestClient(main.app) as client:
            yield client, main
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _query_body(path: str, text: str) -> Dict[str, Any]:
    if path == "/query":
        return {"query": text}
    return {"query": text, "top_k": 5, "history": []}


def _swap_model_factory(factory):
    """Patch the module attribute `_get_faithfulness_model` imports at call time.

    ``results/metrics.py:34`` does ``from sentence_transformers import
    SentenceTransformer`` INSIDE the function, so patching the module attribute
    is what takes effect. Returns a restore callable.
    """
    import sentence_transformers

    saved_cls = sentence_transformers.SentenceTransformer
    sentence_transformers.SentenceTransformer = factory
    return lambda: setattr(sentence_transformers, "SentenceTransformer", saved_cls)


# ---------------------------------------------------------------------------
# C2 - no model load inside a request, on BOTH endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/query", "/v2/query"])
def test_second_request_does_not_load_model(client_and_main, path, caplog):
    """C2: request #2 on each route must not pay the model load again.

    The load is what "Loading weights" comes from, so the primary evidence is a
    count of real constructions across the two requests: request #1 must build
    exactly one model and request #2 must build none. A latency-only assertion
    would pass on a warm page cache even if the model were rebuilt, so the count
    is the assertion; the log scan and the timing are corroboration.

    The cache is reset first, so request #1 genuinely pays the cost - otherwise
    a model cached by an earlier test would make the count trivially 0 and the
    test vacuous.
    """
    from results import metrics as M

    client, _ = client_and_main

    saved_model = M._faithfulness_model
    M._faithfulness_model = None

    constructions: List[int] = []
    real_cls = __import__("sentence_transformers").SentenceTransformer

    def _counting_factory(*a, **kw):
        constructions.append(1)
        return real_cls(*a, **kw)

    restore = _swap_model_factory(_counting_factory)
    try:
        with caplog.at_level(logging.INFO):
            caplog.clear()
            t0 = time.perf_counter()
            r1 = client.post(path, json=_query_body(path, _text_for(path) + " [c2-a]"))
            first_s = time.perf_counter() - t0
            n_after_first = len(constructions)

            caplog.clear()  # only records BETWEEN the two requests count
            t1 = time.perf_counter()
            r2 = client.post(path, json=_query_body(path, _text_for(path) + " [c2-b]"))
            second_s = time.perf_counter() - t1
            between = [r.getMessage() for r in caplog.records if "Loading weights" in r.getMessage()]
    finally:
        restore()
        M._faithfulness_model = saved_model

    assert r1.status_code == 200, (path, r1.status_code, r1.text[:400])
    assert r2.status_code == 200, (path, r2.status_code, r2.text[:400])

    # Request #1 must actually have reached the scoring path, or the rest of the
    # test proves nothing.
    assert n_after_first == 1, (
        f"{path}: expected request #1 to construct exactly one model, "
        f"got {n_after_first} - did the request reach _safe_groundedness_score?"
    )
    assert len(constructions) == 1, (
        f"{path}: request #2 constructed another model ({n_after_first} -> "
        f"{len(constructions)})"
    )
    assert not between, f"{path}: 'Loading weights' between the requests: {between[:3]}"
    assert second_s < first_s, (
        f"{path}: request #2 ({second_s:.3f}s) was not faster than #1 ({first_s:.3f}s)"
    )
    print(
        f"\n{path}: req#1={first_s:.3f}s req#2={second_s:.3f}s "
        f"constructions={len(constructions)} loading-weights-between=0"
    )


# ---------------------------------------------------------------------------
# C3 - the loop keeps serving while the scoring call is in flight
# ---------------------------------------------------------------------------


def test_concurrent_requests_do_not_sum_their_wall_time(client_and_main):
    """C3: N concurrent requests must cost about ONE stall, not N of them.

    Why this is measured against a baseline rather than against the sum of the
    request durations: when requests serialize, the queued request's own measured
    duration grows to include the waiting, so `sum` inflates to roughly
    `wall + T` and comparing `wall < 0.9 * sum` stays true even though the
    requests were fully serialized. Measured on a known-blocking handler
    (2 concurrent, 1s stall): wall=2.004s, per-request=[1.002, 2.003] - the naive
    sum check passes there, so it does not discriminate. A single-request
    baseline does: serialized N requests take ~N*T, overlapped ones take ~T.
    """
    from results import metrics as M

    client, _ = client_and_main

    STALL_S = 1.5
    CONCURRENT = 3

    saved_model = M._faithfulness_model
    M._faithfulness_model = None

    restore = _swap_model_factory(lambda *a, **kw: _SlowModel(STALL_S))

    def _one(path: str, tag: str) -> Tuple[int, float]:
        t = time.perf_counter()
        try:
            resp = client.post(path, json=_query_body(path, _text_for(path) + f" [{tag}]"))
            return resp.status_code, time.perf_counter() - t
        except Exception:  # a transport failure is a result, not a crash
            return -1, time.perf_counter() - t

    try:
        # Baseline: what ONE request costs with the stall in place. This is the
        # unit the concurrent batch is compared against, and it is measured, not
        # assumed, so the assertion does not depend on machine speed.
        base_status, base_s = _one("/v2/query", "c3-baseline")
        assert base_status == 200, base_status
        assert base_s > STALL_S * 0.8, (
            f"baseline request was {base_s:.3f}s, expected the {STALL_S}s stall to "
            f"dominate - the scoring path was probably not reached"
        )

        results: List[Tuple[str, int, float]] = []
        started = threading.Event()

        def _fire(path: str, tag: str):
            started.wait()
            status, duration = _one(path, tag)
            results.append((path, status, duration))

        paths = ["/v2/query", "/query", "/v2/query"][:CONCURRENT]
        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=_fire, args=(path, f"c3-{i}"))
            for i, path in enumerate(paths)
        ]
        for th in threads:
            th.start()
        started.set()
        for th in threads:
            th.join()
        wall_s = time.perf_counter() - t0
    finally:
        restore()
        M._faithfulness_model = saved_model

    assert len(results) == CONCURRENT, results
    assert all(s == 200 for _, s, _ in results), results

    durations = [d for _, _, d in results]
    print(
        f"\nbaseline(1 req)={base_s:.3f}s  concurrent({CONCURRENT}) wall={wall_s:.3f}s "
        f"per-request={[round(d, 3) for d in durations]}"
    )

    # Serialized would put the batch at ~CONCURRENT * base_s; overlapped puts it
    # at roughly base_s, since every request stalls in parallel.
    budget = base_s * 1.5
    assert wall_s < budget, (
        f"{CONCURRENT} concurrent requests took {wall_s:.3f}s, but one request "
        f"takes {base_s:.3f}s (budget {budget:.3f}s) - they appear serialized, "
        f"so the scoring call is not off the event loop"
    )


def test_loop_stays_live_during_scoring(client_and_main):
    """C3, sharper form: a cheap GET must be answered WHILE scoring runs.

    The concurrency test could in principle pass on a fast machine. This one
    cannot: a fixed stall is forced into the scoring call, and a trivial
    ``/metrics`` request is issued repeatedly *during* it. If the scoring call
    ran on the event loop, ``/metrics`` could not be answered until it finished.
    """
    from results import metrics as M

    client, _ = client_and_main

    saved_model = M._faithfulness_model
    M._faithfulness_model = None

    restore = _swap_model_factory(lambda *a, **kw: _SlowModel(3.0))

    probe_latencies: List[float] = []
    probe_errors: List[str] = []
    stop = threading.Event()
    # Set immediately before the slow request is posted. Only probes STARTED
    # after this instant count: a probe that completed before the request began
    # proves nothing about the loop staying live during the scoring call.
    window_start: List[float] = []

    def _probe():
        while not stop.is_set():
            t = time.perf_counter()
            try:
                r = client.get("/metrics")
                finished = time.perf_counter()
                if window_start and t >= window_start[0]:
                    probe_latencies.append(finished - t)
                if r.status_code != 200:
                    probe_errors.append(f"status {r.status_code}")
            except Exception as exc:
                probe_errors.append(repr(exc))

    try:
        prober = threading.Thread(target=_probe)
        prober.start()
        window_start.append(time.perf_counter())
        t0 = time.perf_counter()
        r = client.post("/v2/query", json=_query_body("/v2/query", ENUM_QUERY + " [c3-live]"))
        slow_s = time.perf_counter() - t0
        stop.set()
        prober.join()
    finally:
        restore()
        M._faithfulness_model = saved_model

    assert r.status_code == 200, r.text[:400]
    assert not probe_errors, probe_errors[:3]
    assert slow_s > 2.5, f"expected the forced stall to dominate, got {slow_s:.3f}s"
    assert len(probe_latencies) >= 5, (
        f"only {len(probe_latencies)} probe request(s) were served during the "
        f"{slow_s:.3f}s scoring window - the loop looks blocked"
    )
    print(
        f"\nslow request={slow_s:.3f}s cheap /metrics served DURING it: "
        f"{len(probe_latencies)}x max={max(probe_latencies)*1000:.1f}ms"
    )


def test_setup_failure_still_restores_the_environment(drill, tmp_path_factory, monkeypatch):
    """The fixture's restore must cover the SETUP path, not just the test body.

    Why this test exists: the fixture pins ``EMBEDDING_DIM`` (and eleven other
    vars) in ``os.environ`` before the app is built. An earlier version guarded
    only the ``yield``, which meant a failure *during setup* - a model that will
    not download, a missing seeded artifact, a bad workdir - escaped with the
    drill's env still installed. The next module in the process then saw
    ``EMBEDDING_DIM=640`` and failed with "expected 640, got 1024": 27 failures
    across the suite, none of them in this file. Guarding the yield alone cannot
    establish the invariant, because the dangerous window opens before it.

    So this test forces ``_prepare()`` to raise and asserts the environment is
    still put back. It fails if the fixture regresses to guarding only the yield.
    """
    from fastapi.testclient import TestClient  # noqa: F401  (fixture parity)

    workdir = tmp_path_factory.mktemp("t46-setup-failure")
    scenario = {
        "workers": 1,
        "duration_s": 0.0,
        "rate_rps": 0.0,
        "publish_at_s": 1e9,
        "rollback_at_s": 1e9,
        "query_delay_s": 0.0,
        "seed": 29,
    }

    # Snapshot the real environment BEFORE anything mutates it, so the
    # assertion compares against the true pre-fixture state.
    before = {k: os.environ.get(k) for k in (
        "APP_ENV", "LOG_LEVEL", "LANCEDB_URI", "LANCEDB_TABLE", "EMBEDDING_DIM",
        "CANONICAL_STORE_ROOT", "USE_HYBRID_RETRIEVAL", "RATE_LIMIT_PER_MINUTE",
        "RERANKER_ENABLED", "RERANKER_ADAPTIVE_CONFIDENCE",
        "CURRENT_ADMISSION_YEAR", "RAG_CONF_THRESHOLD",
    )}

    async def _boom(*_a, **_kw):
        raise RuntimeError("forced setup failure (t46 env-restore test)")

    # Break the app build, then drive the fixture's own generator by hand.
    monkeypatch.setattr(drill, "_build_quiet_app", _boom, raising=True)

    fixture_gen = client_and_main.__wrapped__(drill, tmp_path_factory)
    try:
        with pytest.raises(RuntimeError, match="forced setup failure"):
            next(fixture_gen)
    finally:
        fixture_gen.close()

    after = {k: os.environ.get(k) for k in before}
    assert after == before, (
        "the fixture leaked environment changes out of a FAILED setup: "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )
