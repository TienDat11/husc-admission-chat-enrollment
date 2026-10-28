"""Ticket 46: `_safe_groundedness_score` must not pay the model-load cost per request.

The failure this guards against is silent: `SentenceTransformer(...)` *succeeds*
in ~10-15s, so the `try/except` fallback in `_safe_groundedness_score` never
fires and the cost only shows up as latency. A test that only asserts "the score
is a float" would pass on the broken code, so every criterion here is measured
against the cost itself.

Run:  python -m pytest tests/integration/test_groundedness_model_cache.py -q
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

RAG_ROOT = Path(__file__).resolve().parents[2]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from results import metrics as M  # noqa: E402

# The model load is the thing under test, so it cannot be mocked away. One
# process-wide build is expected; the assertions are about the SECOND call.
ANSWER = "Diem chuan nganh Cong nghe thong tin nam 2026 la 26.5 diem."
SOURCES = [
    "Diem chuan nganh Cong nghe thong tin nam 2026 la 26.5 diem theo phuong thuc xet tuyen THPT.",
    "Hoc phi nganh Quan tri kinh doanh nam 2026 la 18 trieu dong moi nam.",
]


@pytest.fixture(scope="module", autouse=True)
def _reset_model_cache():
    """Start from an unloaded cache so call #1 is a genuine first call."""
    M._faithfulness_model = None
    yield


def test_second_call_is_an_order_of_magnitude_faster():
    """Criterion 1: call #2 in the same process must be >=10x faster than #1."""
    t0 = time.perf_counter()
    first = M.faithfulness_score(ANSWER, SOURCES)
    first_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    second = M.faithfulness_score(ANSWER, SOURCES)
    second_s = time.perf_counter() - t1

    assert isinstance(first, float) and isinstance(second, float)
    # Same input, same model => identical score (criterion 4, zero tolerance).
    assert first == second

    speedup = first_s / max(second_s, 1e-9)
    print(
        f"\ncall#1={first_s:.3f}s  call#2={second_s:.3f}s  speedup={speedup:.1f}x"
    )
    # The load dominates call #1 by ~10s; the margin is generous so a loaded CI
    # box does not flake, while a per-call load (~10s both times) fails hard.
    assert speedup >= 10.0, f"call#2 only {speedup:.1f}x faster - model not cached?"
    assert second_s < 1.0, f"call#2 took {second_s:.3f}s - still loading the model"


def test_semantics_unchanged_vs_uncached_reference():
    """Criterion 4: caching must not move the score at all.

    Recomputes the same cosine-similarity definition with a freshly built model
    and compares. This is the check that a `cache` refactor did not quietly
    change what the number means.
    """
    cached = M.faithfulness_score(ANSWER, SOURCES)

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    fresh = SentenceTransformer(M.FAITHFULNESS_MODEL_NAME)
    emb = fresh.encode([ANSWER] + SOURCES, normalize_embeddings=True)
    expected = float(
        np.max(cosine_similarity(emb[0].reshape(1, -1), emb[1:])[0])
    )

    assert cached == pytest.approx(expected, abs=0.0), (
        f"cached={cached!r} uncached={expected!r}"
    )


def test_failure_is_not_cached():
    """Criterion 5: a load failure must not poison the cache.

    `_get_faithfulness_model` documents "failures are never cached". If it were
    cached, one transient disk error would disable groundedness for the process
    lifetime.
    """
    saved = M._faithfulness_model
    M._faithfulness_model = None
    try:
        # Simulate an unimportable sentence_transformers by pointing the name at
        # a module that does not exist for the duration of the call.
        import builtins

        real_import = builtins.__import__

        def _boom(name, *a, **kw):
            if name == "sentence_transformers":
                raise ImportError("simulated missing dependency")
            return real_import(name, *a, **kw)

        builtins.__import__ = _boom
        try:
            # `faithfulness_score` swallows ImportError and returns its default.
            assert M.faithfulness_score(ANSWER, SOURCES) == 0.5
            assert M._faithfulness_model is None, "failure was cached!"
        finally:
            builtins.__import__ = real_import
    finally:
        M._faithfulness_model = saved


def test_concurrent_first_callers_build_one_model():
    """The double-checked lock must not let two threads each build a model.

    Criterion 3's premise: construction is the expensive part, so concurrent
    first callers must not each pay it. Counts real constructions.
    """
    saved = M._faithfulness_model
    M._faithfulness_model = None
    constructions = []
    real_get = M._get_faithfulness_model

    import sentence_transformers

    real_cls = sentence_transformers.SentenceTransformer

    class Counting(real_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            constructions.append(1)
            super().__init__(*a, **kw)

    sentence_transformers.SentenceTransformer = Counting
    try:
        barrier = threading.Barrier(4)
        results: list[float] = []

        def _worker():
            barrier.wait()
            results.append(M.faithfulness_score(ANSWER, SOURCES))

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(constructions) == 1, (
            f"{len(constructions)} models built by 4 concurrent callers"
        )
        assert len(results) == 4 and len(set(results)) == 1
    finally:
        sentence_transformers.SentenceTransformer = real_cls
        M._faithfulness_model = saved


def test_sync_wrapper_does_not_block_the_event_loop():
    """Criterion 3: a concurrent request must not wait for the scoring call.

    Distinguishes "cached for speed" from "off the event loop": `lru_cache`
    alone would still block the loop on the first load. The call sites use
    `asyncio.to_thread`, so a slow scoring call must let the loop keep running.
    """
    from src.main import _safe_groundedness_score

    chunks = [{"text": t} for t in SOURCES]

    async def main() -> tuple[float, int]:
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        ticker = asyncio.create_task(_ticker())
        # Force a slow scoring call: drop the cache so this thread rebuilds.
        saved = M._faithfulness_model
        M._faithfulness_model = None
        try:
            t0 = time.perf_counter()
            score = await asyncio.to_thread(
                _safe_groundedness_score, ANSWER, chunks
            )
            elapsed = time.perf_counter() - t0
        finally:
            M._faithfulness_model = saved
            ticker.cancel()

        # The loop must have spun many times DURING the blocking call.
        assert elapsed > 0.5, "expected the uncached load to be slow"
        return score, ticks

    score, ticks = asyncio.run(main())
    print(f"\nuncached scoring call kept the loop alive: {ticks} ticks")
    assert isinstance(score, float)
    # If the call blocked the loop, no ticker iteration could run and this is 0.
    assert ticks >= 5, (
        f"event loop only ran {ticks} iterations during a {0.0}s blocking call "
        "- the sync scoring call is still blocking"
    )
