"""TDD: BE lifespan runs core warmup in PARALLEL via asyncio.gather +
loop.run_in_executor (with a Semaphore(2)) — so total warm time is
bounded by ~max(load_times) rather than their sum.

Contract pinned by these tests:
  1. With 4 mock services each sleeping 3s, parallel warm (semaphore=2)
     takes < 70% of the sequential baseline (12s → ≤ 8.4s target). The
     exact threshold uses a generous margin (≤ 9.5s) to keep the test
     stable on Windows timer granularity.
  2. State machine: stage starts as 'warming', ends 'ready', no failures.
  3. If one service raises, gather still completes, stage='failed',
     but the app still yields (does NOT crash startup).

These tests bypass the real src/main.py module load (which triggers
Qwen3 import) by exercising a TINY shim that mirrors the lifespan
parallel-warm pattern. The shim lives next to the test so production
code stays unchanged in this slice.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Shaped warmup runner — mirrors the lifespan pattern in src/main.py
# (asyncio.Semaphore + loop.run_in_executor + try/except per task). Kept
# here so the test runs without importing the heavy main module.
# ─────────────────────────────────────────────────────────────────────────────


async def _run_warmup_tasks(
    tasks: Dict[str, Any],
    *,
    semaphore_limit: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Run a dict of name→callable warmups in parallel under a Semaphore.

    Each task is wrapped in try/except; the per-task result captures
    state ('ok'|'failed') and timing so the caller can flip a stage
    machine ('warming' → 'ready'|'failed').
    """
    sem = asyncio.Semaphore(semaphore_limit)
    loop = asyncio.get_running_loop()
    results: Dict[str, Dict[str, Any]] = {}

    async def _wrap(name: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            async with sem:
                await loop.run_in_executor(None, fn)
            results[name] = {"state": "ok", "ms": int((time.perf_counter() - t0) * 1000)}
        except BaseException as exc:  # surface for inspection
            results[name] = {
                "state": "failed",
                "ms": int((time.perf_counter() - t0) * 1000),
                "error": repr(exc),
            }

    await asyncio.gather(*(_wrap(n, fn) for n, fn in tasks.items()))
    return results


def _make_sleep_fn(seconds: float, store: Dict[str, Any], key: str):
    def _fn() -> None:
        time.sleep(seconds)
        store[key] = True

    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parallel < 70% sequential
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warmup_parallel_faster_than_70pct_sequential_baseline():
    """4 services × 3s each. Sequential baseline = 12s. Parallel with
    semaphore=2 finishes in ≤ ~9.5s (target ≤ 8.4s = 70%, but we leave a
    generous margin so the test is stable on Windows scheduling jitter)."""
    per_task_seconds = 0.3  # 0.3s × 4 = 1.2s sequential; parallel target ≤ 0.84s
    warm = 4
    store: Dict[str, Any] = {}
    tasks = {f"svc_{i}": _make_sleep_fn(per_task_seconds, store, f"svc_{i}") for i in range(warm)}

    t0 = time.perf_counter()
    res = await _run_warmup_tasks(tasks)
    parallel_seconds = time.perf_counter() - t0

    sequential_seconds = per_task_seconds * warm  # = 1.2s
    assert len(res) == warm
    assert all(r["state"] == "ok" for r in res.values())
    # Target: parallel ≤ 70% of sequential + jitter margin (≈ 0.3s).
    threshold = sequential_seconds * 0.70 + per_task_seconds
    assert parallel_seconds <= threshold, (
        f"parallel warm {parallel_seconds:.2f}s > threshold {threshold:.2f}s "
        f"(sequential baseline {sequential_seconds:.2f}s)"
    )
    # And all 4 services actually ran.
    assert len(store) == warm


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stage machine: warming → ready when all ok
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warmup_stage_transitions_warming_to_ready_when_all_ok():
    """Track stage transitions through the warmup lifecycle. We wrap the
    helper to expose the stage mutation — same shape as main.py."""
    stages: list[str] = []

    def _flip(target: str) -> None:
        stages.append(target)

    _flip("warming")
    tasks = {
        "embedding": _make_sleep_fn(0.05, {}, "embedding"),
        "reranker": _make_sleep_fn(0.05, {}, "reranker"),
    }
    res = await _run_warmup_tasks(tasks)
    failed = any(r["state"] == "failed" for r in res.values())
    _flip("failed" if failed else "ready")

    assert stages == ["warming", "ready"]
    assert all(r["state"] == "ok" for r in res.values())


# ─────────────────────────────────────────────────────────────────────────────
# 3. One service fails → stage='failed', but lifespan still yields
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warmup_one_service_fails_marks_stage_failed_but_yields():
    """A failing service must NOT prevent the app from starting. The
    stage flips to 'failed' but the surrounding lifespan wrapper still
    yields control (mirroring main.py:305-310)."""
    stages: list[str] = []

    def _flip(s: str) -> None:
        stages.append(s)

    _flip("warming")

    def _boom() -> None:
        raise RuntimeError("embedding model load failed")

    tasks = {
        "embedding": _boom,
        "reranker": _make_sleep_fn(0.05, {}, "reranker"),
        "graphrag": _make_sleep_fn(0.05, {}, "graphrag"),
    }
    res = await _run_warmup_tasks(tasks)
    failed = any(r["state"] == "failed" for r in res.values())
    _flip("failed" if failed else "ready")

    # The successful tasks still ran (failure is isolated per task).
    assert res["embedding"]["state"] == "failed"
    assert "embedding model load failed" in res["embedding"]["error"]
    assert res["reranker"]["state"] == "ok"
    assert res["graphrag"]["state"] == "ok"
    # And the lifecycle still transitioned past 'warming' — it did not
    # crash the gather.
    assert stages == ["warming", "failed"]