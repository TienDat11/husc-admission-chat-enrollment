"""Tests for RequestTracker (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import asyncio
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "src" / "app" / "request_tracker.py"


@pytest.fixture
def rt_module():
    spec = importlib.util.spec_from_file_location("request_tracker", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_initial_state_is_drained(rt_module):
    rt = rt_module.RequestTracker()
    assert rt.in_flight == 0
    drained = await rt.drain(timeout=0.5)
    assert drained is True


@pytest.mark.asyncio
async def test_track_increments_then_decrements(rt_module):
    rt = rt_module.RequestTracker()

    async def use_one():
        async with rt.track():
            assert rt.in_flight == 1
            await asyncio.sleep(0.01)

    await use_one()
    assert rt.in_flight == 0


@pytest.mark.asyncio
async def test_drain_returns_true_when_no_inflight(rt_module):
    rt = rt_module.RequestTracker()
    drained = await rt.drain(timeout=0.5)
    assert drained is True


@pytest.mark.asyncio
async def test_drain_returns_false_on_timeout_when_request_holding(rt_module):
    rt = rt_module.RequestTracker()
    holding = asyncio.Event()
    release = asyncio.Event()

    async def long_request():
        async with rt.track():
            holding.set()
            await release.wait()

    task = asyncio.create_task(long_request())
    await holding.wait()
    drained = await rt.drain(timeout=0.1)
    assert drained is False
    release.set()
    await task
    # And after the request releases, drain should immediately succeed.
    drained2 = await rt.drain(timeout=0.5)
    assert drained2 is True


@pytest.mark.asyncio
async def test_concurrent_acquire_release(rt_module):
    rt = rt_module.RequestTracker()

    async def worker():
        async with rt.track():
            await asyncio.sleep(0.01)

    # Fire 20 concurrent tracked workers; max in_flight should reach 20.
    tasks = [asyncio.create_task(worker()) for _ in range(20)]
    # Give a moment for all to enter.
    await asyncio.sleep(0.005)
    assert rt.in_flight > 0
    await asyncio.gather(*tasks)
    assert rt.in_flight == 0


@pytest.mark.asyncio
async def test_drain_succeeds_once_all_release(rt_module):
    rt = rt_module.RequestTracker()
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker():
        async with rt.track():
            started.set()
            await release.wait()

    t1 = asyncio.create_task(worker())
    t2 = asyncio.create_task(worker())
    await started.wait()
    assert rt.in_flight >= 1

    # Schedule release before drain so drain succeeds within timeout.
    async def release_soon():
        await asyncio.sleep(0.05)
        release.set()

    asyncio.create_task(release_soon())
    drained = await rt.drain(timeout=2.0)
    assert drained is True
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_in_flight_property_is_readonly_snapshot(rt_module):
    rt = rt_module.RequestTracker()
    snap1 = rt.in_flight
    assert snap1 == 0
    async with rt.track():
        snap2 = rt.in_flight
        assert snap2 == 1
    snap3 = rt.in_flight
    assert snap3 == 0
