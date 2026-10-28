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


@pytest.mark.asyncio
async def test_gate_closes_admission_for_the_body(rt_module):
    """Admission is closed from before the body until after it returns.

    This is the property the swap depends on: a request admitted inside the body
    would read the runtime being replaced.
    """
    rt = rt_module.RequestTracker()
    observed: list[bool] = []

    async with rt.gate(timeout=0.5) as drained:
        observed.append(rt.admitting)
        assert drained is True
    observed.append(rt.admitting)

    assert observed == [False, True]


@pytest.mark.asyncio
async def test_gate_blocks_track_until_it_reopens(rt_module):
    """A ``track()`` arriving mid-gate waits for the reopen instead of proceeding.

    The count it would have incremented is the one the drain just proved zero.
    """
    rt = rt_module.RequestTracker()
    entered = asyncio.Event()

    async def late_request():
        async with rt.track():
            entered.set()

    async with rt.gate(timeout=0.5):
        task = asyncio.create_task(late_request())
        # Let the task reach the admission wait; it must not get in.
        await asyncio.sleep(0.01)
        assert not entered.is_set()
        assert rt.in_flight == 0

    await asyncio.wait_for(task, timeout=1.0)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_gate_drains_before_yielding(rt_module):
    """The drain happens before the body: in-flight work is gone when it starts."""
    rt = rt_module.RequestTracker()
    release = asyncio.Event()

    async def in_flight_request():
        async with rt.track():
            await release.wait()

    task = asyncio.create_task(in_flight_request())
    await asyncio.sleep(0.01)
    assert rt.in_flight == 1

    async def release_soon():
        await asyncio.sleep(0.05)
        release.set()

    asyncio.create_task(release_soon())
    async with rt.gate(timeout=2.0) as drained:
        assert drained is True
        assert rt.in_flight == 0
    await task


@pytest.mark.asyncio
async def test_gate_reopens_admission_when_the_body_raises(rt_module):
    """A raising body must not strand admission closed — every later request waits forever."""
    rt = rt_module.RequestTracker()

    with pytest.raises(RuntimeError, match="swap blew up"):
        async with rt.gate(timeout=0.5):
            raise RuntimeError("swap blew up")

    assert rt.admitting is True
    # Proof it is usable, not merely flagged open.
    async with rt.track():
        assert rt.in_flight == 1


@pytest.mark.asyncio
async def test_gate_reports_false_on_timeout_and_still_reopens(rt_module):
    """On timeout the caller is told, and admission is reopened regardless."""
    rt = rt_module.RequestTracker()
    release = asyncio.Event()

    async def stuck_request():
        async with rt.track():
            await release.wait()

    task = asyncio.create_task(stuck_request())
    await asyncio.sleep(0.01)

    async with rt.gate(timeout=0.05) as drained:
        assert drained is False
        assert rt.admitting is False

    assert rt.admitting is True
    release.set()
    await task
