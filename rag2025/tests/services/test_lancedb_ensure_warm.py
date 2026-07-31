"""TDD: `LanceDBAdapter.ensure_warm()` is idempotent + thread-safe.

Contract pinned by these tests:
  1. First call constructs a real adapter (via the provided factory).
  2. Subsequent calls return the SAME instance without calling the factory.
  3. Concurrent calls from N threads only invoke the factory once
     (threading.Lock guards the init).
  4. If the factory raises on the first call, the exception propagates
     and the next call re-tries (no cached failure).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.infrastructure.lancedb_adapter as adapter_mod  # noqa: E402
from src.infrastructure.lancedb_adapter import LanceDBAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts with a clean singleton state (no leftover instance
    or factory from a prior run)."""
    adapter_mod._reset_ensure_warm_for_tests()
    yield
    adapter_mod._reset_ensure_warm_for_tests()


def test_ensure_warm_calls_factory_exactly_once_sequential():
    """Sequential calls return the same instance; factory is called once."""
    fake = MagicMock(return_value=MagicMock(spec=LanceDBAdapter))
    adapter_mod.configure_ensure_warm_factory(fake)

    a = adapter_mod.ensure_warm()
    b = adapter_mod.ensure_warm()
    c = adapter_mod.ensure_warm()

    assert a is b is c, "ensure_warm must return the same instance"
    assert fake.call_count == 1, f"factory called {fake.call_count} times"


def test_ensure_warm_concurrent_only_inits_once():
    """20 threads racing into ensure_warm still only construct one adapter."""
    started = threading.Event()
    fake = MagicMock(side_effect=lambda: (started.set(), MagicMock(spec=LanceDBAdapter))[1])
    adapter_mod.configure_ensure_warm_factory(fake)

    n = 20
    barrier = threading.Barrier(n)
    results: List = [None] * n
    errors: List[BaseException] = []

    def _worker(i: int) -> None:
        try:
            barrier.wait()
            results[i] = adapter_mod.ensure_warm()
        except BaseException as exc:  # surface in main thread
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    ids = {id(r) for r in results if r is not None}
    assert len(ids) == 1, f"expected single instance, got {len(ids)}"
    assert fake.call_count == 1, f"factory called {fake.call_count} times under concurrency"


def test_ensure_warm_propagates_init_failure_and_retries():
    """A failing first attempt must NOT cache the failure — the next call
    retries the factory."""
    state = {"fail": True, "calls": 0}

    def _factory():
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("simulated boot error")
        return MagicMock(spec=LanceDBAdapter)

    adapter_mod.configure_ensure_warm_factory(_factory)

    with pytest.raises(RuntimeError, match="simulated boot error"):
        adapter_mod.ensure_warm()
    # Failure must NOT be cached.
    state["fail"] = False
    got = adapter_mod.ensure_warm()
    assert got is not None
    assert state["calls"] == 2, "second attempt must call factory again"


def test_ensure_warm_is_a_module_level_callable():
    """Other modules import `ensure_warm` directly from the adapter module."""
    assert hasattr(adapter_mod, "ensure_warm")
    assert callable(adapter_mod.ensure_warm)


def test_ensure_warm_default_factory_builds_adapter(monkeypatch):
    """Without an installed factory, ensure_warm falls back to the real
    LanceDBAdapter (via a no-arg constructor). We monkeypatch the
    constructor to avoid hitting a real LanceDB URI."""
    built = MagicMock(spec=LanceDBAdapter)
    monkeypatch.setattr(adapter_mod, "LanceDBAdapter", lambda: built)

    # No factory installed → fallback path.
    out = adapter_mod.ensure_warm()
    assert out is built, "default factory should use LanceDBAdapter()"
    out2 = adapter_mod.ensure_warm()
    assert out2 is out, "must return cached instance"