"""Integration: 10 concurrent /query during /admin/reload-table — no errors, < 30s.

@spec(S13.4) verifies M1+F1+F4 atomic swap under load.
"""
from __future__ import annotations
import asyncio
import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


SRC = Path(__file__).resolve().parents[2] / "src"
ADMIN_PATH = SRC / "routers" / "admin.py"
RT_PATH = SRC / "app" / "request_tracker.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def admin_module():
    return _load_module("admin_router_loadtest", ADMIN_PATH)


@pytest.fixture
def rt_module():
    return _load_module("request_tracker_loadtest", RT_PATH)


def _build_app(admin_module, rt_module, *, query_delay_s: float = 0.05):
    """Build a FastAPI app with the admin router + a mock /query endpoint.

    The /query endpoint sleeps to simulate real work and uses the
    RequestTracker via middleware to register itself as in-flight.
    """
    app = FastAPI()
    tracker = rt_module.RequestTracker()
    app.state.request_tracker = tracker

    class _Retriever:
        def __init__(self, name: str = "v1") -> None:
            self.name = name
            self.serve_count = 0

        async def healthcheck(self) -> bool:
            return True

        async def query(self, q: str) -> dict:
            self.serve_count += 1
            await asyncio.sleep(query_delay_s)
            return {"answer": f"[{self.name}] {q}"}

        async def close(self) -> None:
            return None

    app.state.retriever = _Retriever("blue_v1")

    async def builder(table_name: str):
        return _Retriever(f"blue_{table_name}")

    app.state.build_retriever = builder

    @app.middleware("http")
    async def track_request(request: Request, call_next):
        # Skip admin endpoints to avoid self-deadlock during drain.
        if request.url.path.startswith("/admin/"):
            return await call_next(request)
        async with tracker.track():
            return await call_next(request)

    @app.get("/query")
    async def query(q: str = "ping"):
        retriever = app.state.retriever
        result = await retriever.query(q)
        return {"served_by": retriever.name, **result}

    app.include_router(admin_module.router)
    return app


def test_reload_under_concurrent_query_load(admin_module, rt_module, monkeypatch):
    """Fire 10 concurrent /query while /admin/reload-table runs. Expect:
       - 0 errors
       - All queries served by either blue_v1 (old) or blue_v2 (new), no other names
       - Total wall-clock < 30000ms
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")

    app = _build_app(admin_module, rt_module, query_delay_s=0.02)
    client = TestClient(app)

    NUM_QUERIES = 10
    started = time.monotonic()
    errors: list[str] = []
    served_by: list[str] = []
    served_lock = threading.Lock()

    def fire_query(i: int):
        try:
            r = client.get(f"/query?q=ping_{i}")
            if r.status_code != 200:
                with served_lock:
                    errors.append(f"q{i}: status={r.status_code} body={r.text}")
                return
            with served_lock:
                served_by.append(r.json()["served_by"])
        except Exception as exc:  # pragma: no cover — exercised on real failure
            with served_lock:
                errors.append(f"q{i}: {exc!r}")

    def fire_reload():
        try:
            r = client.post(
                "/admin/reload-table",
                json={"table_name": "husc_v2026_blue", "drain_timeout_s": 5.0},
                headers={"X-Admin-Token": "tok"},
            )
            if r.status_code != 200:
                with served_lock:
                    errors.append(f"reload: status={r.status_code} body={r.text}")
        except Exception as exc:  # pragma: no cover
            with served_lock:
                errors.append(f"reload: {exc!r}")

    with ThreadPoolExecutor(max_workers=NUM_QUERIES + 1) as pool:
        futures = [pool.submit(fire_query, i) for i in range(NUM_QUERIES)]
        # Schedule reload partway through the burst.
        time.sleep(0.005)
        futures.append(pool.submit(fire_reload))
        for fut in as_completed(futures):
            fut.result()

    elapsed_ms = (time.monotonic() - started) * 1000.0

    assert not errors, f"Errors during reload-under-load: {errors!r}"
    assert len(served_by) == NUM_QUERIES, f"Expected {NUM_QUERIES} successful queries, got {len(served_by)}"
    # Each query was served by either the old or the new retriever — never a torn state.
    valid_names = {"blue_v1", "blue_husc_v2026_blue"}
    assert all(name in valid_names for name in served_by), f"Unexpected served_by: {set(served_by) - valid_names}"
    assert elapsed_ms < 30000, f"Total elapsed {elapsed_ms}ms exceeds 30s SLA"


def test_reload_succeeds_with_zero_inflight(admin_module, rt_module, monkeypatch):
    """Sanity baseline: with no in-flight queries, reload completes fast."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module, rt_module)
    client = TestClient(app)

    started = time.monotonic()
    r = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue", "drain_timeout_s": 5.0},
        headers={"X-Admin-Token": "tok"},
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["drained"] is True
    assert body["in_flight_at_swap"] == 0
    assert elapsed_ms < 5000, f"Idle reload took {elapsed_ms}ms (expected <5s)"
