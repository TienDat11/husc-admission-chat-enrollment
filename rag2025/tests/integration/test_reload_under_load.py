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


def _build_app(
    admin_module,
    rt_module,
    *,
    query_delay_s: float = 0.05,
    query_gate: threading.Event | None = None,
):
    """Build a FastAPI app with the admin router + a mock /query endpoint.

    The /query endpoint sleeps to simulate real work and uses the
    RequestTracker via middleware to register itself as in-flight.

    ``query_gate``, when given, holds every /query inside the tracked section
    until the test sets the event — a request that is *really* in flight for as
    long as the test needs, which is what a drain timeout has to be proven
    against. The wait runs in a worker thread (``asyncio.to_thread``) so the
    held request occupies the tracker while the loop stays free for the admin
    request that must time out against it.
    """
    app = FastAPI()
    tracker = rt_module.RequestTracker()
    app.state.request_tracker = tracker

    class _Retriever:
        def __init__(self, name: str = "v1") -> None:
            self.name = name
            self.serve_count = 0
            self.closed = False

        async def healthcheck(self) -> bool:
            return True

        async def query(self, q: str) -> dict:
            if self.closed:
                # What closing a retriever under a reader costs for real:
                # LanceDBRetriever.close() nulls ``_adapter`` and the query path
                # dereferences it. Modelled, so "the old retriever is still
                # queryable" is an assertion and not a comment.
                raise RuntimeError(f"retriever {self.name} is closed")
            self.serve_count += 1
            await asyncio.sleep(query_delay_s)
            return {"answer": f"[{self.name}] {q}"}

        async def close(self) -> None:
            self.closed = True

    app.state.retriever = _Retriever("blue_v1")

    # The retriever the query path reads. Deliberately NOT ``app.state.retriever``:
    # the real query paths (/query, /v2, /v3) read module globals in main.py, and
    # reload-table swaps through main.py's seam. When this test's /query read
    # ``app.state.retriever`` instead, it asserted the one binding the route could
    # update by itself — which is how the P1 bug (route reports success, queries
    # keep the old retriever) survived here.
    runtime: dict[str, _Retriever] = {"retriever": app.state.retriever}
    # What a query reads, reachable from the test so "the old retriever is still
    # the one serving" is asserted against the binding that matters, not against
    # ``app.state.retriever``.
    app.state.runtime = runtime

    # Observed from inside the swap seam, which the route calls inside the gate:
    # the swap must run with admission already closed. If it ran with admission
    # open, a request admitted in that window would read the retriever step 5
    # closes.
    swap_observations: list[dict[str, object]] = []
    app.state.swap_observations = swap_observations

    def _swap(new_retriever):
        previous = runtime["retriever"]
        runtime["retriever"] = new_retriever
        app.state.retriever = new_retriever
        swap_observations.append({"admitting": tracker.admitting})
        return {
            "previous_retriever": previous,
            "hybrid_rebuilt": False,
            "pipeline_rebound": False,
        }

    app._swap_runtime_services = _swap

    built: list[_Retriever] = []
    app.state.built_retrievers = built

    async def builder(table_name: str):
        retriever = _Retriever(f"blue_{table_name}")
        built.append(retriever)
        return retriever

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
        retriever = runtime["retriever"]
        if query_gate is not None:
            # Held *inside* the tracked section, in a worker thread so the event
            # loop stays free: this request is genuinely in flight, which is the
            # only way to make a drain time out for real rather than by faking
            # ``drain()``'s return value.
            await asyncio.to_thread(query_gate.wait)
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
    reload_bodies: list[dict] = []
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
                return
            with served_lock:
                reload_bodies.append(r.json())
        except Exception as exc:  # pragma: no cover
            with served_lock:
                errors.append(f"reload: {exc!r}")

    # ``with`` matters: a bare ``TestClient(app)`` starts a *fresh* event loop per
    # request, so the tracker's events bind to whichever loop first waited on them
    # and the reload's ``resume()`` — on another loop — never wakes the blocked
    # queries. That harness is why this test could pass while the drain was in
    # fact timing out at the full 5s every run. One client, one loop, real drain.
    with TestClient(app) as client:
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

    # The drain must actually complete, not expire. ``drain_ms`` is the route's
    # own measurement of it, and the discriminator: with 20ms queries a real drain
    # finishes in ~10-20ms (measured), whereas under a per-request-event-loop
    # harness it burned the whole 5s bound and *still* reported drained=True — the
    # wakeup from a query's loop sat unprocessed in the reload's ready queue until
    # the timeout handle woke the selector. ``drained`` alone would not catch that,
    # which is why the bound is on the measured duration.
    assert len(reload_bodies) == 1, f"reload did not return 200: {errors!r}"
    body = reload_bodies[0]
    assert body["drained"] is True, body
    assert body["in_flight_at_swap"] == 0, body
    assert body["drain_ms"] < 1000, (
        f"drain took {body['drain_ms']}ms for queries that finish in 20ms — it "
        f"expired instead of draining"
    )
    # And the swap ran inside the gate: admission closed, so no request could be
    # admitted between the drain and the swap.
    assert app.state.swap_observations == [{"admitting": False}], app.state.swap_observations
    assert elapsed_ms < 5000, (
        f"Total elapsed {elapsed_ms}ms — the reload burned its 5s drain bound "
        f"instead of draining the 20ms queries"
    )


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

def _hold_one_query(client: TestClient, app, gate: threading.Event):
    """Start a /query that stays in flight until ``gate`` is set; return its future.

    The future is handed back rather than resolved so the caller can assert on
    the world *while the request is still in flight* — which is the only state a
    drain timeout exists in. Waiting for admission (rather than sleeping) keeps
    the test from racing the middleware.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(client.get, "/query?q=held")
    deadline = time.monotonic() + 10.0
    while app.state.request_tracker.in_flight < 1:
        if time.monotonic() > deadline:  # pragma: no cover — harness failure
            gate.set()
            raise AssertionError("held /query never became in-flight")
        time.sleep(0.005)
    return pool, future

def test_reload_refuses_on_drain_timeout_and_keeps_the_old_release_serving(
    admin_module, rt_module, monkeypatch
):
    """FIX-01: a drain that times out is a refusal — nothing swapped, nothing closed.

    The held request is really in flight (parked in a worker thread inside the
    tracker's section), so the drain really times out. What must hold on the way
    out, in order of what the old "proceeding with swap" branch broke:

    1. 409, not 200 — the operator is told the swap did not happen;
    2. the swap seam was never called, so every query path still reads the old
       release;
    3. the old retriever was not closed — the request still reading it answers
       normally once it is released, which is the AttributeError the tracker's
       docstring documents;
    4. admission was reopened, so the failed reload did not leave the process
       refusing queries: a following idle reload succeeds.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    gate = threading.Event()
    app = _build_app(admin_module, rt_module, query_gate=gate)
    old_retriever = app.state.runtime["retriever"]

    with TestClient(app) as client:
        pool, held = _hold_one_query(client, app, gate)
        try:
            r = client.post(
                "/admin/reload-table",
                json={"table_name": "husc_v2026_blue", "drain_timeout_s": 0.3},
                headers={"X-Admin-Token": "tok"},
            )

            # 1. Refused, and refused as a conflict with a retry hint — not a 200
            #    that happens to carry drained=False.
            assert r.status_code == 409, r.text
            assert r.headers.get("Retry-After") == "1", dict(r.headers)
            body = r.json()
            assert "drain timed out" in body["detail"], body
            assert "force_while_in_flight=true" in body["detail"], body
            assert "active_table" not in body, body

            # 2. No swap: the seam was never entered and the binding every query
            #    reads is untouched.
            assert app.state.swap_observations == [], app.state.swap_observations
            assert app.state.runtime["retriever"] is old_retriever
            assert app.state.retriever is old_retriever

            # 3. The old retriever was not closed under its reader...
            assert old_retriever.closed is False
            # ...and the candidate built for the swap, which never landed, was
            # closed rather than leaked.
            assert [c.closed for c in app.state.built_retrievers] == [True]

            # 4. Admission is open again while the drain is still outstanding:
            #    the refusal path must not leave the gate shut.
            assert app.state.request_tracker.admitting is True
            assert app.state.request_tracker.in_flight == 1
        finally:
            # Released before the assertions on the held response so a failure
            # here can never mask the assertion that actually failed.
            gate.set()
            pool.shutdown(wait=True)

        # The reader that was in flight the whole time answers, from the release
        # the refusal kept serving.
        held_response = held.result()
        assert held_response.status_code == 200, held_response.text
        assert held_response.json()["served_by"] == "blue_v1"
        assert old_retriever.serve_count == 1

        # And the process is still able to reload once the load has drained.
        retry = client.post(
            "/admin/reload-table",
            json={"table_name": "husc_v2026_blue", "drain_timeout_s": 5.0},
            headers={"X-Admin-Token": "tok"},
        )
        assert retry.status_code == 200, retry.text
        assert retry.json()["drained"] is True
        assert retry.json()["forced_while_in_flight"] is False

def test_forced_reload_swaps_past_the_timeout_and_leaves_the_old_retriever_open(
    admin_module, rt_module, monkeypatch
):
    """The emergency is a named flag, not a meaning read into the timeout.

    ``force_while_in_flight=true`` is the only way past a timed-out drain. It
    swaps (so the flag is not decorative), reports that it forced, and leaves the
    old retriever open — closing it is exactly what would kill the reader the
    caller chose to move under.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    gate = threading.Event()
    app = _build_app(admin_module, rt_module, query_gate=gate)
    old_retriever = app.state.runtime["retriever"]

    with TestClient(app) as client:
        pool, held = _hold_one_query(client, app, gate)
        try:
            r = client.post(
                "/admin/reload-table",
                json={
                    "table_name": "husc_v2026_blue",
                    "drain_timeout_s": 0.3,
                    "force_while_in_flight": True,
                },
                headers={"X-Admin-Token": "tok"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["drained"] is False, body
            assert body["forced_while_in_flight"] is True, body
            assert body["in_flight_at_swap"] == 1, body
            assert app.state.runtime["retriever"] is not old_retriever
            # A forced swap must NOT close the retriever its in-flight reader
            # holds — that is the failure the flag accepts, not one it creates.
            assert old_retriever.closed is False
        finally:
            gate.set()
            pool.shutdown(wait=True)

        # The reader survived the forced swap and was answered by the old release.
        held_response = held.result()
        assert held_response.status_code == 200, held_response.text
        assert held_response.json()["served_by"] == "blue_v1"

        # New requests read the release the forced swap installed.
        after = client.get("/query?q=after")
        assert after.status_code == 200, after.text
        assert after.json()["served_by"] == "blue_husc_v2026_blue"
