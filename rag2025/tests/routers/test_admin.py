"""Tests for admin router /admin/reload-table (TDD V5-R030 1:1 mapping).

@spec(S13.4)
"""
from __future__ import annotations
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


SRC = Path(__file__).resolve().parents[2] / "src"
ADMIN_PATH = SRC / "routers" / "admin.py"


@pytest.fixture
def admin_module():
    spec = importlib.util.spec_from_file_location("admin_router_test", ADMIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTracker:
    """Minimal stand-in for ``RequestTracker``: the route needs ``gate`` + ``in_flight``.

    ``gate`` has to be a real async context manager. The route wraps the swap in
    ``async with tracker.gate(...)``, so a MagicMock raises TypeError instead of
    exercising the route — and a fake that swallowed the timeout argument would
    hide the route passing its own default instead of the client's.
    """

    def __init__(self, *, drain_returns: bool = True) -> None:
        self._drain_returns = drain_returns
        self.in_flight = 0
        self.gate_timeouts: list[float] = []

    @asynccontextmanager
    async def gate(self, timeout: float):
        self.gate_timeouts.append(timeout)
        yield self._drain_returns


def _build_app(admin_module, *, drain_returns: bool = True, healthcheck_returns: bool = True, build_raises: bool = False):
    app = FastAPI()
    tracker = _FakeTracker(drain_returns=drain_returns)

    class _Retriever:
        async def healthcheck(self) -> bool:
            return healthcheck_returns

        async def close(self) -> None:
            return None

    async def _builder(table_name: str):
        if build_raises:
            raise RuntimeError("simulated builder failure")
        return _Retriever()

    app.state.request_tracker = tracker
    app.state.build_retriever = _builder
    app.state.retriever = _Retriever()

    # reload-table swaps through the app's runtime seam (main.py's
    # ``swap_runtime_services``, exposed as ``app._swap_runtime_services``) rather
    # than assigning ``app.state.retriever`` itself — assigning the attribute was
    # the P1 bug: every query path reads a module global, so the route reported a
    # successful reload while queries kept the old retriever. This app is mounted
    # by hand with no main.py behind it, so it provides the seam's contract
    # (rebind what a query reads, hand back what was swapped out). The real seam,
    # observed by a real query, is asserted in tests/routers/test_admin_rollback.py.
    def _swap(new_retriever):
        previous = app.state.retriever
        app.state.retriever = new_retriever
        return {
            "previous_retriever": previous,
            "hybrid_rebuilt": False,
            "pipeline_rebound": False,
        }

    app._swap_runtime_services = _swap
    app.include_router(admin_module.router)
    return app


def test_route_bounds_the_gate_by_the_payload_drain_timeout(admin_module, monkeypatch):
    """The client's ``drain_timeout_s`` bounds the gate; absent, the 25s default does."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module)
    client = TestClient(app)

    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue", "drain_timeout_s": 7.5},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 200, res.text
    assert app.state.request_tracker.gate_timeouts == [7.5]

    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 200, res.text
    assert app.state.request_tracker.gate_timeouts == [7.5, 25.0]


def test_post_without_auth_returns_403(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-tok")
    app = _build_app(admin_module)
    client = TestClient(app)
    res = client.post("/admin/reload-table", json={"table_name": "husc_v2026_blue"})
    # Header dependency missing → 422 (FastAPI) or 403; both acceptable as auth failures.
    assert res.status_code in {403, 422}


def test_post_with_wrong_token_returns_403(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-tok")
    app = _build_app(admin_module)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "wrong-tok"},
    )
    assert res.status_code == 403


def test_post_returns_403_when_token_env_unset(admin_module, monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    app = _build_app(admin_module)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "any"},
    )
    assert res.status_code == 403


def test_post_with_valid_auth_returns_200(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["active_table"] == "husc_v2026_blue"
    assert body["drained"] is True
    assert "drain_ms" in body
    assert "total_ms" in body


def test_post_returns_503_when_healthcheck_fails(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module, healthcheck_returns=False)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 503
    assert "healthcheck" in res.json()["detail"].lower()


def test_post_returns_503_when_builder_raises(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module, build_raises=True)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 503
    assert "Builder error" in res.json()["detail"]


def test_drain_timeout_refuses_the_swap(admin_module, monkeypatch):
    """A drain that times out is a 409, and nothing is swapped.

    This test used to assert the opposite — ``200`` with ``drained=False``, the
    route's "proceeding with swap" branch. That branch was the defect FIX-01
    fixes: it closed a retriever a request was still reading, which the tracker's
    own docstring records as an AttributeError for that reader. The behaviour is
    now asserted end-to-end against a genuinely in-flight request in
    ``tests/integration/test_reload_under_load.py``; here the fake tracker is
    enough to pin the contract, because the route's decision is a pure function
    of ``drained``.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module, drain_returns=False)  # drain returned False (timed out)
    old_retriever = app.state.retriever
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 409, res.text
    body = res.json()
    assert "drain timed out" in body["detail"], body
    # The swap seam never ran, so the retriever a query reads is the old one.
    assert app.state.retriever is old_retriever
    assert res.headers.get("Retry-After") == "1", dict(res.headers)

def test_forced_reload_proceeds_on_a_drain_timeout(admin_module, monkeypatch):
    """``force_while_in_flight`` is the only way past a timed-out drain."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module, drain_returns=False)
    old_retriever = app.state.retriever
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue", "force_while_in_flight": True},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["drained"] is False
    assert body["forced_while_in_flight"] is True
    assert body["active_table"] == "husc_v2026_blue"
    assert app.state.retriever is not old_retriever


def test_drain_timeout_clamped_at_30s(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "x", "drain_timeout_s": 60.0},
        headers={"X-Admin-Token": "tok"},
    )
    # Pydantic validator should reject > 30s.
    assert res.status_code == 422


def test_swap_replaces_app_state_retriever(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module)
    old_retriever = app.state.retriever
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 200
    # Reference must have changed.
    assert app.state.retriever is not old_retriever


def test_post_503_when_app_state_missing_dependencies(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = FastAPI()
    app.include_router(admin_module.router)
    # No request_tracker / build_retriever set → 503.
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "x"},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 503
    assert "request_tracker" in res.json()["detail"] or "build_retriever" in res.json()["detail"]
