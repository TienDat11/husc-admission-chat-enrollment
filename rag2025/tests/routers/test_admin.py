"""Tests for admin router /admin/reload-table (TDD V5-R030 1:1 mapping).

@spec(S13.4)
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def _build_app(admin_module, *, drain_returns: bool = True, healthcheck_returns: bool = True, build_raises: bool = False):
    app = FastAPI()
    tracker = MagicMock()
    tracker.drain = AsyncMock(return_value=drain_returns)
    tracker.in_flight = 0

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
    app.include_router(admin_module.router)
    return app


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


def test_drain_timeout_proceeds_with_swap(admin_module, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    app = _build_app(admin_module, drain_returns=False)  # drain returned False (timed out)
    client = TestClient(app)
    res = client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_blue"},
        headers={"X-Admin-Token": "tok"},
    )
    # Even though drain timed out, swap proceeds with status 200 and drained=False.
    assert res.status_code == 200
    body = res.json()
    assert body["drained"] is False
    assert body["active_table"] == "husc_v2026_blue"


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
