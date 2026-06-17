"""TDD: admin warmup must route RAMCLOUDS_API_KEY through settings.

BUG-7 LOW: admin.py:64 must not read os.environ directly for the provider
key; it should use config.settings.settings.RAMCLOUDS_API_KEY.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


def _build_app(admin_module):
    app = FastAPI()
    tracker = MagicMock()
    tracker.drain = AsyncMock(return_value=True)
    tracker.in_flight = 0

    class _Retriever:
        async def healthcheck(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    async def _builder(table_name: str):
        return _Retriever()

    app.state.request_tracker = tracker
    app.state.build_retriever = _builder
    app.state.retriever = _Retriever()
    app.include_router(admin_module.router)
    return app


def test_admin_warmup_routes_key_through_settings(admin_module, monkeypatch):
    """Warmup must decide LLM ping based on settings.RAMCLOUDS_API_KEY, not
    a direct os.environ read.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    # Make sure the key value comes from settings, not raw env.
    monkeypatch.setenv("RAMCLOUDS_API_KEY", "sk-from-env")

    from config.settings import settings as live_settings

    with patch.object(live_settings, "RAMCLOUDS_API_KEY", "sk-from-settings"):
        app = _build_app(admin_module)
        client = TestClient(app)

        # Patch AppServices inside the warmup function's import namespace.
        import app.lifecycle as lifecycle_mod
        with patch.object(lifecycle_mod, "AppServices") as mock_svc:
            embedding = MagicMock()
            embedding.encode_query = MagicMock()
            llm = MagicMock()
            llm.chat = AsyncMock(return_value=MagicMock())
            svc = MagicMock()
            svc.get_embedding.return_value = embedding
            svc.get_llm_client.return_value = llm
            mock_svc.return_value = svc

            res = client.post("/admin/warmup", headers={"X-Admin-Token": "tok"})

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "warmed"
    # LLM was pinged because settings.RAMCLOUDS_API_KEY was set.
    llm.chat.assert_awaited_once()
