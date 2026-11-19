"""/health/ready uses the shared app.state.db_engine with a 2s bound on the release lookup.

Cases (a)-(c) use a bare FastAPI app with the health router and a fake engine, and
monkeypatch ``infrastructure.db.active_release_id`` because ``health_ready`` imports
it inside the function body (each call re-reads the module attribute).

Case (d) needs the live database and skips with a reason when DATABASE_URL is unset.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrastructure import db as db_module  # noqa: E402
from infrastructure.db import WRITE_FREEZE, db_engine_lifespan  # noqa: E402
from routers import health  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_freeze():
    WRITE_FREEZE.release()
    yield
    WRITE_FREEZE.release()


def _bare_app(engine) -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    app.state.db_engine = engine
    return app


def test_engine_none_returns_503(monkeypatch) -> None:
    async def _should_not_run(engine):  # pragma: no cover - must not be reached
        raise AssertionError("release lookup must not run without an engine")

    monkeypatch.setattr(db_module, "active_release_id", _should_not_run)
    client = TestClient(_bare_app(None))

    res = client.get("/health/ready")
    assert res.status_code == 503, res.text
    body = res.json()
    assert body["ready"] is False
    assert body["served_release_id"] is None
    assert set(body) == {"ready", "reason", "served_release_id"}


def test_lookup_exception_returns_503(monkeypatch) -> None:
    async def _boom(engine):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(db_module, "active_release_id", _boom)
    client = TestClient(_bare_app(object()))

    res = client.get("/health/ready")
    assert res.status_code == 503, res.text
    body = res.json()
    assert body["ready"] is False
    assert body["served_release_id"] is None
    assert body["reason"] == "release_lookup_failed"


def test_lookup_timeout_returns_503_quickly(monkeypatch) -> None:
    async def _slow(engine):
        await asyncio.sleep(5)
        return "rel-late"

    monkeypatch.setattr(db_module, "active_release_id", _slow)
    client = TestClient(_bare_app(object()))

    started = time.monotonic()
    res = client.get("/health/ready")
    elapsed = time.monotonic() - started

    assert res.status_code == 503, res.text
    assert elapsed < 3.0, f"timeout not enforced: took {elapsed:.2f}s"
    body = res.json()
    assert body["ready"] is False
    assert body["served_release_id"] is None
    assert body["reason"] == "release_lookup_failed"


def test_live_engine_ready_200_with_served_release_id(migrated_database_url, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", migrated_database_url)
    app = FastAPI(lifespan=db_engine_lifespan)
    app.include_router(health.router)

    with TestClient(app) as client:
        assert app.state.db_engine is not None
        res = client.get("/health/ready")

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ready"] is True
    assert "served_release_id" in body
    assert body["reason"] is None
