"""db_engine_lifespan: one shared engine on app.state.db_engine, or None without DATABASE_URL."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.db import db_engine_lifespan  # noqa: E402
from src.routers.admin_jobs import router as jobs_router  # noqa: E402

OPS_TOKEN = "shared-ops-token"
EDITOR_TOKEN = "tok-editor"

DATABASE_URL = os.environ.get("DATABASE_URL", "")
needs_pg = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set (needs migrated PostgreSQL)")


def _configure(monkeypatch) -> None:
    entries = [
        {"token": EDITOR_TOKEN, "actor_id": "cb.editor", "role": "editor"},
    ]
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)
    monkeypatch.delenv("ADMIN_FOUR_EYES", raising=False)


def _app() -> FastAPI:
    application = FastAPI(lifespan=db_engine_lifespan)
    application.include_router(jobs_router)
    return application


def _headers(token: str) -> dict[str, str]:
    return {"X-Admin-Token": token}


def test_no_database_url_sets_engine_none_and_jobs_answer_503(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app = _app()
    with TestClient(app) as client:
        assert app.state.db_engine is None
        response = client.get("/admin/jobs", headers=_headers(EDITOR_TOKEN))
        assert response.status_code == 503


@needs_pg
def test_database_url_sets_shared_engine_and_jobs_answer_200(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    app = _app()
    with TestClient(app) as client:
        assert app.state.db_engine is not None
        response = client.get("/admin/jobs", headers=_headers(EDITOR_TOKEN))
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@needs_pg
def test_engine_is_cleared_after_client_context_exits(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    app = _app()
    with TestClient(app):
        assert app.state.db_engine is not None
    assert app.state.db_engine is None
