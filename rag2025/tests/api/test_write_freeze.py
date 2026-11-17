"""Operator write freeze (A3): WRITE_FREEZE setting, startup engagement, and guard behaviour.

The freeze guard lives in ``api.errors`` and answers admin writes with 503
WRITES_FROZEN while ``infrastructure.db.WRITE_FREEZE`` is active. These tests pin
that the operator reason survives the schema guard and that the STORE_BACKEND
and WRITE_FREEZE settings validate as declared.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
MAIN_PATH = SRC / "main.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import infrastructure.db as db  # noqa: E402
from api.errors import install_error_handling  # noqa: E402
from config.settings import RAGSettings  # noqa: E402

OPERATOR = "operator"
TOKEN = "tok-admin"
RECORD_PATH = "/admin/records/DIEM_CHUAN:NGANH:cntt:2026/edits"


@pytest.fixture(autouse=True)
def _reset_freeze():
    db.WRITE_FREEZE.release()
    yield
    db.WRITE_FREEZE.release()


def _app(tmp_path: Path, monkeypatch) -> FastAPI:
    from rag2025.src.app.lifecycle import configure_canonical_record_store
    from src.routers.admin import router as admin_router

    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps([{"token": TOKEN, "actor_id": "cb.quan-tri", "role": "admin"}]),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")
    application = FastAPI()
    install_error_handling(application)
    application.include_router(admin_router)

    @application.post("/v2/chat")
    def chat():
        return {"answer": "ok"}

    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "canonical_store")
    )
    return application


def _load_main(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location("write_freeze_a3_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_admin_write_answers_503_writes_frozen_with_operator_reason(tmp_path, monkeypatch) -> None:
    client = TestClient(_app(tmp_path, monkeypatch))
    db.WRITE_FREEZE.engage(OPERATOR, {})

    res = client.post(RECORD_PATH, json={"expected_version": 1, "text": "x"}, headers={"X-Admin-Token": TOKEN})

    assert res.status_code == 503, res.text
    body = res.json()
    assert body["code"] == "WRITES_FROZEN"
    assert body["detail"]["reason"] == OPERATOR


def test_frozen_state_does_not_intercept_public_chat(tmp_path, monkeypatch) -> None:
    client = TestClient(_app(tmp_path, monkeypatch))
    db.WRITE_FREEZE.engage(OPERATOR, {})

    res = client.post("/v2/chat", json={"query": "hi"})

    assert res.status_code == 200, res.text


def test_unfrozen_admin_write_is_not_answered_by_freeze_guard(tmp_path, monkeypatch) -> None:
    client = TestClient(_app(tmp_path, monkeypatch))
    assert not db.WRITE_FREEZE.active

    res = client.post(RECORD_PATH, json={"expected_version": 1, "text": "x"}, headers={"X-Admin-Token": TOKEN})

    assert res.status_code != 503, res.text
    body = res.json()
    assert body.get("code") != "WRITES_FROZEN"


def test_store_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        RAGSettings(STORE_BACKEND="mysql")


@pytest.mark.parametrize("value", ["file", "postgres"])
def test_store_backend_accepts_declared_values(value: str) -> None:
    assert RAGSettings(STORE_BACKEND=value).STORE_BACKEND == value


def test_store_backend_and_write_freeze_defaults() -> None:
    settings = RAGSettings()
    assert settings.STORE_BACKEND == "file"
    assert settings.WRITE_FREEZE is False


def test_schema_guard_does_not_release_operator_freeze(monkeypatch) -> None:
    head = "rev_head_a3"

    async def fake_current_revision(engine):
        return head

    monkeypatch.setattr(db, "current_revision", fake_current_revision)
    db.WRITE_FREEZE.engage(OPERATOR, {})

    assert asyncio.run(db.assert_schema_head(object(), expected_head=head)) is True
    assert db.WRITE_FREEZE.active
    assert db.WRITE_FREEZE.reason == OPERATOR


def test_startup_engages_operator_freeze_when_write_freeze_set(monkeypatch) -> None:
    monkeypatch.setenv("WRITE_FREEZE", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    main = _load_main(monkeypatch)

    asyncio.run(main._check_schema_head_at_startup())

    assert db.WRITE_FREEZE.active
    assert db.WRITE_FREEZE.reason == OPERATOR


def test_startup_leaves_freeze_clear_when_write_freeze_unset(monkeypatch) -> None:
    monkeypatch.delenv("WRITE_FREEZE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    main = _load_main(monkeypatch)

    asyncio.run(main._check_schema_head_at_startup())

    assert not db.WRITE_FREEZE.active
