"""Schema-revision write freeze (plan U2): a DB at an old revision freezes admin writes.

Without DATABASE_URL only the in-process freeze tests run; the revision test needs
the live database and skips with a reason otherwise.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.db import SCHEMA_REVISION_REASON, WRITE_FREEZE, assert_schema_head, head_revision

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"


@pytest.fixture(autouse=True)
def _reset_freeze():
    WRITE_FREEZE.release()
    yield
    WRITE_FREEZE.release()


def _load_main(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location("schema_guard_u2_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engaged_freeze_returns_503_writes_frozen_for_admin_writes(monkeypatch) -> None:
    main = _load_main(monkeypatch)
    WRITE_FREEZE.engage(SCHEMA_REVISION_REASON, {"expected": "x", "current": "y"})
    client = TestClient(main.app)

    res = client.post("/admin/records", json={})
    assert res.status_code == 503, res.text
    body = res.json()
    assert body["code"] == "WRITES_FROZEN"
    assert body["detail"]["reason"] == "schema_revision"
    assert res.headers.get("Retry-After")

    # Public chat and health stay up; login is not a record write.
    assert client.get("/health").status_code == 200
    login = client.post("/admin/login", json={})
    assert login.status_code != 503 or login.json().get("code") != "WRITES_FROZEN"


def test_health_ready_is_503_while_frozen(monkeypatch) -> None:
    main = _load_main(monkeypatch)
    WRITE_FREEZE.engage(SCHEMA_REVISION_REASON, {})
    res = TestClient(main.app).get("/health/ready")
    assert res.status_code == 503, res.text
    assert res.json()["reason"] == SCHEMA_REVISION_REASON


def test_db_at_old_revision_engages_freeze_and_restores_head(database_url: str, migrated_database_url: str) -> None:
    head = head_revision()

    async def set_version(value: str) -> None:
        engine = create_async_engine(database_url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": value})
        finally:
            await engine.dispose()

    async def check_mismatch() -> bool:
        engine = create_async_engine(database_url)
        try:
            return await assert_schema_head(engine)
        finally:
            await engine.dispose()

    asyncio.run(set_version("0000_legacy"))
    try:
        assert asyncio.run(check_mismatch()) is False
        assert WRITE_FREEZE.active and WRITE_FREEZE.reason == SCHEMA_REVISION_REASON
        assert WRITE_FREEZE.detail == {"expected": head, "current": "0000_legacy"}
    finally:
        asyncio.run(set_version(head))

    assert asyncio.run(check_mismatch()) is True
    assert not WRITE_FREEZE.active
