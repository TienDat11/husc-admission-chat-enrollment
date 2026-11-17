"""Startup builds the record store named by STORE_BACKEND (A3 cutover, U3).

``file`` (the default) keeps the file-backed ``CanonicalRecordStore``.
``postgres`` builds ``SqlRecordStore`` from DATABASE_URL and hands that same
object to every consumer configured in the same function. A missing
DATABASE_URL under ``postgres`` fails startup and never echoes a URL.

Only the last test touches PostgreSQL; it uses the throwaway-database
``fresh_db`` fixture and skips without DATABASE_URL.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.canonical import CanonicalRecord  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.infrastructure.identity_store import IdentityStore  # noqa: E402
from src.infrastructure.sql_audit_ledger import SqlAuditLedger  # noqa: E402
from src.infrastructure.sql_record_store import SqlRecordStore  # noqa: E402
from tests.pg_fixtures import fresh_db  # noqa: F401  (fixture)

SECRET_URL = "postgresql+asyncpg://u:secret-pw@127.0.0.1:1/x"


def _stop_runner(app: FastAPI) -> None:
    runner = getattr(app.state, "reindex_runner", None)
    if runner is not None:
        runner.stop()


def _configure(tmp_path: Path, backend: str) -> FastAPI:
    app = FastAPI()
    settings = RAGSettings(
        CANONICAL_STORE_ROOT=tmp_path / "canonical_store",
        STORE_BACKEND=backend,
    )
    configure_canonical_record_store(app, settings=settings)
    return app


def test_file_backend_is_default_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app = _configure(tmp_path, "file")
    try:
        assert isinstance(app.state.canonical_record_store, CanonicalRecordStore)
        assert not isinstance(app.state.canonical_record_store, SqlRecordStore)
    finally:
        _stop_runner(app)


def test_postgres_backend_builds_sql_store_shared_with_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", SECRET_URL)
    calls: list = []

    def _no_connect(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("startup must not open a database connection")

    monkeypatch.setattr(psycopg, "connect", _no_connect)

    app = _configure(tmp_path, "postgres")
    try:
        store = app.state.canonical_record_store
        assert isinstance(store, SqlRecordStore)
        assert app.state.admin_lifecycle_service._store is store
        assert calls == []
    finally:
        _stop_runner(app)


def test_postgres_backend_without_database_url_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app = FastAPI()
    settings = RAGSettings(
        CANONICAL_STORE_ROOT=tmp_path / "canonical_store",
        STORE_BACKEND="postgres",
    )
    with pytest.raises(RuntimeError) as excinfo:
        configure_canonical_record_store(app, settings=settings)

    message = str(excinfo.value)
    assert "STORE_BACKEND" in message
    assert "DATABASE_URL" in message
    assert "secret-pw" not in message
    assert not hasattr(app.state, "canonical_record_store")


def test_postgres_backend_without_database_url_never_echoes_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "   ")
    app = FastAPI()
    settings = RAGSettings(
        CANONICAL_STORE_ROOT=tmp_path / "canonical_store",
        STORE_BACKEND="postgres",
    )
    with pytest.raises(RuntimeError) as excinfo:
        configure_canonical_record_store(app, settings=settings)
    assert "DATABASE_URL" in str(excinfo.value)
    assert "secret-pw" not in str(excinfo.value)


def test_postgres_backend_writes_and_reads_through_app_state(tmp_path, monkeypatch, fresh_db):  # noqa: F811
    monkeypatch.setenv("DATABASE_URL", fresh_db)

    app = _configure(tmp_path, "postgres")
    try:
        store = app.state.canonical_record_store
        assert isinstance(store, SqlRecordStore)
        record_id = "HOC_PHI:NGANH:cntt:2026:all:all"
        draft = CanonicalRecord(
            record_id=record_id,
            version=1,
            record_type="HOC_PHI",
            admission_year=2026,
            effective_from=date(2026, 1, 1),
            status="draft",
            authority_level="husc_canonical",
            text="wiring probe",
            authored_by="alice",
            admission_round="all",
        )
        store.write_record(draft)
        chain = store.get_record(record_id)
        assert chain.latest.text == "wiring probe"
    finally:
        _stop_runner(app)


def test_file_backend_keeps_the_file_audit_ledger(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app = _configure(tmp_path, "file")
    try:
        assert isinstance(app.state.audit_ledger, AuditLedger)
        assert not isinstance(app.state.audit_ledger, SqlAuditLedger)
        assert app.state.audit_ledger._path == tmp_path / "canonical_store" / "audit" / "ledger.jsonl"
        assert not hasattr(app.state, "identity_store")
    finally:
        _stop_runner(app)


def test_postgres_backend_builds_sql_audit_ledger_without_connecting(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", SECRET_URL)

    def _no_connect(*args, **kwargs):
        raise AssertionError("startup must not open a database connection")

    monkeypatch.setattr(psycopg, "connect", _no_connect)
    app = _configure(tmp_path, "postgres")
    try:
        assert isinstance(app.state.audit_ledger, SqlAuditLedger)
        assert app.state.admin_lifecycle_service._ledger is app.state.audit_ledger
        assert not (tmp_path / "canonical_store" / "audit" / "ledger.jsonl").exists()
    finally:
        _stop_runner(app)


def test_postgres_backend_attaches_file_identity_store_under_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", SECRET_URL)
    app = _configure(tmp_path, "postgres")
    try:
        identity = app.state.identity_store
        assert isinstance(identity, IdentityStore)
        assert identity.path.is_relative_to(tmp_path / "canonical_store")
    finally:
        _stop_runner(app)


def test_postgres_backend_without_database_url_names_the_setting_for_the_ledger_too(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app = FastAPI()
    settings = RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "canonical_store", STORE_BACKEND="postgres")
    with pytest.raises(RuntimeError) as excinfo:
        configure_canonical_record_store(app, settings=settings)
    assert str(excinfo.value) == "STORE_BACKEND=postgres requires DATABASE_URL to be set"
    assert not hasattr(app.state, "audit_ledger")
