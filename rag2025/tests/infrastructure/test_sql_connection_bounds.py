"""Every SQL connection the stores open is bounded.

Client side: connect timeout and TCP keepalive, passed as libpq keyword options.
Server side: ``SET LOCAL statement_timeout`` as the first statement of the
transaction, so the caller's queries run under it. No ``options`` startup
parameter is sent, since a transaction-mode pooler may refuse it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import (  # noqa: E402
    sql_audit_ledger,
    sql_identity_repo,
    sql_record_store,
    sql_source_document_registry,
)


class _Recorder:
    def __init__(self) -> None:
        self.connects: list[dict] = []
        self.statements: list[str] = []


class _FakeConnection:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self._recorder.statements.append("<commit>")
        return False

    def execute(self, sql, *args, **kwargs):
        self._recorder.statements.append(sql)
        return self

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "store_cls, method",
    [
        (sql_record_store.SqlRecordStore, "_connect"),
        (sql_audit_ledger.SqlAuditLedger, "_connect"),
        (sql_identity_repo.SqlIdentityRepository, "_txn"),
        (sql_source_document_registry.SqlSourceDocumentRegistry, "_connect"),
    ],
)
def test_every_sql_connection_is_bounded_without_startup_options(
    store_cls, method, monkeypatch
):
    recorder = _Recorder()

    def fake_connect(conninfo, **kwargs):
        recorder.connects.append(kwargs)
        return _FakeConnection(recorder)

    monkeypatch.setattr(sql_record_store.psycopg, "connect", fake_connect)

    store = object.__new__(store_cls)
    store._conninfo = "postgresql://example.invalid/db"
    with getattr(store, method)() as conn:
        conn.execute("SELECT 1")

    kwargs = recorder.connects[0]
    assert kwargs["connect_timeout"] == 10
    assert kwargs["keepalives"] == 1
    assert kwargs["keepalives_idle"] == 30
    assert kwargs["keepalives_interval"] == 10
    assert kwargs["keepalives_count"] == 3
    assert "options" not in kwargs
    assert recorder.statements[0] == (
        f"SET LOCAL statement_timeout = '{sql_record_store.STATEMENT_TIMEOUT_MS}ms'"
    )
    assert recorder.statements[1] == "SELECT 1"
