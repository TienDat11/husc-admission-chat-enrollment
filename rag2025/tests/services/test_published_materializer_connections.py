"""Materializing a SQL store opens a bounded number of connections, not one per record.

The store is the real ``SqlRecordStore``; only ``_connect`` is replaced, by a spy that
hands out a fake connection answering the SQL the store issues. Each ``_connect()`` call
is one database connection in production, so counting them counts connections.
"""

from __future__ import annotations

import contextlib
from datetime import date, datetime, timezone

import pytest
from psycopg.types.json import Jsonb

from src.domain.canonical import CanonicalRecord
from src.infrastructure import sql_record_store as sql
from src.services.published_materializer import PublishedMaterializer

DAY = date(2026, 1, 1)
PUB = datetime(2026, 2, 1, tzinfo=timezone.utc)
DOC = "doc-1"


def _record(record_id: str, version: int = 1, **overrides: object) -> CanonicalRecord:
    fields: dict = dict(
        record_id=record_id,
        version=version,
        record_type="DIEM_CHUAN",
        admission_year=2026,
        effective_from=DAY,
        status="published",
        authority_level="husc_canonical",
        text=f"text of {record_id}@{version}",
        authored_by="alice",
        admission_round="all",
        source_document_id=DOC,
        published_at=PUB,
    )
    fields.update(overrides)
    return CanonicalRecord(**fields)


def _row(record: CanonicalRecord) -> dict:
    """The row dict psycopg would return for ``record``'s stored form."""
    return {
        key: (value.obj if isinstance(value, Jsonb) else value)
        for key, value in sql._to_params(record).items()
    }


class _Cursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConnection:
    """Answers the statements SqlRecordStore's read paths issue, from ``rows``."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def execute(self, sql_text: str, params=None):
        text = " ".join(sql_text.split())
        if "GROUP BY record_id" in text:
            ids = sorted({r["record_id"] for r in self._rows})
            return _Cursor({"record_id": i} for i in ids)
        if text.startswith("SELECT version FROM canonical_records WHERE record_id"):
            found = sorted((r for r in self._rows if r["record_id"] == params[0]),
                           key=lambda r: r["version"])
            return _Cursor({"version": r["version"]} for r in found)
        if "WHERE record_id = %s AND version = %s" in text:
            return _Cursor(r for r in self._rows
                           if r["record_id"] == params[0] and r["version"] == params[1])
        if text.endswith('ORDER BY record_id COLLATE "C", version'):
            return _Cursor(sorted(self._rows, key=lambda r: (r["record_id"], r["version"])))
        if "WHERE status = 'published'" in text:
            return _Cursor(r for r in self._rows if r["status"] == "published")
        if "supersession_retractions" in text:
            return _Cursor([])
        raise AssertionError(f"unexpected SQL in materializer test: {text}")


def _sql_store(records: list[CanonicalRecord], monkeypatch) -> tuple[sql.SqlRecordStore, list[int]]:
    """A real SqlRecordStore whose ``_connect`` counts each connection it opens."""
    store = object.__new__(sql.SqlRecordStore)
    store._conninfo = "postgresql://unused.invalid/db"
    rows = [_row(r) for r in records]
    opened: list[int] = []

    @contextlib.contextmanager
    def connect():
        opened.append(1)
        yield _FakeConnection(rows)

    monkeypatch.setattr(store, "_connect", connect)
    return store, opened


class _PerRecordOnly:
    """The store surface a file store offers: no bulk read, only per-record calls."""

    def __init__(self, store: sql.SqlRecordStore) -> None:
        self._store = store

    def supersession_index(self):
        return self._store.supersession_index()

    def record_ids(self):
        return self._store.record_ids()

    def versions(self, record_id):
        return self._store.versions(record_id)

    def get_version(self, record_id, version):
        return self._store.get_version(record_id, version)


def _records(n: int) -> list[CanonicalRecord]:
    records = [_record(f"R{i:04d}") for i in range(n)]
    # One record with an in-flight edit on top of its published version, so the
    # live-version walk is exercised, not only the single-version path.
    records.append(_record("EDITED", 1))
    records.append(_record("EDITED", 2, status="draft", source_document_id=None,
                           published_at=None))
    return records


@pytest.mark.parametrize("n", [3, 200])
def test_materialize_opens_at_most_two_connections_whatever_the_record_count(
    n, tmp_path, monkeypatch
):
    store, opened = _sql_store(_records(n), monkeypatch)

    result = PublishedMaterializer(store).materialize("rel-connections", tmp_path)

    assert len(result.admitted) == n + 1
    assert len(opened) <= 2


def test_the_spy_counts_the_per_record_path_so_the_bound_is_real(tmp_path, monkeypatch):
    store, opened = _sql_store(_records(20), monkeypatch)

    PublishedMaterializer(_PerRecordOnly(store)).materialize("rel-connections", tmp_path)

    assert len(opened) > 2


def test_bulk_and_per_record_paths_write_the_same_release(tmp_path, monkeypatch):
    store, _ = _sql_store(_records(12), monkeypatch)

    bulk = PublishedMaterializer(store).materialize("rel-same", tmp_path / "bulk")
    per_record = PublishedMaterializer(_PerRecordOnly(store)).materialize(
        "rel-same", tmp_path / "per-record"
    )

    assert bulk.records_path.read_bytes() == per_record.records_path.read_bytes()
    assert bulk.manifest == per_record.manifest
    assert [r.record_id for r in bulk.excluded] == [r.record_id for r in per_record.excluded]
