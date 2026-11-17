"""SqlRecordStore parity and behaviour (A3, U3).

Pure checks run everywhere. PostgreSQL checks use the throwaway-database
``fresh_db`` fixture from tests/scripts/test_migrate_apply.py and skip without
DATABASE_URL. The parity test runs the real migrator over a file-store fixture
and compares every read answer with the file store, after the documented id
remap.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
import pytest

from scripts import migrate_file_stores as migrate
from src.domain.canonical import CanonicalRecord, RecordStatus
from src.infrastructure.canonical_record_store import (
    CanonicalRecordStore,
    RecordNotFoundError,
    RecordStoreError,
    RecordVersionConflictError,
)
import src.infrastructure.sql_record_store as sql_module
from src.infrastructure.sql_record_store import SqlRecordStore, _conninfo
from tests.scripts.test_migrate_apply import _apply, _write_json, fresh_db  # noqa: F401  (fixture)

DOC = "doc-a"
DAY = date(2026, 1, 1)
PUB = datetime(2026, 2, 1, 9, 30, tzinfo=timezone.utc)
H = hashlib.sha256(b"all").hexdigest()[:12]  # the migrator's methods_hash for a bare "all" method
A = f"DIEM_CHUAN:NGANH:cntt:2026:all:{H}"
B = f"DIEM_CHUAN:NGANH:cntt:2027:all:{H}"
D = f"THOI_GIAN:NGANH:cntt:2026:all:{H}"
C_LEGACY = "HOC_PHI:NGANH:cntt:2026:all"  # 5-part legacy id: the migrator remaps it
C = "HOC_PHI:NGANH:cntt:2026:all:all"
E = "HOC_PHI:NGANH:cntt:2027:all:all"  # supersedes the remapped C_LEGACY
RETRACT_REASON = "claim withdrawn by the issuing office"


def _base(record_id: str, version: int, **overrides: object) -> CanonicalRecord:
    fields: dict = dict(
        record_id=record_id,
        version=version,
        record_type="DIEM_CHUAN",
        admission_year=2026,
        effective_from=DAY,
        status="draft",
        authority_level="husc_canonical",
        text=f"text of {record_id}@{version}",
        authored_by="alice",
        admission_round="all",
    )
    fields.update(overrides)
    return CanonicalRecord(**fields)


def _published(record_id: str, version: int, **overrides: object) -> CanonicalRecord:
    fields: dict = {"status": "published", "source_document_id": DOC, "published_at": PUB}
    fields.update(overrides)
    return _base(record_id, version, **fields)


def _pg(url: str):
    return psycopg.connect(_conninfo(url))


def _register(url: str, doc_id: str = DOC) -> None:
    with _pg(url) as conn:
        conn.execute(
            "INSERT INTO source_documents (document_id, sha256, kind, storage_uri, original_name, uploaded_by) "
            "VALUES (%s, %s, 'pdf', 'uploads/x.pdf', 'x.pdf', 'alice')",
            (doc_id, bytes.fromhex("cd" * 32)),
        )


def _document(root: Path, doc_id: str = DOC) -> None:
    _write_json(root / "documents" / f"{doc_id}_d.json", {
        "document_id": doc_id, "sha256": "cd" * 32, "path": "uploads/tuyensinh2026.pdf",
        "state": "registered", "registered_by": "alice",
        "registered_at": "2026-01-02T03:04:05", "updated_at": "2026-01-03T00:00:00+00:00",
    })


def _legacy_root(root: Path) -> None:
    """Several records, multiple versions, a published supersession chain (B supersedes A)."""
    _document(root)
    store = CanonicalRecordStore(root)
    store.write_record(_base(A, 1, text="a v1 draft"))
    store.write_record(_published(A, 2, text="a v2 published"))
    store.write_record(_published(B, 1, admission_year=2027, supersedes=A, authored_by="bob", text="b"))
    store.write_record(_published(C_LEGACY, 1, record_type="HOC_PHI", text="c"))
    store.write_record(_base(D, 1, record_type="THOI_GIAN", authored_by="carol", text="d draft"))


def _mapped_reference(ref: str, idmap: dict) -> str:
    name, sep, version = ref.rpartition("@")
    if not sep:
        return idmap.get(ref, ref)
    return f"{idmap.get(name, name)}@{version}"


def _mapped(record: CanonicalRecord, idmap: dict) -> CanonicalRecord:
    update = {"record_id": idmap.get(record.record_id, record.record_id)}
    if record.supersedes:
        update["supersedes"] = _mapped_reference(record.supersedes, idmap)
    return record.model_copy(update=update)


def _assert_parity(file: CanonicalRecordStore, sql: SqlRecordStore, idmap: dict) -> None:
    for legacy in file.record_ids():
        new = idmap.get(legacy, legacy)
        assert sql.versions(new) == file.versions(legacy), legacy
        f_chain, s_chain = file.get_record(legacy), sql.get_record(new)
        assert _mapped(f_chain.latest, idmap) == s_chain.latest, legacy
        assert tuple(_mapped(r, idmap) for r in f_chain.prior) == s_chain.prior, legacy
        for version in file.versions(legacy):
            assert _mapped(file.get_version(legacy, version), idmap) == sql.get_version(new, version)
            expected = dataclasses.replace(
                file.resolve_provenance(f"{legacy}@{version}"), record_id=new
            )
            assert sql.resolve_provenance(f"{new}@{version}") == expected
        assert sql.is_superseded(new) == file.is_superseded(legacy), legacy
        f_sup = file.superseding_record(legacy)
        assert (None if f_sup is None else _mapped(f_sup, idmap)) == sql.superseding_record(new), legacy
        assert sql.superseding_versions(new) == tuple(
            _mapped(r, idmap) for r in file.superseding_versions(legacy)
        ), legacy
        assert sql.retracted_supersession_keys(new) == file.retracted_supersession_keys(legacy), legacy
    f_index = file.supersession_index()
    assert sql.supersession_index() == {
        idmap.get(name, name): _mapped(rec, idmap) for name, rec in f_index.items()
    }
    assert sql.record_ids() == tuple(sorted(idmap.get(x, x) for x in file.record_ids()))


# ── pure checks (no PostgreSQL) ──────────────────────────────────────────


def test_constructor_never_connects(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("constructor must not connect")

    monkeypatch.setattr(psycopg, "connect", boom)
    SqlRecordStore("postgresql+psycopg://rag@localhost:5439/husc")


def test_constructor_needs_a_url_from_argument_or_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError):
        SqlRecordStore()
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://rag@localhost:5439/husc")
    SqlRecordStore()


def test_url_conversion_drops_the_driver_suffix():
    assert _conninfo("postgresql+asyncpg://rag@localhost:5439/husc") == "postgresql://rag@localhost:5439/husc"
    assert _conninfo("postgresql+psycopg://rag@localhost:5439/husc") == "postgresql://rag@localhost:5439/husc"


def test_url_conversion_touches_only_the_scheme_not_the_password():
    url = "postgresql+asyncpg://rag:pw+asyncpg+psycopg@localhost:5439/husc"
    assert _conninfo(url) == "postgresql://rag:pw+asyncpg+psycopg@localhost:5439/husc"


def test_unreachable_database_is_a_store_error_without_secrets():
    store = SqlRecordStore("postgresql+psycopg://rag:hunter2secret@127.0.0.1:1/husc")
    with pytest.raises(RecordStoreError) as info:
        store.versions("x")
    text = str(info.value)
    assert "unavailable" in text
    for secret in ("hunter2secret", "127.0.0.1", "rag"):
        assert secret not in text


def test_connections_carry_a_connect_timeout(monkeypatch):
    seen: dict = {}
    real_connect = psycopg.connect

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(psycopg, "connect", spy)
    with pytest.raises(RecordStoreError):
        SqlRecordStore("postgresql+psycopg://rag@127.0.0.1:1/husc").versions("x")
    assert seen.get("connect_timeout") == 10


# ── PostgreSQL checks ────────────────────────────────────────────────────


def test_parity_with_file_store_after_migration(tmp_path, fresh_db):  # noqa: F811
    root = tmp_path / "root"
    _legacy_root(root)
    report = _apply(root, fresh_db)
    idmap = report["id_map"]["records"]
    assert idmap[C_LEGACY] == C  # the remap path is exercised, not only identity ids
    file = CanonicalRecordStore(root)
    sql = SqlRecordStore(fresh_db)

    _assert_parity(file, sql, idmap)
    assert sql.is_superseded(A) is True

    # The migrator cannot import retractions (superseded_version None), so apply
    # the same retraction to both stores after migration and compare again.
    args = dict(reason=RETRACT_REASON, retracted_by="alice", at="2026-03-01T00:00:00+00:00")
    assert file.retract_supersession(A, f"{B}@1", **args) is True
    assert sql.retract_supersession(A, f"{B}@1", **args) is True
    assert file.retract_supersession(A, f"{B}@1", **args) is False
    assert sql.retract_supersession(A, f"{B}@1", **args) is False
    _assert_parity(file, sql, idmap)
    assert sql.is_superseded(A) is False


def test_write_and_edit_round_trip_is_lossless(fresh_db):  # noqa: F811
    _register(fresh_db)
    sql = SqlRecordStore(fresh_db)
    v1 = _base(
        A, 1,
        authored_by=None,
        admission_round=None,
        published_at=datetime(2026, 2, 1, 9, 30),  # naive timestamp must stay naive
        note="extra field kept verbatim",
        nested={"k": [1, "ă"]},
    )
    assert sql.write_record(v1) == v1
    assert sql.get_version(A, 1) == v1

    # A draft-to-draft edit would be a second open draft (uq_one_open_draft), so edit to published.
    edited = sql.edit_record(A, status="published", source_document_id=DOC, text="a v2")
    assert edited.version == 2
    assert edited == v1.model_copy(update={"version": 2, "status": RecordStatus.PUBLISHED,
                                           "source_document_id": DOC, "text": "a v2"})
    assert sql.get_version(A, 2) == edited
    assert sql.get_record(A).latest == edited
    assert sql.get_record(A).prior == (v1,)


def test_duplicate_version_raises_conflict_and_keeps_history(fresh_db):  # noqa: F811
    sql = SqlRecordStore(fresh_db)
    sql.write_record(_base(A, 1))
    with pytest.raises(RecordVersionConflictError):
        sql.write_record(_base(A, 1, text="rewrite attempt"))
    assert sql.versions(A) == (1,)
    assert sql.get_version(A, 1).text == f"text of {A}@1"


def test_unknown_ids_raise_not_found_like_the_file_store(tmp_path, fresh_db):  # noqa: F811
    file = CanonicalRecordStore(tmp_path / "empty")
    sql = SqlRecordStore(fresh_db)
    for store in (file, sql):
        with pytest.raises(RecordNotFoundError):
            store.get_record("nope")
        with pytest.raises(RecordNotFoundError):
            store.get_version("nope", 1)
        with pytest.raises(RecordNotFoundError):
            store.superseding_record("nope")
        with pytest.raises(RecordNotFoundError):
            store.resolve_provenance("nope@1")
    assert sql.versions("nope") == ()
    assert sql.next_version("nope") == 1


def test_two_threads_writing_the_same_new_version_one_wins(fresh_db):  # noqa: F811
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def attempt(text: str) -> None:
        store = SqlRecordStore(fresh_db)
        barrier.wait()
        try:
            store.write_record(_base(A, 1, text=text))
            outcomes.append("written")
        except RecordVersionConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=attempt, args=(f"writer {n}",)) for n in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["conflict", "written"]
    sql = SqlRecordStore(fresh_db)
    assert sql.versions(A) == (1,)  # no partial second row


def test_record_ids_order_matches_the_file_store(tmp_path, fresh_db):  # noqa: F811
    # No case-only pairs: the file store maps ids to percent-encoded dirs, which collide on a
    # case-insensitive filesystem (Windows), so "a:x" and "A:x" cannot both be in one fixture.
    ids = ["B:x", "a:x", "a_x", "Z:x", "é:x", "DIEM_CHUAN:z", "aa:1"]
    file = CanonicalRecordStore(tmp_path / "order")
    sql = SqlRecordStore(fresh_db)
    for record_id in reversed(ids):
        file.write_record(_base(record_id, 1))
        sql.write_record(_base(record_id, 1))
    assert sql.record_ids() == file.record_ids() == tuple(sorted(ids))


def test_publishing_archives_the_previous_published_version(fresh_db):  # noqa: F811
    _register(fresh_db)
    sql = SqlRecordStore(fresh_db)
    sql.write_record(_published(A, 1))
    sql.write_record(_published(A, 2, text="a v2"))
    assert sql.get_version(A, 1).status is RecordStatus.ARCHIVED
    assert sql.get_version(A, 2).status is RecordStatus.PUBLISHED
    with _pg(fresh_db) as conn:
        row = conn.execute(
            "SELECT archive_reason FROM canonical_records WHERE record_id = %s AND version = 1", (A,)
        ).fetchone()
    assert row[0] == "superseded"


def test_second_open_draft_raises_conflict(fresh_db):  # noqa: F811
    sql = SqlRecordStore(fresh_db)
    sql.write_record(_base(A, 1))
    with pytest.raises(RecordVersionConflictError):
        sql.write_record(_base(A, 2))
    assert sql.versions(A) == (1,)


def test_unregistered_source_document_raises_store_error(fresh_db):  # noqa: F811
    sql = SqlRecordStore(fresh_db)
    with pytest.raises(RecordStoreError):
        sql.write_record(_published(A, 1, source_document_id="missing-doc"))
    assert sql.versions(A) == ()


def test_retract_rejects_a_short_reason_and_a_bare_claim(fresh_db):  # noqa: F811
    sql = SqlRecordStore(fresh_db)
    sql.write_record(_base(A, 1))
    with pytest.raises(RecordStoreError):
        sql.retract_supersession(A, f"{B}@1", reason="short", retracted_by="alice", at="2026-03-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        sql.retract_supersession(A, B, reason=RETRACT_REASON, retracted_by="alice", at="2026-03-01T00:00:00+00:00")


def test_failed_insert_rolls_back_the_archive_of_the_previous_version(fresh_db):  # noqa: F811
    _register(fresh_db)
    sql = SqlRecordStore(fresh_db)
    sql.write_record(_published(A, 1))
    # A NUL byte is rejected while the INSERT is being sent, after the archive UPDATE ran.
    with pytest.raises(RecordStoreError):
        sql.write_record(_published(A, 2, text="bad\x00text"))
    assert sql.get_version(A, 1).status is RecordStatus.PUBLISHED
    assert sql.versions(A) == (1,)


def test_unique_violation_on_insert_maps_to_conflict(fresh_db, monkeypatch):  # noqa: F811
    sql = SqlRecordStore(fresh_db)
    sql.write_record(_base(A, 1, text="first"))
    # Simulate losing the race: the pre-check sees no stored versions, so the INSERT hits the primary key.
    monkeypatch.setattr(sql_module, "_version_numbers", lambda conn, record_id: ())
    with pytest.raises(RecordVersionConflictError):
        sql.write_record(_base(A, 1, text="second"))
    monkeypatch.undo()
    assert sql.versions(A) == (1,)
    assert sql.get_version(A, 1).text == "first"


def test_supersession_of_a_remapped_legacy_record_survives_migration(tmp_path, fresh_db):  # noqa: F811
    root = tmp_path / "root"
    _document(root)
    file = CanonicalRecordStore(root)
    file.write_record(_published(C_LEGACY, 1, record_type="HOC_PHI", text="c"))
    file.write_record(_published(
        E, 1, record_type="HOC_PHI", admission_year=2027, supersedes=f"{C_LEGACY}@1", authored_by="bob", text="e",
    ))
    report = _apply(root, fresh_db)
    idmap = report["id_map"]["records"]
    assert idmap[C_LEGACY] == C

    sql = SqlRecordStore(fresh_db)
    _assert_parity(file, sql, idmap)
    assert sql.is_superseded(C) is True
    assert sql.superseding_record(C).record_id == idmap.get(E, E)
