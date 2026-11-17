"""SqlAuditLedger over ``audit_events`` (U3 task 4b).

Pure tests run without a database. PostgreSQL tests use the ``fresh_db`` fixture
from tests/scripts/test_migrate_apply.py (a throwaway migrated database) and skip
without DATABASE_URL.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row
from sqlalchemy.ext.asyncio import create_async_engine

from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.infrastructure.audit_ledger import GENESIS_HASH, AuditLedger, AuditLedgerError, ChainVerification
from src.infrastructure.sql_audit_ledger import LEGACY_REQUEST_ID, SqlAuditLedger, _columns
from src.infrastructure.sql_audit_log import append_audit_event
from src.infrastructure.sql_record_store import _conninfo
from tests.scripts.test_migrate_apply import _apply, fresh_db  # noqa: F401  (fixture)

T0 = datetime(2026, 11, 17, 9, 30, tzinfo=timezone.utc)


def _event(i: int = 0, **overrides) -> AuditEvent:
    fields = dict(
        action=AuditAction.EDIT,
        actor_id="alice",
        target=f"rec-{i}",
        timestamp=T0 + timedelta(seconds=i),
        role="editor",
        before_version="1",
        after_version="2",
    )
    fields.update(overrides)
    return AuditEvent(**fields)


def _raw_rows(url: str) -> list[dict]:
    with psycopg.connect(_conninfo(url), row_factory=dict_row) as conn:
        return conn.execute(
            "SELECT seq, request_id, prev_hash, entry_hash FROM audit_events ORDER BY seq"
        ).fetchall()


def _tamper(url: str, statement: str) -> None:
    """Run ``statement`` with the append-only triggers switched off, then restore them."""
    with psycopg.connect(_conninfo(url), autocommit=True) as conn:
        conn.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
        try:
            conn.execute(statement)
        finally:
            conn.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")


def _file_ledger(root: Path) -> list[AuditEvent]:
    """A three-entry file ledger under ``root`` (the migrator's input shape)."""
    events = [
        _event(0),
        _event(1, action=AuditAction.ROLLBACK, reason="Hoàn tác do sai số liệu", before_version="rec-1@2", after_version="rec-1@1"),
        _event(2, action=AuditAction.PUBLISH, target_kind=TargetKind.RELEASE, target="rel-1", release_id="rel-1"),
    ]
    file_ledger = AuditLedger(root / "audit" / "ledger.jsonl")
    for event in events:
        file_ledger.append(event)
    return events


# -- pure ----------------------------------------------------------------------


def test_naive_timestamp_is_stored_as_utc():
    cols = _columns(_event(timestamp=datetime(2026, 1, 2, 3, 4)))
    assert cols["created_at"] == datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert cols["created_at"].tzinfo is not None


def test_missing_role_is_stored_as_unknown():
    assert _columns(_event(role=None))["role"] == "unknown"


def test_digit_versions_are_ints_and_other_versions_go_to_extras():
    cols = _columns(_event(before_version="7", after_version="subject@editor"))
    assert cols["before_version"] == 7
    assert cols["after_version"] is None
    assert cols["before"] is None
    assert cols["after"] == {"audit_extras": {"after_version": "subject@editor"}}


def test_leading_zero_and_out_of_range_versions_are_not_ints():
    cols = _columns(_event(before_version="007", after_version="3000000000"))
    assert cols["before_version"] is None
    assert cols["after_version"] is None
    assert cols["after"] == {"audit_extras": {"before_version": "007", "after_version": "3000000000"}}


def test_extras_carry_only_non_default_keys():
    assert _columns(_event())["after"] is None
    cols = _columns(_event(entity="custom-entity", source_ip="10.0.0.1", review_id="rv-1"))
    assert cols["after"] == {
        "audit_extras": {"entity": "custom-entity", "source_ip": "10.0.0.1", "review_id": "rv-1"}
    }


def test_constructor_never_connects(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("constructor must not connect")

    monkeypatch.setattr(psycopg, "connect", boom)
    SqlAuditLedger("postgresql+psycopg://rag@localhost:5439/husc")


def test_constructor_needs_a_url_from_argument_or_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError):
        SqlAuditLedger()


def test_unreachable_database_is_a_ledger_error_without_secrets():
    ledger = SqlAuditLedger("postgresql+psycopg://rag:hunter2secret@127.0.0.1:1/husc")
    with pytest.raises(AuditLedgerError) as info:
        ledger.verify()
    message = str(info.value)
    assert "unavailable" in message
    for secret in ("hunter2secret", "127.0.0.1", "rag"):
        assert secret not in message


# -- PostgreSQL ----------------------------------------------------------------


def test_empty_table_verifies_with_zero_checked(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    assert ledger.verify() == ChainVerification(ok=True, checked=0, broken_index=None, reason=None, last_hash=None)
    assert ledger.verify_detailed()["legacy_checked"] == 0


def test_three_appends_verify_and_round_trip_every_field(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    events = [
        _event(
            0,
            action=AuditAction.CREATE,
            before_version=None,
            after_version="1",
            reason="Tạo hồ sơ tuyển sinh năm 2026",
            entity="custom-entity",
            source_ip="10.0.0.7",
            review_id="rv-9",
        ),
        _event(
            1,
            action=AuditAction.PUBLISH,
            target_kind=TargetKind.RELEASE,
            target="rel-2026-1",
            role="approver",
            release_id="rel-2026-1",
            reason="Công bố đợt 1",
        ),
        _event(
            2,
            action=AuditAction.ROLLBACK,
            actor_id="carol",
            role="admin",
            before_version="rec-2@2",
            after_version="rec-2@1",
            reason="Hoàn tác do sai số liệu",
        ),
    ]
    entries = [ledger.append(event) for event in events]

    assert [entry.seq for entry in entries] == [1, 2, 3]
    assert entries[0].prev_hash == GENESIS_HASH
    assert entries[1].prev_hash == entries[0].entry_hash
    for entry in entries:
        assert len(entry.entry_hash) == 64 and entry.entry_hash == entry.entry_hash.lower()

    result = ledger.verify()
    assert result.ok is True
    assert result.checked == 3
    assert result.broken_index is None
    assert result.last_hash == entries[-1].entry_hash

    stored = ledger.entries()
    assert [entry.event for entry in stored] == events
    assert [entry.entry_hash for entry in stored] == [entry.entry_hash for entry in entries]


def test_record_permission_change_round_trips(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    entry = ledger.record_permission_change(
        actor_id="root",
        actor_role="admin",
        subject_id="dana",
        before_role=None,
        after_role="editor",
        reason="Cấp quyền biên tập",
        source_ip="10.1.1.1",
        four_eyes=True,
    )
    stored = ledger.get(entry.event.event_id)
    assert stored is not None
    assert stored.event == entry.event
    assert stored.event.target_kind is TargetKind.SUBJECT
    assert stored.event.before_version == "dana@none"
    assert stored.event.after_version == "dana@editor"
    assert stored.event.review_id == "four_eyes=on"
    assert ledger.verify().ok


def test_find_and_get_filter_the_chain(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    events = [
        _event(1, actor_id="alice"),
        _event(2, actor_id="bob", action=AuditAction.PUBLISH, release_id="rel-1", target_kind=TargetKind.RELEASE),
        _event(1, actor_id="carol", action=AuditAction.ROLLBACK, reason="Hoàn tác"),
    ]
    for event in events:
        ledger.append(event)

    assert [entry.seq for entry in ledger.find(target="rec-1")] == [1, 3]
    assert [entry.seq for entry in ledger.find(actor_id="bob")] == [2]
    assert [entry.seq for entry in ledger.find(action="rollback")] == [3]
    assert [entry.seq for entry in ledger.find(action=AuditAction.PUBLISH)] == [2]
    assert ledger.find(action="no_such_action") == ()
    assert ledger.get(events[1].event_id).seq == 2
    assert ledger.get(uuid.uuid4().hex) is None


def test_rows_from_async_writer_and_sql_ledger_interleave_and_verify(fresh_db):
    async_url = fresh_db.replace("+psycopg", "+asyncpg", 1)
    ledger = SqlAuditLedger(fresh_db)
    ledger.append(_event(0))

    async def async_append() -> None:
        engine = create_async_engine(async_url)
        try:
            await append_audit_event(
                engine,
                {
                    "aggregate_type": "record",
                    "aggregate_id": "rec-async",
                    "actor_id": "async-writer",
                    "role": "editor",
                    "action": "edit",
                    "before": None,
                    "after": None,
                    "reason": None,
                    "before_version": 3,
                    "after_version": 4,
                    "release_id": None,
                    "request_id": "req-1",
                },
            )
        finally:
            await engine.dispose()

    asyncio.run(async_append())
    ledger.append(_event(1))

    result = ledger.verify()
    assert result.ok is True
    assert result.checked == 3


def test_concurrent_threads_keep_one_chain(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for k in range(5):
                ledger.append(_event(n * 5 + k, actor_id=f"writer-{n}"))
        except BaseException as exc:  # noqa: BLE001 - surface any failure to the test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    rows = _raw_rows(fresh_db)
    assert len(rows) == 20
    assert len({row["seq"] for row in rows}) == 20
    prev_hashes = [bytes(row["prev_hash"]) for row in rows if row["prev_hash"] is not None]
    assert len(prev_hashes) == 19
    assert len(set(prev_hashes)) == 19
    result = ledger.verify()
    assert result.ok is True
    assert result.checked == 20


def test_tampered_row_breaks_at_its_index(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    for i in range(3):
        ledger.append(_event(i))
    _tamper(fresh_db, "UPDATE audit_events SET reason = 'x' WHERE seq = 2")

    result = ledger.verify()
    assert result.ok is False
    assert result.broken_index == 1
    assert result.checked == 1


def test_invalid_enum_value_in_a_row_does_not_crash_verify(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    ledger.append(_event(0))
    _tamper(fresh_db, "UPDATE audit_events SET action = 'bogus' WHERE seq = 1")

    result = ledger.verify()
    assert result.ok is False
    assert result.broken_index == 0


def test_legacy_marker_after_a_non_legacy_row_is_broken(fresh_db):
    ledger = SqlAuditLedger(fresh_db)
    for i in range(3):
        ledger.append(_event(i))
    _tamper(fresh_db, f"UPDATE audit_events SET request_id = '{LEGACY_REQUEST_ID}' WHERE seq = 2")

    result = ledger.verify()
    assert result.ok is False
    assert result.broken_index == 1


def test_migrated_chain_verifies_after_sql_appends(tmp_path, fresh_db):
    events = _file_ledger(tmp_path)
    _apply(tmp_path, fresh_db)
    ledger = SqlAuditLedger(fresh_db)
    ledger.append(_event(10))
    ledger.append(_event(11))

    detailed = ledger.verify_detailed()
    assert detailed["ok"] is True
    assert detailed["checked"] == 5
    assert detailed["legacy_checked"] == 3

    stored = ledger.entries()
    assert [entry.event for entry in stored[:3]] == events
    assert ledger.verify().ok is True


def test_migrated_legacy_prev_hash_edit_breaks_the_chain(tmp_path, fresh_db):
    _file_ledger(tmp_path)
    _apply(tmp_path, fresh_db)
    ledger = SqlAuditLedger(fresh_db)
    ledger.append(_event(10))
    ledger.append(_event(11))
    _tamper(fresh_db, "UPDATE audit_events SET prev_hash = sha256('x'::bytea) WHERE seq = 2")

    result = ledger.verify()
    assert result.ok is False
    assert result.broken_index == 1


def test_migrated_chain_breaks_on_tampered_first_sql_row(tmp_path, fresh_db):
    _file_ledger(tmp_path)
    _apply(tmp_path, fresh_db)
    ledger = SqlAuditLedger(fresh_db)
    ledger.append(_event(10))
    ledger.append(_event(11))
    _tamper(fresh_db, "UPDATE audit_events SET reason = 'x' WHERE seq = 4")

    detailed = ledger.verify_detailed()
    assert detailed["ok"] is False
    assert detailed["broken_index"] == 3
    assert detailed["legacy_checked"] == 3
    assert detailed["checked"] == 3
