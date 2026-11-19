"""Real-run PostgreSQL write path of scripts/migrate_file_stores.py (U3, Task 3a).

Pure CLI/report checks run everywhere. Tests that write PostgreSQL need
DATABASE_URL and skip otherwise. Each PostgreSQL test builds its own throwaway
database (CREATE DATABASE on the 'postgres' db of the same server, alembic
upgrade head) and drops it at teardown, so the 'husc' data is never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest

import scripts.migrate_file_stores as migrate
from scripts.migrate_file_stores import MigrationError
from src.domain.canonical.record import RecordStatus
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from tests.infrastructure.conftest import run_alembic
from tests.pg_fixtures import fresh_db  # noqa: F401  (re-exported fixture)
from tests.scripts.test_migrate_cli import (
    _PLAIN_ALICE,
    _PLAIN_BOB,
    STORE_ORDER,
    _identity_fixture,
    _multi_store,
    _rec,
    _tree,
)

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
RECORD_KEY = "CHI_TIEU:nganh:7220104:2026:all:all"
TABLES = [
    "source_documents", "admin_identities", "releases", "canonical_records",
    "reindex_outbox", "audit_events", "supersession_retractions",
    "conflict_entries", "idempotency_keys", "unanswered_questions", "chat_feedback",
]


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _write_lines(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs), encoding="utf-8")


def _full_fixture(root: Path) -> None:
    """One legacy file (or more) for each of the 12 stores."""
    _multi_store(root)  # record (source doc-1), unanswered question, identities, 3 audit events
    _write_json(root / "documents" / "doc-1_d.json", {
        "document_id": "doc-1", "sha256": "cd" * 32, "path": "uploads/tuyensinh2026.pdf",
        "state": "registered", "registered_by": "alice",
        "registered_at": "2026-01-02T03:04:05", "updated_at": "2026-01-03T00:00:00+00:00",
    })
    blob = b"%PDF-1.4 tuyensinh2026"
    (root / "source_documents").mkdir(parents=True, exist_ok=True)
    (root / "source_documents" / "tuyensinh2026.pdf").write_bytes(blob)
    _write_json(root / "documents" / "doc-up_d.json", {
        "document_id": "uploads/tuyensinh2026.pdf", "sha256": hashlib.sha256(blob).hexdigest(),
        "path": "uploads/tuyensinh2026.pdf", "state": "registered", "registered_by": "alice",
        "registered_at": "2026-01-02T03:04:05", "updated_at": "2026-01-03T00:00:00+00:00",
    })
    # Child listed before its parent: the release insert order must follow parent_release_id.
    _write_json(root / "releases" / "a-child" / "release_manifest.json", {
        "release_id": "rel-2026-2", "parent_release_id": "rel-2026-1", "lancedb_table": "chunks_2026",
        "lancedb_version": 4, "graph_path": "graph/a.json", "created_at": "2026-01-05T00:00:00+00:00",
    })
    _write_json(root / "releases" / "rel-2026-1" / "release_manifest.json", {
        "release_id": "rel-2026-1", "parent_release_id": None, "lancedb_table": "chunks_2026",
        "lancedb_version": 3, "graph_path": None, "created_at": "2026-01-04T00:00:00+00:00",
    })
    _write_lines(root / "reindex_outbox.jsonl", [{
        "entry_id": "rel-2026-1|" + RECORD_KEY + "@1", "record_id": RECORD_KEY, "version": 1,
        "release_id": "rel-2026-1", "appended_at": "2026-01-05T01:00:00+00:00", "action": "publish",
    }])
    _write_json(root / "conflicts" / "queue.json", {"items": [{
        "conflict_id": "conf-1", "status": "open", "coordinate_key": RECORD_KEY,
        "record_ids": [RECORD_KEY], "conflict_type": "duplicate_coordinate",
    }]})
    _write_json(root / "idempotency" / f"{'ab' * 32}.json", {
        "state": "complete", "actor_id": "alice", "operation": "publish",
        "request_fingerprint": "ef" * 32, "status_code": 200, "response": {"ok": True},
        "created_at": "2026-01-06T00:00:00+00:00",
    })
    _write_lines(root / "unanswered" / "feedback.jsonl", [{
        "trace_id": "trace-f1", "helpful": True, "reason": None,
        "answered_at": "2026-01-07T00:00:00+00:00", "linked_item": False,
    }])


def _counts(url: str) -> dict[str, int]:
    with psycopg.connect(_conninfo(url)) as conn:
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


def _conninfo(sync_url: str) -> str:
    return sync_url.replace("+psycopg", "", 1)


def _apply(root: Path, url: str) -> dict:
    with psycopg.connect(_conninfo(url)) as conn:
        return migrate.apply_to_database(root, conn, now=NOW)


def _stores(report: dict) -> dict[str, dict]:
    return {s["name"]: s for s in report["stores"]}


# ── CLI and pure checks (no PostgreSQL) ──────────────────────────────────


def test_sync_database_url_swaps_async_driver_for_psycopg():
    assert migrate.sync_database_url("postgresql+asyncpg://rag@localhost:5439/husc") == (
        "postgresql+psycopg://rag@localhost:5439/husc"
    )


def test_real_run_without_confirm_frozen_exits_2_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://rag@localhost:5439/husc")
    root = tmp_path / "root"
    _multi_store(root)
    before = _tree(root)
    report_path = tmp_path / "r.json"
    assert migrate.main(["--report", str(report_path), "--root", str(root)]) == 2
    err = capsys.readouterr().err
    assert "--confirm-frozen" in err
    assert not report_path.exists()
    assert _tree(root) == before


def test_real_run_without_database_url_exits_2_and_never_connects(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("must not connect")

    monkeypatch.setattr(migrate.psycopg, "connect", boom)
    report_path = tmp_path / "r.json"
    code = migrate.main(["--confirm-frozen", "--report", str(report_path), "--root", str(tmp_path)])
    assert code == 2
    assert "DATABASE_URL" in capsys.readouterr().err
    assert not report_path.exists()


def test_dry_run_never_connects_even_with_a_database_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://rag@localhost:5439/husc")

    def boom(*args, **kwargs):
        raise AssertionError("dry run must not connect")

    monkeypatch.setattr(migrate.psycopg, "connect", boom)
    root = tmp_path / "root"
    _multi_store(root)
    assert migrate.main(["--dry-run", "--report", str(tmp_path / "r.json"), "--root", str(root)]) == 0


def test_dry_run_with_scrub_identities_exits_2_and_keeps_identities(tmp_path):
    root = tmp_path / "root"
    _multi_store(root)
    path = root / "identity" / "identities.json"
    before = path.read_bytes()
    code = migrate.main([
        "--dry-run", "--scrub-identities", "--report", str(tmp_path / "r.json"), "--root", str(root),
    ])
    assert code == 2
    assert path.read_bytes() == before


# ── PostgreSQL write path ────────────────────────────────────────────────


def test_twelve_store_fixture_applies_and_counts_match(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    report = _apply(root, fresh_db)
    assert report["dry_run"] is False
    assert report["run_at"] == NOW.isoformat()
    stores = _stores(report)
    assert list(stores) == STORE_ORDER
    for name, s in stores.items():
        assert s["target_rows"] == s["source_rows"], (name, s)
        assert s["mismatch"] == 0, (name, s)
        expected_inserted = 0 if name == "source_document_blobs" else s["source_rows"]
        assert s["inserted"] == expected_inserted, (name, s)
    assert _counts(fresh_db)["canonical_records"] == 1
    assert _counts(fresh_db)["audit_events"] == 3
    assert report["defaulted"]["records"]["authored_by"] == 1
    assert report["defaulted"]["releases"]["status"] == 2
    assert report["skipped"] == {"idempotency": 0}


def test_apply_twice_inserts_nothing_the_second_time(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    first = _apply(root, fresh_db)
    counts = _counts(fresh_db)
    second = _apply(root, fresh_db)
    assert _counts(fresh_db) == counts
    assert all(s["inserted"] == 0 for s in second["stores"])
    for a, b in zip(first["stores"], second["stores"]):
        assert a["target_rows"] == b["target_rows"], (a["name"], a, b)
        assert a["mismatch"] == b["mismatch"] == 0


def test_audit_hashes_are_byte_exact_and_seq_is_ledger_order_with_setval(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    _apply(root, fresh_db)
    ledger = [json.loads(line) for line in (root / "audit" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        rows = conn.execute(
            "SELECT seq, event_id::text, prev_hash, entry_hash FROM audit_events ORDER BY seq"
        ).fetchall()
        next_seq = conn.execute(
            "SELECT nextval(pg_get_serial_sequence('audit_events', 'seq'))"
        ).fetchone()[0]
    assert [r[0] for r in rows] == [1, 2, 3]
    assert next_seq == 4
    for row, entry in zip(rows, ledger):
        assert bytes(row[3]) == bytes.fromhex(entry["entry_hash"])
        assert row[1] == str(uuid.UUID(hex=entry["event"]["event_id"]))
    assert rows[0][2] is None  # genesis prev_hash stays NULL
    for prev_row, row, entry in zip(rows, rows[1:], ledger[1:]):
        assert bytes(row[2]) == bytes.fromhex(entry["prev_hash"]) == bytes(prev_row[3])


def test_duplicate_actor_rolls_back_every_store(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    path = root / "identity" / "identities.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    duplicate = dict(doc["entries"][0])
    duplicate["subject_id"] = str(uuid.uuid4())
    doc["entries"].append(duplicate)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(MigrationError) as exc:
        _apply(root, fresh_db)
    assert "identities" in str(exc.value)
    assert "alice" in str(exc.value)
    assert all(count == 0 for count in _counts(fresh_db).values())


def test_retraction_without_superseded_version_raises_and_rolls_back(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    _write_json(root / "supersession_retractions" / "d1" / "r1.json", {
        "target_record_id": RECORD_KEY,
        "superseding_version_key": RECORD_KEY + "@2",
        "retracted_by": "alice",
        "at": "2026-01-08T00:00:00+00:00",
        "reason": "sai so lieu nguon",
    })
    with pytest.raises(MigrationError) as exc:
        _apply(root, fresh_db)
    assert "retractions" in str(exc.value)
    assert RECORD_KEY in str(exc.value)
    assert all(count == 0 for count in _counts(fresh_db).values())


def test_dangling_source_document_id_becomes_null_with_payload_key_and_counter(tmp_path, fresh_db):
    root = tmp_path / "root"
    CanonicalRecordStore(root).write_record(
        _rec("CHI_TIEU:nganh:7220104:2026:dangling").model_copy(update={"source_document_id": "doc-missing"})
    )
    report = _apply(root, fresh_db)
    assert report["dangling_source_document_id"] == 1
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        source, legacy = conn.execute(
            "SELECT source_document_id, structured_payload->>'legacy_source_document_id' FROM canonical_records"
        ).fetchone()
    assert source is None
    assert legacy == "doc-missing"


def _register_doc_1(root: Path) -> None:
    _write_json(root / "documents" / "doc-1_d.json", {
        "document_id": "doc-1", "sha256": "cd" * 32, "path": "uploads/tuyensinh2026.pdf",
        "state": "registered", "registered_by": "alice",
        "registered_at": "2026-01-02T03:04:05", "updated_at": "2026-01-03T00:00:00+00:00",
    })


def test_superseded_published_row_with_dangling_source_is_archived_not_aborted(tmp_path, fresh_db):
    root = tmp_path / "root"
    _register_doc_1(root)
    rid = "CHI_TIEU:nganh:7220104:2026:supersede"
    store = CanonicalRecordStore(root)
    store.write_record(_rec(rid).model_copy(update={
        "status": RecordStatus.PUBLISHED, "published_at": NOW, "source_document_id": "doc-missing",
    }))
    store.write_record(_rec(rid).model_copy(update={
        "version": 2, "status": RecordStatus.PUBLISHED,
        "published_at": datetime(2026, 11, 18, tzinfo=timezone.utc),
    }))
    report = _apply(root, fresh_db)
    assert report["dangling_source_document_id"] == 1
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        rows = conn.execute(
            "SELECT version, status, source_document_id::text, "
            "structured_payload->>'legacy_source_document_id', structured_payload->>'legacy_status' "
            "FROM canonical_records ORDER BY version"
        ).fetchall()
    assert rows == [
        (1, "archived", None, "doc-missing", "published"),
        (2, "published", "doc-1", None, None),
    ]


def test_single_published_row_with_dangling_source_still_aborts_and_rolls_back(tmp_path, fresh_db):
    root = tmp_path / "root"
    _register_doc_1(root)
    CanonicalRecordStore(root).write_record(_rec("CHI_TIEU:nganh:7220104:2026:stays").model_copy(update={
        "status": RecordStatus.PUBLISHED, "published_at": NOW, "source_document_id": "doc-missing",
    }))
    with pytest.raises(MigrationError) as exc:
        _apply(root, fresh_db)
    assert "doc-missing" in str(exc.value)
    assert all(count == 0 for count in _counts(fresh_db).values())


def test_superseded_row_without_original_status_records_draft_as_legacy_status(tmp_path, fresh_db):
    root = tmp_path / "root"
    _register_doc_1(root)
    rid = "CHI_TIEU:nganh:7220104:2026:nostatus"
    store = CanonicalRecordStore(root)
    store.write_record(_rec(rid))
    store.write_record(_rec(rid).model_copy(update={"version": 2}))
    v1 = root / "records" / f"rec-{quote(rid, safe='')}_r" / "v0001.json"
    payload = json.loads(v1.read_text(encoding="utf-8"))
    payload["status"] = None  # legacy row with no status: the effective default is draft
    v1.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _apply(root, fresh_db)
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        rows = conn.execute(
            "SELECT version, status, structured_payload->>'legacy_status' "
            "FROM canonical_records ORDER BY version"
        ).fetchall()
    assert rows == [(1, "archived", "draft"), (2, "draft", None)]


def test_in_progress_idempotency_claim_is_skipped_and_counted(tmp_path, fresh_db):
    root = tmp_path / "root"
    _write_json(root / "idempotency" / f"{'cd' * 32}.json", {
        "state": "in_progress", "actor_id": "alice", "operation": "publish",
        "request_fingerprint": "ef" * 32, "created_at": "2026-01-06T00:00:00+00:00",
    })
    report = _apply(root, fresh_db)
    assert report["skipped"] == {"idempotency": 1}
    assert _counts(fresh_db)["idempotency_keys"] == 0


def test_identities_file_is_byte_identical_after_a_normal_real_run(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    path = root / "identity" / "identities.json"
    before = path.read_bytes()
    _apply(root, fresh_db)
    assert path.read_bytes() == before


def test_cli_real_run_writes_report_and_scrub_rewrites_identities_atomically(tmp_path, fresh_db, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    root = tmp_path / "root"
    _full_fixture(root)
    report_path = tmp_path / "out" / "r.json"
    code = migrate.main([
        "--confirm-frozen", "--scrub-identities", "--report", str(report_path), "--root", str(root),
        "--database-url", fresh_db,
    ])
    assert code == 0
    text = report_path.read_text(encoding="utf-8")
    assert _PLAIN_ALICE not in text and _PLAIN_BOB not in text
    assert json.loads(text)["dry_run"] is False

    identities = root / "identity" / "identities.json"
    assert not (root / "identity" / "identities.json.tmp").exists()
    doc = json.loads(identities.read_text(encoding="utf-8"))
    by_actor = {e["actor_id"]: e for e in doc["entries"]}
    assert by_actor["alice"]["credential"] == "scrubbed:" + hashlib.sha256(_PLAIN_ALICE.encode("utf-8")).hexdigest()
    assert by_actor["bob"]["credential"] == "scrubbed:" + hashlib.sha256(_PLAIN_BOB.encode("utf-8")).hexdigest()
    assert by_actor["alice"]["password_hash"] == "scrypt$16384$8$1$c2FsdA==$aGFzaA=="
    assert _PLAIN_ALICE not in identities.read_text(encoding="utf-8")


def test_no_plaintext_token_in_report_or_exception(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    report_text = json.dumps(_apply(root, fresh_db), default=str)
    assert _PLAIN_ALICE not in report_text and _PLAIN_BOB not in report_text

    dup_root = tmp_path / "dup"
    _identity_fixture(dup_root)
    path = dup_root / "identity" / "identities.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    extra = dict(doc["entries"][0])
    extra["subject_id"] = str(uuid.uuid4())
    doc["entries"].append(extra)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(MigrationError) as exc:
        _apply(dup_root, fresh_db)
    assert _PLAIN_ALICE not in str(exc.value) and _PLAIN_BOB not in str(exc.value)


# ── Fail-closed exit, redaction, audit chain guard, scrub durability ─────


def _hold_audit_lock(url: str):
    conn = psycopg.connect(_conninfo(url), autocommit=True)
    conn.execute("SELECT pg_advisory_lock(hashtext('audit_chain'))")
    return conn


def _insert_app_audit_row(url: str) -> None:
    with psycopg.connect(_conninfo(url), autocommit=True) as conn:
        conn.execute(
            "INSERT INTO audit_events (event_id, aggregate_type, aggregate_id, actor_id, role, action, entry_hash) "
            "VALUES (%s, 'app', 'x', 'app', 'editor', 'publish', %s)",
            (str(uuid.uuid4()), b"\x00" * 32),
        )


def test_scrub_keeps_the_original_file_mode(tmp_path):
    root = tmp_path / "root"
    _identity_fixture(root)
    path = root / "identity" / "identities.json"
    os.chmod(path, 0o600)
    assert migrate.scrub_identities_file(root) == 2
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_scrub_fsyncs_the_tmp_file(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _identity_fixture(root)
    synced: list[int] = []
    real_fsync = os.fsync

    def spy(fd):
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(migrate.os, "fsync", spy)
    assert migrate.scrub_identities_file(root) == 2
    assert synced


def test_failed_scrub_leaves_identities_byte_identical_and_says_data_is_committed(
    tmp_path, fresh_db, capsys, monkeypatch
):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    root = tmp_path / "root"
    _full_fixture(root)
    path = root / "identity" / "identities.json"
    before = path.read_bytes()
    (root / "identity" / "identities.json.tmp").mkdir()  # blocks the tmp write
    code = migrate.main([
        "--confirm-frozen", "--scrub-identities", "--report", str(tmp_path / "out" / "r.json"),
        "--root", str(root), "--database-url", fresh_db,
    ])
    assert code == 2
    assert path.read_bytes() == before
    assert "data committed; identities scrub failed - re-run is safe" in capsys.readouterr().err
    assert _counts(fresh_db)["admin_identities"] == 2


def test_write_error_text_names_sqlstate_and_table_not_row_values(fresh_db):
    marker = "MARKER-ROWVALUE-7f3a"
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        with conn.cursor() as cur:
            with pytest.raises(MigrationError) as exc:
                migrate._write(
                    cur, migrate._Tally(NOW), "outbox", "reindex_outbox",
                    {"event_id": marker}, "(event_id)", "k1",
                )
    text = str(exc.value)
    assert marker not in text
    assert "22P02" in text
    assert "reindex_outbox" in text


def test_main_prints_reduced_text_for_a_psycopg_error(tmp_path, fresh_db, capsys, monkeypatch):
    marker = "MARKER-PGERR-91c2"
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        try:
            conn.execute("SELECT %s::uuid", (marker,))
        except psycopg.Error as exc:
            pg_error = exc

    def fail(*args, **kwargs):
        raise pg_error

    monkeypatch.setattr(migrate, "run_real_run", fail)
    root = tmp_path / "root"
    _multi_store(root)
    code = migrate.main([
        "--confirm-frozen", "--report", str(tmp_path / "r.json"), "--root", str(root),
        "--database-url", fresh_db,
    ])
    err = capsys.readouterr().err
    assert code == 2
    assert marker not in err
    assert "22P02" in err


def test_in_progress_claim_is_skipped_without_a_mismatch_and_exits_0(tmp_path, fresh_db):
    root = tmp_path / "root"
    _write_json(root / "idempotency" / f"{'cd' * 32}.json", {
        "state": "in_progress", "actor_id": "alice", "operation": "publish",
        "request_fingerprint": "ef" * 32, "created_at": "2026-01-06T00:00:00+00:00",
    })
    report_path = tmp_path / "r.json"
    code = migrate.main([
        "--confirm-frozen", "--report", str(report_path), "--root", str(root),
        "--database-url", fresh_db,
    ])
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    idem = _stores(report)["idempotency"]
    assert (idem["source_rows"], idem["target_rows"], idem["mismatch"]) == (1, 0, 0)
    assert report["skipped"] == {"idempotency": 1}


def test_silent_on_conflict_skip_exits_2_and_names_the_store(tmp_path, fresh_db, capsys):
    root = tmp_path / "root"
    same = {
        "trace_id": "trace-dup", "helpful": True, "reason": None,
        "answered_at": "2026-01-07T00:00:00+00:00", "linked_item": False,
    }
    _write_lines(root / "unanswered" / "feedback.jsonl", [same, dict(same)])
    report_path = tmp_path / "r.json"
    code = migrate.main([
        "--confirm-frozen", "--report", str(report_path), "--root", str(root),
        "--database-url", fresh_db,
    ])
    assert code == 2
    assert "feedback" in capsys.readouterr().err
    assert report_path.exists()  # the report is still written
    assert _stores(json.loads(report_path.read_text(encoding="utf-8")))["feedback"]["mismatch"] == 1


def test_audit_rerun_refuses_app_rows_outside_the_ledger_and_rolls_back(tmp_path, fresh_db):
    root = tmp_path / "root"
    _full_fixture(root)
    _apply(root, fresh_db)
    _insert_app_audit_row(fresh_db)
    counts = _counts(fresh_db)
    with pytest.raises(MigrationError) as exc:
        _apply(root, fresh_db)
    assert "audit_events already has rows not in the ledger" in str(exc.value)
    assert _counts(fresh_db) == counts


def test_audit_lock_timeout_is_a_clear_error_and_rolls_back_every_store(tmp_path, fresh_db, monkeypatch):
    monkeypatch.setattr(migrate, "_AUDIT_LOCK_TIMEOUT", "500ms", raising=False)
    root = tmp_path / "root"
    _full_fixture(root)
    holder = _hold_audit_lock(fresh_db)
    outcome: dict[str, BaseException] = {}

    def run() -> None:
        try:
            _apply(root, fresh_db)
        except BaseException as exc:  # captured for the assertions below
            outcome["exc"] = exc

    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        worker.join(60)
        assert not worker.is_alive(), "apply blocked on the audit lock with no timeout"
    finally:
        holder.execute("SELECT pg_advisory_unlock(hashtext('audit_chain'))")
        holder.close()
    worker.join(60)
    exc = outcome.get("exc")
    assert isinstance(exc, MigrationError), exc
    assert "audit" in str(exc) and "lock" in str(exc)
    assert all(count == 0 for count in _counts(fresh_db).values())


def test_published_record_with_dangling_source_raises_and_rolls_back(tmp_path, fresh_db):
    # Deliberate deviation from the brief: the brief asked for the source FK to be nulled,
    # but DDL CHECK (status <> 'published' OR source_document_id IS NOT NULL) forbids NULL,
    # so a published record that cites a missing source document aborts the run.
    root = tmp_path / "root"
    _full_fixture(root)
    CanonicalRecordStore(root).write_record(
        _rec("CHI_TIEU:nganh:7220104:2026:published").model_copy(update={
            "status": RecordStatus.PUBLISHED, "published_at": NOW, "source_document_id": "doc-missing",
        })
    )
    with pytest.raises(MigrationError) as exc:
        _apply(root, fresh_db)
    assert "records" in str(exc.value)
    assert "published" in str(exc.value)
    assert all(count == 0 for count in _counts(fresh_db).values())


def test_defaulted_sentinels_cover_each_filled_gap(tmp_path, fresh_db, monkeypatch):
    root = tmp_path / "root"
    _full_fixture(root)
    doc_path = root / "documents" / "doc-1_d.json"
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    for key in ("registered_by", "registered_at", "updated_at"):
        doc.pop(key)
    doc_path.write_text(json.dumps(doc), encoding="utf-8")
    _write_json(root / "supersession_retractions" / "d1" / "r1.json", {
        "target_record_id": RECORD_KEY,
        "superseding_version_key": RECORD_KEY + "@2",
        "reason": "short",
    })
    real_load = migrate.load_retractions

    def with_known_superseded_version(root_arg):
        rows = real_load(root_arg)
        for row in rows:
            row["superseded_version"] = 1
        return rows

    monkeypatch.setattr(migrate, "load_retractions", with_known_superseded_version)
    report = _apply(root, fresh_db)
    defaulted = report["defaulted"]
    assert defaulted["source_documents"]["uploaded_by"] == 1
    assert defaulted["source_documents"]["created_at"] == 1
    assert defaulted["source_documents"]["updated_at"] == 1
    assert defaulted["records"]["authored_by"] == 1
    assert defaulted["records"]["created_at"] == 1
    assert defaulted["releases"]["created_by"] == 2
    assert defaulted["identities"]["created_by"] == 2
    assert defaulted["retractions"]["revoked_by"] == 1
    assert defaulted["retractions"]["revoked_at"] == 1
    assert defaulted["retractions"]["reason"] == 1
    assert "feedback_id" not in defaulted.get("feedback", {})
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        assert conn.execute("SELECT uploaded_by FROM source_documents").fetchone()[0] == "legacy-migration"
        assert conn.execute("SELECT authored_by FROM canonical_records").fetchone()[0] == "legacy-migration"
        revoked_by, reason = conn.execute(
            "SELECT revoked_by, reason FROM supersession_retractions"
        ).fetchone()
    assert revoked_by == "legacy-migration"
    assert reason == "legacy migration: short"
    assert len(reason) >= 10


def _row(record_id: str, version: int, status: str | None) -> dict:
    return {"record_id": record_id, "version": version, "status": status}


def test_normalise_keeps_the_newest_open_and_newest_published_version_per_record():
    rows = [
        _row("A", 1, "draft"),
        _row("A", 2, "review"),
        _row("A", 3, "draft"),  # below the newest published: superseded
        _row("A", 4, "published"),
        _row("A", 5, "published"),  # older published: superseded
        _row("A", 6, "published"),  # newest published: kept
        _row("B", 1, "published"),
        _row("B", 2, None),  # status None is a draft after defaulting; above the published: kept
        _row("C", 1, "archived"),  # untouched
        _row("E", 1, "published"),
        _row("E", 2, "draft"),  # open above the newest published: kept
    ]
    out, count = migrate._supersede_extra_open_and_published(rows)
    statuses = {(r["record_id"], r["version"]): r["status"] for r in out}
    assert statuses == {
        ("A", 1): "archived",
        ("A", 2): "archived",
        ("A", 3): "archived",
        ("A", 4): "archived",
        ("A", 5): "archived",
        ("A", 6): "published",
        ("B", 1): "published",
        ("B", 2): None,
        ("C", 1): "archived",
        ("E", 1): "published",
        ("E", 2): "draft",
    }
    assert count == 5
    superseded = [r for r in out if r.get("_superseded")]
    assert {(r["record_id"], r["version"]) for r in superseded} == {
        ("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5),
    }
    assert {(r["record_id"], r["version"]): r["_legacy_status"] for r in superseded} == {
        ("A", 1): "draft", ("A", 2): "review", ("A", 3): "draft",
        ("A", 4): "published", ("A", 5): "published",
    }
    assert [r for r in rows if "_superseded" in r] == []  # input not mutated


def test_normalise_archives_every_open_version_below_the_newest_published_in_a_chain():
    rows = [
        _row("A", 1, "draft"),
        _row("A", 2, "review"),
        _row("A", 3, "approved"),
        _row("A", 4, "published"),
    ]
    out, count = migrate._supersede_extra_open_and_published(rows)
    assert [(r["version"], r["status"]) for r in out] == [
        (1, "archived"), (2, "archived"), (3, "archived"), (4, "published"),
    ]
    assert count == 3


def test_normalise_is_a_no_op_for_a_clean_chain():
    rows = [_row("A", 1, "archived"), _row("A", 2, "published"), _row("A", 3, "draft")]
    out, count = migrate._supersede_extra_open_and_published(rows)
    assert out == rows
    assert count == 0


def test_normalise_report_keys_are_capped_and_counted():
    rows = [_row("A", v, "draft") for v in range(1, 252)]
    keys = migrate._normalised_report([f"A@{v}" for v in range(1, 252)])
    assert keys["records"] == 251
    assert len(keys["keys"]) == 200
    assert keys["keys"][0] == "A@1"
    superseded = migrate._superseded_keys(migrate._supersede_extra_open_and_published(rows)[0])  # 251 open drafts: only the newest is kept
    assert len(superseded) == 250
    assert "A@251" not in superseded


def test_normalise_chain_writes_one_current_version_with_legacy_status(tmp_path, fresh_db):
    root = tmp_path / "root"
    _write_json(root / "documents" / "doc-1_d.json", {
        "document_id": "doc-1", "sha256": "cd" * 32, "path": "uploads/tuyensinh2026.pdf",
        "state": "registered", "registered_by": "alice",
        "registered_at": "2026-01-02T03:04:05", "updated_at": "2026-01-03T00:00:00+00:00",
    })
    base = _rec("CHI_TIEU:nganh:7220104:2026:chain")
    store = CanonicalRecordStore(root)
    store.write_record(base)  # v1 draft
    store.write_record(base.model_copy(update={"version": 2, "status": RecordStatus.REVIEW}))
    store.write_record(base.model_copy(update={"version": 3, "status": RecordStatus.APPROVED}))
    store.write_record(base.model_copy(update={
        "version": 4, "status": RecordStatus.PUBLISHED, "source_document_id": "doc-1",
        "published_at": datetime(2026, 2, 1, tzinfo=timezone.utc),
    }))

    dry = migrate.build_report(root)
    assert dry["normalised"]["records"] == 3

    report = _apply(root, fresh_db)
    for name, s in _stores(report).items():
        assert s["mismatch"] == 0, (name, s)
    assert report["normalised"]["records"] == 3
    assert [k.rsplit("@", 1)[1] for k in report["normalised"]["keys"]] == ["1", "2", "3"]
    with psycopg.connect(_conninfo(fresh_db)) as conn:
        rows = conn.execute(
            "SELECT version, status, archive_reason, structured_payload->>'legacy_status' "
            "FROM canonical_records ORDER BY version"
        ).fetchall()
    assert rows == [
        (1, "archived", "superseded", "draft"),
        (2, "archived", "superseded", "review"),
        (3, "archived", "superseded", "approved"),
        (4, "published", None, None),
    ]
