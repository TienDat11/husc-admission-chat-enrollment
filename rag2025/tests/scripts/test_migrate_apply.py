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
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest
from sqlalchemy.engine import make_url

import scripts.migrate_file_stores as migrate
from scripts.migrate_file_stores import MigrationError
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from tests.infrastructure.conftest import run_alembic
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
    """One legacy file (or more) for each of the 11 loader stores."""
    _multi_store(root)  # record (source doc-1), unanswered question, identities, 3 audit events
    _write_json(root / "documents" / "doc-1_d.json", {
        "document_id": "doc-1", "sha256": "cd" * 32, "path": "uploads/tuyensinh2026.pdf",
        "state": "registered", "registered_by": "alice",
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


@pytest.fixture
def fresh_db():
    """Yields a psycopg-form sync URL of a brand-new migrated database."""
    base = os.getenv("DATABASE_URL", "").strip()
    if not base:
        pytest.skip("DATABASE_URL not set; PostgreSQL apply tests skipped")
    admin_url = _conninfo(make_url(migrate.sync_database_url(base)).set(database="postgres").render_as_string(hide_password=False))
    try:
        admin = psycopg.connect(admin_url, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable at DATABASE_URL: {exc}")
    name = "a3apply_" + uuid.uuid4().hex[:12]
    with admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    target = make_url(base).set(database=name)
    run_alembic(target.render_as_string(hide_password=False), "upgrade", "head")
    try:
        yield migrate.sync_database_url(target.render_as_string(hide_password=False))
    finally:
        with psycopg.connect(admin_url, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


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
        assert s["inserted"] == s["source_rows"], (name, s)
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
