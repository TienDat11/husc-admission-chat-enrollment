"""Task 2a (U3): coordinate-key derivation, records loader, retractions loader.

Fixtures are written with the real CanonicalRecordStore so the on-disk shape is
the real one, not an invented one. Malformed files are produced by editing the
real bytes after the store wrote them.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pytest

from scripts.migrate_file_stores import (
    MigrationError,
    derive_coordinate_key,
    load_audit,
    load_identities,
    load_records,
    load_retractions,
    load_source_documents,
    load_unanswered,
    split_version_key,
)
from src.domain.admin_identity import Role
from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.domain.canonical.record import (
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from src.infrastructure.identity_store import IdentityEntry, IdentityStore
from src.infrastructure.source_document_registry import SourceDocumentRegistry


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _rec(
    record_id: str,
    *,
    record_type: RecordType = RecordType.CHI_TIEU,
    year: int = 2026,
    version: int = 1,
    round_: str | None = None,
    status: RecordStatus = RecordStatus.DRAFT,
    published_at: datetime | None = None,
    methods: tuple[str, ...] = (),
    **extra: object,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=record_id,
        record_type=record_type,
        version=version,
        admission_year=year,
        effective_from=date(year, 1, 1),
        status=status,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="doc-1",
        published_at=published_at,
        admission_round=round_,
        admission_methods=list(methods),
        text="noi dung",
        **extra,
    )


def _store(tmp_path: Path) -> tuple[CanonicalRecordStore, Path]:
    root = tmp_path / "store"
    return CanonicalRecordStore(root), root


def _version_path(root: Path, legacy: str, version: int) -> Path:
    return root / "records" / f"rec-{quote(legacy, safe='')}_r" / f"v{version:04d}.json"


def _edit(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _payload(record_id: str, **overrides) -> dict:
    base = {
        "record_id": record_id,
        "record_type": "CHI_TIEU",
        "admission_year": 2026,
        "admission_round": None,
        "admission_methods": [],
    }
    base.update(overrides)
    return base


# ── derive_coordinate_key ────────────────────────────────────────────────


def test_five_part_id_gives_all_round_and_all_methods():
    key = derive_coordinate_key(_payload("CHI_TIEU:nganh:7220104:2026:canon-lietke29"))
    assert key == "CHI_TIEU:nganh:7220104:2026:all:all"


def test_six_part_id_hashes_its_method_token():
    key = derive_coordinate_key(
        _payload(
            "DIEM_CHUAN:nganh:7220104:2024:hoc_ba:ttts-p61",
            record_type="DIEM_CHUAN",
            admission_year=2024,
        )
    )
    assert key == f"DIEM_CHUAN:nganh:7220104:2024:all:{_sha12('hoc_ba')}"


def test_nonempty_admission_methods_are_sorted_and_hashed():
    key = derive_coordinate_key(
        _payload(
            "CHI_TIEU:nganh:7220104:2026:x",
            admission_methods=["xet_tuyen", "hoc_ba"],
        )
    )
    assert key == f"CHI_TIEU:nganh:7220104:2026:all:{_sha12('hoc_ba|xet_tuyen')}"


def test_non_empty_admission_round_is_reflected_in_the_key():
    key = derive_coordinate_key(
        _payload("CHI_TIEU:nganh:7220104:2026:x", admission_round="1")
    )
    assert key == "CHI_TIEU:nganh:7220104:2026:1:all"


@pytest.mark.parametrize(
    "payload",
    [
        _payload("CHI_TIEU:nganh:7220104:2025:x"),  # year differs from payload
        _payload("CHI_TIEU:nganh::2026:x"),  # empty subject
        _payload("CHI_TIEU:nganh:7220104:abc:x"),  # unparseable year
        _payload("CHI_TIEU"),  # too few parts
        {k: v for k, v in _payload("CHI_TIEU:nganh:7220104:2026:x").items() if k != "record_id"},
    ],
)
def test_missing_or_inconsistent_coordinate_gives_none(payload):
    assert derive_coordinate_key(payload) is None


# ── records loader ───────────────────────────────────────────────────────


def test_clean_record_takes_coordinate_key_and_keeps_legacy_id(tmp_path):
    store, root = _store(tmp_path)
    legacy = "DIEM_CHUAN:nganh:7220104:2026:hoc_ba:ttts-p61"
    store.write_record(_rec(legacy, record_type=RecordType.DIEM_CHUAN))

    out = load_records(root)

    key = f"DIEM_CHUAN:nganh:7220104:2026:all:{_sha12('hoc_ba')}"
    assert len(out.rows) == 1
    row = out.rows[0]
    assert row["record_id"] == key
    assert row["coordinate_key"] == key
    assert row["status"] == "draft"
    assert row["structured_payload"]["legacy_record_id"] == legacy
    assert out.id_map == {legacy: key}
    assert out.quarantined_rows == 0
    assert len(out.rows) == out.source_rows


def test_record_with_mismatched_year_is_unanchored_and_counted(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2025:x"
    store.write_record(_rec(legacy, year=2026))

    out = load_records(root)

    assert len(out.rows) == out.source_rows == 1
    row = out.rows[0]
    assert row["record_id"] == f"unanchored:{legacy}"
    assert row["coordinate_key"] == row["record_id"]
    assert row["status"] == "draft"
    assert "UNANCHORED_DRAFT" in row["structured_payload"]["blockers"]
    assert out.quarantined_rows == 1
    assert out.id_map == {legacy: f"unanchored:{legacy}"}


def test_published_without_source_document_is_unanchored(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:canon-a"
    store.write_record(_rec(legacy))
    # The store refuses a published record without provenance, so the stored
    # bytes are edited directly to simulate the legacy shape.
    _edit(
        _version_path(root, legacy, 1),
        lambda p: p.update(status="published", source_document_id=None),
    )

    out = load_records(root)

    assert out.rows[0]["record_id"] == f"unanchored:{legacy}"
    assert out.rows[0]["status"] == "draft"
    assert out.quarantined_rows == 1
    assert len(out.rows) == out.source_rows == 1


def test_duplicate_drafts_keep_first_sorted_and_quarantine_the_rest(tmp_path):
    store, root = _store(tmp_path)
    first = "CHI_TIEU:nganh:7220104:2026:canon-a"
    second = "CHI_TIEU:nganh:7220104:2026:canon-b"
    store.write_record(_rec(first))
    store.write_record(_rec(second))
    key = "CHI_TIEU:nganh:7220104:2026:all:all"

    out = load_records(root)

    by_legacy = {r["structured_payload"]["legacy_record_id"]: r for r in out.rows}
    assert by_legacy[first]["record_id"] == key
    assert by_legacy[second]["record_id"] == f"unanchored:{second}"
    assert out.quarantined_rows == 1
    assert len(out.rows) == out.source_rows == 2


def test_published_beats_draft_on_the_same_coordinate(tmp_path):
    store, root = _store(tmp_path)
    draft = "CHI_TIEU:nganh:7220104:2026:canon-a"  # sorts first
    published = "CHI_TIEU:nganh:7220104:2026:canon-b"
    store.write_record(_rec(draft))
    store.write_record(
        _rec(
            published,
            status=RecordStatus.PUBLISHED,
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    key = "CHI_TIEU:nganh:7220104:2026:all:all"

    out = load_records(root)

    by_legacy = {r["structured_payload"]["legacy_record_id"]: r for r in out.rows}
    assert by_legacy[published]["record_id"] == key
    assert by_legacy[published]["status"] == "published"
    assert by_legacy[draft]["record_id"] == f"unanchored:{draft}"
    assert out.quarantined_rows == 1


def test_latest_published_at_wins_among_published_duplicates(tmp_path):
    store, root = _store(tmp_path)
    older = "CHI_TIEU:nganh:7220104:2026:canon-a"  # sorts first
    newer = "CHI_TIEU:nganh:7220104:2026:canon-b"
    store.write_record(
        _rec(
            older,
            status=RecordStatus.PUBLISHED,
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    store.write_record(
        _rec(
            newer,
            status=RecordStatus.PUBLISHED,
            published_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
    )
    key = "CHI_TIEU:nganh:7220104:2026:all:all"

    out = load_records(root)

    by_legacy = {r["structured_payload"]["legacy_record_id"]: r for r in out.rows}
    assert by_legacy[newer]["record_id"] == key
    assert by_legacy[older]["record_id"] == f"unanchored:{older}"
    assert out.quarantined_rows == 1
    assert len(out.rows) == out.source_rows == 2


def test_corrupt_record_file_raises_migration_error_naming_the_path(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:canon-a"
    store.write_record(_rec(legacy))
    bad = _version_path(root, legacy, 1)
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(MigrationError) as info:
        load_records(root)

    assert str(bad) in str(info.value)


# ── per-legacy-id decision (I-1) ─────────────────────────────────────────


def test_multi_version_split_quarantines_all_versions_of_the_legacy_id(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(_rec(legacy))  # v1 draft, anchored on its own
    store.write_record(_rec(legacy, version=2))  # v2 draft, then made published
    _edit(
        _version_path(root, legacy, 2),
        lambda p: p.update(status="published", source_document_id=""),
    )

    out = load_records(root)

    assert [r["record_id"] for r in out.rows] == [f"unanchored:{legacy}"] * 2
    assert [r["version"] for r in out.rows] == [1, 2]
    assert out.id_map == {legacy: f"unanchored:{legacy}"}
    assert out.quarantined_rows == 2
    assert len(out.rows) == out.source_rows == 2


def test_versions_with_different_round_use_the_lowest_version_key(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(_rec(legacy))  # v1: round None -> "all"
    store.write_record(_rec(legacy, version=2, round_="1"))  # v2: round "1"

    out = load_records(root)

    key = "CHI_TIEU:nganh:7220104:2026:all:all"
    assert [r["record_id"] for r in out.rows] == [key, key]
    assert [r["coordinate_key"] for r in out.rows] == [key, key]
    # Key columns of every row agree with the emitted coordinate_key, not with
    # the version's own payload.
    for row in out.rows:
        assert row["record_type"] == "CHI_TIEU"
        assert row["scope"] == "nganh"
        assert row["subject"] == "7220104"
        assert row["admission_year"] == 2026
        assert row["admission_round"] == "all"
        assert row["methods_hash"] == "all"
    assert out.id_map == {legacy: key}
    assert out.quarantined_rows == 0


def test_two_files_with_the_same_legacy_version_raise(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(_rec(legacy))
    v1 = _version_path(root, legacy, 1)
    (v1.parent / "v1.json").write_bytes(v1.read_bytes())  # same version, other name

    with pytest.raises(MigrationError):
        load_records(root)


def test_two_directories_with_the_same_record_id_raise(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(_rec(legacy))
    other = root / "records" / "rec-copy_r"
    other.mkdir(parents=True)
    (other / "v0001.json").write_bytes(_version_path(root, legacy, 1).read_bytes())

    with pytest.raises(MigrationError):
        load_records(root)


def test_load_records_twice_gives_identical_output(tmp_path):
    store, root = _store(tmp_path)
    store.write_record(_rec("CHI_TIEU:nganh:7220104:2026:canon-a"))
    store.write_record(_rec("CHI_TIEU:nganh:7220104:2026:canon-b"))
    store.write_record(_rec("CHI_TIEU:nganh:7220104:2025:c", year=2026))

    first = load_records(root)
    second = load_records(root)

    assert first.rows == second.rows
    assert first.id_map == second.id_map
    assert first.quarantined_rows == second.quarantined_rows


# ── malformed-but-valid JSON wraps to MigrationError (I-2) ───────────────


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(record_id=123),
        lambda p: p.update(version=None),
        lambda p: p.update(version="1"),
        lambda p: p.update(admission_methods=[1, "a"]),
        lambda p: p.update(status="published", source_document_id=5),
        lambda p: p.update(status="published", source_document_id=["x"]),
    ],
    ids=[
        "record_id-int",
        "version-None",
        "version-str",
        "methods-mixed-types",
        "source_document_id-int",
        "source_document_id-list",
    ],
)
def test_malformed_record_payload_raises_migration_error_naming_path(tmp_path, mutate):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(_rec(legacy))
    bad = _version_path(root, legacy, 1)
    _edit(bad, mutate)

    with pytest.raises(MigrationError) as info:
        load_records(root)

    assert str(bad) in str(info.value)


def test_record_file_that_is_a_json_list_raises_migration_error(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(_rec(legacy))
    bad = _version_path(root, legacy, 1)
    bad.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(MigrationError) as info:
        load_records(root)

    assert str(bad) in str(info.value)


# ── legacy extras and blockers (E) ───────────────────────────────────────


def test_unmapped_legacy_keys_are_kept_under_legacy_extra(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:x"
    store.write_record(
        _rec(
            legacy,
            values={"kind": "x"},
            extractor="pdf_milestone",
            seed_version=3,
            raw_source_text="nguyen van",
        )
    )

    out = load_records(root)

    structured = out.rows[0]["structured_payload"]
    assert structured["values"] == {"kind": "x"}
    assert structured["legacy_record_id"] == legacy
    extra = structured["legacy_extra"]
    assert extra["extractor"] == "pdf_milestone"
    assert extra["seed_version"] == 3
    assert extra["raw_source_text"] == "nguyen van"
    assert "values" not in extra
    assert "legacy_record_id" not in extra


# ── retractions loader ───────────────────────────────────────────────────


def test_split_version_key_parses_record_and_version():
    assert split_version_key("a@3") == ("a", 3)
    assert split_version_key("CHI_TIEU:nganh:7220104:2026:x") == (
        "CHI_TIEU:nganh:7220104:2026:x",
        None,
    )


def test_retraction_loader_splits_superseding_version_key(tmp_path):
    store, root = _store(tmp_path)
    store.retract_supersession(
        "old-rec",
        "a@3",
        reason="superseded in error by a@3",
        retracted_by="admin",
        at="2026-02-01T00:00:00Z",
    )

    rows = load_retractions(root)

    assert len(rows) == 1
    row = rows[0]
    assert row["target_record_id"] == "old-rec"
    assert row["superseding_record_id"] == "a"
    assert row["superseding_version"] == 3
    assert row["revoked_by"] == "admin"
    assert row["revoked_at"] == "2026-02-01T00:00:00Z"


def test_bare_id_retraction_has_no_superseding_version(tmp_path):
    store, root = _store(tmp_path)
    store.retract_supersession(
        "old-rec",
        "a",
        reason="superseded in error by a",
        retracted_by="admin",
        at="2026-02-01T00:00:00Z",
    )

    rows = load_retractions(root)

    assert rows[0]["superseding_record_id"] == "a"
    assert rows[0]["superseding_version"] is None


def _raw_retraction(root: Path, text: str) -> Path:
    path = root / "supersession_retractions" / "rec-old_r" / "claim.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_corrupt_retraction_file_raises_migration_error_naming_path(tmp_path):
    _, root = _store(tmp_path)
    bad = _raw_retraction(root, "{not json")

    with pytest.raises(MigrationError) as info:
        load_retractions(root)

    assert str(bad) in str(info.value)


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"target_record_id": "t"}',
        '{"target_record_id": "t", "superseding_version_key": 5}',
        '{"superseding_version_key": "a@3"}',
        "[1, 2]",
    ],
    ids=["empty", "missing-claim", "int-claim", "missing-target", "json-list"],
)
def test_malformed_retraction_payload_raises_migration_error_naming_path(tmp_path, text):
    _, root = _store(tmp_path)
    bad = _raw_retraction(root, text)

    with pytest.raises(MigrationError) as info:
        load_retractions(root)

    assert str(bad) in str(info.value)


# ---- Task 2b-1 (U3): releases, outbox, conflicts, idempotency, unanswered, feedback ----
# Fixtures write the on-disk shapes the real writers produce. Loaders are looked
# up on the module at call time so the 2a tests above are unaffected.
import scripts.migrate_file_stores as migrate_mod  # noqa: E402

CCCD = "012345678901"
EMAIL = "student@example.com"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _jsonl(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _manifest(release_id: str) -> dict:
    return {
        "release_id": release_id,
        "parent_release_id": None,
        "lancedb_table": "chunks",
        "lancedb_version": 3,
        "lancedb_tag": "tag",
        "graph_path": "graph/graph.graphml",
        "corpus_sha256": "a" * 64,
        "created_at": "2026-11-01T00:00:00+00:00",
        "chunk_count": 10,
        "document_count": 2,
    }


def _assert_error_names(fn, path, line=None):
    with pytest.raises(MigrationError) as excinfo:
        fn()
    msg = str(excinfo.value)
    assert str(path) in msg
    if line is not None:
        assert f"line {line}" in msg


def _outbox_row(n: int) -> dict:
    return {
        "entry_id": f"e{n}",
        "record_id": f"rec{n}",
        "version": n,
        "record_version_key": f"rec{n}:{n}",
        "affected_record_ids": [f"rec{n}"],
        "stamp": {"n": n},
        "appended_at": "2026-11-01T00:00:00+00:00",
        "release_id": None,
        "action": "publish",
    }


def _conflict_item(cid: str) -> dict:
    return {
        "conflict_id": cid,
        "record_type": "program",
        "ma_nganh": "7310",
        "admission_year": 2026,
        "field": "tuition",
        "statements": {},
        "origin": "seed_report",
        "status": "open",
        "resolution": None,
    }


def _idem_complete() -> dict:
    return {
        "state": "complete",
        "actor_id": "a1",
        "operation": "chat",
        "request_fingerprint": "f" * 64,
        "status_code": 200,
        "response": {"ok": True},
        "created_at": "2026-11-01T00:00:00+00:00",
        "completed_at": "2026-11-01T00:00:01+00:00",
    }


def _unanswered_row(n: int, question: str = "q") -> dict:
    return {
        "trace_id": f"t{n}",
        "question": question,
        "asked_at": "2026-11-01T00:00:00+00:00",
        "route": "rag",
        "status_code": "NO_ANSWER",
        "resolved_at": None,
        "resolved_by_record_id": None,
        "out_of_scope": False,
    }


def test_load_releases_reads_each_release_manifest(tmp_path):
    _write(tmp_path / "releases" / "REL1" / "release_manifest.json", json.dumps(_manifest("REL1")))
    _write(tmp_path / "releases" / "REL2" / "release_manifest.json", json.dumps(_manifest("REL2")))
    out = migrate_mod.load_releases(tmp_path)
    assert out.source_rows == 2
    assert len(out.rows) == 2
    assert [r["release_id"] for r in out.rows] == ["REL1", "REL2"]
    assert out.rows[0]["lancedb_table"] == "chunks"
    assert out.rows[0]["manifest"]["release_id"] == "REL1"


def test_load_releases_missing_dir_is_empty(tmp_path):
    out = migrate_mod.load_releases(tmp_path)
    assert (out.rows, out.source_rows, out.unfillable) == ([], 0, [])


def test_load_releases_corrupt_manifest_names_path(tmp_path):
    path = _write(tmp_path / "releases" / "REL1" / "release_manifest.json", "{not json")
    _assert_error_names(lambda: migrate_mod.load_releases(tmp_path), path)


def test_load_releases_not_null_gap_is_unfillable(tmp_path):
    _write(tmp_path / "releases" / "REL1" / "release_manifest.json", json.dumps(_manifest("REL1")))
    out = migrate_mod.load_releases(tmp_path)
    assert out.rows[0]["status"] is None
    assert {"legacy_id": "REL1", "column": "status"} in out.unfillable


def test_load_outbox_maps_entries_to_rows(tmp_path):
    _write(tmp_path / "reindex_outbox.jsonl", _jsonl([_outbox_row(1), _outbox_row(2)]))
    out = migrate_mod.load_outbox(tmp_path)
    assert out.source_rows == 2
    assert len(out.rows) == 2
    assert out.rows[1]["event_id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "husc-outbox:e2"))
    assert out.rows[1]["record_version"] == 2


def test_load_outbox_corrupt_line_names_path_and_line(tmp_path):
    path = _write(tmp_path / "reindex_outbox.jsonl", _jsonl([_outbox_row(1)]) + "{bad\n")
    _assert_error_names(lambda: migrate_mod.load_outbox(tmp_path), path, line=2)


def test_load_outbox_missing_file_is_empty(tmp_path):
    out = migrate_mod.load_outbox(tmp_path)
    assert (out.rows, out.source_rows) == ([], 0)


def test_load_conflicts_maps_items(tmp_path):
    payload = {"schema_version": 1, "items": [_conflict_item("c1"), _conflict_item("c2")]}
    _write(tmp_path / "conflicts" / "queue.json", json.dumps(payload))
    out = migrate_mod.load_conflicts(tmp_path)
    assert out.source_rows == 2
    assert [r["conflict_id"] for r in out.rows] == ["c1", "c2"]
    assert out.rows[0]["status"] == "open"


def test_load_conflicts_wrong_top_level_type_names_path(tmp_path):
    path = _write(tmp_path / "conflicts" / "queue.json", json.dumps([1, 2]))
    _assert_error_names(lambda: migrate_mod.load_conflicts(tmp_path), path)


def test_load_conflicts_not_null_gap_is_unfillable(tmp_path):
    payload = {"schema_version": 1, "items": [_conflict_item("c1")]}
    _write(tmp_path / "conflicts" / "queue.json", json.dumps(payload))
    out = migrate_mod.load_conflicts(tmp_path)
    assert out.rows[0]["coordinate_key"] is None
    assert {"legacy_id": "c1", "column": "coordinate_key"} in out.unfillable


def test_load_idempotency_maps_complete_and_in_progress(tmp_path):
    _write(tmp_path / "idempotency" / "aaa.json", json.dumps(_idem_complete()))
    in_progress = {
        "state": "in_progress",
        "actor_id": "a1",
        "operation": "chat",
        "request_fingerprint": "f" * 64,
        "created_at": "2026-11-01T00:00:00+00:00",
    }
    _write(tmp_path / "idempotency" / "bbb.json", json.dumps(in_progress))
    out = migrate_mod.load_idempotency(tmp_path)
    assert out.source_rows == 2
    assert len(out.rows) == 2
    assert {"legacy_id": "bbb", "column": "status_code"} in out.unfillable


def test_load_idempotency_corrupt_file_names_path(tmp_path):
    path = _write(tmp_path / "idempotency" / "bad.json", "{oops")
    _assert_error_names(lambda: migrate_mod.load_idempotency(tmp_path), path)


def test_load_unanswered_maps_lines(tmp_path):
    _write(tmp_path / "unanswered" / "queue.jsonl", _jsonl([_unanswered_row(1), _unanswered_row(2)]))
    out = migrate_mod.load_unanswered(tmp_path)
    assert out.source_rows == 2
    assert len(out.rows) == 2
    assert out.rows[0]["trace_id"] == "t1"
    assert out.rows[0]["route"] == "rag"


def test_load_unanswered_redacts_cccd_and_email(tmp_path):
    question = f"my cccd {CCCD} and mail {EMAIL}"
    _write(tmp_path / "unanswered" / "queue.jsonl", _jsonl([_unanswered_row(1, question)]))
    out = migrate_mod.load_unanswered(tmp_path)
    assert out.redacted_rows >= 1
    dumped = json.dumps(out.rows, ensure_ascii=False)
    assert CCCD not in dumped
    assert EMAIL not in dumped


def test_load_unanswered_corrupt_line_names_path_and_line(tmp_path):
    path = _write(tmp_path / "unanswered" / "queue.jsonl", _jsonl([_unanswered_row(1)]) + "{x\n")
    _assert_error_names(lambda: migrate_mod.load_unanswered(tmp_path), path, line=2)


def test_load_unanswered_is_deterministic(tmp_path):
    _write(tmp_path / "unanswered" / "queue.jsonl", _jsonl([_unanswered_row(1), _unanswered_row(2)]))
    assert migrate_mod.load_unanswered(tmp_path).rows == migrate_mod.load_unanswered(tmp_path).rows


def test_load_feedback_redacts_reason_and_flags_missing_pk(tmp_path):
    row = {
        "trace_id": "t1",
        "helpful": False,
        "reason": f"contact {EMAIL} cccd {CCCD}",
        "answered_at": "2026-11-01T00:00:00+00:00",
        "linked_item": True,
    }
    _write(tmp_path / "unanswered" / "feedback.jsonl", _jsonl([row]))
    out = migrate_mod.load_feedback(tmp_path)
    assert out.redacted_rows >= 1
    dumped = json.dumps(out.rows, ensure_ascii=False)
    assert CCCD not in dumped
    assert EMAIL not in dumped
    assert out.rows[0]["feedback_id"] is None
    assert any(u["column"] == "feedback_id" for u in out.unfillable)


# ---- Task 2b-1 fix round 1 ----
import subprocess  # noqa: E402
import sys  # noqa: E402
import uuid  # noqa: E402


def _conflict_file_with(tmp_path, item: dict) -> Path:
    return _write(
        tmp_path / "conflicts" / "queue.json",
        json.dumps({"schema_version": 1, "items": [item]}),
    )


def test_load_conflicts_redacts_resolution_reason_and_fills_override_fields(tmp_path):
    item = _conflict_item("c1")
    item["resolution"] = {
        "reason": f"officer note cccd {CCCD}",
        "resolved_by": "officer1",
        "resolved_by_role": "admin",
        "resolved_at": "2026-11-02T00:00:00+00:00",
        "event_id": "ev1",
    }
    _conflict_file_with(tmp_path, item)
    out = migrate_mod.load_conflicts(tmp_path)
    assert out.redacted_rows == 1
    assert CCCD not in json.dumps(out.rows, ensure_ascii=False)
    assert out.rows[0]["override_reason"] is not None
    assert out.rows[0]["resolved_by"] == "officer1"
    assert out.rows[0]["resolved_at"] == "2026-11-02T00:00:00+00:00"


def test_load_conflicts_keeps_legacy_item_minus_resolution_in_field_diffs(tmp_path):
    item = _conflict_item("c1")
    item["resolution"] = None
    _conflict_file_with(tmp_path, item)
    out = migrate_mod.load_conflicts(tmp_path)
    expected = {k: v for k, v in item.items() if k != "resolution"}
    assert out.rows[0]["field_diffs"] == {"legacy_item": expected}
    assert out.rows[0]["field_diffs"]["legacy_item"]["record_type"] == "program"


def test_load_conflicts_non_dict_resolution_names_path(tmp_path):
    item = _conflict_item("c1")
    item["resolution"] = [1]
    path = _conflict_file_with(tmp_path, item)
    _assert_error_names(lambda: migrate_mod.load_conflicts(tmp_path), path)


def test_load_outbox_event_id_is_deterministic_uuid_with_id_map(tmp_path):
    row = _outbox_row(1)
    row["entry_id"] = "REL1|rec1@1"
    _write(tmp_path / "reindex_outbox.jsonl", _jsonl([row]))
    first = migrate_mod.load_outbox(tmp_path)
    second = migrate_mod.load_outbox(tmp_path)
    assert first.rows[0]["event_id"] == second.rows[0]["event_id"]
    uuid.UUID(first.rows[0]["event_id"])
    assert first.id_map == {"REL1|rec1@1": first.rows[0]["event_id"]}


def test_load_unanswered_non_string_question_names_path_and_line(tmp_path):
    bad = json.dumps({"trace_id": "a", "question": 5}) + "\n"
    path = _write(tmp_path / "unanswered" / "queue.jsonl", _jsonl([_unanswered_row(1)]) + bad)
    _assert_error_names(lambda: migrate_mod.load_unanswered(tmp_path), path, line=2)


def test_load_feedback_non_string_reason_names_path_and_line(tmp_path):
    bad = json.dumps({"trace_id": "a", "helpful": True, "reason": 5, "answered_at": "x"}) + "\n"
    path = _write(tmp_path / "unanswered" / "feedback.jsonl", bad)
    _assert_error_names(lambda: migrate_mod.load_feedback(tmp_path), path, line=1)


def test_migrate_module_loads_without_src_on_sys_path():
    rag_root = Path(__file__).resolve().parents[2]
    script = (
        "import importlib.util, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "spec = importlib.util.spec_from_file_location('mfs', sys.argv[2])\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['mfs'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "mod.derive_coordinate_key({})\n"
        "assert not any(p.replace('\\\\', '/').endswith('/rag2025/src') for p in sys.path)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script, str(rag_root), str(rag_root / "scripts" / "migrate_file_stores.py")],
        capture_output=True,
        text=True,
        cwd=str(rag_root),
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Task 2b-2 (U3): source documents, identities, audit ledger loaders.
# Fixtures use the real writers (SourceDocumentRegistry, IdentityStore,
# AuditLedger); only the corruption cases edit bytes by hand.
# ---------------------------------------------------------------------------

_DOC_COLUMNS = {
    "document_id", "sha256", "kind", "storage_uri", "original_name", "page_count",
    "admission_year", "status", "uploaded_by", "created_at", "updated_at",
}
# admin_identities: created_at/updated_at are DB-defaulted (now()), so the loader
# leaves them out and lets the insert take the default.
_IDENTITY_COLUMNS = {
    "subject_id", "actor_id", "role", "password_hash", "header_token_hash",
    "disabled", "created_by",
}
_AUDIT_COLUMNS = {
    "event_id", "seq", "aggregate_type", "aggregate_id", "actor_id", "role",
    "action", "before", "after", "reason", "before_version", "after_version",
    "release_id", "request_id", "created_at", "prev_hash", "entry_hash",
}
_SHA_A = hashlib.sha256(b"doc-a").hexdigest()
_SHA_B = hashlib.sha256(b"doc-b").hexdigest()
_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _doc_fixture(root: Path) -> SourceDocumentRegistry:
    # The real registry only registers PDF ids, so the .docx and .jsonl#slug
    # entries are real-shaped copies of a stored entry with the id swapped.
    reg = SourceDocumentRegistry(root, root / "repo")
    reg.create("new_data/a.pdf", sha256=_SHA_A, file_present=True, actor_id="ops-1", now=_NOW)
    reg.create("new_data/d.pdf", sha256=None, file_present=False, actor_id="ops-1", now=_NOW)
    reg.create("new_data/r.pdf", sha256=_SHA_A, file_present=True, actor_id="ops-2", now=_NOW)
    reg.retire("new_data/r.pdf", actor_id="ops-2", now=_NOW)
    for new_id, name in (("new_data/b.docx", "legacy-b"), ("rag2025/data/c.jsonl#slug", "legacy-c")):
        payload = json.loads(reg.get("new_data/a.pdf").to_json())
        payload.update(document_id=new_id, path=new_id)
        (root / "documents" / f"doc-{name}_d.json").write_text(json.dumps(payload), encoding="utf-8")
    return reg


def test_source_documents_one_row_per_entry_in_document_id_order(tmp_path):
    _doc_fixture(tmp_path)
    result = load_source_documents(tmp_path)
    assert result.source_rows == 5
    assert [r["document_id"] for r in result.rows] == [
        "new_data/a.pdf", "new_data/b.docx", "new_data/d.pdf", "new_data/r.pdf",
        "rag2025/data/c.jsonl#slug",
    ]
    assert all(set(r) == _DOC_COLUMNS for r in result.rows)


def test_source_documents_field_mapping(tmp_path):
    reg = _doc_fixture(tmp_path)
    by_id = {r["document_id"]: r for r in load_source_documents(tmp_path).rows}
    a = by_id["new_data/a.pdf"]
    assert a["sha256"] == bytes.fromhex(_SHA_A)
    assert a["kind"] == "pdf"
    assert a["original_name"] == "a.pdf"
    assert a["storage_uri"] == reg.get("new_data/a.pdf").path
    assert a["status"] == "uploaded"
    assert a["uploaded_by"] == "ops-1"
    assert a["created_at"] == _NOW
    assert a["page_count"] is None and a["admission_year"] is None
    assert by_id["new_data/b.docx"]["kind"] == "docx"


def test_source_documents_retired_maps_to_archived(tmp_path):
    _doc_fixture(tmp_path)
    row = {r["document_id"]: r for r in load_source_documents(tmp_path).rows}["new_data/r.pdf"]
    assert row["status"] == "archived"
    assert row["kind"] == "pdf"
    assert row["uploaded_by"] == "ops-2"
    assert {r["status"] for r in load_source_documents(tmp_path).rows} == {"uploaded", "archived"}


def test_source_documents_unknown_kind_and_missing_sha_are_unfillable(tmp_path):
    _doc_fixture(tmp_path)
    result = load_source_documents(tmp_path)
    by_id = {r["document_id"]: r for r in result.rows}
    assert by_id["new_data/b.docx"]["kind"] == "docx"
    assert by_id["rag2025/data/c.jsonl#slug"]["kind"] is None
    assert by_id["rag2025/data/c.jsonl#slug"]["original_name"] == "c.jsonl"
    assert {"legacy_id": "rag2025/data/c.jsonl#slug", "column": "kind"} in result.unfillable
    assert by_id["new_data/d.pdf"]["sha256"] is None
    assert {"legacy_id": "new_data/d.pdf", "column": "sha256"} in result.unfillable


def test_source_documents_missing_store_is_empty(tmp_path):
    result = load_source_documents(tmp_path / "nothing-here")
    assert result.rows == [] and result.source_rows == 0


_PLAIN_ALICE = "alice-plaintext-credential-XYZ"
_PLAIN_BOB = "bob-plaintext-credential-QRS"
_ALICE_SUBJECT = "6f1c2a4e-9d3b-4c1a-8e2f-1a2b3c4d5e6f"
_PWHASH = "scrypt$16384$8$1$c2FsdA==$aGFzaA=="


def _identity_fixture(root: Path) -> None:
    IdentityStore(root).seed([
        IdentityEntry(
            subject_id=_ALICE_SUBJECT, actor_id="alice", role=Role.EDITOR,
            credential=_PLAIN_ALICE, password_hash=_PWHASH,
        ),
        IdentityEntry(
            subject_id="bob-legacy", actor_id="bob", role=Role.ADMIN,
            credential=_PLAIN_BOB, disabled=True,
        ),
    ])


def test_identities_field_mapping_and_digest(tmp_path):
    _identity_fixture(tmp_path)
    result = load_identities(tmp_path)
    assert result.source_rows == 2
    assert [r["actor_id"] for r in result.rows] == ["alice", "bob"]
    assert all(set(r) == _IDENTITY_COLUMNS for r in result.rows)
    alice, bob = result.rows
    assert alice["subject_id"] == _ALICE_SUBJECT
    assert alice["role"] == "editor"
    assert alice["password_hash"] == _PWHASH
    assert alice["header_token_hash"] == hashlib.sha256(_PLAIN_ALICE.encode("utf-8")).digest()
    assert alice["disabled"] is False
    assert bob["disabled"] is True
    assert bob["password_hash"] is None
    assert alice["created_by"] is None
    assert {"legacy_id": "alice", "column": "created_by"} in result.unfillable


def test_identities_never_carry_plaintext_credential(tmp_path):
    _identity_fixture(tmp_path)
    result = load_identities(tmp_path)
    assert _PLAIN_ALICE not in str(result.rows)
    assert _PLAIN_BOB not in str(result.rows)
    assert _PLAIN_ALICE not in repr(result) and _PLAIN_BOB not in repr(result)
    for row in result.rows:
        for value in row.values():
            assert _PLAIN_ALICE not in repr(value)
            assert _PLAIN_BOB not in repr(value)


def test_identities_non_uuid_subject_remapped_deterministically(tmp_path):
    _identity_fixture(tmp_path)
    first = load_identities(tmp_path)
    second = load_identities(tmp_path)
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "husc-identity:bob-legacy"))
    bob = next(r for r in first.rows if r["actor_id"] == "bob")
    assert bob["subject_id"] == expected
    assert uuid.UUID(bob["subject_id"]) is not None
    assert first.id_map["bob-legacy"] == expected
    assert [r["subject_id"] for r in second.rows] == [r["subject_id"] for r in first.rows]


def test_identities_missing_or_empty_store_is_empty(tmp_path):
    assert load_identities(tmp_path).rows == []
    path = IdentityStore(tmp_path).path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    result = load_identities(tmp_path)
    assert result.rows == [] and result.source_rows == 0


@pytest.mark.parametrize(
    "doc",
    [
        {"schema_version": "2.0", "entries": []},
        {"schema_version": "1.0", "entries": "not-a-list"},
        {"schema_version": "1.0", "entries": ["not-a-dict"]},
    ],
)
def test_identities_malformed_document_raises_naming_path(tmp_path, doc):
    path = IdentityStore(tmp_path).path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(MigrationError, match="identities.json"):
        load_identities(tmp_path)


def _event(n: int, **overrides: object) -> AuditEvent:
    values = dict(
        action=AuditAction.PUBLISH,
        actor_id="alice",
        role="editor",
        target=f"rec-{n}",
        target_kind=TargetKind.RELEASE,
        timestamp=datetime(2026, 1, n, tzinfo=timezone.utc),
        before_version=str(n),
        after_version=str(n + 1),
        reason=f"reason {n}",
        release_id="rel-1",
    )
    values.update(overrides)
    return AuditEvent(**values)


def _ledger_path(root: Path) -> Path:
    return root / "audit" / "ledger.jsonl"


def _ledger_fixture(root: Path) -> list:
    ledger = AuditLedger(_ledger_path(root))
    return [ledger.append(_event(n)) for n in (1, 2, 3)]


def _rewrite_ledger(root: Path, mutate) -> None:
    path = _ledger_path(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    mutate(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_audit_round_trip_keeps_hash_bytes(tmp_path):
    entries = _ledger_fixture(tmp_path)
    lines = [json.loads(x) for x in _ledger_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    result = load_audit(tmp_path)
    assert result.source_rows == 3
    assert all(set(r) == _AUDIT_COLUMNS for r in result.rows)
    assert [r["seq"] for r in result.rows] == [1, 2, 3]
    assert lines[0]["prev_hash"] == "0" * 64
    assert result.rows[0]["prev_hash"] is None
    for row, line, entry in zip(result.rows, lines, entries):
        assert row["entry_hash"] == bytes.fromhex(line["entry_hash"])
        assert row["entry_hash"] == bytes.fromhex(entry.entry_hash)
        assert row["event_id"] == str(uuid.UUID(hex=line["event"]["event_id"]))
    assert result.rows[1]["prev_hash"] == bytes.fromhex(lines[0]["entry_hash"])
    assert result.rows[2]["prev_hash"] == bytes.fromhex(lines[1]["entry_hash"])


def test_audit_row_field_mapping(tmp_path):
    _ledger_fixture(tmp_path)
    row = load_audit(tmp_path).rows[0]
    assert row["aggregate_type"] == "release"
    assert row["aggregate_id"] == "rec-1"
    assert row["actor_id"] == "alice"
    assert row["role"] == "editor"
    assert row["action"] == "publish"
    assert row["before"] is None and row["after"] is None
    assert row["before_version"] == 1 and row["after_version"] == 2
    assert row["reason"] == "reason 1"
    assert row["release_id"] == "rel-1"
    assert row["request_id"] is None
    assert row["created_at"] == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_audit_non_digit_versions_kept_as_legacy(tmp_path):
    ledger = AuditLedger(_ledger_path(tmp_path))
    ledger.append(_event(1, before_version="v1", after_version="7"))
    ledger.append(_event(2, before_version=None, after_version="v9"))
    row1, row2 = load_audit(tmp_path).rows
    assert row1["before_version"] is None
    assert row1["before"] == {"legacy_version": "v1"}
    assert row1["after_version"] == 7 and row1["after"] is None
    assert row2["before_version"] is None and row2["before"] is None
    assert row2["after_version"] is None
    assert row2["after"] == {"legacy_version": "v9"}


def test_audit_missing_role_is_unfillable(tmp_path):
    AuditLedger(_ledger_path(tmp_path)).append(_event(1, role=None))
    result = load_audit(tmp_path)
    event_id = result.rows[0]["event_id"]
    assert result.rows[0]["role"] is None
    assert {"legacy_id": event_id, "column": "role"} in result.unfillable


@pytest.mark.parametrize("bad", ["a" * 63, "g" * 64, "a" * 66])
def test_audit_malformed_entry_hash_names_file_and_line(tmp_path, bad):
    _ledger_fixture(tmp_path)

    def mutate(lines):
        doc = json.loads(lines[1])
        doc["entry_hash"] = bad
        lines[1] = json.dumps(doc)

    _rewrite_ledger(tmp_path, mutate)
    with pytest.raises(MigrationError, match=r"ledger\.jsonl: line 2:"):
        load_audit(tmp_path)


@pytest.mark.parametrize(("index", "line_no"), [(0, 1), (2, 3)])
def test_audit_broken_prev_hash_linkage_raises(tmp_path, index, line_no):
    _ledger_fixture(tmp_path)

    def mutate(lines):
        doc = json.loads(lines[index])
        doc["prev_hash"] = "ab" * 32
        lines[index] = json.dumps(doc)

    _rewrite_ledger(tmp_path, mutate)
    with pytest.raises(MigrationError, match=rf"ledger\.jsonl: line {line_no}:"):
        load_audit(tmp_path)


def test_audit_non_increasing_seq_raises(tmp_path):
    _ledger_fixture(tmp_path)

    def mutate(lines):
        doc = json.loads(lines[2])
        doc["seq"] = 2
        lines[2] = json.dumps(doc)

    _rewrite_ledger(tmp_path, mutate)
    with pytest.raises(MigrationError, match=r"ledger\.jsonl: line 3:"):
        load_audit(tmp_path)


def test_audit_invalid_event_id_raises(tmp_path):
    _ledger_fixture(tmp_path)

    def mutate(lines):
        doc = json.loads(lines[0])
        doc["event"]["event_id"] = "not-a-uuid"
        lines[0] = json.dumps(doc)

    _rewrite_ledger(tmp_path, mutate)
    with pytest.raises(MigrationError, match=r"ledger\.jsonl: line 1:"):
        load_audit(tmp_path)


def test_audit_blank_line_raises_with_line_number(tmp_path):
    _ledger_fixture(tmp_path)
    _rewrite_ledger(tmp_path, lambda lines: lines.insert(1, ""))
    with pytest.raises(MigrationError, match=r"ledger\.jsonl: line 2:"):
        load_audit(tmp_path)


def test_audit_missing_ledger_is_empty(tmp_path):
    result = load_audit(tmp_path)
    assert result.rows == [] and result.source_rows == 0


# -- Fix round 1: event_id type check; newline-only splitting -----------------


@pytest.mark.parametrize("bad_id", [5, ["x"], None, "zz"])
def test_audit_non_string_or_invalid_event_id_raises_migration_error(tmp_path, bad_id):
    _ledger_fixture(tmp_path)

    def mutate(lines):
        doc = json.loads(lines[0])
        doc["event"]["event_id"] = bad_id
        lines[0] = json.dumps(doc)

    _rewrite_ledger(tmp_path, mutate)
    with pytest.raises(MigrationError, match=r"ledger\.jsonl: line 1:"):
        load_audit(tmp_path)


def test_audit_reason_with_unicode_line_separators_round_trips(tmp_path):
    # The real writer keeps U+2028 and U+0085 raw (ensure_ascii=False); they must
    # not split the ledger line.
    reason = "a b \x85c"
    entry = AuditLedger(_ledger_path(tmp_path)).append(_event(1, reason=reason))
    result = load_audit(tmp_path)
    assert len(result.rows) == 1
    assert result.rows[0]["reason"] == reason
    assert result.rows[0]["entry_hash"] == bytes.fromhex(entry.entry_hash)


def test_unanswered_question_with_line_separators_loads_as_one_row(tmp_path):
    from src.services.unanswered_queue import UnansweredQueueService

    queue_path = tmp_path / "unanswered" / "queue.jsonl"
    UnansweredQueueService(queue_path=queue_path).push(
        "hoc phi  bao nhieu \x85 vay", "trace-sep", "route", "OUT_OF_SCOPE"
    )
    stored = json.loads(queue_path.read_text(encoding="utf-8").split("\n")[0])
    result = load_unanswered(tmp_path)
    assert len(result.rows) == 1
    assert result.rows[0]["question"] == stored["question"]


def test_audit_crlf_terminated_ledger_still_loads(tmp_path):
    entries = _ledger_fixture(tmp_path)
    path = _ledger_path(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    result = load_audit(tmp_path)
    assert len(result.rows) == 3
    assert [r["entry_hash"] for r in result.rows] == [bytes.fromhex(e.entry_hash) for e in entries]
