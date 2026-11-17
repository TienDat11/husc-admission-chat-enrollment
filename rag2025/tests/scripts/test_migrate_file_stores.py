"""Task 2a (U3): coordinate-key derivation, records loader, retractions loader.

Fixtures are written with the real CanonicalRecordStore so the on-disk shape is
the real one, not an invented one. Malformed files are produced by editing the
real bytes after the store wrote them.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pytest

from scripts.migrate_file_stores import (
    MigrationError,
    derive_coordinate_key,
    load_records,
    load_retractions,
    split_version_key,
)
from src.domain.canonical.record import (
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore


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
    assert out.rows[1]["event_id"] == "e2"
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
