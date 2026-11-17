"""Task 2a (U3): coordinate-key derivation, records loader, retractions loader.

Fixtures are written with the real CanonicalRecordStore so the on-disk shape is
the real one, not an invented one.
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
    status: RecordStatus = RecordStatus.DRAFT,
    published_at: datetime | None = None,
    methods: tuple[str, ...] = (),
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=record_id,
        record_type=record_type,
        version=1,
        admission_year=year,
        effective_from=date(year, 1, 1),
        status=status,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="doc-1",
        published_at=published_at,
        admission_methods=list(methods),
        text="noi dung",
    )


def _store(tmp_path: Path) -> tuple[CanonicalRecordStore, Path]:
    root = tmp_path / "store"
    return CanonicalRecordStore(root), root


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
    store.write_record(_rec("CHI_TIEU:nganh:7220104:2025:x", year=2026))

    out = load_records(root)

    assert len(out.rows) == out.source_rows == 1
    row = out.rows[0]
    assert row["record_id"] == "unanchored:CHI_TIEU:nganh:7220104:2025:x"
    assert row["coordinate_key"] == row["record_id"]
    assert row["status"] == "draft"
    assert "UNANCHORED_DRAFT" in row["structured_payload"]["blockers"]
    assert out.quarantined_rows == 1


def test_published_without_source_document_is_unanchored(tmp_path):
    store, root = _store(tmp_path)
    legacy = "CHI_TIEU:nganh:7220104:2026:canon-a"
    store.write_record(_rec(legacy))
    # The store refuses a published record without provenance, so the stored
    # bytes are edited directly to simulate the legacy shape.
    path = root / "records" / f"rec-{quote(legacy, safe='')}_r" / "v0001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "published"
    payload["source_document_id"] = None
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

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
    bad = root / "records" / f"rec-{quote(legacy, safe='')}_r" / "v0001.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(MigrationError) as info:
        load_records(root)

    assert str(bad) in str(info.value)


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
