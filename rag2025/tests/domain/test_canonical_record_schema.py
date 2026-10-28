"""TDD: rag2025/src/domain/canonical — versioned canonical admissions record.

Ticket 02 acceptance:
- a record carries the spec's minimum field set;
- a reversed effective window is rejected at construction;
- a ``published`` record without a source document is rejected;
- ``record_id`` + ``version`` are stable across repeated serialize/deserialize;
- Vietnamese text round-trips byte-faithful (bytes compared, not normalized
  strings);
- a record written by an older schema still reads after a field was added.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical import (  # noqa: E402
    SCHEMA_VERSION,
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
    from_canonical_json,
    to_canonical_json,
)

# The minimum field set named in the spec (Implementation Decisions →
# "Metadata/effective year"). Every one of these must be carried by a record.
SPEC_FIELDS = (
    "record_id",
    "record_type",
    "version",
    "admission_year",
    "effective_from",
    "effective_to",
    "status",
    "authority_level",
    "source_document_id",
    "source_url",
    "source_page",
    "published_at",
    "supersedes",
    "locale",
)


def _record(**overrides) -> CanonicalRecord:
    """A minimal valid published record; each test overrides what it probes."""
    fields = {
        "record_id": "DIEM_CHUAN:NGANH:cntt:2026",
        "record_type": RecordType.DIEM_CHUAN,
        "version": 1,
        "admission_year": 2026,
        "effective_from": date(2026, 6, 1),
        "effective_to": date(2026, 12, 31),
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 12,
        "published_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "supersedes": "DIEM_CHUAN:NGANH:cntt:2025",
        "text": "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.",
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


def test_record_carries_the_spec_minimum_field_set():
    dumped = _record().model_dump()
    missing = [name for name in SPEC_FIELDS if name not in dumped]
    assert missing == [], f"record does not carry required fields: {missing}"
    assert _record().locale == "vi"


def test_record_cannot_be_constructed_without_a_year_or_effective_window():
    with pytest.raises(ValidationError):
        _record(admission_year=None)
    with pytest.raises(ValidationError):
        _record(effective_from=None)


def test_admission_year_outside_the_sane_range_is_rejected():
    # Bounded year follows the repo's existing convention for year fields
    # (src/models/chunk.py ChunkMetadataV3.data_year is likewise range-bound).
    # Lower/upper edges are accepted; one step outside each is rejected, so a
    # regression that widens the bound to any int fails here.
    for year in (2000, 2100):
        assert _record(admission_year=year).admission_year == year
    for year in (1999, 2101):
        with pytest.raises(ValidationError):
            _record(admission_year=year)


def test_reversed_effective_window_is_rejected():
    with pytest.raises(ValidationError):
        _record(
            effective_from=date(2026, 12, 31),
            effective_to=date(2026, 6, 1),
        )


def test_effective_window_boundaries_are_accepted():
    # Same-day window: a fact valid for exactly one day.
    same_day = _record(effective_from=date(2026, 6, 1), effective_to=date(2026, 6, 1))
    assert same_day.effective_from == same_day.effective_to

    # Open-ended window: still in force, no end date.
    open_ended = _record(effective_to=None)
    assert open_ended.effective_to is None


def test_published_record_without_source_document_is_rejected():
    # Absent and blank provenance are the same defect: a published fact with
    # nothing to trace back to. Whitespace-only must not slip past a
    # truthiness check.
    for missing in (None, "   "):
        with pytest.raises(ValidationError):
            _record(source_document_id=missing)


def test_unpublished_record_may_lack_a_source_document():
    draft = _record(
        status=RecordStatus.DRAFT,
        source_document_id=None,
        source_url=None,
        source_page=None,
        published_at=None,
    )
    assert draft.status is RecordStatus.DRAFT


def test_authority_levels_rank_in_the_specs_source_priority_order():
    # Spec "Thứ tự ưu tiên nguồn": canonical > HUSC official > Hue University /
    # ministry > prior year > derived FAQ. The tie is intentional (one tier).
    assert AuthorityLevel.HUSC_CANONICAL.rank == 1
    assert AuthorityLevel.HUSC_CANONICAL.rank < AuthorityLevel.HUSC_OFFICIAL.rank
    assert AuthorityLevel.HUSC_OFFICIAL.rank < AuthorityLevel.HUE_UNIVERSITY.rank
    assert AuthorityLevel.HUE_UNIVERSITY.rank == AuthorityLevel.MINISTRY.rank
    assert AuthorityLevel.MINISTRY.rank < AuthorityLevel.PRIOR_YEAR_OFFICIAL.rank
    assert AuthorityLevel.PRIOR_YEAR_OFFICIAL.rank < AuthorityLevel.DERIVED_FAQ.rank


def test_record_id_and_version_are_stable_across_repeated_roundtrip():
    record = _record()
    reference = to_canonical_json(record)

    current = record
    for _ in range(5):
        current = from_canonical_json(to_canonical_json(current))
        assert current.record_id == record.record_id
        assert current.version == record.version
        assert current.record_version_key == record.record_version_key
        assert current == record
        assert to_canonical_json(current) == reference


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_vietnamese_text_roundtrips_byte_faithful(form):
    text = "Học phí ngành Kỹ thuật phần mềm năm 2026: 18.000.000 đồng/năm."
    text = unicodedata.normalize(form, text)
    raw = text.encode("utf-8")

    payload = to_canonical_json(_record(text=text))

    # Stored as raw UTF-8, never \uXXXX escapes.
    assert raw in payload
    # Never rewritten to the other normalization form.
    other = "NFD" if form == "NFC" else "NFC"
    assert unicodedata.normalize(other, text).encode("utf-8") != raw
    assert from_canonical_json(payload).text.encode("utf-8") == raw


def test_record_written_by_an_older_schema_still_reads():
    payload = json.loads(to_canonical_json(_record()))
    # Simulate a writer that predates the current schema: no schema_version and
    # none of the fields added since 1.0.
    for dropped in ("schema_version", "source_page", "supersedes", "effective_to"):
        del payload[dropped]

    record = CanonicalRecord.model_validate(payload)

    assert record.schema_version == SCHEMA_VERSION
    assert record.source_page is None
    assert record.supersedes is None
    assert record.effective_to is None
    assert record.record_id == "DIEM_CHUAN:NGANH:cntt:2026"
    assert record.version == 1


def test_field_added_by_a_newer_writer_survives_a_roundtrip():
    payload = json.loads(to_canonical_json(_record()))
    payload["quota"] = 120  # written by a newer schema this version doesn't know

    record = from_canonical_json(json.dumps(payload, ensure_ascii=False))

    assert record.quota == 120
    assert json.loads(to_canonical_json(record))["quota"] == 120
