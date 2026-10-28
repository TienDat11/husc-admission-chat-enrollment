"""TDD: rag2025/src/domain/canonical/diff.py — ticket 25 criteria 1 and 2.

The diff is computed server-side so every consumer replays the same result.
Acceptance probed here:
- a diff compares versions of ONE record; cross-record diffs raise;
- ``version`` is excluded — a version bump alone is not a diff;
- changes are ordered by field name (deterministic, client-independent);
- ``has_high_impact`` is True iff at least one change is high-impact;
- old/new values are JSON-safe (dates/enums as wire strings, whole diff
  serializes without a second conversion);
- ``baseline=None`` (never-published record) reports ``baseline_version``
  None and the draft's full content as changes against nothing — it does
  NOT invent an empty baseline record to diff against.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.canonical.diff import diff_records
from src.domain.canonical.high_impact import HIGH_IMPACT_VERSION


def _record(**overrides) -> CanonicalRecord:
    """Same minimal valid published record as the schema tests; overrides probe."""
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


def test_cross_record_diff_is_refused():
    baseline = _record(record_id="DIEM_CHUAN:NGANH:cntt:2026")
    draft = _record(record_id="HOC_PHI:NGANH:cntt:2026")
    with pytest.raises(ValueError, match="one record"):
        diff_records(baseline, draft)


def test_version_bump_alone_is_not_a_diff():
    """Every edit bumps ``version``; reporting it would bury the real change."""
    baseline = _record(version=1)
    draft = _record(version=2)
    diff = diff_records(baseline, draft)
    assert diff.changes == ()
    assert diff.has_high_impact is False
    assert diff.baseline_version == 1
    assert diff.draft_version == 2


def test_changes_are_ordered_by_field_name():
    draft = _record(
        version=2,
        text="Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 24,0 điểm.",
        admission_year=2027,
        effective_from=date(2026, 7, 1),
    )
    diff = diff_records(_record(), draft)
    fields = [c.field for c in diff.changes]
    assert set(fields) == {"admission_year", "effective_from", "text"}
    assert fields == sorted(fields)


def test_has_high_impact_true_only_when_a_change_is_high_impact():
    # status is bookkeeping (NON_HIGH_IMPACT): published -> draft alone.
    low_only = diff_records(_record(), _record(version=2, status=RecordStatus.DRAFT))
    assert [c.field for c in low_only.changes] == ["status"]
    assert low_only.changes[0].high_impact is False
    assert low_only.has_high_impact is False
    assert low_only.high_impact_changes == ()

    # one high-impact change flips the flag and appears in high_impact_changes.
    mixed = diff_records(
        _record(),
        _record(version=2, status=RecordStatus.DRAFT, text="Điểm chuẩn là 24,0."),
    )
    assert mixed.has_high_impact is True
    assert [c.field for c in mixed.high_impact_changes] == ["text"]


def test_values_are_json_safe_wire_strings():
    diff = diff_records(
        _record(),
        _record(
            version=2,
            effective_from=date(2026, 7, 5),
            authority_level=AuthorityLevel.MINISTRY,
        ),
    )
    by_field = {c.field: c for c in diff.changes}
    assert by_field["effective_from"].old == "2026-06-01"
    assert by_field["effective_from"].new == "2026-07-05"
    assert by_field["authority_level"].old == "husc_official"
    assert by_field["authority_level"].new == "ministry"
    # the whole payload serializes with no second conversion step
    json.dumps(diff.to_dict(), ensure_ascii=False)


def test_new_record_has_no_invented_baseline():
    """baseline=None: the diff says "no published version" (baseline_version
    None) and shows the draft's content as changes against nothing."""
    draft = _record()
    diff = diff_records(None, draft)
    assert diff.baseline_version is None
    assert diff.draft_version == 1
    assert diff.changes  # full content is shown; nothing is invented
    assert all(c.old is None for c in diff.changes)
    assert "version" not in [c.field for c in diff.changes]
    assert diff.has_high_impact is True


def test_extra_fields_are_compared_and_default_to_high_impact():
    """extra="allow" means the compared shape is the record's own; an
    unclassified extra must surface, not vanish."""
    diff = diff_records(_record(), _record(version=2, ghi_chu="xét tuyển sớm"))
    extra = [c for c in diff.changes if c.field == "ghi_chu"]
    assert len(extra) == 1
    assert extra[0].old is None
    assert extra[0].new == "xét tuyển sớm"
    assert extra[0].high_impact is True


def test_diff_carries_the_classification_version():
    diff = diff_records(_record(), _record(version=2))
    assert diff.high_impact_version == HIGH_IMPACT_VERSION
    assert diff.to_dict()["high_impact_version"] == HIGH_IMPACT_VERSION
