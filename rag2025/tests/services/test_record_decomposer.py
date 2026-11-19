"""Record decomposer (plan §PL-6.6): legal articles, tables and unanchored notices."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.identity import parse_coordinate_key, pick_record_type  # noqa: E402
from src.domain.canonical.record import RecordStatus  # noqa: E402
from src.services.record_decomposer import (  # noqa: E402
    UNANCHORED_DRAFT,
    decompose,
    decompose_with_blockers,
)

LEGAL = [
    "Điều 1. Phạm vi điều chỉnh\nQuy chế này quy định tuyển sinh đại học.",
    "Điều 2. Đối tượng áp dụng\nÁp dụng cho thí sinh dự tuyển.",
]
TABLE = [
    "Mã ngành 7220201 Kỹ thuật phần mềm chỉ tiêu 120\n"
    "7340101 Quản trị kinh doanh chỉ tiêu 90"
]
NOTICE = ["Thông báo lịch nộp hồ sơ tuyển sinh, vui lòng đăng ký trước ngày 30 tháng 6."]


def _decompose(pages):
    return decompose(pages, source_document_id="doc-1", authored_by="cb", admission_year=2026)


def _value(x) -> str:
    return str(getattr(x, "value", x))


def _subjects(records, rtype: str) -> list[str]:
    return sorted(
        parse_coordinate_key(r.record_id).subject
        for r in records
        if _value(r.record_type) == rtype
    )


def test_legal_pages_become_one_dieu_kien_per_article() -> None:
    records = _decompose(LEGAL)
    assert _subjects(records, "DIEU_KIEN") == ["dieu_1", "dieu_2"]
    assert all(r.status == RecordStatus.DRAFT for r in records)


def test_table_pages_become_nganh_and_chi_tieu_per_code() -> None:
    records = _decompose(TABLE)
    assert _subjects(records, "NGANH") == ["7220201", "7340101"]
    assert _subjects(records, "CHI_TIEU") == ["7220201", "7340101"]


def test_unanchored_notice_is_one_draft_with_blocker() -> None:
    decomposition = decompose_with_blockers(
        NOTICE, source_document_id="doc-1", authored_by="cb", admission_year=2026
    )
    assert len(decomposition.records) == 1
    record = decomposition.records[0]
    assert _value(record.record_type) == "THONG_BAO"
    assert not record.record_id.startswith("upload:")
    assert UNANCHORED_DRAFT in decomposition.blockers


def test_decomposition_is_deterministic() -> None:
    assert _decompose(LEGAL) == _decompose(LEGAL)


def test_priority_picks_chi_tieu_over_thoi_gian() -> None:
    assert _value(pick_record_type(["THOI_GIAN", "CHI_TIEU"])) == "CHI_TIEU"
