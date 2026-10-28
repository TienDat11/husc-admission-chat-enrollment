"""T05: DOCX headings reach chunk metadata as ``section``.

The citation reader needs a section name per chunk: the chunker takes the
closest heading that precedes the unit's start offset inside the record text.
A record without headings (a PDF, or a seeded reference record) leaves the key
absent so the FE falls back to the page number.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.canonical.semantic_units import SemanticUnit, UnitKind  # noqa: E402
from src.services.canonical_chunk_builder import _chunk_for  # noqa: E402
from src.services.review_preview import _dedup_sources  # noqa: E402

RELEASE_ID = "R-2026-10-01"

TEXT = (
    "Phuong thuc xet tuyen\n\n"
    "Thi sinh dang ky xet tuyen theo phuong thuc hoc ba THPT.\n\n"
    "Dieu kien xet hoc ba\n\n"
    "Diem trung binh mon Toan khong duoi 6.0 diem."
)

HEADINGS = (
    {"text": "Phuong thuc xet tuyen", "offset": 0},
    {"text": "Dieu kien xet hoc ba", "offset": 104},
)


def _record(**overrides) -> CanonicalRecord:
    fields = {
        "record_id": "TB:XET_TUYEN:2026",
        "record_type": RecordType.THONG_BAO,
        "version": 1,
        "admission_year": 2026,
        "effective_from": date(2026, 1, 1),
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": "uploads/abc123.docx",
        "source_page": None,
        "text": TEXT,
        "docx_headings": HEADINGS,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


def _unit(start: int, value: str = "unit text") -> SemanticUnit:
    return SemanticUnit(
        record_id="TB:XET_TUYEN:2026",
        record_type=RecordType.THONG_BAO.value,
        field="text",
        slot="0",
        value=value,
        kind=UnitKind.TEXT,
        source_span=(start, start + len(value)),
    )


def test_chunk_before_first_heading_has_no_section():
    # Only headings that precede the unit count: a unit at the very start
    # sits under no heading yet.
    record = _record()
    chunk = _chunk_for(record, _unit(0), RELEASE_ID)
    assert "section" not in chunk["metadata"]


def test_chunk_picks_nearest_preceding_heading():
    record = _record()
    first = _chunk_for(record, _unit(30), RELEASE_ID)
    assert first["metadata"]["section"] == "Phuong thuc xet tuyen"
    second = _chunk_for(record, _unit(120), RELEASE_ID)
    assert second["metadata"]["section"] == "Dieu kien xet hoc ba"


def test_chunk_without_headings_has_no_section():
    # A PDF or seeded reference record carries no headings: the key stays
    # absent so the FE falls back to the page number.
    bare = CanonicalRecord.model_validate(
        {k: v for k, v in _record().model_dump(mode="json").items()
         if k != "docx_headings"}
    )
    assert getattr(bare, "docx_headings", None) is None
    chunk = _chunk_for(bare, _unit(30), RELEASE_ID)
    assert "section" not in chunk["metadata"]


def test_dedup_sources_carries_page_section_title():
    class _Doc:
        source = "REC:1"
        metadata = {
            "source_document_id": "uploads/abc123.docx",
            "source_page": 24,
            "section": "Dieu kien xet hoc ba",
            "record_id": "REC:1",
            "record_type": "THONG_BAO",
            "effective_from": "2026-01-01",
            "effective_to": None,
            "authority_level": "husc_official",
        }

    sources = _dedup_sources([_Doc()])
    assert sources[0]["page"] == 24
    assert sources[0]["section"] == "Dieu kien xet hoc ba"
    assert sources[0]["title"] == "Dieu kien xet hoc ba"


def test_dedup_sources_without_section_title_is_none():
    class _Doc:
        source = "REC:2"
        metadata = {
            "source_document_id": "new_data/a.pdf",
            "source_page": 3,
            "record_id": "REC:2",
            "record_type": "THONG_BAO",
            "effective_from": "2026-01-01",
            "effective_to": None,
            "authority_level": "husc_official",
        }

    sources = _dedup_sources([_Doc()])
    assert sources[0]["page"] == 3
    assert sources[0]["section"] is None
    assert sources[0]["title"] is None
