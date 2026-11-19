"""Chunk identity is taken over NFC text; the preview is byte-stable across runs.

An NFC copy and an NFD copy of one record are the same fact, so they must share
chunk ids and content hashes. A preview (unpublished records allowed) must
produce the same output every time it runs.
"""

from __future__ import annotations

import json
import sys
import unicodedata
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
from src.services.canonical_chunk_builder import build_chunk_index  # noqa: E402
from src.services.chunk_identity import normalize_for_hash  # noqa: E402

RELEASE_ID = "R-2026-09-01"
TO_HOP_TEXT = (
    "Tổ hợp xét tuyển năm 2026 ngành Quản lý văn hóa (mã ngành 7229042): "
    "C00, C03, C19, X70, D14."
)
TO_HOP_VALUES = {
    "chi_tieu": 30,
    "ma_nganh": "7229042",
    "ten_nganh": "Quản lý văn hóa",
    "to_hop_codes": ["C00", "C03", "C19", "X70", "D14"],
}


def _record(**overrides) -> CanonicalRecord:
    fields = {
        "record_id": "TO_HOP:NGANH:7229042:2026",
        "record_type": RecordType.TO_HOP,
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_CANONICAL,
        "source_document_id": "THONG_TIN_TUYEN_SINH_202611.pdf",
        "source_page": 25,
        "text": TO_HOP_TEXT,
        "values": TO_HOP_VALUES,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


def _dump(chunks: list[dict]) -> str:
    return json.dumps(chunks, sort_keys=True, ensure_ascii=False)


def test_nfc_and_nfd_copies_of_one_record_share_chunk_ids(tmp_path):
    nfc = unicodedata.normalize("NFC", TO_HOP_TEXT)
    nfd = unicodedata.normalize("NFD", TO_HOP_TEXT)
    assert nfc != nfd

    from_nfc = build_chunk_index([_record(text=nfc)], tmp_path / "nfc", RELEASE_ID)
    from_nfd = build_chunk_index([_record(text=nfd)], tmp_path / "nfd", RELEASE_ID)

    assert normalize_for_hash(nfc) == normalize_for_hash(nfd)
    assert sorted(from_nfc.chunk_ids) == sorted(from_nfd.chunk_ids)
    assert [c["text"] for c in from_nfc.chunks] == [c["text"] for c in from_nfd.chunks]


def test_preview_output_is_byte_identical_across_runs(tmp_path):
    draft = _record(status=RecordStatus.DRAFT)

    first = build_chunk_index([draft], tmp_path / "run1", RELEASE_ID, allow_unpublished=True)
    second = build_chunk_index([draft], tmp_path / "run2", RELEASE_ID, allow_unpublished=True)

    assert _dump(first.chunks) == _dump(second.chunks)
    assert first.chunk_ids == second.chunk_ids
