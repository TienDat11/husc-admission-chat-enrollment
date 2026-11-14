"""U1 / PL-6.4 #2: CanonicalRecord normalizes every string input to NFC.

Invariant: CanonicalRecord(**nfd_payload) == CanonicalRecord(**nfc_payload),
including strings nested inside extra fields such as ``structured_payload``.
"""
from __future__ import annotations

import sys
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
    to_canonical_json,
)


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _payload(text: str) -> dict:
    return {
        "record_id": "DIEM_CHUAN:NGANH:cntt:2026",
        "record_type": RecordType.DIEM_CHUAN,
        "version": 1,
        "admission_year": 2026,
        "effective_from": date(2026, 6, 1),
        "status": RecordStatus.DRAFT,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "text": text,
    }


def test_nfd_text_is_stored_as_nfc():
    text = "Điều kiện tuyển sinh năm 2026"
    record = CanonicalRecord(**_payload(_nfd(text)))
    assert record.text == unicodedata.normalize("NFC", text)


def test_nfd_and_nfc_payloads_build_equal_records():
    text = "Điểm chuẩn ngành Công nghệ thông tin"
    nfd = CanonicalRecord(**_payload(_nfd(text)))
    nfc = CanonicalRecord(**_payload(unicodedata.normalize("NFC", text)))
    assert nfd == nfc


def test_nested_extra_strings_are_nfc():
    text = "Học phí tín chỉ"
    payload = _payload(unicodedata.normalize("NFC", text))
    payload["structured_payload"] = {
        "label": _nfd("Học phí"),
        "rows": [{"note": _nfd("Điều 5")}],
    }
    record = CanonicalRecord(**payload)
    extra = record.model_dump()["structured_payload"]
    assert extra["label"] == unicodedata.normalize("NFC", "Học phí")
    assert extra["rows"][0]["note"] == unicodedata.normalize("NFC", "Điều 5")


def test_nfd_payload_serializes_to_same_bytes_as_nfc():
    text = "Chỉ tiêu xét tuyển"
    nfd = CanonicalRecord(**_payload(_nfd(text)))
    nfc = CanonicalRecord(**_payload(unicodedata.normalize("NFC", text)))
    assert to_canonical_json(nfd) == to_canonical_json(nfc)


def test_list_string_fields_are_nfc():
    payload = _payload("nội dung")
    payload["major_codes"] = [_nfd("Đà Nẵng")]
    record = CanonicalRecord(**payload)
    assert record.major_codes == [unicodedata.normalize("NFC", "Đà Nẵng")]
