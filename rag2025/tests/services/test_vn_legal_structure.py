from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.vn_legal_structure import extract_doc_numbers, parse_structure


SAMPLE = (
    "Điều 4a. Quy định chung\n"
    "1. Nội dung khoản 1.\n"
    "2. Nội dung khoản 2.\n"
    "d) Điểm d nội dung.\n"
)


def test_dieu_khoan_diem() -> None:
    units = parse_structure(SAMPLE)

    assert len(units) == 4
    assert [(unit.kind, unit.label, unit.parent_label) for unit in units] == [
        ("dieu", "Điều 4a", None),
        ("khoan", "khoản 1", "Điều 4a"),
        ("khoan", "khoản 2", "Điều 4a"),
        ("diem", "điểm d", "khoản 2"),
    ]


def test_doc_number_bgddt() -> None:
    text = "Thông tư số 08/2022/TT-BGDĐT ngày 06/6/2022"

    assert extract_doc_numbers(text) == ["08/2022/TT-BGDĐT"]


def test_doc_number_husc() -> None:
    text = "Quyết định số 1234/QĐ-DHKH"

    assert extract_doc_numbers(text) == ["1234/QĐ-DHKH"]


def test_nfc_normalization() -> None:
    nfc_units = parse_structure(SAMPLE)
    nfd_units = parse_structure(unicodedata.normalize("NFD", SAMPLE))

    assert [
        (unit.kind, unicodedata.normalize("NFC", unit.label), unit.number)
        for unit in nfc_units
    ] == [
        (unit.kind, unicodedata.normalize("NFC", unit.label), unit.number)
        for unit in nfd_units
    ]
