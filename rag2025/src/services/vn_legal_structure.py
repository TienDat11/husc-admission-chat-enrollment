"""Rule-based Vietnamese legal structure parser; no NLP model.
Boilerplate Vietnamese legal text is highly repetitive, so regex gives high precision (spec section 7.1).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

RX_DOC_CENTRAL = re.compile(
    r"\b(\d{1,4}\s*/\s*\d{4}\s*/\s*"
    r"(?:TT-BGDĐT|TT-BGDDT|NĐ-CP|ND-CP|QĐ-TTg|QD-TTg|QH\d{2}|"
    r"TT-[A-ZĐ]{2,8}|QĐ-[A-ZĐ]{2,8}|NQ-[A-ZĐ]{2,8}))\b"
)
RX_DOC_LOCAL = re.compile(
    r"\b(\d{1,5}\s*/\s*(?:QĐ|TB|CV|KH|HD|TTr)\s*-\s*"
    r"(?:ĐHKH|DHKH|ĐHH|DHH|BGDĐT|BGDDT|[A-ZĐ]{2,10}))\b"
)
RX_CHUONG = re.compile(r"^\s*Chương\s+([IVXLCDM]+)\b", re.M)
RX_MUC = re.compile(r"^\s*Mục\s+(\d+)\b", re.M)
RX_DIEU = re.compile(r"^\s*Điều\s+(\d{1,3}[a-zđ]?)\s*[.．:]", re.M)
RX_KHOAN = re.compile(r"^\s*(\d{1,2})\s*[.．]\s+(?=[A-ZĐÀ-Ỹ“\"])", re.M)
RX_DIEM = re.compile(r"^\s*([a-zđăâêôơư])\s*\)\s+", re.M)


@dataclass(frozen=True)
class LegalUnit:
    kind: Literal["chuong", "muc", "dieu", "khoan", "diem"]
    label: str
    number: str
    text: str
    span: tuple[int, int]
    parent_label: str | None


def extract_doc_numbers(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for rx in (RX_DOC_CENTRAL, RX_DOC_LOCAL):
        for match in rx.finditer(text):
            value = re.sub(r"\s+", "", match.group(1))
            if value not in seen:
                seen.add(value)
                out.append(value)
    return out


def parse_structure(text: str) -> list[LegalUnit]:
    normalized = unicodedata.normalize("NFC", text)
    patterns = (
        ("chuong", RX_CHUONG, "Chương"),
        ("muc", RX_MUC, "Mục"),
        ("dieu", RX_DIEU, "Điều"),
        ("khoan", RX_KHOAN, "khoản"),
        ("diem", RX_DIEM, "điểm"),
    )
    matches = []
    for kind, rx, prefix in patterns:
        for match in rx.finditer(normalized):
            number = match.group(1)
            label = f"{prefix} {number}"
            matches.append((match.start(), kind, label, number))
    matches.sort(key=lambda item: item[0])
    current: dict[str, str | None] = {kind: None for kind, _, _ in patterns}
    order = [kind for kind, _, _ in patterns]
    units: list[LegalUnit] = []
    for index, (start, kind, label, number) in enumerate(matches):
        level = order.index(kind)
        parent = next((current[k] for k in reversed(order[:level]) if current[k]), None)
        end = matches[index + 1][0] if index + 1 < len(matches) else len(normalized)
        units.append(LegalUnit(kind, label, number, normalized[start:end], (start, end), parent))
        for lower in order[level + 1 :]:
            current[lower] = None
        current[kind] = label
    return units
