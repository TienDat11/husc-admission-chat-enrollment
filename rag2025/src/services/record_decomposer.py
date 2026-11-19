"""Decompose an uploaded document into draft canonical records (plan §PL-6.6).

Three shapes, decided from the text alone:

1. Legal text (has ``Điều N.`` headings): one ``DIEU_KIEN`` record per article,
   subject ``dieu_<N>``.
2. Tables (major codes ``7\\d{6}``): one ``NGANH`` record per code, plus one
   record per column type the code's page carries (``DIEM_CHUAN``,
   ``CHI_TIEU``, ``HOC_PHI``, ``TO_HOP``, ``THOI_GIAN``).
3. Neither: one ``THONG_BAO`` record flagged ``UNANCHORED_DRAFT``. It is a
   draft that a reviewer must anchor before it can be submitted.

Pure: no I/O. The caller writes the records.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src.domain.canonical.identity import (
    ALL_METHODS,
    RECORD_TYPE_PRIORITY,
    coordinate_key,
)
from src.domain.canonical.record import (
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.services.page_router import MAJOR_CODE_RX
from src.services.vn_legal_structure import RX_DIEU, extract_doc_numbers, parse_structure

UNANCHORED_DRAFT = "UNANCHORED_DRAFT"
LEGAL_SCOPE_DEFAULT = "phap_quy"
TABLE_SCOPE_DEFAULT = "nganh"
UNANCHORED_SCOPE_DEFAULT = "chung"

COLUMN_TYPES: tuple[tuple[re.Pattern[str], RecordType], ...] = (
    (re.compile(r"điểm chuẩn|điểm trúng tuyển", re.IGNORECASE), RecordType.DIEM_CHUAN),
    (re.compile(r"chỉ tiêu", re.IGNORECASE), RecordType.CHI_TIEU),
    (re.compile(r"học phí", re.IGNORECASE), RecordType.HOC_PHI),
    (re.compile(r"tổ hợp", re.IGNORECASE), RecordType.TO_HOP),
    (re.compile(r"thời gian", re.IGNORECASE), RecordType.THOI_GIAN),
)


@dataclass(frozen=True)
class Decomposition:
    records: tuple[CanonicalRecord, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class _Context:
    source_document_id: str
    authored_by: str
    admission_year: int
    effective_from: date
    authority_level: AuthorityLevel
    scope: str | None  # from the document's own number, if it states one


def _page_starts(pages: Sequence[str]) -> list[int]:
    starts: list[int] = []
    offset = 0
    for text in pages:
        starts.append(offset)
        offset += len(text) + 2  # the "\n\n" join
    return starts


def _page_of(offset: int, starts: Sequence[int]) -> int:
    return bisect_right(starts, offset)  # 1-based


def _legal_records(
    joined: str, starts: Sequence[int], ctx: _Context
) -> list[CanonicalRecord]:
    units = parse_structure(joined)
    boundaries = sorted(
        unit.span[0] for unit in units if unit.kind in ("chuong", "muc", "dieu")
    )
    bodies: dict[str, list[str]] = {}
    first_page: dict[str, int] = {}
    for unit in units:
        if unit.kind != "dieu":
            continue
        start = unit.span[0]
        end = next((b for b in boundaries if b > start), len(joined))
        number = unit.number.lower()
        bodies.setdefault(number, []).append(joined[start:end].strip())
        first_page.setdefault(number, _page_of(start, starts))
    return [
        _record(
            ctx,
            record_type=RecordType.DIEU_KIEN,
            subject=f"dieu_{number}",
            text="\n\n".join(bodies[number]),
            page=first_page[number],
            scope=ctx.scope or LEGAL_SCOPE_DEFAULT,
        )
        for number in bodies
    ]


def _column_types(page_text: str) -> set[RecordType]:
    return {rtype for rx, rtype in COLUMN_TYPES if rx.search(page_text)}


def _table_records(pages: Sequence[str], ctx: _Context) -> list[CanonicalRecord]:
    # (record_type, code) -> (first page, the page's lines carrying the code)
    slots: dict[tuple[RecordType, str], tuple[int, list[str]]] = {}
    for page_index, page_text in enumerate(pages, start=1):
        page_types = _column_types(page_text)
        for line in page_text.splitlines():
            for code in dict.fromkeys(MAJOR_CODE_RX.findall(line)):
                for rtype in (RecordType.NGANH, *page_types):
                    page, lines = slots.get((rtype, code), (page_index, []))
                    lines.append(line.strip())
                    slots[(rtype, code)] = (page, lines)

    ordered = sorted(
        slots.items(),
        key=lambda item: (RECORD_TYPE_PRIORITY.index(item[0][0]), item[0][1]),
    )
    return [
        _record(
            ctx,
            record_type=rtype,
            subject=code,
            text="\n".join(lines),
            page=page,
            scope=ctx.scope or TABLE_SCOPE_DEFAULT,
            major_codes=[code],
        )
        for (rtype, code), (page, lines) in ordered
    ]


def _record(
    ctx: _Context,
    *,
    record_type: RecordType,
    subject: str,
    text: str,
    page: int | None,
    scope: str,
    major_codes: list[str] | None = None,
) -> CanonicalRecord:
    key = coordinate_key(
        record_type=record_type,
        scope=scope,
        subject=subject,
        admission_year=ctx.admission_year,
        methods_hash_value=ALL_METHODS,
    )
    return CanonicalRecord(
        record_id=key,
        record_type=record_type,
        version=1,
        admission_year=ctx.admission_year,
        effective_from=ctx.effective_from,
        status=RecordStatus.DRAFT,
        authority_level=ctx.authority_level,
        source_document_id=ctx.source_document_id,
        source_page=page,
        authored_by=ctx.authored_by,
        text=text,
        major_codes=major_codes or [],
    )


def decompose_with_blockers(
    pages: Sequence[str],
    *,
    source_document_id: str,
    authored_by: str,
    admission_year: int,
    effective_from: date | None = None,
    authority_level: AuthorityLevel = AuthorityLevel.HUSC_OFFICIAL,
) -> Decomposition:
    texts = [unicodedata.normalize("NFC", page or "") for page in pages]
    joined = "\n\n".join(texts)
    if not joined.strip():
        return Decomposition((), ())

    doc_numbers = extract_doc_numbers(joined)
    ctx = _Context(
        source_document_id=source_document_id,
        authored_by=authored_by,
        admission_year=admission_year,
        effective_from=effective_from or date(admission_year, 1, 1),
        authority_level=authority_level,
        scope=doc_numbers[0] if doc_numbers else None,
    )
    starts = _page_starts(texts)

    records: list[CanonicalRecord] = []
    if RX_DIEU.search(joined):
        records.extend(_legal_records(joined, starts, ctx))
    records.extend(_table_records(texts, ctx))

    if records:
        return Decomposition(tuple(records), ())

    # Nothing anchors the text: keep it as a draft that must be anchored before
    # it can be submitted. The subject is derived from the content, so the same
    # text always maps to the same coordinate.
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    unanchored = _record(
        ctx,
        record_type=RecordType.THONG_BAO,
        subject=f"tb_{digest}",
        text=joined.strip(),
        page=None,
        scope=ctx.scope or UNANCHORED_SCOPE_DEFAULT,
    )
    return Decomposition((unanchored,), (UNANCHORED_DRAFT,))


def decompose(
    pages: Sequence[str],
    *,
    source_document_id: str,
    authored_by: str,
    admission_year: int,
    effective_from: date | None = None,
    authority_level: AuthorityLevel = AuthorityLevel.HUSC_OFFICIAL,
) -> list[CanonicalRecord]:
    return list(
        decompose_with_blockers(
            pages,
            source_document_id=source_document_id,
            authored_by=authored_by,
            admission_year=admission_year,
            effective_from=effective_from,
            authority_level=authority_level,
        ).records
    )
