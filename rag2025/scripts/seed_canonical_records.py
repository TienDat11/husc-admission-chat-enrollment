"""Seed the canonical record store with real admissions facts (ticket 28).

Reads the three official documents in ``new_data/`` and the existing chunk
corpus in ``rag2025/data/chunked/``, and writes one ``draft``
:class:`CanonicalRecord` per extracted fact into the versioned record store
(:mod:`src.infrastructure.canonical_record_store`), plus a machine-readable
report next to it.

Design rules
------------
* **No fabricated facts.** Every value written here is either read out of a
  source document (with its page) or out of a corpus chunk (with its file and
  ``chunk_id``). A cell that cannot be parsed produces a *warning* and a record
  whose field is explicitly absent (``values[field] is None`` +
  ``missing_fields``); it is never silently dropped and never filled with a
  neighbouring row's value.
* **pypdf tier only.** ``docling`` is not installed in this venv, so the parse
  goes through ``src.services.pdf_pipeline.parse_pdf(..., force="pypdf")``
  (tier 0). Per-page text is read with ``pypdf`` directly because the pipeline
  returns one joined string and provenance needs real page numbers.
* **Append-only, idempotent.** A ``record_id`` is a deterministic function of
  (fact type, ngành, admission year, source). Re-running the seeder on the same
  inputs writes nothing new: an existing version with identical bytes is
  skipped, and a version with *different* bytes is reported, never overwritten
  (the store is append-only by design).
* **draft only.** Nothing here publishes; ``status`` is ``draft`` for every
  record and no pointer/manifest is touched.

Usage::

    cd rag2025
    python scripts/seed_canonical_records.py            # write the real store
    python scripts/seed_canonical_records.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_RAG2025_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _RAG2025_ROOT.parent
if str(_RAG2025_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG2025_ROOT))

from src.domain.canonical import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
    to_canonical_json,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.pdf_pipeline import parse_pdf  # noqa: E402

SEED_VERSION = "t28-v1"


def _default_store_root() -> Path:
    """The store the APP reads, not a parallel one.

    This used to be hardcoded to ``data/canonical_records``, which is not the
    directory :mod:`src.app.lifecycle` wires the admin lifecycle to. The app
    reads ``RAGSettings.CANONICAL_STORE_ROOT`` (default
    ``data/canonical_store``), so a seeded store was invisible to every admin
    route and to the published materializer: the seeder reported 487 records
    while the app served an empty store. Resolving the setting here keeps one
    source of truth; ``--store-root`` still overrides for tests and drills.

    ``config.settings`` is imported lazily so a bad/missing ``.env`` surfaces
    as a clear error at run time rather than at import time.
    """
    from config.settings import RAGSettings

    return Path(RAGSettings().CANONICAL_STORE_ROOT)


DEFAULT_STORE_ROOT = _default_store_root()
REPORT_FILENAME = "seed_report.json"

# --------------------------------------------------------------------------
# Sources (repo-relative, forward slashes; the ticket's mandatory list)
# --------------------------------------------------------------------------
TTTS_PDF = "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"
HD2026_PDF = "new_data/2304_BGDDT-GDDH_HD tuyển sinh ĐH, CĐ 2026.pdf"
TT06_PDF = "new_data/Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf"

CORPUS_CANONICAL = "rag2025/data/chunked/husc_canonical_2026.jsonl"
CORPUS_LIETKE29 = "rag2025/data/chunked/husc_lietke29nganh_fix.jsonl"
CORPUS_DIEM2025 = "rag2025/data/chunked/husc_diem_ranking_2025.jsonl"
CORPUS_NGANHMOI = "rag2025/data/chunked/husc_nganh_moi_fix.jsonl"
CORPUS_TIMELINE = "rag2025/data/chunked/husc_2026_timeline_hocba_v1.jsonl"
CORPUS_THONGBAO = "rag2025/data/chunked/husc_thongbao_fix_2026.jsonl"

# Cross-check inputs — never record sources, only verification evidence.
FIXTURE_DIR = "rag2025/tests/fixtures/tables"
XCHECK_TUITION = "rag2025/data/major_codes/husc_tuition_2026_official.json"
XCHECK_DIEM_HISTORY = "rag2025/data/major_codes/diem_chuan_history.json"

HUSC_2026 = 2026
TTTS_PAGES_CHI_TIEU = (24, 25, 26, 27, 28, 29)
TTTS_PAGES_HOC_PHI = (54, 55)
TTTS_PAGES_DIEM = (61, 62, 63)
HD2026_MILESTONE_PAGES = (2, 3, 10)

_MIN_MA_LEN = 6


# --------------------------------------------------------------------------
# small text helpers
# --------------------------------------------------------------------------
def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slug(text: str) -> str:
    """Lowercase ASCII slug usable inside a ``record_id``."""
    text = text.strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "x"


def _vnd(text: str) -> int | None:
    """``'17.485.000'`` -> ``17485000`` (Vietnamese thousands separator)."""
    digits = text.replace(".", "").replace(",", "").strip()
    return int(digits) if digits.isdigit() else None


def _score(text: str) -> float | None:
    """``'16,00'`` -> ``16.0`` (Vietnamese decimal comma)."""
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _int(text: str) -> int | None:
    digits = text.replace(".", "").strip()
    return int(digits) if digits.isdigit() else None


def _year_text(text: str) -> float | None:
    """``'4,5 năm'`` -> ``4.5``; ``'4 năm'`` -> ``4.0``."""
    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _first_page_hit(pages: dict[int, str], needle: str, pages_to_search) -> int | None:
    for page in pages_to_search:
        if needle in pages[page]:
            return page
    return None


# --------------------------------------------------------------------------
# fact container
# --------------------------------------------------------------------------
@dataclass
class Fact:
    """One extracted fact, before it becomes a :class:`CanonicalRecord`."""

    record_id: str
    record_type: RecordType
    admission_year: int
    text: str
    source_document_id: str
    authority: AuthorityLevel
    extractor: str
    source_page: int | None = None
    source_url: str | None = None
    source_chunk_id: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    effective_basis: str = "admission_year_convention"
    values: dict = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    raw_source_text: str = ""

    @property
    def source_kind(self) -> str:
        return "pdf" if self.source_page is not None else "corpus"

    def window(self) -> tuple[date, date | None]:
        if self.effective_from is not None:
            return self.effective_from, self.effective_to
        return date(self.admission_year, 1, 1), date(self.admission_year, 12, 31)

    def to_record(self) -> CanonicalRecord:
        start, end = self.window()
        # A field the extractor could not read is *explicitly missing*: it may
        # never carry a value, or a downstream reader would treat a guess as a
        # fact. Any stray value for a declared-missing field is dropped here.
        values = dict(self.values)
        for field_name in self.missing_fields:
            values[field_name] = None
        return CanonicalRecord(
            record_id=self.record_id,
            record_type=self.record_type,
            version=1,
            admission_year=self.admission_year,
            effective_from=start,
            effective_to=end,
            status=RecordStatus.DRAFT,
            authority_level=self.authority,
            source_document_id=self.source_document_id,
            source_url=self.source_url,
            source_page=self.source_page,
            locale="vi",
            text=self.text,
            values=values,
            missing_fields=list(self.missing_fields),
            warnings=list(self.warnings),
            source_kind=self.source_kind,
            source_chunk_id=self.source_chunk_id,
            raw_source_text=self.raw_source_text[:600],
            extractor=self.extractor,
            seed_version=SEED_VERSION,
            effective_window_basis=self.effective_basis,
        )


@dataclass
class Warning_:
    code: str
    message: str
    source_document_id: str
    source_page: int | None = None
    source_chunk_id: str | None = None
    record_id: str | None = None


# --------------------------------------------------------------------------
# PDF text access (pypdf tier of the pipeline)
# --------------------------------------------------------------------------
class PdfText:
    """Per-page text of one PDF plus the pipeline's document-level parse."""

    def __init__(self, repo_root: Path, document_id: str, cache_dir: Path) -> None:
        self.document_id = document_id
        self.path = repo_root / document_id
        result = parse_pdf(self.path, cache_dir, force="pypdf")
        self.parser = result.parser
        self.page_count = result.page_count
        self.needs_review = result.needs_review
        from pypdf import PdfReader  # local import: only pypdf is installed

        reader = PdfReader(str(self.path))
        self._reader = reader
        self.pages: dict[int, str] = {
            index: (page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        }

    def lines(self, page: int) -> list[str]:
        return [line.strip() for line in self.pages[page].split("\n")]

    def chunks(self, page: int) -> list[tuple[float, float, str]]:
        """``(x, y, text)`` per printed fragment, in text-matrix coordinates.

        pypdf reports the text matrix position of every show-text operation, so
        a fragment's column and line on the printed page survive even where a
        line-oriented read merges merged table cells. Used by the điểm trúng
        tuyển extractor, whose table merges cells vertically.
        """
        out: list[tuple[float, float, str]] = []

        def visit(text: str, cm, tm, font_dict, font_size) -> None:
            if text.strip():
                out.append((round(tm[4], 1), round(tm[5], 1), text.strip()))

        self._reader.pages[page - 1].extract_text(visitor_text=visit)
        return out

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


# Repeated table furniture at the top of every page of a long table (the page
# number plus the header cells). Lines are compared exactly — pypdf emits each
# header cell as its own line — so a data line is never dropped for merely
# containing one of these words.
_PAGE_TOP_FURNITURE = frozenset(
    {
        "STT",
        "Khối",
        "ngành",
        "Mã ngành/",
        "Mã ngành",
        "CTĐT",
        "Cụm",
        "Tên ngành",
        "Tên ngành/Chương trình",
        "đào tạo",
        "Học phí/",
        "tín chỉ",
        "Dự kiến học phí/",
        "năm học",
    }
)


def _split_rows(
    pdf: PdfText,
    pages: tuple[int, ...],
    first_stt: int,
    row_start: re.Pattern[str],
    stop_at_section_header: bool = False,
    start_marker: str | None = None,
    end_marker: str | None = None,
    end_pattern: re.Pattern[str] | None = None,
) -> list[tuple[int, int, str]]:
    """Blocks of consecutive rows keyed by their printed STT.

    A row begins on a line whose leading number equals the next expected STT
    (``row_start``); every following line belongs to that row until the next
    row begins. ``start_marker`` skips page furniture before the table starts,
    ``end_marker`` / ``end_pattern`` stop the run (an exact substring / a regex
    matched against the line). Returns ``(page, stt, joined_block_text)`` in
    document order.

    The leading run of each page is dropped: a long table repeats its header and
    its page number at the top of every page, and leaving the page number inside
    the previous row's block is how a printed "chỉ tiêu 62" cell turns into
    "62 25" and reads as 25 (row 3, pages 24-25 of the chỉ tiêu table).
    """
    out: list[tuple[int, int, str]] = []
    expected = first_stt
    current_stt: int | None = None
    current: list[str] = []
    current_page: int | None = None
    started = start_marker is None

    def flush() -> None:
        nonlocal current_stt, current, current_page
        if current_stt is not None:
            out.append((current_page or 0, current_stt, _norm_ws(" ".join(current))))
        current_stt, current, current_page = None, [], None

    for page in pages:
        at_page_top = True
        for line in pdf.lines(page):
            if at_page_top:
                if line == "" or line == str(page) or line in _PAGE_TOP_FURNITURE:
                    continue
                at_page_top = False
            if not started:
                if start_marker is not None and start_marker in line:
                    started = True
                continue
            if end_marker is not None and end_marker in line:
                flush()
                return out
            if end_pattern is not None and end_pattern.match(line):
                flush()
                return out
            if stop_at_section_header and re.match(r"^[IVX]+\s*\.\s", line):
                flush()
                return out
            match = row_start.match(line)
            if match and int(match.group(1)) == expected:
                flush()
                current_stt, current_page = expected, page
                current = [line]
                expected += 1
            elif current_stt is not None:
                current.append(line)
    flush()
    return out


# --------------------------------------------------------------------------
# PDF extractors — THONG_TIN_TUYEN_SINH_202611.pdf
# --------------------------------------------------------------------------
_MA_RE = re.compile(r"(?<![\d.])(\d{6,7}[A-Z]{0,3})(?![\d])")
_TO_HOP_RE = re.compile(r"\b([A-Z]\d{2})\b")
_SUBJECT_ITEM_RE = re.compile(r"(?:^|\s)\d\.\s+([^\d]+?)(?=\s+\d\.\s|\s+[A-Z]\d{2}\b|$)")
_ROW_STT_ANY = re.compile(r"^(\d{1,2})\s+(?=\S)")
_ROW_STT_HOCPHI = re.compile(r"^(\d{1,2})\s+(IV|V|VII)\s+(?=\S)")
_ROW_STT_BARE = re.compile(r"^(\d{1,2})\s*$")
_SCORE_RE = re.compile(
    r"\b(HB|THPT|NL|KH|KHHB|KHĐT)\s+(\d{1,4})\s+(\d{1,4})\s+(\d{1,2},\d{2})"
    r"(?:\s+(\d{1,4})\s+(\d{1,4})\s+(\d{1,2},\d{2}))?"
)
_SUBROW_RE = re.compile(r"(\d{1,4})\s+(\d{1,4})\s+(\d{1,2},\d{2})\s*$")


def extract_pdf_chi_tieu_tables(pdf: PdfText, warn: list[Warning_]) -> list[Fact]:
    """Pages 24-29: mã ngành + tổ hợp xét tuyển + chỉ tiêu, one row per ngành."""
    rows = _split_rows(
        pdf,
        TTTS_PAGES_CHI_TIEU,
        first_stt=1,
        row_start=_ROW_STT_ANY,
        start_marker="TRƯỜNG ĐẠI HỌCKHOA HỌC",
    )
    facts: list[Fact] = []
    for page, stt, block in rows:
        ma_match = _MA_RE.search(block)
        ma = ma_match.group(1) if ma_match else None
        # the ngành name is what sits between the STT and the mã ngành
        name = None
        if ma:
            head = block.split(ma)[0]
            name = _norm_ws(re.sub(r"^\d{1,2}\s+", "", head)).strip(" -–—")
        chi_tieu = None
        numbers = re.findall(r"(?<![\d.,])(\d{2,4})(?![\d.,])", block.split(ma or "\0")[-1])
        if numbers:
            chi_tieu = int(numbers[-1])
        codes = _TO_HOP_RE.findall(block)
        subjects = [_norm_ws(s) for s in _SUBJECT_ITEM_RE.findall(block)]
        raw = block[:400]

        missing: list[str] = []
        if not ma:
            missing.append("ma_nganh")
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"chỉ tiêu table row {stt} (page {page}): no mã ngành cell found",
                    pdf.document_id,
                    page,
                )
            )
        if chi_tieu is None:
            missing.append("chi_tieu")
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"chỉ tiêu table row {stt} (page {page}): no chỉ tiêu cell found",
                    pdf.document_id,
                    page,
                )
            )
        if not codes:
            missing.append("to_hop_codes")
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"chỉ tiêu table row {stt} (page {page}): tổ hợp cell is empty on the "
                    "page — record written with to_hop_codes missing",
                    pdf.document_id,
                    page,
                )
            )
        if codes and subjects and len(codes) != len(subjects):
            missing.append("to_hop_subjects")
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"chỉ tiêu table row {stt} (page {page}): {len(codes)} tổ hợp codes vs "
                    f"{len(subjects)} subject lines — subjects dropped, codes kept",
                    pdf.document_id,
                    page,
                )
            )
        key = (ma or f"khong-ma-{_slug(name or str(stt))}").lower()
        base = {
            "ma_nganh": ma,
            "ten_nganh": name,
            "chi_tieu": chi_tieu,
            "to_hop_codes": codes or None,
            "to_hop_subjects": subjects if (codes and len(codes) == len(subjects)) else None,
        }
        common = dict(
            source_document_id=pdf.document_id,
            authority=AuthorityLevel.HUSC_OFFICIAL,
            source_page=page,
            source_chunk_id=None,
            raw_source_text=raw,
        )
        facts.append(
            Fact(
                record_id=f"NGANH:nganh:{key}:{HUSC_2026}:ttts-p{page}",
                record_type=RecordType.NGANH,
                admission_year=HUSC_2026,
                text=(
                    f"Ngành {name or '(không rõ tên)'}, mã ngành {ma or '(trang không in mã)'} "
                    "— Trường Đại học Khoa học, Đại học Huế, tuyển sinh năm 2026."
                ),
                extractor="pdf_chi_tieu_table/nganh",
                values={"ma_nganh": ma, "ten_nganh": name, "stt_tren_bang": stt},
                missing_fields=tuple(missing),
                **common,
            )
        )
        facts.append(
            Fact(
                record_id=f"CHI_TIEU:nganh:{key}:{HUSC_2026}:ttts-p{page}",
                record_type=RecordType.CHI_TIEU,
                admission_year=HUSC_2026,
                text=(
                    f"Chỉ tiêu tuyển sinh năm 2026 ngành {name or ma or stt}"
                    f" (mã ngành {ma or 'không in trên trang'}): "
                    f"{chi_tieu if chi_tieu is not None else 'không đọc được'}."
                ),
                extractor="pdf_chi_tieu_table/chi_tieu",
                values=dict(base, stt_tren_bang=stt),
                missing_fields=tuple(missing),
                **common,
            )
        )
        facts.append(
            Fact(
                record_id=f"TO_HOP:nganh:{key}:{HUSC_2026}:ttts-p{page}",
                record_type=RecordType.TO_HOP,
                admission_year=HUSC_2026,
                text=(
                    f"Tổ hợp xét tuyển năm 2026 ngành {name or ma or stt}"
                    f" (mã ngành {ma or 'không in trên trang'}): "
                    + (", ".join(codes) if codes else "không đọc được từ bảng")
                    + (
                        " — "
                        + "; ".join(f"{c}: {s}" for c, s in zip(codes, subjects))
                        if codes and len(codes) == len(subjects)
                        else ""
                    )
                ),
                extractor="pdf_chi_tieu_table/to_hop",
                values=dict(base, stt_tren_bang=stt),
                missing_fields=tuple(missing),
                **common,
            )
        )
    return facts


_AMOUNT_TOKEN_RE = re.compile(r"(?<![\d.])(?:\d{1,3}(?:\.\d{3})+|0)(?![\d])")


def extract_pdf_hoc_phi(pdf: PdfText, warn: list[Warning_]) -> list[Fact]:
    """Pages 54-55: học phí/tín chỉ and dự kiến học phí/năm học per ngành.

    The HUSC section of this table ends on page 55 with its footnote (lines
    starting ``-``) and the next school's section (``* Trường …``), so the run
    stops at that marker instead of swallowing the next school's table. The two
    amounts are taken as the last two money-shaped tokens of the row, because
    the footnote text that follows the last row also contains numbers.
    """
    rows = _split_rows(
        pdf,
        TTTS_PAGES_HOC_PHI,
        first_stt=1,
        row_start=_ROW_STT_HOCPHI,
        start_marker="* Trường Đại học Khoa học",
        end_pattern=re.compile(r"^\*\s*Trường\b"),
    )
    facts: list[Fact] = []
    for page, stt, block in rows:
        khoi = block.split()[1] if len(block.split()) > 1 else None
        ma_match = _MA_RE.search(block)
        ma = ma_match.group(1) if ma_match else None
        name = per_credit = per_year = None
        amounts = list(_AMOUNT_TOKEN_RE.finditer(block))
        if ma and len(amounts) >= 2:
            per_credit = amounts[-2].group(0)
            per_year = amounts[-1].group(0)
            tail = block[block.index(ma) + len(ma) : amounts[-2].start()]
            name = _norm_ws(tail) or None
        missing: list[str] = []
        if not ma:
            missing.append("ma_nganh")
        if name is None:
            missing.append("ten_nganh")
        if per_credit is None:
            missing.append("hoc_phi_tin_chi")
        if per_year is None:
            missing.append("hoc_phi_nam")
        if missing:
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"học phí table row {stt} (page {page}): could not read "
                    f"{', '.join(missing)} — record written with those fields missing",
                    pdf.document_id,
                    page,
                )
            )
        key = (ma or f"khong-ma-{_slug(name or str(stt))}").lower()
        facts.append(
            Fact(
                record_id=f"HOC_PHI:nganh:{key}:{HUSC_2026}:ttts-p{page}",
                record_type=RecordType.HOC_PHI,
                admission_year=HUSC_2026,
                text=(
                    f"Học phí năm học 2026-2027 ngành {name or ma or stt}"
                    f" (mã ngành {ma or 'không in trên trang'}"
                    + (f", khối {khoi}" if khoi else "")
                    + "): "
                    + (
                        f"{per_credit} đồng/tín chỉ, dự kiến {per_year} đồng/năm học."
                        if per_credit and per_year
                        else "không đọc được từ bảng."
                    )
                ),
                extractor="pdf_hoc_phi_table",
                values={
                    "ma_nganh": ma,
                    "ten_nganh": name,
                    "khoi_nganh": khoi,
                    "hoc_phi_tin_chi": per_credit,
                    "hoc_phi_tin_chi_vnd": _vnd(per_credit) if per_credit else None,
                    "hoc_phi_nam": per_year,
                    "hoc_phi_nam_vnd": _vnd(per_year) if per_year else None,
                    "stt_tren_bang": stt,
                },
                missing_fields=tuple(missing),
                source_document_id=pdf.document_id,
                authority=AuthorityLevel.HUSC_OFFICIAL,
                source_page=page,
                raw_source_text=block[:400],
            )
        )
    return facts


# --------------------------------------------------------------------------
# điểm trúng tuyển table (pages 61-63) — read by printed column
# --------------------------------------------------------------------------
# The table draws one program row as *two* printed lines (the "học bạ" line and
# the "THPT" line) and merges the chỉ tiêu / số nhập học cells across both, so a
# line-oriented read can neither place a cell in its column nor say which
# phương thức a score belongs to. pypdf's text matrix reports the printed x/y of
# every fragment, and the printed header row sits at the same x as its column
# ('Chỉ' x=279.6, 'Số' 318.4/478.8, 'Điểm' 351.7/510.0, 'Phương' 231.9/392.2,
# 'Năm 2024' 303.1, 'Năm 2025' 461.1), so the windows below are measured from
# the header itself. Attribution rules, each one printed evidence:
#   * a row is anchored on its printed mã ngành cell;
#   * a 2024 score belongs to the phương thức whose printed label (HB / THPT) it
#     sits beside (within _DIEM_SCORE_TOLERANCE pt) — every row prints both;
#   * a 2024 phương thức label belongs to the row whose centre it is nearest to
#     (the label of a two-line row prints 18.5 pt above or below that centre, and
#     rows are 34 pt apart, so a per-row window would also catch the neighbour);
#   * the 2025 phương thức cell is a *merged* cell drawn inside its own frame, so
#     the y pypdf reports for it belongs to that frame rather than to the page
#     (rows 1, 8 and 18 print one; their fragments land at y=301.2, 387.3, 424.6
#     while the rows sit at 505.3, 685.9, 683.0). It is therefore read from the
#     row's printed block in the reading-order text, and cross-checked against the
#     cells printed in the 2025 column — a row that claims one without a printed
#     cell to back it, or a printed cell no row claims, is warned about;
#   * a score printed with '*' is that phương thức's đợt bổ sung score (page 63
#     footnote: "(*): Điểm trúng tuyển đợt bổ sung");
#   * a cell that matches no column, or a score that sits beside no label, is
#     warned about — never guessed, never borrowed from the neighbouring row.
_DIEM_COLUMNS: dict[str, tuple[float, float]] = {
    "stt": (24.0, 40.0),
    "ma_nganh": (44.0, 108.0),
    "ten_nganh": (108.0, 228.0),
    "phuong_thuc_2024": (228.0, 268.0),
    "chi_tieu_2024": (270.0, 302.0),
    "so_nhap_hoc_2024": (303.0, 337.0),
    "diem_2024": (338.0, 366.0),
    "phuong_thuc_2025": (384.0, 426.0),
    "chi_tieu_2025": (427.0, 459.0),
    "so_nhap_hoc_2025": (460.0, 496.0),
    "diem_2025": (497.0, 532.0),
}
_DIEM_SECTION_MARKER = "Đại học Khoa học"
_DIEM_ROW_HALF = 30.0  # every cell of a row lies within 30 pt of its mã ngành
_DIEM_LABEL_ROW_TOLERANCE = 30.0  # a 2024 label sits 8.5-24.3 pt from its row
_DIEM_LABEL_TIE = 2.0  # a cell equidistant from two rows belongs to neither
_DIEM_SCORE_TOLERANCE = 11.0  # a 2024 score sits ~9 pt from its label
_DIEM_2025_TOLERANCE = 15.0  # the single 2025 score cell sits at the centre
_METHODS: dict[str, tuple[str, str]] = {
    "HB": ("hoc_ba", "Học bạ (HB)"),
    "THPT": ("thi_thpt", "Thi tốt nghiệp THPT"),
}
_METHOD_LABELS: dict[str, str] = {key: label for key, label in _METHODS.values()}
_RE_CHUNK_MA = re.compile(r"^\d{6,7}[A-Z]{0,4}$")
_RE_CHUNK_INT = re.compile(r"^\d{1,4}$")
_RE_CHUNK_SCORE = re.compile(r"^\d{1,2},\d{2}\*?$")
_RE_NAME_VARIANT = re.compile(
    r"\(\s*(?:CTĐT|chương trình đào tạo)\s+([^)]+?)\s*\)\s*$", re.IGNORECASE
)
_RE_NUMERIC_LINE = re.compile(r"^[\d\s,.*()]+$")
_RE_MA_TOKEN = re.compile(r"(?<![\d.])\d{6,7}[A-Z]{0,4}(?![\d])")
_RE_LABEL_TOKEN = re.compile(r"^[A-ZĐ]{2,6}$")
_RE_PRINTED_SCORE = re.compile(r"\d{1,2},\d{2}\*?")
_DEFAULT_PROGRAM_VARIANTS = frozenset({"cu-nhan", "cu-nhan-chinh-quy", "chinh-quy"})


@dataclass
class _DiemRow:
    """One printed program row of the điểm trúng tuyển table."""

    page: int
    y: float
    ma: str | None
    name: str | None
    name_by_coordinates: str | None
    raw_block: str
    stt: str | None
    labels_2024: list[tuple[str, float]]  # (phương thức key, label y)
    label_2025: tuple[str | None, str] | None  # (phương thức key or None, printed)
    scores_2024: dict[str, list[tuple[float, float, str]]]  # key -> [(y, x, text)]
    scores_2025: list[tuple[float, float, str]]
    cells: dict[str, tuple[float, float, str] | None]


def _diem_column(x: float) -> str | None:
    """Which printed column the fragment at ``x`` belongs to."""
    for column, (low, high) in _DIEM_COLUMNS.items():
        if low <= x <= high:
            return column
    return None


def _printed_lines(
    fragments: list[tuple[float, float, str]], joiner: str = " "
) -> list[tuple[float, str]]:
    """Fragments of one column merged per printed line, top line first.

    pypdf splits a printed cell across text-showing runs ('THP' + 'T',
    'L' + 'ịch sử'), so a cell is only whole once its fragments on one line are
    joined — with a space inside a name, with nothing inside a label.
    """
    lines: list[tuple[float, str]] = []
    parts: list[str] = []
    line_y: float | None = None
    for x, y, text in sorted(fragments, key=lambda f: (-f[1], f[0])):
        if line_y is None or abs(y - line_y) <= 1.0:
            parts.append(text)
            line_y = y if line_y is None else line_y
        else:
            lines.append((line_y, _norm_ws(joiner.join(parts))))
            parts, line_y = [text], y
    if parts and line_y is not None:
        lines.append((line_y, _norm_ws(joiner.join(parts))))
    return lines


def _nearest_anchor(y: float, anchor_ys: list[float]) -> tuple[int | None, float, float]:
    """Nearest row centre to ``y``: ``(index, distance, runner-up distance)``.

    The runner-up distance is how a cell that sits between two rows — and so
    cannot be attributed to either — is told apart from one that clearly belongs
    to a row.
    """
    if not anchor_ys:
        return None, float("inf"), float("inf")
    ranked = sorted(range(len(anchor_ys)), key=lambda index: abs(y - anchor_ys[index]))
    best = abs(y - anchor_ys[ranked[0]])
    second = abs(y - anchor_ys[ranked[1]]) if len(ranked) > 1 else float("inf")
    return ranked[0], best, second


def _plain_row(
    ma: str, occurrence: int, lines: list[str]
) -> tuple[str | None, list[str]]:
    """The ngành name and the whole printed block of the row carrying ``ma``.

    The name is read from the reading-order text, where pypdf re-joins what the
    coordinate view splits ('L' + 'ịch sử' -> 'Lịch sử', 'Truyền thôn' + 'g số'
    -> 'Truyền thông số'); the numbers are read from the coordinate view.
    """
    seen = 0
    for index, line in enumerate(lines):
        if not re.search(rf"(?<![\d.]){re.escape(ma)}(?![\d])", line):
            continue
        seen += 1
        if seen != occurrence:
            continue
        block: list[str] = []
        for following in lines[index:]:
            stripped = following.strip()
            if block and (_RE_MA_TOKEN.search(stripped) or stripped.startswith("(*")):
                break
            block.append(stripped)
            if len(block) > 14:
                break
        while block and (not block[-1] or block[-1].isdigit()):
            block.pop()
        if not block:
            return None, []
        head = block[0][block[0].index(ma) + len(ma) :].strip()
        parts = [token for token in head.split() if not _RE_LABEL_TOKEN.match(token)]
        for following in block[1:]:
            if _RE_NUMERIC_LINE.match(following) or following in _METHODS:
                break
            parts.append(following)
        return (_norm_ws(" ".join(parts)) or None), block
    return None, []


def _plain_block_label(block: list[str]) -> str | None:
    """The row's 2025 phương thức token, read from the row's printed block.

    The merged 2025 "phương thức" cell is drawn inside its own cell frame, so the
    y pypdf's visitor reports for it belongs to that frame rather than to the
    page: the three printed ones land at y=301.2, 387.3 and 424.6 while their
    rows sit at 505.3, 685.9 and 683.0. The layout pass behind ``pdf.lines``
    places the token in the right row's block, so that is what this reads — a
    standalone method line printed after the row's 2024 score and before its 2025
    cells. The 2024 labels print attached to their own score line ('THPT 16,25'),
    so they are not standalone.
    """
    cells = [line for line in block if line]
    for index in range(1, len(cells) - 1):
        token = cells[index]
        if not _RE_LABEL_TOKEN.match(token):
            continue
        if _RE_PRINTED_SCORE.search(cells[index - 1]) and re.match(
            r"^\d{1,4}\b", cells[index + 1]
        ):
            return token
    return None


def _printed_2025_labels(
    visible: list[tuple[int, float, float, str]]
) -> list[str]:
    """The phương thức tokens printed in the table's 2025 column."""
    return [
        text
        for _, text in _printed_lines(
            [
                (x, y, text)
                for _, x, y, text in visible
                if _diem_column(x) == "phuong_thuc_2025"
            ],
            joiner="",
        )
        if _RE_LABEL_TOKEN.match(text)
    ]


def _diem_table_start(
    chunks: list[tuple[float, float, str]],
    page: int,
    warn: list[Warning_],
    document_id: str,
) -> int | None:
    """Index of the table's first printed chunk on ``page``, or None to skip it.

    Page 61 opens with the tail of the *previous* school's table (its rows 18-20)
    and every page repeats the header row, so the HUSC table starts either right
    after the "Đại học Khoa học" section marker or at the first printed mã ngành.
    Without the marker the two tables cannot be told apart, so the page is skipped
    rather than misread.
    """
    if page == TTTS_PAGES_DIEM[0]:
        marker = [
            index
            for index, (_, _, text) in enumerate(chunks)
            if _DIEM_SECTION_MARKER in text
        ]
        if marker:
            return max(marker) + 1
        warn.append(
            Warning_(
                "section_marker_missing",
                f"điểm trúng tuyển table: the HUSC section marker "
                f"({_DIEM_SECTION_MARKER}) was not found on page {page}, so the rows of "
                "the preceding school's table cannot be told apart from HUSC's; the page "
                "is skipped rather than misread",
                document_id,
                page,
            )
        )
        return None
    for index, (x, _, text) in enumerate(chunks):
        if _diem_column(x) == "ma_nganh" and _RE_CHUNK_MA.match(text):
            return index
    return None


def _diem_rows(pdf: PdfText, warn: list[Warning_]) -> list[_DiemRow]:
    """Every HUSC program row of the điểm trúng tuyển table, cells by column."""
    rows: list[_DiemRow] = []
    for page in TTTS_PAGES_DIEM:
        chunks = pdf.chunks(page)  # visitor order == printed reading order
        floor = _diem_table_start(chunks, page, warn, pdf.document_id)
        if floor is None:
            continue
        visible = [
            (index, x, y, text)
            for index, (x, y, text) in enumerate(chunks)
            if index >= floor
        ]
        anchors = [
            (y, text)
            for _, x, y, text in visible
            if _diem_column(x) == "ma_nganh" and _RE_CHUNK_MA.match(text)
        ]
        if not anchors:
            warn.append(
                Warning_(
                    "table_rows_missing",
                    f"điểm trúng tuyển table: no mã ngành row found on page {page}",
                    pdf.document_id,
                    page,
                )
            )
            continue
        rows.extend(_diem_page_rows(pdf, page, visible, anchors, warn))
    return rows


def _assign_diem_labels_2024(
    visible: list[tuple[int, float, float, str]],
    anchor_ys: list[float],
    warn: list[Warning_],
    document_id: str,
    page: int,
) -> dict[int, list[tuple[str, float]]]:
    """Each 2024 phương thức label of one page, given to the row it sits beside.

    A per-row window would let a neighbouring row's label in: rows 9 and 10 of
    page 62 are 34 pt apart while a label prints 8.5-18.5 pt from its own row's
    centre. Fragments are joined without a separator because pypdf splits the
    label itself ('THP' + 'T').
    """
    out: dict[int, list[tuple[str, float]]] = {}
    for y, text in _printed_lines(
        [
            (x, y, text)
            for _, x, y, text in visible
            if _diem_column(x) == "phuong_thuc_2024"
        ],
        joiner="",
    ):
        if text not in _METHODS:
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"điểm trúng tuyển table (page {page}): the 2024 phương thức cell "
                    f"'{text}' at y={y} is not one of the printed methods "
                    f"({', '.join(_METHODS)}) — the scores beside it are left unclaimed",
                    document_id,
                    page,
                )
            )
            continue
        index, best, second = _nearest_anchor(y, anchor_ys)
        if index is None or best > _DIEM_LABEL_ROW_TOLERANCE:
            warn.append(
                Warning_(
                    "unattributed_table_cell",
                    f"điểm trúng tuyển table (page {page}): the 2024 label '{text}' at "
                    f"y={y} sits beside no row of the HUSC table",
                    document_id,
                    page,
                )
            )
            continue
        if second - best <= _DIEM_LABEL_TIE:
            warn.append(
                Warning_(
                    "ambiguous_table_cell",
                    f"điểm trúng tuyển table (page {page}): the 2024 label '{text}' at "
                    f"y={y} is {best} pt from one row and {second} pt from the next, so it "
                    "is not attributed",
                    document_id,
                    page,
                )
            )
            continue
        out.setdefault(index, []).append((_METHODS[text][0], y))
    return out


def _assign_diem_scores(
    rows: list[_DiemRow],
    visible: list[tuple[int, float, float, str]],
    anchor_ys: list[float],
    warn: list[Warning_],
    document_id: str,
    page: int,
) -> None:
    """Give every printed score to the row and the label it is printed beside."""
    labels = [
        (index, method_key, label_y)
        for index, row in enumerate(rows)
        for method_key, label_y in row.labels_2024
    ]
    scores_2024 = [
        (x, y, text)
        for _, x, y, text in visible
        if _diem_column(x) == "diem_2024" and _RE_CHUNK_SCORE.match(text)
    ]
    for x, y, text in scores_2024:
        nearest: tuple[int, str, float] | None = None
        for index, method_key, label_y in labels:
            delta = abs(y - label_y)
            if delta <= _DIEM_SCORE_TOLERANCE and (
                nearest is None or delta < abs(y - nearest[2])
            ):
                nearest = (index, method_key, label_y)
        if nearest is None:
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"điểm trúng tuyển table (page {page}): the 2024 cell '{text}' printed "
                    f"at x={x} y={y} sits beside no HB/THPT label — no record claims it",
                    document_id,
                    page,
                )
            )
            continue
        index, method_key, _ = nearest
        rows[index].scores_2024.setdefault(method_key, []).append((y, x, text))

    for x, y, text in [
        (x, y, text)
        for _, x, y, text in visible
        if _diem_column(x) == "diem_2025" and _RE_CHUNK_SCORE.match(text)
    ]:
        index, best, second = _nearest_anchor(y, anchor_ys)
        if index is None or best > _DIEM_2025_TOLERANCE:
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"điểm trúng tuyển table (page {page}): the 2025 cell '{text}' printed "
                    f"at x={x} y={y} lies beside no mã ngành row — no record claims it",
                    document_id,
                    page,
                )
            )
            continue
        if second - best <= _DIEM_LABEL_TIE:
            warn.append(
                Warning_(
                    "ambiguous_table_cell",
                    f"điểm trúng tuyển table (page {page}): the 2025 cell '{text}' at "
                    f"y={y} is {best} pt from one row and {second} pt from the next, so it "
                    "is not attributed",
                    document_id,
                    page,
                )
            )
            continue
        rows[index].scores_2025.append((y, x, text))


def _diem_page_rows(
    pdf: PdfText,
    page: int,
    visible: list[tuple[int, float, float, str]],
    anchors: list[tuple[float, str]],
    warn: list[Warning_],
) -> list[_DiemRow]:
    """Build one page's rows: cells by printed column, scores by their label."""
    anchor_ys = [y for y, _ in anchors]
    labels_2024 = _assign_diem_labels_2024(
        visible, anchor_ys, warn, pdf.document_id, page
    )
    available = _printed_2025_labels(visible)
    rows: list[_DiemRow] = []
    names: dict[str, int] = {}
    for position, (y0, ma) in enumerate(anchors):
        names[ma] = names.get(ma, 0) + 1
        name, block = _plain_row(ma, names[ma], pdf.lines(page))
        fragments = [
            (x, y, text) for _, x, y, text in visible if abs(y - y0) <= _DIEM_ROW_HALF
        ]
        name_by_coordinates = (
            " ".join(
                text
                for _, text in _printed_lines(
                    [
                        (x, y, text)
                        for x, y, text in fragments
                        if _diem_column(x) == "ten_nganh"
                    ]
                )
            )
            or None
        )
        cells: dict[str, tuple[float, float, str] | None] = {}
        for column in (
            "chi_tieu_2024",
            "so_nhap_hoc_2024",
            "chi_tieu_2025",
            "so_nhap_hoc_2025",
        ):
            candidates = [
                (abs(y - y0), (x, y, text))
                for x, y, text in fragments
                if _diem_column(x) == column and _RE_CHUNK_INT.match(text)
            ]
            cells[column] = (
                min(candidates, key=lambda item: item[0])[1] if candidates else None
            )
        stt_cells = [
            (x, y, text)
            for x, y, text in fragments
            if _diem_column(x) == "stt" and _RE_CHUNK_INT.match(text)
        ]
        stt = (
            min(stt_cells, key=lambda cell: abs(cell[1] - y0))[2] if stt_cells else None
        )
        token = _plain_block_label(block)
        if token is None:
            label_2025 = None
        elif token in available:
            available.remove(token)
            label_2025 = (_METHODS[token][0] if token in _METHODS else None, token)
        else:
            label_2025 = None
            warn.append(
                Warning_(
                    "unattributed_table_cell",
                    f"điểm trúng tuyển table row {stt} (page {page}): the row's block prints "
                    f"the 2025 phương thức '{token}' but no such cell is printed in the "
                    "2025 column of the HUSC table — the row's 2025 phương thức is unset",
                    pdf.document_id,
                    page,
                )
            )
        if label_2025 is not None and label_2025[0] is None:
            warn.append(
                Warning_(
                    "unparsed_table_cell",
                    f"điểm trúng tuyển table row {stt} (page {page}): the 2025 phương thức "
                    f"'{label_2025[1]}' is not one of the printed methods "
                    f"({', '.join(_METHODS)}) — the record keeps the printed token and "
                    "leaves the method key unset",
                    pdf.document_id,
                    page,
                )
            )
        rows.append(
            _DiemRow(
                page=page,
                y=y0,
                ma=ma,
                name=name,
                name_by_coordinates=name_by_coordinates,
                raw_block=" | ".join(block),
                stt=stt,
                labels_2024=labels_2024.get(position, []),
                label_2025=label_2025,
                scores_2024={},
                scores_2025=[],
                cells=cells,
            )
        )
    _assign_diem_scores(rows, visible, anchor_ys, warn, pdf.document_id, page)
    for leftover in available:
        warn.append(
            Warning_(
                "unattributed_table_cell",
                f"điểm trúng tuyển table (page {page}): the 2025 phương thức cell "
                f"'{leftover}' is printed in the 2025 column but belongs to no row of the "
                "HUSC table",
                pdf.document_id,
                page,
            )
        )
    _check_diem_cells_accounted(rows, visible, anchor_ys, warn, pdf.document_id, page)
    return rows


def _check_diem_cells_accounted(
    rows: list[_DiemRow],
    visible: list[tuple[int, float, float, str]],
    anchor_ys: list[float],
    warn: list[Warning_],
    document_id: str,
    page: int,
) -> None:
    """Every printed cell of the table must land in exactly one row's record.

    The rows are read by column, so a cell that matches no column — or an integer
    cell of a row whose mã ngành was not detected — would otherwise be dropped
    without a trace.
    """
    for column in (
        "chi_tieu_2024",
        "so_nhap_hoc_2024",
        "chi_tieu_2025",
        "so_nhap_hoc_2025",
    ):
        printed = [
            (x, y, text)
            for _, x, y, text in visible
            if _diem_column(x) == column and _RE_CHUNK_INT.match(text)
        ]
        claimed = {
            (round(cell[0], 1), round(cell[1], 1), cell[2])
            for row in rows
            for cell in [row.cells.get(column)]
            if cell is not None
        }
        for x, y, text in printed:
            if (round(x, 1), round(y, 1), text) in claimed:
                continue
            warn.append(
                Warning_(
                    "unattributed_table_cell",
                    f"điểm trúng tuyển table (page {page}): the {column} cell '{text}' at "
                    f"x={x} y={y} was not attributed to any row",
                    document_id,
                    page,
                )
            )

def _skeleton(text: str) -> str:
    """Diacritic-free alphanumeric form, for comparing two spellings of a name."""
    return _slug(text).replace("-", "")


def _diem_key(row: _DiemRow) -> str:
    """``record_id`` fragment for one row: mã ngành plus any non-default variant."""
    base = (row.ma or f"khong-ma-{_slug(row.name or f'y{int(row.y)}')}").lower()
    variant = _RE_NAME_VARIANT.search(row.name or "")
    if variant:
        slug = _slug(variant.group(1))
        if slug not in _DEFAULT_PROGRAM_VARIANTS:
            base = f"{base}-{slug[:40]}"
    return base


def _split_diem_scores(
    scores: list[tuple[float, float, str]],
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """The row's main score and its đợt bổ sung scores (printed with ``*``)."""
    main = [score for score in scores if not score[2].endswith("*")]
    bo_sung = [score for score in scores if score[2].endswith("*")]
    return main, bo_sung


def _printed_method_label(label_2025: tuple[str | None, str] | None) -> str | None:
    """The phương thức a row's 2025 score is printed under, as the page prints it.

    A token the table does not use for a known phương thức is kept verbatim
    rather than mapped to the nearest known one.
    """
    if label_2025 is None:
        return None
    key, printed = label_2025
    return _METHOD_LABELS.get(key, printed)


def extract_pdf_diem_chuan(pdf: PdfText, warn: list[Warning_]) -> list[Fact]:
    """Pages 61-63: điểm trúng tuyển / chỉ tiêu / số nhập học 2024 and 2025.

    One record per (ngành, year, phương thức) with a score. Where the printed
    page does not state the phương thức of a score (the 2025 column is blank on
    most rows), the record says so instead of assuming one.
    """
    facts: list[Fact] = []
    for row in _diem_rows(pdf, warn):
        key = _diem_key(row)
        for method_key, label_y in row.labels_2024:
            scores = row.scores_2024.get(method_key, [])
            if not scores:
                warn.append(
                    Warning_(
                        "blank_table_cell",
                        f"điểm trúng tuyển table row {row.stt} (page {row.page}): the 2024 "
                        f"{method_key} label is printed but its score cell is blank",
                        pdf.document_id,
                        row.page,
                    )
                )
                continue
            facts.append(
                _diem_fact(
                    pdf,
                    row,
                    key,
                    year=2024,
                    method_key=method_key,
                    method_label=_METHOD_LABELS[method_key],
                    method_basis="printed_label_2024_column",
                    scores=scores,
                    label_y=label_y,
                    warn=warn,
                )
            )
        for y, x, text in row.scores_2025:
            facts.append(
                _diem_fact(
                    pdf,
                    row,
                    key,
                    year=2025,
                    method_key=row.label_2025[0] if row.label_2025 else None,
                    method_label=_printed_method_label(row.label_2025),
                    method_basis=(
                        "printed_label_2025_column"
                        if row.label_2025
                        else "phuong_thuc_khong_in_tren_trang"
                    ),
                    scores=[(y, x, text)],
                    label_y=None,
                    warn=warn,
                )
            )
        if not row.labels_2024 and not row.scores_2025:
            warn.append(
                Warning_(
                    "table_row_without_cells",
                    f"điểm trúng tuyển table row {row.stt} (page {row.page}): no 2024 "
                    "phương thức label and no 2025 score could be read",
                    pdf.document_id,
                    row.page,
                )
            )
    return facts


def _diem_fact(
    pdf: PdfText,
    row: _DiemRow,
    key: str,
    *,
    year: int,
    method_key: str | None,
    method_label: str | None,
    method_basis: str,
    scores: list[tuple[float, float, str]],
    label_y: float | None,
    warn: list[Warning_],
) -> Fact:
    """One điểm trúng tuyển record, with every unread cell left explicitly absent."""
    main, bo_sung = _split_diem_scores(scores)
    missing: list[str] = []
    diem_text: str | None = None
    if len(main) == 1:
        diem_text = main[0][2]
    else:
        missing.append("diem_trung_tuyen")
        warn.append(
            Warning_(
                "unparsed_table_cell",
                f"điểm trúng tuyển table row {row.stt} (page {row.page}): {len(main)} "
                f"unmarked {year} score cells found"
                + (f" ({', '.join(score[2] for score in main)})" if main else "")
                + " — the record is written with điểm trúng tuyển missing",
                pdf.document_id,
                row.page,
            )
        )
    if method_key is None:
        missing.append("phuong_thuc")
    chi_tieu_key = f"chi_tieu_{year}"
    so_nhap_hoc_key = f"so_nhap_hoc_{year}"
    chi_tieu_cell = row.cells.get(chi_tieu_key)
    so_nhap_hoc_cell = row.cells.get(so_nhap_hoc_key)
    chi_tieu = _int(chi_tieu_cell[2]) if chi_tieu_cell else None
    so_nhap_hoc = _int(so_nhap_hoc_cell[2]) if so_nhap_hoc_cell else None
    if chi_tieu_cell is None:
        missing.append("chi_tieu")
        warn.append(
            Warning_(
                "unparsed_table_cell",
                f"điểm trúng tuyển table row {row.stt} (page {row.page}): no chỉ tiêu {year} "
                "cell found — the record is written with that field missing",
                pdf.document_id,
                row.page,
            )
        )
    if so_nhap_hoc_cell is None:
        missing.append("so_nhap_hoc")
        warn.append(
            Warning_(
                "unparsed_table_cell",
                f"điểm trúng tuyển table row {row.stt} (page {row.page}): no số nhập học "
                f"{year} cell found — the record is written with that field missing",
                pdf.document_id,
                row.page,
            )
        )
    if row.name and row.name_by_coordinates:
        if _skeleton(row.name) != _skeleton(row.name_by_coordinates):
            warn.append(
                Warning_(
                    "nganh_name_disagreement",
                    f"điểm trúng tuyển table row {row.stt} (page {row.page}): the ngành name "
                    f"reads '{row.name}' in the reading-order text but "
                    f"'{row.name_by_coordinates}' by printed column — record keeps both",
                    pdf.document_id,
                    row.page,
                )
            )
    if not row.name:
        missing.append("ten_nganh")
        warn.append(
            Warning_(
                "unparsed_table_cell",
                f"điểm trúng tuyển table row {row.stt} (page {row.page}): the ngành name "
                "could not be read — the record is written with that field missing",
                pdf.document_id,
                row.page,
            )
        )
    attribution = {
        "page": row.page,
        "row_center_y": row.y,
        "columns": {
            column: {"x": cell[0], "y": cell[1], "text": cell[2]}
            for column, cell in (
                (chi_tieu_key, chi_tieu_cell),
                (so_nhap_hoc_key, so_nhap_hoc_cell),
            )
            if cell is not None
        },
        "scores": {
            "cells": [
                {"x": x, "y": y, "text": text} for y, x, text in sorted(scores, key=lambda s: -s[0])
            ],
            "method_label_y": label_y,
        },
    }
    detail = [
        f"phương thức {method_label}" if method_label else "phương thức không in trên trang",
        f"điểm trúng tuyển {diem_text if diem_text else '(thiếu)'}",
    ]
    if bo_sung:
        detail.append("đợt bổ sung " + "; ".join(score[2] for score in bo_sung))
    detail.append(f"chỉ tiêu {chi_tieu if chi_tieu is not None else '(thiếu)'}")
    detail.append(f"số nhập học {so_nhap_hoc if so_nhap_hoc is not None else '(thiếu)'}")
    values = {
        "ma_nganh": row.ma,
        "ten_nganh": row.name,
        "ten_nganh_theo_toa_do": row.name_by_coordinates,
        "nam": year,
        "phuong_thuc_key": method_key,
        "phuong_thuc": method_label,
        "phuong_thuc_basis": method_basis,
        "diem_trung_tuyen": diem_text,
        "diem_trung_tuyen_num": _score(diem_text) if diem_text else None,
        "diem_trung_tuyen_bo_sung": [score[2] for score in bo_sung] or None,
        "diem_trung_tuyen_bo_sung_num": [
            _score(score[2]) for score in bo_sung
        ]
        or None,
        "chi_tieu": chi_tieu,
        "so_nhap_hoc": so_nhap_hoc,
        "stt_tren_bang": row.stt,
        "cell_attribution": attribution,
    }
    return Fact(
        record_id=(
            f"DIEM_CHUAN:nganh:{key}:{year}:{method_key or 'khong-xac-dinh'}:ttts-p{row.page}"
        ),
        record_type=RecordType.DIEM_CHUAN,
        admission_year=year,
        authority=AuthorityLevel.PRIOR_YEAR_OFFICIAL,
        text=(
            f"Điểm trúng tuyển năm {year} ngành {row.name or row.ma or row.stt} "
            f"(mã ngành {row.ma or 'trang không in mã'}): " + "; ".join(detail) + "."
        ),
        extractor="pdf_diem_trung_tuyen_table/printed_column",
        values=values,
        missing_fields=tuple(missing),
        source_document_id=pdf.document_id,
        source_page=row.page,
        raw_source_text=row.raw_block,
    )


# --------------------------------------------------------------------------
# PDF extractors — Bộ GD&ĐT documents
# --------------------------------------------------------------------------
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/20\d{2}\b")
_CUE_RE = re.compile(r"(trước|từ ngày|đến|chậm nhất|hoàn thành|trong khoảng)", re.I)
_CITATION_RE = re.compile(r"(Thông tư|Nghị định|Quyết định|Công văn)\s+số", re.I)


def _sentences(text: str) -> list[str]:
    flat = _norm_ws(text)
    parts = re.split(r"(?<=[.;:])\s+", flat)
    out = []
    for part in parts:
        part = _norm_ws(re.sub(r"^[a-zđ]\)\s*", "", part))
        if 25 <= len(part) <= 400:
            out.append(part)
    return out


def _milestone_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    day, month, year = (int(part) for part in match.group(0).split("/"))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_pdf_milestones(pdf: PdfText, warn: list[Warning_]) -> list[Fact]:
    """Student-facing admission milestones in the ministry guidance letter."""
    facts: list[Fact] = []
    seen: set[str] = set()
    for page in HD2026_MILESTONE_PAGES:
        for sentence in _sentences(pdf.pages[page]):
            if not _DATE_RE.search(sentence) or not _CUE_RE.search(sentence):
                continue
            if _CITATION_RE.search(sentence):
                continue
            if "thí sinh" not in sentence.lower():
                continue
            if sentence in seen:
                continue
            seen.add(sentence)
            when = _milestone_date(sentence)
            slug = _slug(sentence)[:60]
            facts.append(
                Fact(
                    record_id=f"THOI_GIAN:toan-truong:{slug}:2026:hd2026-p{page}",
                    record_type=RecordType.THOI_GIAN,
                    admission_year=2026,
                    authority=AuthorityLevel.MINISTRY,
                    text=sentence,
                    extractor="pdf_milestone_sentence",
                    values={
                        "kind": "moc_tuyen_sinh",
                        "scope": "toan_quoc",
                        "moc_ngay": when.isoformat() if when else None,
                    },
                    effective_from=when,
                    effective_to=when,
                    effective_basis="stated_in_source",
                    source_document_id=pdf.document_id,
                    source_page=page,
                    raw_source_text=sentence,
                )
            )
    return facts


def extract_pdf_regulation_effective_date(pdf: PdfText) -> list[Fact]:
    """Thông tư 06/2026: its own effective date, printed on page 1."""
    page = _first_page_hit(pdf.pages, "hiệu lực thi hành", (1,))
    if page is None:
        return []
    match = re.search(
        r"có hiệu lực thi hành kể từ ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})",
        _norm_ws(pdf.pages[page]),
    )
    if not match:
        return []
    when = date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    sentence = (
        f"Thông tư 06/2026/TT-BGDĐT ban hành Quy chế tuyển sinh đại học năm 2026 "
        f"có hiệu lực thi hành kể từ ngày {when.strftime('%d/%m/%Y')} "
        "(thay thế Thông tư 08/2022/TT-BGDĐT và Thông tư 06/2025/TT-BGDĐT)."
    )
    return [
        Fact(
            record_id="THOI_GIAN:toan-truong:hieu-luc-tt06-2026:2026:tt06-p1",
            record_type=RecordType.THOI_GIAN,
            admission_year=2026,
            authority=AuthorityLevel.MINISTRY,
            text=sentence,
            extractor="pdf_regulation_effective_date",
            values={
                "kind": "moc_tuyen_sinh",
                "scope": "toan_quoc",
                "van_ban": "Thông tư 06/2026/TT-BGDĐT",
                "moc_ngay": when.isoformat(),
            },
            effective_from=when,
            effective_basis="stated_in_source",
            source_document_id=pdf.document_id,
            source_page=page,
            raw_source_text=_norm_ws(pdf.pages[page])[:400],
        )
    ]


# --------------------------------------------------------------------------
# Corpus extractors
# --------------------------------------------------------------------------
def _corpus_chunks(repo_root: Path, relpath: str) -> list[dict]:
    """Chunks of one corpus file, de-duplicated by ``chunk_id`` (first wins)."""
    path = repo_root / relpath
    seen: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id and chunk_id not in seen:
            seen[chunk_id] = chunk
    return list(seen.values())


def _corpus_meta(chunk: dict) -> dict:
    meta = chunk.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _corpus_source_url(chunk: dict) -> str | None:
    meta = _corpus_meta(chunk)
    url = meta.get("source_url") or chunk.get("source")
    if isinstance(url, str) and url.startswith("http"):
        return url
    crumbs = chunk.get("breadcrumbs")
    if isinstance(crumbs, list) and crumbs and isinstance(crumbs[0], str):
        if crumbs[0].startswith("http"):
            return crumbs[0]
    return None


_NGANH_CHUNK_RE = re.compile(
    r"^Ngành (?P<ten>.+?) — Mã ngành: (?P<ma>[0-9A-Z]+)\. "
    r"Chỉ tiêu năm 2026: (?P<ct>[\d.]+)\. "
    r"Tổ hợp xét tuyển: (?P<th>[^.]*)\. "
    r"Khối học phí: KHỐI (?P<khoi>[IVX]+) \((?P<hp>[\d,]+) VNĐ/tín chỉ\)\. "
    r"Thời gian đào tạo: (?P<tg>[\d.,]+) năm\."
)


def extract_corpus_nganh_chunks(
    repo_root: Path, relpath: str, warn: list[Warning_]
) -> list[Fact]:
    """``husc_nganh_<mã>_2026`` chunks: one curated fact set per ngành."""
    facts: list[Fact] = []
    for chunk in _corpus_chunks(repo_root, relpath):
        chunk_id = str(chunk.get("chunk_id") or "")
        if not re.fullmatch(r"husc_nganh_[0-9A-Za-z]+_2026", chunk_id):
            continue
        text = _norm_ws(str(chunk.get("text") or ""))
        match = _NGANH_CHUNK_RE.match(text)
        if not match:
            warn.append(
                Warning_(
                    "unparsed_corpus_chunk",
                    f"{relpath}#{chunk_id}: chunk does not match the curated ngành "
                    "pattern — no records emitted from it",
                    f"{relpath}#{chunk_id}",
                    source_chunk_id=chunk_id,
                )
            )
            continue
        ten, ma = match.group("ten"), match.group("ma")
        chi_tieu = _int(match.group("ct"))
        codes = [c.strip() for c in match.group("th").split(",") if c.strip()]
        khoi = match.group("khoi")
        per_credit = _vnd(match.group("hp"))
        years = _year_text(match.group("tg"))
        key = ma.lower()
        src = f"{relpath}#{chunk_id}"
        common = dict(
            source_document_id=src,
            source_url=_corpus_source_url(chunk),
            source_chunk_id=chunk_id,
            authority=AuthorityLevel.HUSC_CANONICAL,
            raw_source_text=text,
        )
        facts.append(
            Fact(
                record_id=f"NGANH:nganh:{key}:2026:canon-nganh",
                record_type=RecordType.NGANH,
                admission_year=2026,
                text=f"Ngành {ten}, mã ngành {ma} — HUSC tuyển sinh 2026.",
                extractor="corpus_nganh_chunk/nganh",
                values={"ma_nganh": ma, "ten_nganh": ten},
                **common,
            )
        )
        facts.append(
            Fact(
                record_id=f"CHI_TIEU:nganh:{key}:2026:canon-nganh",
                record_type=RecordType.CHI_TIEU,
                admission_year=2026,
                text=f"Chỉ tiêu tuyển sinh năm 2026 ngành {ten} (mã ngành {ma}): {chi_tieu}.",
                extractor="corpus_nganh_chunk/chi_tieu",
                values={"ma_nganh": ma, "ten_nganh": ten, "chi_tieu": chi_tieu},
                **common,
            )
        )
        facts.append(
            Fact(
                record_id=f"TO_HOP:nganh:{key}:2026:canon-nganh",
                record_type=RecordType.TO_HOP,
                admission_year=2026,
                text=(
                    f"Tổ hợp xét tuyển năm 2026 ngành {ten} (mã ngành {ma}): "
                    + ", ".join(codes)
                    + "."
                ),
                extractor="corpus_nganh_chunk/to_hop",
                values={"ma_nganh": ma, "ten_nganh": ten, "to_hop_codes": codes or None},
                **common,
            )
        )
        facts.append(
            Fact(
                record_id=f"HOC_PHI:nganh:{key}:2026:canon-nganh",
                record_type=RecordType.HOC_PHI,
                admission_year=2026,
                text=(
                    f"Học phí năm học 2026-2027 ngành {ten} (mã ngành {ma}): khối {khoi}, "
                    f"{match.group('hp')} VNĐ/tín chỉ."
                ),
                extractor="corpus_nganh_chunk/hoc_phi",
                values={
                    "ma_nganh": ma,
                    "ten_nganh": ten,
                    "khoi_nganh": khoi,
                    "hoc_phi_tin_chi_vnd": per_credit,
                },
                **common,
            )
        )
        facts.append(
            Fact(
                record_id=f"THOI_GIAN:nganh:{key}:2026:canon-nganh",
                record_type=RecordType.THOI_GIAN,
                admission_year=2026,
                text=(
                    f"Thời gian đào tạo ngành {ten} (mã ngành {ma}): "
                    f"{match.group('tg')} năm."
                ),
                extractor="corpus_nganh_chunk/thoi_gian_dao_tao",
                values={
                    "kind": "thoi_gian_dao_tao",
                    "scope": "nganh",
                    "ma_nganh": ma,
                    "ten_nganh": ten,
                    "thoi_gian_dao_tao_nam": years,
                },
                **common,
            )
        )
    return facts


_LIETKE_ITEM_RE = re.compile(
    r"^(?P<stt>\d+)\.\s+(?P<ten>.+?)\s+\(mã (?P<ma>[0-9A-Z]+)\)\s+-\s+"
    r"chỉ tiêu (?P<ct>[\d.]+),\s+tổ hợp (?P<th>.+)\.$"
)


def extract_corpus_lietke29(
    repo_root: Path, relpath: str, warn: list[Warning_]
) -> list[Fact]:
    """``liet_ke_28_nganh_2026_v2``: the 29 ngành/CTĐT list with mã + chỉ tiêu."""
    facts: list[Fact] = []
    for chunk in _corpus_chunks(repo_root, relpath):
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id != "liet_ke_28_nganh_2026_v2":
            continue
        src = f"{relpath}#{chunk_id}"
        url = _corpus_source_url(chunk)
        raw = str(chunk.get("text") or "")
        for line in raw.splitlines():
            line = _norm_ws(line)
            if not re.match(r"^\d+\.\s", line):
                continue
            match = _LIETKE_ITEM_RE.match(line)
            if not match:
                warn.append(
                    Warning_(
                        "unparsed_corpus_chunk",
                        f"{src}: list item not parseable, no record emitted: {line[:120]}",
                        src,
                        source_chunk_id=chunk_id,
                    )
                )
                continue
            ten = _norm_ws(match.group("ten"))
            ma = match.group("ma")
            chi_tieu = _int(match.group("ct"))
            codes = [c.strip() for c in match.group("th").split(",") if c.strip()]
            key = ma.lower()
            common = dict(
                source_document_id=src,
                source_url=url,
                source_chunk_id=chunk_id,
                authority=AuthorityLevel.HUSC_CANONICAL,
                raw_source_text=line,
            )
            facts.append(
                Fact(
                    record_id=f"NGANH:nganh:{key}:2026:canon-lietke29",
                    record_type=RecordType.NGANH,
                    admission_year=2026,
                    text=f"Ngành {ten}, mã ngành {ma} — HUSC tuyển sinh 2026.",
                    extractor="corpus_lietke29/nganh",
                    values={"ma_nganh": ma, "ten_nganh": ten},
                    **common,
                )
            )
            facts.append(
                Fact(
                    record_id=f"CHI_TIEU:nganh:{key}:2026:canon-lietke29",
                    record_type=RecordType.CHI_TIEU,
                    admission_year=2026,
                    text=(
                        f"Chỉ tiêu tuyển sinh năm 2026 ngành {ten} (mã ngành {ma}): "
                        f"{chi_tieu}."
                    ),
                    extractor="corpus_lietke29/chi_tieu",
                    values={"ma_nganh": ma, "ten_nganh": ten, "chi_tieu": chi_tieu},
                    **common,
                )
            )
            facts.append(
                Fact(
                    record_id=f"TO_HOP:nganh:{key}:2026:canon-lietke29",
                    record_type=RecordType.TO_HOP,
                    admission_year=2026,
                    text=(
                        f"Tổ hợp xét tuyển năm 2026 ngành {ten} (mã ngành {ma}): "
                        + ", ".join(codes)
                        + "."
                    ),
                    extractor="corpus_lietke29/to_hop",
                    values={"ma_nganh": ma, "ten_nganh": ten, "to_hop_codes": codes or None},
                    **common,
                )
            )
    return facts


_TOHOP_CHUNK_RE = re.compile(
    r"^Tổ hợp xét tuyển (?P<code>[A-Z0-9]+): (?P<subjects>[^.]+)\. "
    r"Các ngành dùng tổ hợp này tại HUSC 2026: (?P<nganh>.+)\.$"
)


def extract_corpus_tohop_catalog(repo_root: Path, relpath: str) -> list[Fact]:
    """``tohop_<mã>`` chunks: what each tổ hợp code means (code -> subjects)."""
    facts: list[Fact] = []
    for chunk in _corpus_chunks(repo_root, relpath):
        chunk_id = str(chunk.get("chunk_id") or "")
        if not re.fullmatch(r"tohop_[a-z0-9]+", chunk_id):
            continue
        text = _norm_ws(str(chunk.get("text") or ""))
        match = _TOHOP_CHUNK_RE.match(text)
        if not match:
            continue
        code = match.group("code")
        subjects = _norm_ws(match.group("subjects"))
        facts.append(
            Fact(
                record_id=f"TO_HOP:to-hop:{code.lower()}:2026:canon-tohop",
                record_type=RecordType.TO_HOP,
                admission_year=2026,
                authority=AuthorityLevel.HUSC_CANONICAL,
                text=f"Tổ hợp xét tuyển {code} gồm các môn: {subjects}.",
                extractor="corpus_tohop_catalog",
                values={
                    "to_hop_code": code,
                    "to_hop_subjects": subjects,
                    "nganh_abbrev_text": _norm_ws(match.group("nganh")),
                },
                source_document_id=f"{relpath}#{chunk_id}",
                source_url=_corpus_source_url(chunk),
                source_chunk_id=chunk_id,
                raw_source_text=text,
            )
        )
    return facts


_DIEM_ITEM_RE = re.compile(
    r"^\s*(?P<stt>\d+)\.\s+(?P<ten>.+?)\s+\(mã (?P<ma>[0-9A-Z]+)\):\s+"
    r"(?P<diem>\d+(?:\.\d+)?)\s+điểm\s+—\s+tổ hợp\s+(?P<th>.+)$"
)


def extract_corpus_diem_2025(
    repo_root: Path, relpath: str, warn: list[Warning_]
) -> list[Fact]:
    """``diem_chuan_2025_full``: 2025 điểm chuẩn by phương thức (THPT, học bạ)."""
    facts: list[Fact] = []
    for chunk in _corpus_chunks(repo_root, relpath):
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id != "diem_chuan_2025_full":
            continue
        src = f"{relpath}#{chunk_id}"
        url = _corpus_source_url(chunk)
        method = None
        for line in str(chunk.get("text") or "").splitlines():
            stripped = _norm_ws(line)
            if "PHƯƠNG THỨC XÉT ĐIỂM THI THPT" in stripped:
                method = ("thi_thpt", "điểm thi THPT")
                continue
            if "PHƯƠNG THỨC XÉT HỌC BẠ" in stripped:
                method = ("hoc_ba", "học bạ THPT")
                continue
            if method is None or not re.match(r"^\d+\.\s", stripped):
                continue
            match = _DIEM_ITEM_RE.match(stripped)
            if not match:
                warn.append(
                    Warning_(
                        "unparsed_corpus_chunk",
                        f"{src}: điểm chuẩn item not parseable, no record emitted: "
                        f"{stripped[:120]}",
                        src,
                        source_chunk_id=chunk_id,
                    )
                )
                continue
            ma = match.group("ma")
            ten = _norm_ws(match.group("ten"))
            diem = match.group("diem")
            codes = [c.strip() for c in match.group("th").split(",") if c.strip()]
            facts.append(
                Fact(
                    record_id=f"DIEM_CHUAN:nganh:{ma.lower()}:2025:canon-diem-{method[0]}",
                    record_type=RecordType.DIEM_CHUAN,
                    admission_year=2025,
                    authority=AuthorityLevel.PRIOR_YEAR_OFFICIAL,
                    text=(
                        f"Điểm chuẩn năm 2025 ngành {ten} (mã ngành {ma}), phương thức "
                        f"{method[1]}: {diem} điểm — tổ hợp {', '.join(codes)}."
                    ),
                    extractor=f"corpus_diem_2025/{method[0]}",
                    values={
                        "ma_nganh": ma,
                        "ten_nganh": ten,
                        "phuong_thuc": method[1],
                        "phuong_thuc_key": method[0],
                        "diem_trung_tuyen": diem,
                        "diem_trung_tuyen_num": float(diem),
                        "to_hop_codes": codes or None,
                    },
                    source_document_id=src,
                    source_url=url,
                    source_chunk_id=chunk_id,
                    raw_source_text=stripped,
                )
            )
    return facts


_NGANHMOI_BLOCK_RE = re.compile(
    r"(?P<stt>\d+)\.\s+(?P<ten>[^\n]+?)\n\s*- Mã ngành: (?P<ma>[0-9A-Z]+)\n"
    r"\s*- Thời gian đào tạo: (?P<tg>[^\n]+?)\n"
    r"\s*- Chỉ tiêu tuyển sinh: (?P<ct>\d+)\n"
    r"\s*- Tổ hợp xét tuyển: (?P<th>[^\n]+)"
)


def extract_corpus_nganh_moi(repo_root: Path, relpath: str) -> list[Fact]:
    """``husc_nganh_moi_2026``: the three programs opened in 2026."""
    facts: list[Fact] = []
    for chunk in _corpus_chunks(repo_root, relpath):
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id != "husc_nganh_moi_2026":
            continue
        src = f"{relpath}#{chunk_id}"
        url = _corpus_source_url(chunk)
        raw = str(chunk.get("text") or "")
        for match in _NGANHMOI_BLOCK_RE.finditer(raw):
            ten = _norm_ws(match.group("ten"))
            ma = match.group("ma")
            key = ma.lower()
            common = dict(
                source_document_id=src,
                source_url=url,
                source_chunk_id=chunk_id,
                authority=AuthorityLevel.HUSC_CANONICAL,
                raw_source_text=_norm_ws(match.group(0)),
            )
            facts.append(
                Fact(
                    record_id=f"NGANH:nganh:{key}:2026:canon-nganhmoi",
                    record_type=RecordType.NGANH,
                    admission_year=2026,
                    text=f"Ngành {ten}, mã ngành {ma} — chương trình mới mở năm 2026 tại HUSC.",
                    extractor="corpus_nganh_moi/nganh",
                    values={"ma_nganh": ma, "ten_nganh": ten},
                    **common,
                )
            )
            facts.append(
                Fact(
                    record_id=f"CHI_TIEU:nganh:{key}:2026:canon-nganhmoi",
                    record_type=RecordType.CHI_TIEU,
                    admission_year=2026,
                    text=(
                        f"Chỉ tiêu tuyển sinh năm 2026 ngành {ten} (mã ngành {ma}): "
                        f"{int(match.group('ct'))}."
                    ),
                    extractor="corpus_nganh_moi/chi_tieu",
                    values={
                        "ma_nganh": ma,
                        "ten_nganh": ten,
                        "chi_tieu": int(match.group("ct")),
                    },
                    **common,
                )
            )
            facts.append(
                Fact(
                    record_id=f"TO_HOP:nganh:{key}:2026:canon-nganhmoi",
                    record_type=RecordType.TO_HOP,
                    admission_year=2026,
                    text=(
                        f"Tổ hợp xét tuyển năm 2026 ngành {ten} (mã ngành {ma}): "
                        f"{_norm_ws(match.group('th'))}."
                    ),
                    extractor="corpus_nganh_moi/to_hop",
                    values={
                        "ma_nganh": ma,
                        "ten_nganh": ten,
                        "to_hop_codes": [
                            c.strip() for c in match.group("th").split(",") if c.strip()
                        ],
                    },
                    **common,
                )
            )
            facts.append(
                Fact(
                    record_id=f"THOI_GIAN:nganh:{key}:2026:canon-nganhmoi",
                    record_type=RecordType.THOI_GIAN,
                    admission_year=2026,
                    text=(
                        f"Thời gian đào tạo ngành {ten} (mã ngành {ma}): "
                        f"{_norm_ws(match.group('tg'))}."
                    ),
                    extractor="corpus_nganh_moi/thoi_gian_dao_tao",
                    values={
                        "kind": "thoi_gian_dao_tao",
                        "scope": "nganh",
                        "ma_nganh": ma,
                        "ten_nganh": ten,
                        "thoi_gian_dao_tao_nam": _year_text(match.group("tg")),
                    },
                    **common,
                )
            )
    return facts


def extract_corpus_timeline(repo_root: Path, relpath: str) -> list[Fact]:
    """``qa_lich_xet_hoc_ba_2026_v1``: HUSC's 2026 registration windows."""
    facts: list[Fact] = []
    for chunk in _corpus_chunks(repo_root, relpath):
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id != "qa_lich_xet_hoc_ba_2026_v1":
            continue
        src = f"{relpath}#{chunk_id}"
        url = _corpus_source_url(chunk)
        text = str(chunk.get("text") or "")
        flat = _norm_ws(text)
        windows = [
            (
                "hueuni",
                "Hệ thống Đại học Huế (http://dkxt.hueuni.edu.vn)",
                r"đăng ký từ (\d{2}:\d{2}) ngày (\d{1,2}/\d{1,2}/\d{4}) đến "
                r"(\d{2}:\d{2}) ngày (\d{1,2}/\d{1,2}/\d{4})",
            ),
            (
                "bogddt",
                "Hệ thống Bộ GD&ĐT (https://thisinh.thitotnghiepthpt.edu.vn)",
                r"đăng ký nguyện vọng từ (\d{1,2}/\d{1,2}/\d{4}) đến "
                r"(\d{2}:\d{2}) ngày (\d{1,2}/\d{1,2}/\d{4})",
            ),
        ]
        for key, label, pattern in windows:
            match = re.search(pattern, flat)
            if not match:
                continue
            groups = match.groups()
            dates = [g for g in groups if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", g)]
            start = _date_vn(dates[0]) if dates else None
            end = _date_vn(dates[-1]) if len(dates) > 1 else None
            facts.append(
                Fact(
                    record_id=f"THOI_GIAN:toan-truong:dang-ky-xet-tuyen-{key}:2026:canon-timeline",
                    record_type=RecordType.THOI_GIAN,
                    admission_year=2026,
                    authority=AuthorityLevel.HUSC_CANONICAL,
                    text=(
                        f"Lịch đăng ký xét tuyển đại học chính quy HUSC năm 2026 — "
                        f"{label}: {match.group(0)}."
                    ),
                    extractor="corpus_timeline/window",
                    values={
                        "kind": "moc_tuyen_sinh",
                        "scope": "toan_truong",
                        "he_thong": label,
                        "mo_ta": match.group(0),
                    },
                    effective_from=start,
                    effective_to=end,
                    effective_basis="stated_in_source",
                    source_document_id=src,
                    source_url=url,
                    source_chunk_id=chunk_id,
                    raw_source_text=flat[:400],
                )
            )
    return facts


def _date_vn(text: str) -> date | None:
    parts = text.split("/")
    if len(parts) != 3:
        return None
    try:
        return date(int(parts[2]), int(parts[1]), int(parts[0]))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
@dataclass
class SeedResult:
    store_root: Path
    report: dict
    report_path: Path


def collect_facts(repo_root: Path, cache_dir: Path) -> tuple[list[Fact], list[Warning_], dict]:
    warn: list[Warning_] = []
    facts: list[Fact] = []
    pdf_meta: dict[str, dict] = {}

    for document_id in (TTTS_PDF, HD2026_PDF, TT06_PDF):
        pdf = PdfText(repo_root, document_id, cache_dir)
        pdf_meta[document_id] = {
            "parser": pdf.parser,
            "page_count": pdf.page_count,
            "needs_review": pdf.needs_review,
            "sha256": pdf.sha256(),
        }
        if document_id == TTTS_PDF:
            facts += extract_pdf_chi_tieu_tables(pdf, warn)
            facts += extract_pdf_hoc_phi(pdf, warn)
            facts += extract_pdf_diem_chuan(pdf, warn)
        elif document_id == HD2026_PDF:
            facts += extract_pdf_milestones(pdf, warn)
        else:
            facts += extract_pdf_regulation_effective_date(pdf)

    facts += extract_corpus_nganh_chunks(repo_root, CORPUS_CANONICAL, warn)
    facts += extract_corpus_tohop_catalog(repo_root, CORPUS_CANONICAL)
    facts += extract_corpus_lietke29(repo_root, CORPUS_LIETKE29, warn)
    facts += extract_corpus_diem_2025(repo_root, CORPUS_DIEM2025, warn)
    facts += extract_corpus_nganh_moi(repo_root, CORPUS_NGANHMOI)
    facts += extract_corpus_timeline(repo_root, CORPUS_TIMELINE)
    return facts, warn, pdf_meta


def _distinct(warnings: list[Warning_]) -> list[Warning_]:
    """``warnings`` with byte-identical entries collapsed, order preserved."""
    seen: set[tuple] = set()
    out: list[Warning_] = []
    for item in warnings:
        key = (
            item.code,
            item.message,
            item.source_document_id,
            item.source_page,
            item.source_chunk_id,
            item.record_id,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

def _dedupe(facts: list[Fact], warn: list[Warning_]) -> list[Fact]:
    """Two extractions must never produce two different facts with one id."""
    seen: dict[str, Fact] = {}
    for fact in facts:
        existing = seen.get(fact.record_id)
        if existing is None:
            seen[fact.record_id] = fact
            continue
        if to_canonical_json(existing.to_record()) == to_canonical_json(fact.to_record()):
            continue
        warn.append(
            Warning_(
                "duplicate_record_id",
                f"two different extractions produced record_id {fact.record_id}; "
                "the first was kept",
                fact.source_document_id,
                fact.source_page,
                fact.source_chunk_id,
                fact.record_id,
            )
        )
    return list(seen.values())


def _fact_values_for(facts: list[Fact], record_type: RecordType, field_name: str):
    """(fact, value) pairs of one fact type carrying ``field_name``."""
    return [
        (fact, fact.values.get(field_name))
        for fact in facts
        if fact.record_type is record_type and fact.values.get(field_name) is not None
    ]


def _nganh_coverage(facts: list[Fact], record_type: RecordType) -> dict:
    """Per-ngành coverage of one fact type, split by sub-kind."""
    per_nganh: dict[str, set[str]] = {}
    per_kind: dict[str, set[str]] = {}
    school_level = 0
    for fact in facts:
        if fact.record_type is not record_type:
            continue
        kind = str(fact.values.get("kind") or "nganh_fact")
        ma = fact.values.get("ma_nganh")
        if ma:
            per_nganh.setdefault(str(ma), set()).add(kind)
            per_kind.setdefault(kind, set()).add(str(ma))
        else:
            school_level += 1
    return {
        "records": sum(1 for f in facts if f.record_type is record_type),
        "nganh_count": len(per_nganh),
        "nganh": sorted(per_nganh),
        "by_kind": {k: len(v) for k, v in sorted(per_kind.items())},
        "records_without_ma_nganh": school_level,
    }


def _cross_checks(repo_root: Path, facts: list[Fact], pdf_meta: dict) -> list[dict]:
    """Independent verification of extracted values against hand-checked inputs."""
    checks: list[dict] = []

    # 1. tuition: pages 54-55 vs the user-confirmed canonical tuition file
    tuition_path = repo_root / XCHECK_TUITION
    mismatches: list[dict] = []
    if tuition_path.is_file():
        canonical = json.loads(tuition_path.read_text(encoding="utf-8"))
        by_ma = {m["ma_nganh"]: m for m in canonical["majors"]}
        extracted = {
            fact.values["ma_nganh"]: fact.values
            for fact in facts
            if fact.record_type is RecordType.HOC_PHI
            and fact.source_document_id == TTTS_PDF
            and fact.values.get("ma_nganh")
        }
        for ma, values in extracted.items():
            expected = by_ma.get(ma)
            if expected is None:
                mismatches.append({"ma_nganh": ma, "issue": "not in canonical file"})
                continue
            if expected["vnd_per_credit"] != values.get("hoc_phi_tin_chi_vnd"):
                mismatches.append(
                    {
                        "ma_nganh": ma,
                        "field": "vnd_per_credit",
                        "pdf": values.get("hoc_phi_tin_chi_vnd"),
                        "canonical": expected["vnd_per_credit"],
                    }
                )
            if expected.get("vnd_per_year_est") != values.get("hoc_phi_nam_vnd"):
                mismatches.append(
                    {
                        "ma_nganh": ma,
                        "field": "vnd_per_year_est",
                        "pdf": values.get("hoc_phi_nam_vnd"),
                        "canonical": expected.get("vnd_per_year_est"),
                    }
                )
        checks.append(
            {
                "name": "pdf_hoc_phi_vs_canonical_tuition_file",
                "source": TTTS_PDF,
                "reference": XCHECK_TUITION,
                "rows_compared": len(extracted),
                "reference_rows": len(by_ma),
                "mismatches": mismatches,
                "ok": not mismatches,
            }
        )

    # 2. tuition rows vs the ticket-04 hand-verified fixture (page 55 rows)
    fixture_path = repo_root / FIXTURE_DIR / "ttts2026_p55_hoc_phi_khoa_hoc.json"
    if fixture_path.is_file():
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        expected_cells = []
        for row in fixture["cell_attribution"]:
            cells = {cell["header"]: cell["text"] for cell in row["cells"]}
            ma = cells.get("Mã ngành/CTĐT")
            credit = cells.get("Học phí/tín chỉ")
            per_year = cells.get("Dự kiến học phí/năm học")
            if ma and credit and per_year:
                expected_cells.append((ma, credit, per_year, row["page"]))
        extracted = {
            fact.values["ma_nganh"]: fact.values
            for fact in facts
            if fact.record_type is RecordType.HOC_PHI
            and fact.source_document_id == TTTS_PDF
        }
        mismatches = []
        for ma, per_credit, per_year, page in expected_cells:
            values = extracted.get(ma)
            if values is None:
                mismatches.append({"ma_nganh": ma, "issue": "row not extracted"})
                continue
            got_credit = f"{values.get('hoc_phi_tin_chi_vnd'):,}".replace(",", ".")
            if got_credit != per_credit:
                mismatches.append(
                    {
                        "ma_nganh": ma,
                        "field": "hoc_phi_tin_chi",
                        "fixture": per_credit,
                        "extracted": got_credit,
                    }
                )
            got_year = f"{values.get('hoc_phi_nam_vnd'):,}".replace(",", ".")
            if got_year != per_year:
                mismatches.append(
                    {
                        "ma_nganh": ma,
                        "field": "hoc_phi_nam",
                        "fixture": per_year,
                        "extracted": got_year,
                    }
                )
            if values.get("stt_tren_bang") is None:
                mismatches.append({"ma_nganh": ma, "issue": "no STT on row"})
        checks.append(
            {
                "name": "pdf_hoc_phi_vs_ticket04_fixture_p55",
                "source": TTTS_PDF,
                "reference": f"{FIXTURE_DIR}/ttts2026_p55_hoc_phi_khoa_hoc.json",
                "rows_compared": len(expected_cells),
                "fixture_pages": sorted({page for *_rest, page in expected_cells}),
                "mismatches": mismatches,
                "ok": not mismatches,
            }
        )

    # 3. tổ hợp pairing: pages 24-29 vs the hand-verified p24 fixture
    fixture_path = repo_root / FIXTURE_DIR / "ttts2026_p24_to_hop_mon_merge.json"
    if fixture_path.is_file():
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        pairs = {}
        for row in fixture["cell_attribution"]:
            name = next(
                (c["text"] for c in row["cells"] if c["header"] == "Tổ hợp môn xét tuyển"),
                None,
            )
            code = next(
                (
                    c["text"]
                    for c in row["cells"]
                    if c["header"] == "Mã tổ hợp môn xét tuyển"
                ),
                None,
            )
            if name and code:
                pairs[_norm_ws(re.sub(r"^\d+\.\s*", "", name))] = code
        pdf_pairs: dict[str, str] = {}
        for fact in facts:
            if (
                fact.record_type is RecordType.TO_HOP
                and fact.source_document_id == TTTS_PDF
                and fact.values.get("to_hop_subjects")
            ):
                for code, subject in zip(
                    fact.values["to_hop_codes"], fact.values["to_hop_subjects"]
                ):
                    pdf_pairs[_norm_ws(subject)] = code
        mismatches = [
            {"subjects": subject, "fixture": code, "extracted": pdf_pairs.get(subject)}
            for subject, code in pairs.items()
            if pdf_pairs.get(subject) != code
        ]
        checks.append(
            {
                "name": "pdf_to_hop_pairing_vs_ticket04_fixture_p24",
                "source": TTTS_PDF,
                "reference": f"{FIXTURE_DIR}/ttts2026_p24_to_hop_mon_merge.json",
                "rows_compared": len(pairs),
                "mismatches": mismatches,
                "ok": not mismatches,
            }
        )

    # 4. 2025 điểm chuẩn: corpus chunk vs the generated history file
    history_path = repo_root / XCHECK_DIEM_HISTORY
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        expected = {
            (e["major_code"], e["method"]): float(e["diem_chuan"])
            for e in history["entries"]
            if e.get("year") == 2025
        }
        got = {
            (fact.values["ma_nganh"], fact.values["phuong_thuc_key"]): fact.values[
                "diem_trung_tuyen_num"
            ]
            for fact in facts
            if fact.record_type is RecordType.DIEM_CHUAN
            and fact.source_document_id.startswith(CORPUS_DIEM2025)
        }
        mismatches = [
            {"key": list(key), "history": value, "extracted": got.get(key)}
            for key, value in expected.items()
            if got.get(key) != value
        ]
        checks.append(
            {
                "name": "corpus_diem_2025_vs_diem_chuan_history_file",
                "source": CORPUS_DIEM2025,
                "reference": XCHECK_DIEM_HISTORY,
                "rows_compared": len(expected),
                "mismatches": mismatches,
                "ok": not mismatches,
            }
        )

    # 5. chỉ tiêu: the 25 table rows must sum to the total printed on page 24
    header_line = None
    if pdf_meta.get(TTTS_PDF):
        from pypdf import PdfReader

        reader = PdfReader(str(repo_root / TTTS_PDF))
        page_text = reader.pages[23].extract_text() or ""
        for line in page_text.split("\n"):
            if "KHOA HỌC" in line and re.search(r"\d{3,4}\s*$", line.strip()):
                header_line = _norm_ws(line)
                break
    printed_total = None
    if header_line:
        printed_total = int(re.search(r"(\d{3,4})\s*$", header_line).group(1))
    row_values = [
        fact.values["chi_tieu"]
        for fact in facts
        if fact.record_type is RecordType.CHI_TIEU
        and fact.source_document_id == TTTS_PDF
        and fact.values.get("chi_tieu") is not None
    ]
    extracted_total = sum(row_values) if row_values else None
    checks.append(
        {
            "name": "pdf_chi_tieu_rows_sum_vs_printed_total",
            "source": TTTS_PDF,
            "reference": f"page 24 section header: {header_line}",
            "printed_total": printed_total,
            "extracted_total": extracted_total,
            "rows": len(row_values),
            "mismatches": (
                []
                if printed_total == extracted_total
                else [{"printed": printed_total, "extracted": extracted_total}]
            ),
            "ok": printed_total is not None and printed_total == extracted_total,
        }
    )

    # 6. every ticket-04 fixture page is a real page, and the row anchors it
    #    transcribes are printed there — checked with the same pypdf tier the
    #    extraction uses, so the page numbers this seed stamps are corroborated
    #    by hand-verified evidence (including the p12/p57 fixtures, which
    #    transcribe *other* schools on the same document and therefore yield no
    #    HUSC record).
    fixture_dir = repo_root / FIXTURE_DIR
    if fixture_dir.is_dir():
        from pypdf import PdfReader  # local import: only pypdf is installed

        page_cache: dict[tuple[str, int], str] = {}

        def _fixture_page_text(document_id: str, page: int) -> str:
            key = (document_id, page)
            if key not in page_cache:
                reader = PdfReader(str(repo_root / document_id))
                page_cache[key] = reader.pages[page - 1].extract_text() or ""
            return page_cache[key]

        for fixture_path in sorted(fixture_dir.glob("*.json")):
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            document_id = fixture["source_document"]
            pages = list(fixture["source_pages"])
            text_by_page = {
                page: _norm_ws(_fixture_page_text(document_id, page)) for page in pages
            }
            anchors = [
                row["anchor"] for row in fixture.get("rows", []) if row.get("anchor")
            ]
            missing = [
                anchor
                for anchor in anchors
                if not any(
                    _norm_ws(anchor) in text_by_page.get(page, "") for page in pages
                )
            ]
            checks.append(
                {
                    "name": f"ticket04_fixture_pages_{fixture_path.stem}",
                    "source": document_id,
                    "reference": f"{FIXTURE_DIR}/{fixture_path.name}",
                    "pages": pages,
                    "anchors_checked": len(anchors),
                    "anchors_not_found": missing,
                    "ok": not missing,
                }
            )
    return checks


def _cross_source_conflicts(facts: list[Fact]) -> list[dict]:
    """Same (fact type, ngành, year) stated differently by two sources."""
    conflicts: list[dict] = []
    field_names = (
        "chi_tieu",
        "to_hop_codes",
        "hoc_phi_tin_chi_vnd",
        "diem_trung_tuyen_num",
        "thoi_gian_dao_tao_nam",
        "ten_nganh",
    )
    grouped: dict[tuple, list[Fact]] = {}
    for fact in facts:
        ma = fact.values.get("ma_nganh")
        if not ma:
            continue
        grouped.setdefault((fact.record_type, str(ma), fact.admission_year), []).append(fact)
    for (record_type, ma, year), group in sorted(
        grouped.items(), key=lambda item: (item[0][0].value, item[0][1], item[0][2])
    ):
        if len(group) < 2:
            continue
        for field_name in field_names:
            stated = {
                fact.source_document_id: fact.values.get(field_name)
                for fact in group
                if fact.values.get(field_name) is not None
            }
            if len({json.dumps(v, ensure_ascii=False, sort_keys=True) for v in stated.values()}) > 1:
                conflicts.append(
                    {
                        "record_type": record_type.value,
                        "ma_nganh": ma,
                        "admission_year": year,
                        "field": field_name,
                        "statements": stated,
                    }
                )
    return conflicts


def seed(
    store_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
) -> SeedResult:
    """Extract the facts and write them (plus the report) into ``store_root``."""
    repo = Path(repo_root) if repo_root is not None else _REPO_ROOT
    store_dir = Path(store_root) if store_root is not None else DEFAULT_STORE_ROOT
    cache_dir = store_dir / ".parse_cache"

    facts, warn, pdf_meta = collect_facts(repo, cache_dir)
    facts = _dedupe(facts, warn)
    facts.sort(key=lambda fact: fact.record_id)
    store = CanonicalRecordStore(store_dir)

    written = 0
    already_present = 0
    for fact in facts:
        record = fact.to_record()
        versions = store.versions(fact.record_id)
        if versions:
            stored = store.get_version(fact.record_id, versions[0])
            if to_canonical_json(stored) == to_canonical_json(record):
                already_present += 1
                continue
            warn.append(
                Warning_(
                    "stored_version_differs",
                    f"{fact.record_id}: a stored version exists whose bytes differ from "
                    "this extraction — left untouched (the store is append-only)",
                    fact.source_document_id,
                    fact.source_page,
                    fact.source_chunk_id,
                    fact.record_id,
                )
            )
            already_present += 1
            continue
        if not dry_run:
            store.write_record(record)
        written += 1

    by_type: dict[str, int] = {}
    by_document: dict[str, int] = {}
    for fact in facts:
        by_type[fact.record_type.value] = by_type.get(fact.record_type.value, 0) + 1
        by_document[fact.source_document_id] = by_document.get(fact.source_document_id, 0) + 1

    coverage = {
        record_type.value: _nganh_coverage(facts, record_type)
        for record_type in (
            RecordType.DIEM_CHUAN,
            RecordType.CHI_TIEU,
            RecordType.HOC_PHI,
            RecordType.TO_HOP,
            RecordType.NGANH,
            RecordType.THOI_GIAN,
        )
    }
    conflicts = _cross_source_conflicts(facts)
    checks = _cross_checks(repo, facts, pdf_meta)

    pages_used: dict[str, list[int]] = {}
    for fact in facts:
        if fact.source_page is not None:
            pages_used.setdefault(fact.source_document_id, [])
            if fact.source_page not in pages_used[fact.source_document_id]:
                pages_used[fact.source_document_id].append(fact.source_page)

    report = {
        "report_version": 1,
        "seed_version": SEED_VERSION,
        "store_root": str(store_dir).replace("\\", "/"),
        "schema_version": "1.0",
        "dry_run": dry_run,
        "record_counts": {
            "total": len(facts),
            "written_this_run": written,
            "already_present": already_present,
            "by_fact_type": dict(sorted(by_type.items())),
            "by_source_document": dict(sorted(by_document.items())),
            "by_status": {"draft": len(facts)},
            "published": 0,
        },
        "fact_type_coverage": coverage,
        "sources": [
            {
                "document_id": document_id,
                "kind": "pdf",
                "parser": meta["parser"],
                "page_count": meta["page_count"],
                "needs_review": meta["needs_review"],
                "sha256": meta["sha256"],
                "records": by_document.get(document_id, 0),
                "pages_used": sorted(pages_used.get(document_id, [])),
            }
            for document_id, meta in pdf_meta.items()
        ]
        + [
            {
                "document_id": document_id,
                "kind": "corpus",
                "records": by_document.get(document_id, 0),
                "pages_used": [],
            }
            for document_id in sorted(by_document)
            if document_id not in pdf_meta
        ],
        "warnings": [
            {
                "code": item.code,
                "message": item.message,
                "source_document_id": item.source_document_id,
                "source_page": item.source_page,
                "source_chunk_id": item.source_chunk_id,
                "record_id": item.record_id,
            }
            # One unreadable cell can be reported once per record it blocks (a
            # table row yields a record per phương thức). Identical entries are
            # collapsed so the count of distinct unread cells stays truthful.
            for item in _distinct(
                sorted(
                    warn, key=lambda w: (w.code, w.source_document_id, w.source_page or 0)
                )
            )
        ],
        "cross_source_conflicts": conflicts,
        "cross_checks": checks,
        "provenance": {
            "pdf_records_carry_page": sum(1 for f in facts if f.source_page is not None),
            "corpus_records_have_no_page": sum(1 for f in facts if f.source_page is None),
            "note": (
                "PDF-derived records carry (source_document_id, source_page) and are "
                "round-trip verifiable with pypdf; corpus-derived records carry "
                "(file#chunk_id, source_url) and source_page=null because the chunk "
                "corpus carries no page numbers. No page is ever guessed."
            ),
        },
        "not_extracted": [
            {
                "what": "per-ngành mốc thời gian (deadline theo từng ngành)",
                "why": (
                    "deadlines in the sources are school-wide or national, never per "
                    "ngành; the per-ngành time fact available is thời gian đào tạo"
                ),
                "documents_needed": [
                    "thông báo tuyển sinh theo ngành của HUSC (tuyensinh.husc.edu.vn) "
                    "— không có trong repo"
                ],
            },
            {
                "what": "điểm chuẩn 2026",
                "why": "chưa công bố tại thời điểm tài liệu (corpus: is_pending=true)",
                "documents_needed": ["công bố điểm chuẩn 2026 của HUSC"],
            },
            {
                "what": "học phí khối ngành VI / các khối ngoài IV, V, VII cho HUSC",
                "why": "bảng học phí HUSC (trang 54-55) chỉ có ba khối IV, V, VII",
                "documents_needed": [],
            },
            {
                "what": "husc_thongbao_fix_2026.jsonl",
                "why": (
                    "chỉ chứa lệ phí/hồ sơ/đối tượng xét tuyển thẳng, không thuộc sáu "
                    "loại fact của ticket; file còn lặp chunk_id (đã khử trùng khi đọc) "
                    "và mốc thời gian trùng với husc_2026_timeline_hocba_v1 nhưng "
                    "diễn đạt mềm hơn ('thường từ')"
                ),
                "documents_needed": [],
            },
            {
                "what": "liet_ke_26_nganh_2026 / bang_xep_hang_chi_tieu_2026 (husc_canonical_fix_2026.jsonl)",
                "why": "bản trùng/thiếu so với liet_ke_28_nganh_2026_v2 đã dùng",
                "documents_needed": [],
            },
        ],
        "conventions": {
            "record_id": (
                "{RECORD_TYPE}:nganh:{ma_nganh}:{admission_year}:{source_key} — "
                "deterministic, so a second run on the same input writes nothing"
            ),
            "effective_window": (
                "when the source states a date, effective_from/to are that date "
                "(basis=stated_in_source); otherwise the window is the admission year "
                "(basis=admission_year_convention) — never an invented date"
            ),
            "status": "draft for every record; nothing here publishes",
            "authority": {
                "HUSC_OFFICIAL": TTTS_PDF,
                "MINISTRY": f"{HD2026_PDF}; {TT06_PDF}",
                "HUSC_CANONICAL": f"{CORPUS_CANONICAL}; {CORPUS_LIETKE29}; {CORPUS_NGANHMOI}; {CORPUS_TIMELINE}",
                "PRIOR_YEAR_OFFICIAL": f"{CORPUS_DIEM2025} (2025 điểm chuẩn)",
            },
        },
    }

    report_path = store_dir / REPORT_FILENAME
    if not dry_run:
        store_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return SeedResult(store_root=store_dir, report=report, report_path=report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", default=str(DEFAULT_STORE_ROOT))
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = seed(
        store_root=args.store_root, repo_root=args.repo_root, dry_run=args.dry_run
    )
    counts = result.report["record_counts"]
    print(
        f"records={counts['total']} written={counts['written_this_run']} "
        f"already_present={counts['already_present']} "
        f"warnings={len(result.report['warnings'])} "
        f"conflicts={len(result.report['cross_source_conflicts'])}"
    )
    for name, stats in result.report["fact_type_coverage"].items():
        print(f"  {name:12s} records={stats['records']:4d} ngành={stats['nganh_count']:3d}")
    if not args.dry_run:
        print(f"report -> {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
