"""
Table-aware chunker v2 — implements v4 Phase 3 Hybrid Strategy:
  1. Wide-table bypass (≥8 cols, ≥3 data rows) → single chunk
  2. Program-grouped tables → 1 chunk per program family
  3. Small-table fallback (≤6 data rows) → single chunk
  4. Text sections → smaller chunks (200 tokens) with sentence-boundary respect
  5. Header context always preserved with table data

Target: 1200+ chunks with high quality (no row explosion, no orphaned rows).

Usage:
    cd rag2025
    $env:PYTHONPATH="src;."; python scripts/chunk_table_aware_v2.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

import tiktoken
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from services.temporal_authority import get_current_admission_year

STAGING_HTML = Path(__file__).parent.parent / "data" / "staging" / "html"
STAGING_PDF = Path(__file__).parent.parent / "data" / "staging" / "pdf"
CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"

ENCODING = "cl100k_base"
# Chunk-budget knobs (target size, overlap, hard cap, soft floor) are driven by
# config.settings so .env remains the single authoritative source. Removing the
# previously hardcoded TEXT_CHUNK_SIZE=220 / TEXT_OVERLAP=50 / HARD_MAX_TOKENS=512
# / SOFT_MIN_TOKENS=50 here; each function resolves its knob at call time.
MIN_TOKENS = 25
WIDE_TABLE_MIN_COLS = 8
WIDE_TABLE_MIN_ROWS = 1
SMALL_TABLE_MAX_ROWS = 6
PROGRAM_GROUP_MAX_TOKENS = 500


STOPWORDS = {
    "và", "các", "của", "có", "được", "cho", "trong", "là", "một", "này",
    "để", "với", "theo", "từ", "đã", "sẽ", "không", "khi", "bằng",
}

# ST-6c: a pipe-table row is any line that starts with a pipe (optional leading
# whitespace). Rows are ATOMIC — a chunk boundary must never land inside one
# (mirrors the pipe-row guard in src/chunker.py; fixes chunked_24 chunk_100→101
# dropping a subject-combo table row).
_PIPE_TABLE_LINE_RE = re.compile(r"^\s*\|", re.MULTILINE)

# ── Ticket 32: merged (rowspan) cell provenance ──
# A table that merges an identity cell ("Số TT" / "mã ngành" / "tên ngành")
# vertically prints that value on ONE line of the block; the continuation lines
# reach the parser with those cells empty. Emitted as-is they assert a figure
# (điểm chuẩn / chỉ tiêu / học phí) for a programme the chunk never names —
# confidently wrong. The empty cell has three shapes on the real pages
# (ticket 04 evidence: pages 12, 24, 55, 57 of THONG_TIN_TUYEN_SINH_202611.pdf):
#   * rowspan continuation (p12 rows 4-5, p12_kinhte row 4, p57 row 3): the row
#     is an option of the block above it — inheriting that block's identity is
#     correct;
#   * wrapped header fragment (p12/p55: "(Mã xét", "hợp môn", "đào tạo"): the
#     row is header text, not data — copying it labels a row with header garbage;
#   * genuinely blank data cell (p57 row 1 "Phương thức tuyển sinh"): the page
#     prints it blank — inheriting "THPT" asserts a method the document never
#     states for that programme.
# Only the first shape is inherited, only from a row that owns an identity cell,
# and never across columns; everything else is reported (see RowSpan).
_STT_SHAPE_RE = re.compile(r"^\d{1,3}(?:\s+\S.*)?$")  # "5", "1  Luật", "16  VII"
_CODE_SHAPE_RE = re.compile(r"^\d{6,7}$")  # "7380101"

# ── Ticket 32: the visible marker for a row whose label could not be resolved ──
# An unresolved row is KEPT (deleting it loses the name<->mã ngành link the page
# prints across two rows) but it must not read as a clean data row: the reason it
# could not be attributed is written INTO the chunk ``text``, because metadata is
# invisible on the answer path (llm_generator._build_context reads only data_year
# / notification_id / info_type / source). The wording is digit-free on purpose:
# services/citation.py indexes every number in a chunk as a grounded value, so a
# marker carrying a row/column index would fabricate a citable claim.
ROWSPAN_MARKER_PREFIX = "[KHÔNG XÁC ĐỊNH ĐƯỢC NHÃN"
_ROWSPAN_MARKER_RE = re.compile(r"\s*\[KHÔNG XÁC ĐỊNH ĐƯỢC NHÃN[^\]]*\]")
# A leading enumeration prefix ("3. Ngữ văn") is a combo code, not a figure.
_ORDINAL_PREFIX_RE = re.compile(r"^\s*\d+[.)]\s+")
# span.kind -> why the label could not be resolved (mirrors span.reason, which is
# kept verbatim in metadata for the citation note).
_REASON_CLAUSES = {
    "no-anchor": "không có dòng nào phía trên sở hữu các ô trống",
    "sibling": "dòng trên là một dòng khác của cùng các cột",
    "header": "dòng trên chỉ mang chữ của tiêu đề",
    "wrapped": "dòng trên là mảnh ô bị gãy",
}
_FIGURES_CLAUSE = "số trong dòng không gán được cho ngành nào"

# Wording that only ever appears in a table header on these pages (ticket 04
# hand-transcription). A cell equal to one of these is header text that a
# wrapped header left in a data row — never an identity.
_HEADER_LABELS = frozenset({
    "tt", "stt", "số tt", "ngành", "tên ngành", "chương trình đào tạo",
    "đào tạo", "mã ngành", "mã xét tuyển", "tổ hợp môn",
    "mã tổ hợp môn xét tuyển", "xét tuyển", "phương thức", "chỉ tiêu",
    "dự kiến", "ghi chú", "tín chỉ", "năm học", "học phí", "khối",
    "trình độ",
})
# First words of those headers: a wrapped header leaves single words in data
# cells ("khóa", "tuyển)", "hợp môn"), which are refused as identity values.
# Kept to words no programme/trường name starts with — "Kiến trúc", "Văn học",
# "Giáo dục", "Phân hiệu" must stay inheritable.
_HEADER_WORDS = frozenset({
    "tt", "stt", "số", "ngành", "tên", "chương", "trình", "mã", "tổ", "hợp",
    "phương", "thức", "chỉ", "tiêu", "dự", "đào", "tuyển", "xét", "ghi",
    "chú", "tín", "khối", "năm",
})


def _is_identity_value(value: str, header_values: frozenset[str], lead: bool) -> bool:
    """May ``value`` be inherited into a row's identity column (ticket 32)?

    Refused, in order:
    * a wrapped header fragment — a verbatim header cell, unbalanced parentheses
      such as "(Mã xét", or header wording ("khóa", "tuyển)", "hợp môn");
    * any other cell carrying a digit: a quota, a score, a tuition figure or a
      combo code ("1. Toán, Vật lý, Hóa học", "Đọc: 425 - 450") must never
      become the label of a row. Only the two identity shapes carry digits — a
      mã ngành / mã xét tuyển (6-7 digits) and an ordinal "số TT", optionally
      merged with the programme name in one cell ("1  Luật", "16  VII").
    * a bare word: an identity names a programme or a school, so a lone token is
      a cell fragment, not a label ("Academic", "khóa").

    ``lead`` marks the anchor's own first identity cell, the only column where a
    bare ordinal ("5") is an identity.
    """
    v = " ".join(value.split())
    if not v:
        return False
    if v in header_values:
        return False
    if v.count("(") != v.count(")"):
        return False
    if v.lower() in _HEADER_LABELS:
        return False
    if v.split()[0].lower().strip(".,;:()") in _HEADER_WORDS:
        return False
    if _CODE_SHAPE_RE.match(v):  # mã ngành / mã xét tuyển
        return True
    if _STT_SHAPE_RE.match(v):  # số TT, or "số TT + tên ngành" merged in one cell
        return lead or any(ch.isalpha() for ch in v)
    if any(ch.isdigit() for ch in v):  # a data number is never a label
        return False
    return len(v.split()) >= 2 and any(ch.isalpha() for ch in v)

class RowSpan(NamedTuple):
    """Ticket 32: provenance for one row that leans on a merged (rowspan) cell."""

    row: int  # data-row index inside its table
    anchor_row: int | None  # nearest preceding row that owns an identity cell
    identity: str  # label inherited from the anchor ("" when unresolved)
    cells: dict[int, str]  # identity cells inherited, keyed by column index
    skipped: dict[int, str]  # anchor cells refused by the identity guard
    columns: tuple[int, ...]  # empty identity cells this row leans on
    reason: str  # "" when resolved, otherwise why it could not be
    kind: str = ""  # reason class for an unresolved span (see _REASON_CLAUSES)

    @property
    def resolved(self) -> bool:
        return self.anchor_row is not None


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def detect_year(text: str) -> str:
    _ceiling = get_current_admission_year() + 1
    for m in re.finditer(r"năm\s*(?:học\s*)?(\d{4})", text, re.IGNORECASE):
        y = int(m.group(1))
        if 2024 <= y <= _ceiling:
            return str(y)
    for m in re.finditer(r"(\d{4})\s*[-–]\s*(\d{4})", text):
        y2 = int(m.group(2))
        if 2025 <= y2 <= _ceiling:
            return str(y2)
    return str(get_current_admission_year())


def detect_info_type(text: str, source: str) -> str:
    t = text.lower()
    if "điểm chuẩn" in t or "điểm xét tuyển" in t:
        return "diem_chuan"
    if "học phí" in t or "thu học phí" in t or "tín chỉ" in t:
        return "hoc_phi"
    if "ngành" in t and ("đào tạo" in t or "tuyển sinh" in t or "chỉ tiêu" in t):
        return "nganh_tuyen_sinh"
    if "điều kiện" in t or "hồ sơ" in t or "thủ tục" in t:
        return "dieu_kien_xet_tuyen"
    if "thông tư" in source.lower() or "quy chế" in t:
        return "van_ban_phap_ly"
    if "phương thức" in t or "xét tuyển" in t:
        return "phuong_thuc_xet_tuyen"
    if "lịch" in t or "thời gian" in t or "deadline" in t:
        return "thoi_gian_tuyen_sinh"
    return "thong_tin_chung"


def detect_source_type(filepath: Path) -> str:
    if "thongbao_" in filepath.name:
        return "web_crawl"
    if "pdf_" in filepath.name:
        return "pdf_extract"
    return "unknown"


def extract_sparse_terms(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2][:50]


# ══════════════════════════════════════════
# TABLE PARSING & CHUNKING (v4 Phase 3)
# ══════════════════════════════════════════

class MarkdownTable:
    """Parsed markdown table with header, separator, and data rows."""

    def __init__(self, raw_lines: list[str]):
        self.raw_lines = [l.strip() for l in raw_lines if l.strip().startswith("|")]
        self.header_line = ""
        self.separator_line = ""
        self.data_rows: list[str] = []
        self._parse()

    def _parse(self):
        for line in self.raw_lines:
            if re.match(r"^\|[\s\-|:]+\|$", line):
                self.separator_line = line
            elif not self.header_line:
                self.header_line = line
            else:
                self.data_rows.append(line)

    @property
    def num_cols(self) -> int:
        """Max columns across header AND data rows — handles PDF parsing artifacts
        where header content spans cells, causing header to report fewer cols."""
        header_cols = len([c for c in self.header_line.split("|") if c.strip()]) if self.header_line else 0
        data_cols = max((len([c for c in row.split("|") if c.strip()]) for row in self.data_rows), default=0)
        return max(header_cols, data_cols)

    @property
    def is_wide(self) -> bool:
        return self.num_cols >= WIDE_TABLE_MIN_COLS and len(self.data_rows) >= WIDE_TABLE_MIN_ROWS

    @property
    def is_small(self) -> bool:
        return len(self.data_rows) <= SMALL_TABLE_MAX_ROWS

    def has_program_column(self) -> bool:
        """Check if table has a column that groups rows by program/major."""
        if not self.header_line:
            return False
        header_lower = self.header_line.lower()
        program_keywords = ["ngành", "mã ngành", "chuyên ngành", "ngành học", "trình độ", " chương trình"]
        return any(kw in header_lower for kw in program_keywords)

    def group_row_indices_by_program(self) -> list[list[int]]:
        """Group data-row indices by program prefix (first column content changes)."""
        if not self.data_rows:
            return []

        groups: list[list[int]] = []
        current_group: list[int] = []
        prev_prefix = ""

        for index, row in enumerate(self.data_rows):
            # Extract first non-empty column as group key
            cells = [c for c in self._cells_of(row) if c]
            prefix = cells[0] if cells else ""

            # New group when: prefix changes from non-empty to non-empty
            # OR prefix is non-empty and current group is already started
            if prefix and prev_prefix and prefix != prev_prefix and current_group:
                groups.append(current_group)
                current_group = [index]
            else:
                current_group.append(index)

            if prefix:
                prev_prefix = prefix

        if current_group:
            groups.append(current_group)

        return groups if groups else [list(range(len(self.data_rows)))]

    @staticmethod
    def _cells_of(line: str) -> list[str]:
        """Cells of one pipe row, position preserved (unlike `num_cols`)."""
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def rowspan_provenance(self) -> list[RowSpan]:
        """Ticket 32: resolve every row that leans on a merged (rowspan) cell.

        A row is reported when it carries content while its leading cells — the
        columns left of its first value, i.e. the identity region (Số TT / mã
        ngành / tên ngành) — are empty and something above fills them. Such a
        row is attached to the nearest preceding row that OWNS an identity cell
        (one that starts further left and shares no column with it), and only
        that row's identity cells are inherited — never a header fragment, never
        a data number, never a value from another column. A row that cannot be
        attached that way is reported unresolved rather than silently emitted.

        Anchor-driven, not a blanket rule: a table whose rows all start in the
        same column has no owner to inherit from and yields nothing (the p24
        negative control); a row that fills a column its neighbour also fills
        owns its own identity and is left alone (p57, where two programmes are
        both printed with Số TT "5").
        """
        if not self.data_rows:
            return []

        header_cells = self._cells_of(self.header_line) if self.header_line else []
        header_values = frozenset(c for c in header_cells if c)
        header_populated = {j for j, c in enumerate(header_cells) if c}
        rows = [self._cells_of(row) for row in self.data_rows]
        populated = [{j for j, c in enumerate(row) if c} for row in rows]
        first = [min(cols) if cols else None for cols in populated]

        spans: list[RowSpan] = []
        for i, cols in enumerate(populated):
            first_content = first[i]
            if first_content is None or first_content == 0:
                continue  # owns the leading column: nothing merged to inherit
            filled_above = set(header_populated)
            for k in range(i):
                filled_above |= populated[k]
            columns = tuple(j for j in range(first_content) if j in filled_above)
            if not columns:
                continue  # leading cells empty in every row: not an identity region

            anchor = next(
                (k for k in range(i - 1, -1, -1)
                 if first[k] is not None and first[k] < first_content),
                None,
            )
            if anchor is None:
                spans.append(RowSpan(
                    i, None, "", {}, {}, columns,
                    "no row above starts left of this row, so nothing owns the empty cells",
                    "no-anchor",
                ))
                continue
            if populated[anchor] & cols:
                spans.append(RowSpan(
                    i, None, "", {}, {}, columns,
                    f"row {anchor} above is another row of the same columns "
                    "(this row carries its own identity)",
                    "sibling",
                ))
                continue

            cells: dict[int, str] = {}
            skipped: dict[int, str] = {}
            region = [j for j in range(first_content) if j in populated[anchor]]
            for position, j in enumerate(region):
                value = rows[anchor][j]
                if _is_identity_value(value, header_values, lead=position == 0):
                    cells[j] = value
                else:
                    skipped[j] = value
            # One PDF cell wrapped over several markdown columns arrives as text
            # fragments ("lịch; Kinh" | "tế và quản" | "lý tài" — page 12 of the
            # 2026 prospectus). Such an anchor owns no identity: only a cell
            # with a Số TT / mã ngành marker carries digits past the guard above.
            wrapped_fragment = (
                len(region) > 1
                and bool(cells)
                and not any(ch.isdigit() for value in cells.values() for ch in value)
            )
            if wrapped_fragment:
                skipped.update(cells)
                cells = {}
            if not cells:
                reason = (
                    f"row {anchor} above carries wrapped cell fragments in columns {region} "
                    "with no Số TT / mã ngành marker: not an identity"
                    if wrapped_fragment
                    else f"row {anchor} above carries only header text / data numbers "
                    "in the empty cells"
                )
                spans.append(RowSpan(
                    i, None, "", {}, skipped, columns, reason,
                    "wrapped" if wrapped_fragment else "header",
                ))
                continue
            spans.append(RowSpan(
                i, anchor,
                " / ".join(cells[j] for j in sorted(cells)),
                cells, skipped, columns, "",
            ))

        return spans

    def to_chunk_text(self, rows: list[str] | None = None) -> str:
        """Build chunk text with header + separator + specified rows."""
        rows = rows or self.data_rows
        parts = []
        if self.header_line:
            parts.append(self.header_line)
        if self.separator_line:
            parts.append(self.separator_line)
        parts.extend(rows)
        return "\n".join(parts)


def chunk_table(table: MarkdownTable, base_meta: dict, doc_id: str, enc, start_idx: int) -> list[dict]:
    """Chunk table using v4 Phase 3 hybrid strategy.

    Priority:
    1. Wide-table bypass (≥8 cols, ≥3 rows) → single chunk
    2. Small-table fallback (≤6 rows) → single chunk
    3. Program-grouped tables → 1 chunk per program family
    4. General: pack rows by token budget with header

    T1.8: output is passed through `_enforce_size_gates` (hard cap 512 tokens,
    soft floor 50 tokens) before returning.
    """
    chunks = _chunk_table(table, base_meta, doc_id, enc, start_idx)
    chunks = _enforce_size_gates(chunks, base_meta, doc_id, enc)
    _assert_unresolved_rows_marked(chunks, doc_id)
    return chunks


def _rowspan_metadata(
    spans: list[RowSpan],
    row_indices: list[int],
    annotations: dict[int, dict] | None = None,
) -> dict:
    """Ticket 32: the rowspan provenance belonging to one emitted chunk."""
    wanted = set(row_indices)
    attributed = [
        {
            "row": s.row,
            "anchor_row": s.anchor_row,
            "identity": s.identity,
            "cells": {str(col): value for col, value in sorted(s.cells.items())},
            **({"skipped": {str(col): value for col, value in sorted(s.skipped.items())}} if s.skipped else {}),
        }
        for s in spans if s.resolved and s.row in wanted
    ]
    unresolved = [
        {
            "row": s.row,
            "columns": list(s.columns),
            "reason": s.reason,
            "marker": (annotations or {}).get(s.row, {}).get("marker", ""),
            "figures": (annotations or {}).get(s.row, {}).get("figures", False),
            **({"skipped": {str(col): value for col, value in sorted(s.skipped.items())}} if s.skipped else {}),
        }
        for s in spans if not s.resolved and s.row in wanted
    ]
    if attributed and unresolved:
        return {"rowspan_attribution": attributed, "rowspan_unresolved": unresolved}
    if attributed:
        return {"rowspan_attribution": attributed}
    if unresolved:
        return {"rowspan_unresolved": unresolved}
    return {}


def _absorb_rowspan(target: dict, incoming: dict) -> None:
    """Ticket 32: folding a chunk into its neighbour must not drop its rowspan
    provenance.

    The merged text carries the absorbed chunk's rows too, so its attribution and
    its unresolved reports still belong to the emitted chunk. Dropping them would
    put the row back into the corpus with neither a traceable label nor a warning
    — the silent state this ticket removes. Row indices are table-scoped and every
    chunk merged here comes from one `chunk_table` call for one table, so a row
    can never be claimed twice.
    """
    for key in ("rowspan_attribution", "rowspan_unresolved"):
        entries = incoming.get(key)
        if not entries:
            continue
        kept = list(target.get(key, []))
        rows = {entry["row"] for entry in kept}
        kept.extend(entry for entry in entries if entry["row"] not in rows)
        target[key] = kept


def _row_carries_figures(cells: list[str]) -> bool:
    """Does this row assert a number (điểm chuẩn / chỉ tiêu / học phí)?

    A bare enumeration prefix ("3. Ngữ văn") is a combo code, not a figure, so it
    is stripped before the digit test. Everything else with a digit counts: on the
    real pages those are exactly the quota / score / tuition cells.
    """
    return any(
        any(ch.isdigit() for ch in _ORDINAL_PREFIX_RE.sub("", cell, count=1))
        for cell in cells if cell
    )


def _marker_column(cells: list[str], span: RowSpan) -> int:
    """Where inside the row the marker goes.

    The rightmost EMPTY cell of the leading identity region the row leans on, and
    never column 0: services/citation.py takes ``cells[0]`` as a row's label and
    treats a row as leaning only while ``cells[0]`` is empty, so a marker in the
    first cell would become the row's identity. A declared (non-empty) cell is
    never overwritten either — the marker only ever fills a cell the page prints
    blank. Falls back to appending a trailing cell when the region has no other
    empty slot.
    """
    candidates = [j for j in span.columns if j != 0 and not cells[j]]
    if candidates:
        return max(candidates)
    return len(cells)


def _rowspan_marker(span: RowSpan, cells: list[str]) -> str:
    """The visible text for one unresolved row (see ROWSPAN_MARKER_PREFIX)."""
    clause = _REASON_CLAUSES.get(span.kind, "")
    if _row_carries_figures(cells):
        clause = f"{clause}; {_FIGURES_CLAUSE}" if clause else _FIGURES_CLAUSE
    return f"{ROWSPAN_MARKER_PREFIX}: {clause}]" if clause else ROWSPAN_MARKER_PREFIX + "]"


def _annotated_rows(table: MarkdownTable, spans: list[RowSpan]) -> tuple[list[str], dict[int, dict]]:
    """``table.data_rows`` with a marker inserted for every unresolved row.

    Returns the annotated rows plus the per-row annotation (the marker text and
    whether the row carries figures), so the metadata entry and the visible text
    are built from one source. Line<->row identity is preserved — exactly one
    line per row, no line added or dropped — so ticket 33's row-index handoff
    (``_piece_rows``) keeps mapping pieces to the right table rows, and each row
    stays a single atomic pipe line (ST-6c), so a split carries its marker with
    the row it annotates.

    The marker only ever lands on a cell the row itself declares empty, inside
    the leading identity region it leans on, and never on column 0 unless that
    region is column 0 alone (in which case it is appended past the last cell).
    ``services/citation.py`` reads a row's label from ``cells[0]`` and calls a
    row "leaning" only while ``cells[0]`` is empty, so this placement keeps both
    the label and the leaning-row classification of the annotated row identical
    to the raw row the provenance entries were computed from.
    """
    rows = list(table.data_rows)
    annotations: dict[int, dict] = {}
    for span in spans:
        if span.resolved:
            continue
        cells = MarkdownTable._cells_of(rows[span.row])
        figures = _row_carries_figures(cells)
        marker = _rowspan_marker(span, cells)
        col = _marker_column(cells, span)
        if col < len(cells):
            cells[col] = marker
        else:
            cells.append(marker)
        rows[span.row] = "| " + " | ".join(cells) + " |"
        annotations[span.row] = {"marker": marker, "figures": figures}
    return rows, annotations


def strip_rowspan_markers(text: str) -> str:
    """The chunk text as the page printed it (ticket 32's marker removed).

    Consumers that compare a chunk line with its source row, or that index the
    row's cells, must see the document's own text; the marker is the chunker's
    annotation, not content.
    """
    return _ROWSPAN_MARKER_RE.sub("", text)


def _assert_unresolved_rows_marked(chunks: list[dict], doc_id: str) -> None:
    """Ticket 32 ô 4: no row is emitted without its marker (guard, not luck).

    ``_chunk_table`` filters nothing — every data row of the table reaches a chunk
    — so the only thing standing between an unattributable row and the corpus is
    this invariant: for every emitted chunk, each ``rowspan_unresolved`` entry
    must have its marker present in that chunk's ``text``, as many times as there
    are entries declaring it. Counted per marker (not merely ``in text``) so a
    table with several same-kind unresolved rows cannot pass on one surviving
    marker. Raises instead of warning: a silent row is exactly the state this
    ticket removes, and a warning is invisible to the student.
    """
    for chunk in chunks:
        entries = chunk["metadata"].get("rowspan_unresolved") or []
        needed: dict[str, int] = {}
        for entry in entries:
            marker = str(entry.get("marker") or "")
            needed[marker] = needed.get(marker, 0) + 1
        for marker, count in needed.items():
            if not marker:
                raise RuntimeError(
                    f"Ticket 32: {doc_id}: an unresolved rowspan row carries no marker text"
                )
            found = chunk["text"].count(marker)
            if found < count:
                raise RuntimeError(
                    f"Ticket 32: {doc_id}: {count} unresolved rowspan row(s) declare marker "
                    f"{marker!r} but the emitted text carries it {found} time(s) — a row with "
                    "no traceable label would reach the corpus without saying so"
                )


def _chunk_table(table: MarkdownTable, base_meta: dict, doc_id: str, enc, start_idx: int) -> list[dict]:
    """Inner table-chunking logic (v4 Phase 3 hybrid strategy).

    Ticket 32: the table's rowspan provenance is resolved once here and attached
    to every chunk that carries the rows it belongs to. Rows that lean on a
    merged cell without being attributable are warned about on this path (the
    production path), never emitted silently.
    """
    chunks = []

    spans = table.rowspan_provenance()
    for span in spans:
        if not span.resolved:
            logger.warning(
                "Ticket 32: {} row {} carries values but leans on a merged/rowspan cell it "
                "does not carry (columns {} empty, filled above) and cannot be attributed: {} | {}",
                doc_id, span.row, list(span.columns), span.reason,
                table.data_rows[span.row][:120],
            )

    # Ticket 32: the emitted text is the annotated rows — every unresolved row
    # carries its visible marker. Routing (is_wide / is_small / program grouping)
    # and the provenance scan above stay on the RAW rows, so a marker can never
    # become an inheritable identity or change how the table is chunked.
    annotated, annotations = _annotated_rows(table, spans)
    all_rows = list(range(len(annotated)))

    def emit(idx: int, text: str, rows: list[int], **meta) -> dict:
        """Ticket 33: a table chunk hands the size gate the table-scoped indices
        of the rows it carries, so the gate can keep each provenance entry with
        the piece that actually owns its rows. Private handoff — popped by
        ``_enforce_size_gates`` before the chunk is returned — and only carried
        when there is provenance to scope."""
        chunk = _make_chunk(doc_id, idx, text, base_meta, **meta)
        if any(key.startswith("rowspan_") for key in chunk["metadata"]):
            chunk["_rowspan_row_indices"] = list(rows)
        return chunk

    # ── Rule 1: Wide-table bypass ──
    if table.is_wide:
        text = table.to_chunk_text(annotated)
        tok = count_tokens(text, enc)
        # Even if oversized, keep as single chunk per Phase 3
        chunks.append(emit(
            start_idx, text, all_rows,
            chunk_type="wide_table", table_dims=f"{len(table.data_rows)}x{table.num_cols}",
            token_count=tok, **_rowspan_metadata(spans, all_rows, annotations),
        ))
        return chunks

    # ── Rule 2: Small-table fallback ──
    if table.is_small:
        text = table.to_chunk_text(annotated)
        tok = count_tokens(text, enc)
        if tok >= MIN_TOKENS:
            chunks.append(emit(
                start_idx, text, all_rows,
                chunk_type="small_table", table_dims=f"{len(table.data_rows)}x{table.num_cols}",
                token_count=tok, **_rowspan_metadata(spans, all_rows, annotations),
            ))
        return chunks

    # ── Rule 3: Program-grouped tables ──
    if table.has_program_column():
        groups = table.group_row_indices_by_program()
        idx = start_idx
        for group in groups:
            text = table.to_chunk_text([annotated[j] for j in group])
            tok = count_tokens(text, enc)
            if tok < MIN_TOKENS:
                continue
            chunks.append(emit(
                idx, text, group,
                chunk_type="program_group", table_dims=f"{len(group)}x{table.num_cols}",
                token_count=tok, **_rowspan_metadata(spans, group, annotations),
            ))
            idx += 1
        return chunks

    # ── Rule 4: General row-packing with header ──
    current_rows: list[int] = []
    current_tok = count_tokens(table.header_line + "\n" + table.separator_line, enc) if table.header_line else 0
    idx = start_idx

    for i, row in enumerate(annotated):
        row_tok = count_tokens(row, enc)
        if current_tok + row_tok > PROGRAM_GROUP_MAX_TOKENS and current_rows:
            text = table.to_chunk_text([annotated[j] for j in current_rows])
            chunks.append(emit(
                idx, text, current_rows,
                chunk_type="packed_table", table_dims=f"{len(current_rows)}x{table.num_cols}",
                token_count=count_tokens(text, enc), **_rowspan_metadata(spans, current_rows, annotations),
            ))
            idx += 1
            current_rows = [i]
            current_tok = row_tok + (count_tokens(table.header_line + "\n" + table.separator_line, enc) if table.header_line else 0)
        else:
            current_rows.append(i)
            current_tok += row_tok

    if current_rows:
        text = table.to_chunk_text([annotated[j] for j in current_rows])
        tok = count_tokens(text, enc)
        if tok >= MIN_TOKENS:
            chunks.append(emit(
                idx, text, current_rows,
                chunk_type="packed_table", table_dims=f"{len(current_rows)}x{table.num_cols}",
                token_count=tok, **_rowspan_metadata(spans, current_rows, annotations),
            ))

    return chunks


def _flag_oversized_pipe_rows(pieces: list[str], enc, max_tokens: int) -> None:
    """ST-6c: log a warning for any piece that still exceeds ``max_tokens``.

    These are single atomic pipe-table rows too large to split; keeping them
    whole is deliberate (a row must never be cut mid-row), so the warning
    surfaces the data anomaly without breaking row integrity.
    """
    for p in pieces:
        tok = count_tokens(p, enc)
        if tok > max_tokens:
            logger.warning(
                "ST-6c: pipe-table row exceeds HARD_MAX_TOKENS ({} > {}) and is "
                "kept whole: {}",
                tok, max_tokens, p[:80],
            )


def _split_pieces_and_sources(
    text: str, enc, max_tokens: int,
) -> tuple[list[str], list[str] | None, list[list[int]] | None]:
    """Split an oversized chunk into <=max_tokens pieces, tracking sources.

    Table-looking text is split at row boundaries (header + separator repeated on
    each piece) to preserve table structure; everything else uses a balanced
    token-boundary split so no tiny tail piece is produced. ST-6c: pipe-table
    rows are ATOMIC — a single oversized row is kept whole and flagged, never
    cut mid-row, and any pipe-table text is split at line boundaries before a
    balanced token split is ever used.

    Ticket 33: along with the pieces this returns the line list they were cut
    from and, per piece, the indices of the lines that piece carries. The
    splitter is the only place that knows which line went where, and the size
    gate needs exactly that to keep rowspan provenance inside the piece that
    owns the rows it names. ``sources`` is None when the balanced token split
    was used (there is no line boundary to map) — unreachable for table text,
    which is line-preserving by construction.
    """
    if count_tokens(text, enc) <= max_tokens:
        lines = text.split("\n")
        return [text], lines, [list(range(len(lines)))]

    stripped = text.strip()
    if stripped.startswith("|") and stripped.endswith("|"):
        lines = stripped.split("\n")
        header, separator, data_rows = "", "", []
        header_i, separator_i = -1, -1
        data_idx: list[int] = []
        for i, ln in enumerate(lines):
            if re.match(r"^\|[\s\-|:]+\|$", ln) and not separator:
                separator, separator_i = ln, i
            elif not header and ln.strip().startswith("|"):
                header, header_i = ln, i
            elif ln.strip().startswith("|"):
                data_rows.append(ln)
                data_idx.append(i)
        if header and data_rows:
            head_tok = count_tokens(header + "\n" + separator, enc)
            head_idx = ([header_i] if header_i >= 0 else []) + ([separator_i] if separator_i >= 0 else [])
            pieces, blocks, current, cur_idx, cur_tok = [], [], [], [], head_tok
            for i, row in zip(data_idx, data_rows):
                row_tok = count_tokens(row, enc)
                if cur_tok + row_tok > max_tokens and current:
                    pieces.append("\n".join([header, separator] + current))
                    blocks.append(head_idx + cur_idx)
                    current, cur_idx, cur_tok = [row], [i], head_tok + row_tok
                else:
                    current.append(row)
                    cur_idx.append(i)
                    cur_tok += row_tok
            if current:
                pieces.append("\n".join([header, separator] + current))
                blocks.append(head_idx + cur_idx)
            if all(count_tokens(p, enc) <= max_tokens for p in pieces):
                return pieces, lines, blocks
            # ST-6c: row packing could not fit every row under the cap — at
            # least one row is itself oversized. Rows are atomic, so keep the
            # row-preserving pieces and flag the oversize instead of falling
            # through to a balanced split that would cut a row mid-row.
            _flag_oversized_pipe_rows(pieces, enc, max_tokens)
            return pieces, lines, blocks

    if _PIPE_TABLE_LINE_RE.search(text):
        # ST-6c: pipe-table text (even mixed with surrounding prose) splits at
        # line boundaries so no row is cut, before any balanced token split.
        lines = text.split("\n")
        pieces, blocks, current, cur_idx, cur_tok = [], [], [], [], 0
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            line_tok = count_tokens(line, enc)
            if line_tok > max_tokens:
                # Single row above the cap: keep whole + flag, never cut.
                # Flush any pending rows first so lines that precede this
                # oversized row are not emitted after it (mirrors
                # src/chunker.py:_split_oversized_pipe_safe).
                if current:
                    pieces.append("\n".join(current))
                    blocks.append(cur_idx)
                    current, cur_idx, cur_tok = [], [], 0
                pieces.append(line)
                blocks.append([i])
                _flag_oversized_pipe_rows([line], enc, max_tokens)
                continue
            if cur_tok + line_tok > max_tokens and current:
                pieces.append("\n".join(current))
                blocks.append(cur_idx)
                current, cur_idx, cur_tok = [line], [i], line_tok
            else:
                current.append(line)
                cur_idx.append(i)
                cur_tok += line_tok
        if current:
            pieces.append("\n".join(current))
            blocks.append(cur_idx)
        if pieces:
            return pieces, lines, blocks

    # Balanced token-boundary split (no tiny tail) — no line boundary to map.
    tokens = enc.encode(text)
    n = len(tokens)
    k = (n + max_tokens - 1) // max_tokens
    base, rem = divmod(n, k)
    pieces, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        piece = enc.decode(tokens[start:end]).strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces, None, None


def _split_oversized_preserving(text: str, enc, max_tokens: int) -> list[str]:
    """Split an oversized chunk into <=max_tokens pieces.

    The line- and row-preserving rules live in ``_split_pieces_and_sources``;
    this is the text-only view used by the rechunk scripts.
    """
    return _split_pieces_and_sources(text, enc, max_tokens)[0]


def _piece_rows(
    row_indices: list[int] | None, lines: list[str] | None,
    blocks: list[list[int]] | None, text: str,
) -> list[set[int]] | None:
    """Ticket 33: the table-scoped row indices each split piece carries.

    Row indices are TABLE-scoped (ticket 32's frame of reference): every chunk
    of one ``chunk_table`` call re-emits that table's header, so a piece's rows
    are its chunk's rows minus the leading header/separator lines. Returns None
    when no line mapping exists (the split fell back to a token boundary, or the
    chunk carried no rows), and the caller then leaves provenance untouched.
    """
    if row_indices is None or blocks is None or lines is None:
        return None
    if len(lines) != len(text.split("\n")) or len(lines) < len(row_indices):
        return None
    head = len(lines) - len(row_indices)
    return [
        {row_indices[i - head] for i in block if i >= head}
        for block in blocks
    ]


def _rowspan_scoped_metadata(meta: dict, rows: set[int]) -> dict:
    """Ticket 33: the provenance one split piece may carry.

    A piece carries exactly the entries of the rows it holds. Row indices are
    table-scoped and the pieces of one split hold *disjoint* row sets, so every
    entry has exactly one home: the piece holding the row it labels. Two
    consequences, and they are the two things this ticket asks for:

    * a piece can no longer name a row that fell to a sibling piece — ticket
      32's over-labelling, where a piece claimed provenance for a row it does
      not contain;
    * no entry is dropped anywhere, so the pieces of a split carry exactly the
      entry set of the unsplit chunk.

    The second point is why the rule is stated over the labelled row alone. An
    attribution also names its ``anchor_row``, and a split boundary can fall
    between the two (measured on ``data/staging/pdf/pdf_2.md``: 6 of the 66
    entries in split parents). Requiring the anchor to be in the same piece
    would drop those 6 from every piece at once — putting a leaning row back
    into the corpus with neither attribution nor warning, the exact silent state
    ticket 32 removes. Keeping them with the row they label costs nothing the
    consumer relies on: the anchor is still a row of the same table (ticket 32
    guarantees the anchor is in the chunk the entry was attached to) and the row
    the entry names is always one the piece contains. Empty key sets are
    dropped, matching ``_rowspan_metadata``'s shape.
    """
    scoped = {key: value for key, value in meta.items() if not key.startswith("rowspan_")}
    for key in ("rowspan_attribution", "rowspan_unresolved"):
        entries = meta.get(key)
        if not entries:
            continue
        kept = [entry for entry in entries if entry["row"] in rows]
        if kept:
            scoped[key] = kept
    return scoped


def _enforce_size_gates(
    chunks: list[dict], base_meta: dict, doc_id: str, enc,
    hard_max_tokens: int | None = None, soft_min_tokens: int | None = None,
) -> list[dict]:
    """T1.8: hard cap (split) + soft floor (merge) driven by config.settings.

    Splits oversized chunks (preserving table rows), merges sub-floor chunks into
    a neighbor without exceeding the cap, then re-indexes chunk ids. Knobs resolve
    from config.settings at call time unless explicit values are passed.

    Ticket 33: a split piece carries only the provenance of the rows it holds —
    each entry stays with the piece that owns its rows, and none is dropped.
    """
    hard_max = hard_max_tokens if hard_max_tokens is not None else settings.HARD_MAX_TOKENS
    soft_min = soft_min_tokens if soft_min_tokens is not None else settings.SOFT_MIN_TOKENS

    if not chunks:
        return chunks

    out: list[dict] = []
    for c in chunks:
        row_indices = c.pop("_rowspan_row_indices", None)
        text = c.get("text", "")
        tok = count_tokens(text, enc)
        if tok <= hard_max:
            out.append(c)
            continue
        pieces, lines, blocks = _split_pieces_and_sources(text, enc, hard_max)
        meta = {k: v for k, v in (c.get("metadata") or {}).items() if k != "token_count"}
        rows_per_piece = _piece_rows(row_indices, lines, blocks, text)
        for si, piece in enumerate(pieces):
            piece_meta = meta if rows_per_piece is None else _rowspan_scoped_metadata(meta, rows_per_piece[si])
            out.append(_make_chunk(
                doc_id, len(out), piece, base_meta,
                **piece_meta, split_index=si, token_count=count_tokens(piece, enc),
            ))

    # Soft-floor merge into previous neighbor (keep hard cap).
    merged: list[dict] = []
    for c in out:
        if merged:
            cur_tok = count_tokens(c.get("text", ""), enc)
            prev_tok = count_tokens(merged[-1].get("text", ""), enc)
            if cur_tok < soft_min and prev_tok + cur_tok <= hard_max:
                merged[-1]["text"] = f"{merged[-1].get('text', '')}\n\n{c.get('text', '')}"
                merged[-1]["text_plain"] = merged[-1]["text"]
                prev_meta = merged[-1].setdefault("metadata", {})
                prev_meta["token_count"] = prev_tok + cur_tok
                _absorb_rowspan(prev_meta, c.get("metadata") or {})
                continue
        merged.append(c)

    # Re-index chunk ids sequentially for the caller's doc-level sequencing.
    for i, c in enumerate(merged):
        c["chunk_id"] = i
        c["id"] = f"{doc_id}_chunk_{i}"
    return merged


def _make_chunk(doc_id: str, chunk_id: int, text: str, base_meta: dict, **extra_meta) -> dict:
    """Create a chunk dict with all metadata."""
    meta = {**base_meta, **extra_meta}
    return {
        "id": f"{doc_id}_chunk_{chunk_id}",
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "text_plain": text,
        "metadata": meta,
        "breadcrumbs": [base_meta.get("source", "")],
        "prev_chunk_id": "",
        "next_chunk_id": "",
        "sparse_terms": extract_sparse_terms(text),
    }


# ══════════════════════════════════════════
# TEXT CHUNKING (sentence-boundary aware)
# ══════════════════════════════════════════

# Vietnamese sentence-end markers — expanded for better boundary detection
_SENT_END = re.compile(r"[.!?;:…–—]\s*$")
_SENT_END_SOFT = re.compile(r'[.,:;!?…–—)\]"\']\s*$')
# Vietnamese phrase-end markers that indicate a natural break point
_VI_PHRASE_END = re.compile(
    r"(như\s+sau|bao\s+gồm|bao\s+gồm\s+các|theo\s+sau|được\s+sau|"
    r"như\s+thế\s+nào|được\s+như\s+sau|"
    r"điểm\s+chuẩn|chỉ\s+tiêu|học\s+phí|mã\s+ngành|"
    r"phương\s+thức|xét\s+tuyển|tuyển\s+sinh|"
    r"thời\s+gian|hồ\s+sơ|thủ\s+tục|điều\s+kiện)\s*:\s*$|"
    r"[.:;!?…–—]\s*$"
)


def chunk_text_sections(
    text: str, base_meta: dict, doc_id: str, enc, start_idx: int,
    text_chunk_size: int | None = None, text_overlap: int | None = None,
) -> list[dict]:
    """Split non-table text with sentence-boundary awareness."""
    budget = text_chunk_size if text_chunk_size is not None else settings.CHUNK_SIZE_TOKENS
    overlap = text_overlap if text_overlap is not None else settings.CHUNK_OVERLAP

    # Split by double newlines first
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Further split long paragraphs at sentence boundaries
    parts = []
    for p in paragraphs:
        if count_tokens(p, enc) <= budget:
            parts.append(p)
        else:
            # Split by lines first
            lines = p.split("\n")
            sub_parts = _pack_lines(lines, enc, budget, overlap)
            parts.extend(sub_parts)

    chunks = []
    idx = start_idx
    for part in parts:
        tok = count_tokens(part, enc)
        if tok < MIN_TOKENS:
            continue
        # Try to end at sentence boundary
        part = _trim_to_sentence(part, enc, budget)
        chunks.append(_make_chunk(
            doc_id, idx, part, base_meta,
            chunk_type="text_section", token_count=count_tokens(part, enc),
        ))
        idx += 1

    return chunks


def _pack_lines(
    lines: list[str], enc, text_chunk_size: int | None = None, text_overlap: int | None = None,
) -> list[str]:
    """Pack lines into chunks respecting token budget and sentence boundaries."""
    budget = text_chunk_size if text_chunk_size is not None else settings.CHUNK_SIZE_TOKENS
    overlap = text_overlap if text_overlap is not None else settings.CHUNK_OVERLAP

    parts = []
    current: list[str] = []
    current_tok = 0

    for line in lines:
        line = line.strip()
        if not line:
            if current:
                current.append("")
            continue

        line_tok = count_tokens(line, enc)

        if current_tok + line_tok > budget and current:
            # Try to complete the sentence by scanning ahead
            extended = list(current) + [line]
            extended_tok = current_tok + line_tok
            # Keep adding lines until we hit a sentence boundary or 2x budget
            if extended_tok <= budget * 2.0:
                chunk_text = _complete_sentence(extended, enc, budget * 2.0, budget)
                if chunk_text:
                    parts.append(chunk_text)
                    current = []
                    current_tok = 0
                    continue

            # Normal flush at current boundary
            chunk_text = "\n".join(current).strip()
            parts.append(chunk_text)
            # Start new chunk with overlap
            overlap_lines = _get_overlap_lines(current, enc, overlap)
            current = overlap_lines + [line]
            current_tok = sum(count_tokens(l, enc) for l in current if l.strip())
        else:
            current.append(line)
            current_tok += line_tok

    if current:
        text = "\n".join(current).strip()
        if text:
            parts.append(text)

    return parts


def _complete_sentence(
    lines: list[str], enc, max_tokens: int, text_chunk_size: int | None = None,
) -> str | None:
    """Try to extend lines until a sentence boundary is found within max_tokens.

    Returns the completed chunk text if a boundary is found, None otherwise.
    """
    budget = text_chunk_size if text_chunk_size is not None else settings.CHUNK_SIZE_TOKENS

    result: list[str] = []
    total_tok = 0

    for line in lines:
        line_tok = count_tokens(line, enc)
        if total_tok + line_tok > max_tokens:
            break
        result.append(line)
        total_tok += line_tok
        # Check if this line ends at a sentence/phrase boundary
        stripped = line.strip()
        if stripped and (_SENT_END_SOFT.search(stripped) or _VI_PHRASE_END.search(stripped)):
            text = "\n".join(result).strip()
            # Only return if we actually extended past the original budget somewhat
            if total_tok > budget * 0.8:
                return text

    return None


def _get_overlap_lines(lines: list[str], enc, overlap_tokens: int) -> list[str]:
    """Get last N tokens worth of lines for overlap."""
    if not lines or overlap_tokens == 0:
        return []

    result: list[str] = []
    total_tok = 0
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        tok = count_tokens(line, enc)
        if total_tok + tok > overlap_tokens:
            break
        result.insert(0, line)
        total_tok += tok

    return result


def _trim_to_sentence(text: str, enc, text_chunk_size: int | None = None) -> str:
    """Try to end text at a sentence boundary without exceeding chunk size."""
    budget = text_chunk_size if text_chunk_size is not None else settings.CHUNK_SIZE_TOKENS

    tok = count_tokens(text, enc)
    if tok <= budget:
        return text

    # Find last sentence boundary within budget
    lines = text.split("\n")
    result: list[str] = []
    total_tok = 0

    for line in lines:
        line_tok = count_tokens(line, enc)
        if total_tok + line_tok > budget and result:
            # Check if current result ends at sentence/phrase boundary
            last_line = result[-1].strip()
            if _SENT_END_SOFT.search(last_line) or _VI_PHRASE_END.search(last_line):
                break
            # If not, include this line to try to complete the sentence
            if total_tok + line_tok <= budget * 1.3:  # allow 30% overflow
                result.append(line)
                total_tok += line_tok
            break
        result.append(line)
        total_tok += line_tok

    return "\n".join(result).strip()


def _merge_consecutive_table_chunks(chunks: list[dict], base_meta: dict, doc_id: str, enc) -> list[dict]:
    """Merge consecutive small_table chunks that share the same header row.

    PDF-extracted tables are often split into fragments (2 data rows each)
    with identical headers. Merging them produces coherent multi-row tables
    that are better for retrieval.
    """
    if not chunks:
        return chunks

    merged: list[dict] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        ct = c.get("metadata", {}).get("chunk_type", "")
        text = c.get("text", "")

        # Only merge small_table chunks that are tables
        if ct != "small_table" or not is_table_chunk_inline(text):
            merged.append(c)
            i += 1
            continue

        # Extract header from this chunk
        rows = [l.strip() for l in text.split("\n") if l.strip().startswith("|")]
        if len(rows) < 3:  # need header + separator + at least 1 data row
            merged.append(c)
            i += 1
            continue

        header = rows[0]
        separator = rows[1] if len(rows) > 1 else ""
        data_rows = [r for r in rows[2:] if not re.match(r"^\|[\s\-|:]+\|$", r)]

        # Collect consecutive chunks with the same header
        combined_data = list(data_rows)
        j = i + 1
        while j < len(chunks):
            nxt = chunks[j]
            nxt_ct = nxt.get("metadata", {}).get("chunk_type", "")
            nxt_text = nxt.get("text", "")
            if nxt_ct != "small_table" or not is_table_chunk_inline(nxt_text):
                break
            nxt_rows = [l.strip() for l in nxt_text.split("\n") if l.strip().startswith("|")]
            if len(nxt_rows) < 2 or nxt_rows[0] != header:
                break
            nxt_data = [r for r in nxt_rows[2:] if not re.match(r"^\|[\s\-|:]+\|$", r)]
            # Also include any data rows that appear as row[1] if no separator
            if not nxt_data and len(nxt_rows) >= 2 and not re.match(r"^\|[\s\-|:]+\|$", nxt_rows[1]):
                nxt_data = [nxt_rows[1]]
            combined_data.extend(nxt_data)
            j += 1

        if j == i + 1:
            # No merge happened
            merged.append(c)
            i += 1
            continue

        # Build merged chunk
        parts = []
        if header:
            parts.append(header)
        if separator:
            parts.append(separator)
        parts.extend(combined_data)
        merged_text = "\n".join(parts)
        tok = count_tokens(merged_text, enc)

        # Determine the new chunk_type based on dimensions
        num_cols = len([c for c in header.split("|") if c.strip()]) if header else 0
        num_data = len(combined_data)
        if num_cols >= WIDE_TABLE_MIN_COLS and num_data >= WIDE_TABLE_MIN_ROWS:
            new_ct = "wide_table"
        elif num_data <= SMALL_TABLE_MAX_ROWS:
            new_ct = "small_table"
        else:
            new_ct = "program_group"

        merged.append(_make_chunk(
            doc_id, len(merged), merged_text, base_meta,
            chunk_type=new_ct, table_dims=f"{num_data}x{num_cols}",
            token_count=tok,
        ))
        i = j

    # Re-index chunk IDs
    for idx, c in enumerate(merged):
        c["chunk_id"] = idx
        c["id"] = f"{c['doc_id']}_chunk_{idx}"

    return merged


def is_table_chunk_inline(text: str) -> bool:
    """Quick check if text is a table chunk."""
    t = text.strip()
    return t.startswith("|") and t.endswith("|")


# ══════════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════════

def process_file(filepath: Path, enc, file_idx: int) -> list[dict]:
    """Process a single file with v4 Phase 3 hybrid chunking."""
    logger.info(f"Processing: {filepath.name}")
    text = filepath.read_text(encoding="utf-8")
    if not text.strip():
        return []

    year = detect_year(text)
    source_type = detect_source_type(filepath)

    if source_type == "web_crawl":
        m = re.search(r"thongbao_(\d+)", filepath.name)
        source = f"tuyensinh.husc.edu.vn/thongbao.php?id={m.group(1)}" if m else filepath.stem
    else:
        source = filepath.stem

    info_type = detect_info_type(text, source)

    base_meta = {
        "source": source,
        "source_type": source_type,
        "year": year,
        "data_year": int(year),
        "info_type": info_type,
        "school": "HUSC",
        "audience": "thi_sinh",
        "unit": "HUSC",
    }

    doc_id = f"chunked_{file_idx}"
    all_chunks = []
    chunk_idx = 0

    # Separate text from table blocks — with lookahead for table continuation
    lines = text.split("\n")
    table_buffer: list[str] = []
    text_buffer: list[str] = []
    in_table = False

    def flush_text():
        nonlocal chunk_idx, text_buffer
        if text_buffer:
            txt = "\n".join(text_buffer).strip()
            if txt and count_tokens(txt, enc) >= MIN_TOKENS:
                chunks = chunk_text_sections(txt, base_meta, doc_id, enc, chunk_idx)
                all_chunks.extend(chunks)
                chunk_idx += len(chunks)
            text_buffer = []

    def flush_table():
        nonlocal chunk_idx, table_buffer
        if table_buffer:
            table = MarkdownTable(table_buffer)
            if table.data_rows:  # has at least 1 data row
                chunks = chunk_table(table, base_meta, doc_id, enc, chunk_idx)
                all_chunks.extend(chunks)
                chunk_idx += len(chunks)
            table_buffer = []

    def peek_next_table_header(from_idx: int) -> str:
        """Look ahead past blank lines to see if next content is a table with same structure."""
        j = from_idx
        # Skip blank lines
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            return ""
        first = lines[j].strip()
        if first.startswith("|") and first.endswith("|") and "|" in first[1:-1]:
            return first
        return ""

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]

        if is_table_line:
            if not in_table:
                flush_text()
                in_table = True
            table_buffer.append(stripped)
        else:
            if in_table:
                # Check if this is just a blank line and the next non-blank content
                # is a continuation of the same table (same header row)
                if not stripped:
                    next_header = peek_next_table_header(i + 1)
                    current_header = ""
                    if table_buffer:
                        for tl in table_buffer:
                            if not re.match(r"^\|[\s\-|:]+\|$", tl):
                                current_header = tl
                                break
                    if next_header and current_header and next_header == current_header:
                        # Same header ahead — this is a table continuation, skip blank line
                        i += 1
                        continue
                flush_table()
                in_table = False
            text_buffer.append(lines[i])
        i += 1

    flush_text()
    flush_table()

    logger.info(f"  → {filepath.name}: {len(all_chunks)} chunks (year={year}, type={info_type})")
    return all_chunks


def main():
    logger.info("=== TABLE-AWARE CHUNKING v2 (v4 Phase 3 Hybrid) ===")

    staging_files = []
    if STAGING_HTML.exists():
        staging_files.extend(sorted(STAGING_HTML.glob("*.md")))
    if STAGING_PDF.exists():
        staging_files.extend(sorted(STAGING_PDF.glob("*.md")))

    if not staging_files:
        logger.error("No staging files found!")
        return

    logger.info(f"Found {len(staging_files)} staging files")

    enc = tiktoken.get_encoding(ENCODING)

    # Delete old table-aware chunks (12+)
    for f in sorted(CHUNKED_DIR.glob("chunked_*.jsonl")):
        m = re.search(r"chunked_(\d+)", f.name)
        if m and int(m.group(1)) >= 12 and "enhanced" not in f.name:
            logger.info(f"  Removing old: {f.name}")
            f.unlink()

    file_idx = 12
    all_chunks = []

    for filepath in staging_files:
        chunks = process_file(filepath, enc, file_idx)
        if chunks:
            out_path = CHUNKED_DIR / f"chunked_{file_idx}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            logger.info(f"  → Saved {len(chunks)} chunks to {out_path.name}")
            all_chunks.extend(chunks)
            file_idx += 1

    logger.info(f"=== CHUNKING COMPLETE ===")
    logger.info(f"Total: {len(all_chunks)} chunks from {len(staging_files)} files")

    # Summary by chunk_type
    by_type = {}
    for c in all_chunks:
        ct = c.get("metadata", {}).get("chunk_type", "unknown")
        by_type[ct] = by_type.get(ct, 0) + 1
    logger.info("By chunk_type:")
    for k, v in sorted(by_type.items()):
        logger.info(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"  TABLE-AWARE CHUNKING v2 COMPLETE")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
