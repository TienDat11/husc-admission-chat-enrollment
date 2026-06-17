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
from typing import Any

import tiktoken
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import RAGSettings
from services.temporal_authority import get_current_admission_year

STAGING_HTML = Path(__file__).parent.parent / "data" / "staging" / "html"
STAGING_PDF = Path(__file__).parent.parent / "data" / "staging" / "pdf"
CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"

ENCODING = "cl100k_base"
TEXT_CHUNK_SIZE = 220
TEXT_OVERLAP = 50
MIN_TOKENS = 25
WIDE_TABLE_MIN_COLS = 8
WIDE_TABLE_MIN_ROWS = 1
SMALL_TABLE_MAX_ROWS = 6
PROGRAM_GROUP_MAX_TOKENS = 500


STOPWORDS = {
    "và", "các", "của", "có", "được", "cho", "trong", "là", "một", "này",
    "để", "với", "theo", "từ", "đã", "sẽ", "không", "khi", "bằng",
}


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

    def group_rows_by_program(self) -> list[list[str]]:
        """Group data rows by program prefix (first column content changes)."""
        if not self.data_rows:
            return []

        groups: list[list[str]] = []
        current_group: list[str] = []
        prev_prefix = ""

        for row in self.data_rows:
            # Extract first column as group key
            cells = [c.strip() for c in row.split("|") if c.strip()]
            prefix = cells[0] if cells else ""

            # New group when: prefix changes from non-empty to non-empty
            # OR prefix is non-empty and current group is already started
            if prefix and prev_prefix and prefix != prev_prefix and current_group:
                groups.append(current_group)
                current_group = [row]
            else:
                current_group.append(row)

            if prefix:
                prev_prefix = prefix

        if current_group:
            groups.append(current_group)

        return groups if groups else [self.data_rows]

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
    """
    chunks = []

    # ── Rule 1: Wide-table bypass ──
    if table.is_wide:
        text = table.to_chunk_text()
        tok = count_tokens(text, enc)
        # Even if oversized, keep as single chunk per Phase 3
        chunks.append(_make_chunk(
            doc_id, start_idx, text, base_meta,
            chunk_type="wide_table", table_dims=f"{len(table.data_rows)}x{table.num_cols}",
            token_count=tok,
        ))
        return chunks

    # ── Rule 2: Small-table fallback ──
    if table.is_small:
        text = table.to_chunk_text()
        tok = count_tokens(text, enc)
        if tok >= MIN_TOKENS:
            chunks.append(_make_chunk(
                doc_id, start_idx, text, base_meta,
                chunk_type="small_table", table_dims=f"{len(table.data_rows)}x{table.num_cols}",
                token_count=tok,
            ))
        return chunks

    # ── Rule 3: Program-grouped tables ──
    if table.has_program_column():
        groups = table.group_rows_by_program()
        idx = start_idx
        for group in groups:
            text = table.to_chunk_text(group)
            tok = count_tokens(text, enc)
            if tok < MIN_TOKENS:
                continue
            chunks.append(_make_chunk(
                doc_id, idx, text, base_meta,
                chunk_type="program_group", table_dims=f"{len(group)}x{table.num_cols}",
                token_count=tok,
            ))
            idx += 1
        return chunks

    # ── Rule 4: General row-packing with header ──
    current_rows: list[str] = []
    current_tok = count_tokens(table.header_line + "\n" + table.separator_line, enc) if table.header_line else 0
    idx = start_idx

    for row in table.data_rows:
        row_tok = count_tokens(row, enc)
        if current_tok + row_tok > PROGRAM_GROUP_MAX_TOKENS and current_rows:
            text = table.to_chunk_text(current_rows)
            chunks.append(_make_chunk(
                doc_id, idx, text, base_meta,
                chunk_type="packed_table", table_dims=f"{len(current_rows)}x{table.num_cols}",
                token_count=count_tokens(text, enc),
            ))
            idx += 1
            current_rows = [row]
            current_tok = row_tok + (count_tokens(table.header_line + "\n" + table.separator_line, enc) if table.header_line else 0)
        else:
            current_rows.append(row)
            current_tok += row_tok

    if current_rows:
        text = table.to_chunk_text(current_rows)
        tok = count_tokens(text, enc)
        if tok >= MIN_TOKENS:
            chunks.append(_make_chunk(
                doc_id, idx, text, base_meta,
                chunk_type="packed_table", table_dims=f"{len(current_rows)}x{table.num_cols}",
                token_count=tok,
            ))

    return chunks


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


def chunk_text_sections(text: str, base_meta: dict, doc_id: str, enc, start_idx: int) -> list[dict]:
    """Split non-table text with sentence-boundary awareness."""
    # Split by double newlines first
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Further split long paragraphs at sentence boundaries
    parts = []
    for p in paragraphs:
        if count_tokens(p, enc) <= TEXT_CHUNK_SIZE:
            parts.append(p)
        else:
            # Split by lines first
            lines = p.split("\n")
            sub_parts = _pack_lines(lines, enc)
            parts.extend(sub_parts)

    chunks = []
    idx = start_idx
    for part in parts:
        tok = count_tokens(part, enc)
        if tok < MIN_TOKENS:
            continue
        # Try to end at sentence boundary
        part = _trim_to_sentence(part, enc)
        chunks.append(_make_chunk(
            doc_id, idx, part, base_meta,
            chunk_type="text_section", token_count=count_tokens(part, enc),
        ))
        idx += 1

    return chunks


def _pack_lines(lines: list[str], enc) -> list[str]:
    """Pack lines into chunks respecting token budget and sentence boundaries."""
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

        if current_tok + line_tok > TEXT_CHUNK_SIZE and current:
            # Try to complete the sentence by scanning ahead
            extended = list(current) + [line]
            extended_tok = current_tok + line_tok
            # Keep adding lines until we hit a sentence boundary or 2x budget
            if extended_tok <= TEXT_CHUNK_SIZE * 2.0:
                chunk_text = _complete_sentence(extended, enc, TEXT_CHUNK_SIZE * 2.0)
                if chunk_text:
                    parts.append(chunk_text)
                    current = []
                    current_tok = 0
                    continue

            # Normal flush at current boundary
            chunk_text = "\n".join(current).strip()
            parts.append(chunk_text)
            # Start new chunk with overlap
            overlap_lines = _get_overlap_lines(current, enc, TEXT_OVERLAP)
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


def _complete_sentence(lines: list[str], enc, max_tokens: int) -> str | None:
    """Try to extend lines until a sentence boundary is found within max_tokens.

    Returns the completed chunk text if a boundary is found, None otherwise.
    """
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
            if total_tok > TEXT_CHUNK_SIZE * 0.8:
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


def _trim_to_sentence(text: str, enc) -> str:
    """Try to end text at a sentence boundary without exceeding chunk size."""
    tok = count_tokens(text, enc)
    if tok <= TEXT_CHUNK_SIZE:
        return text

    # Find last sentence boundary within budget
    lines = text.split("\n")
    result: list[str] = []
    total_tok = 0

    for line in lines:
        line_tok = count_tokens(line, enc)
        if total_tok + line_tok > TEXT_CHUNK_SIZE and result:
            # Check if current result ends at sentence/phrase boundary
            last_line = result[-1].strip()
            if _SENT_END_SOFT.search(last_line) or _VI_PHRASE_END.search(last_line):
                break
            # If not, include this line to try to complete the sentence
            if total_tok + line_tok <= TEXT_CHUNK_SIZE * 1.3:  # allow 30% overflow
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
