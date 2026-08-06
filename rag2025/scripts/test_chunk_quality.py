"""
UltraQA — Chunk Quality Deep Test
Tests all aspects of chunking before LanceDB ingest and NER graph build.

Based on v4 plans Phase 2 (content audit) and Phase 3 (chunking redesign):
- D1: Semantic completeness (no orphaned table rows)
- D3: Table extraction quality (wide-table bypass, program grouping)
- D5: Section boundaries (header context preserved)
- D8: Metadata preservation (all required fields present)
- Plus: overlap integrity, dedup, token count sanity, encoding safety

Usage:
    cd rag2025
    $env:PYTHONPATH="src;."; python scripts/test_chunk_quality.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken

CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"
STAGING_DIR = Path(__file__).parent.parent / "data" / "staging"
ENC = tiktoken.get_encoding("cl100k_base")

# ── v4 Phase 3 acceptance criteria ──
REQUIRED_META_FIELDS = {"source", "source_type", "year", "data_year", "info_type", "school", "audience", "unit"}
TABLE_ROW_MIN_TOKENS = 20
TEXT_CHUNK_MIN_TOKENS = 30
HARD_MAX_TOKENS = 512  # T1.8: hard cap — 0 chunks above this is an acceptance gate
MAX_CHUNK_TOKENS = 800  # T1.8: soft ceiling — chunks above this are warned
WIDE_TABLE_COLS = 8
WIDE_TABLE_MIN_ROWS = 1


def load_all_chunks() -> list[dict]:
    """Load all chunks from JSONL files."""
    chunks = []
    for f in sorted(CHUNKED_DIR.glob("chunked_*.jsonl")):
        if "enhanced" in f.name or "test" in f.name:
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def is_table_chunk(chunk: dict) -> bool:
    text = chunk.get("text", "")
    return text.strip().startswith("|") and text.strip().endswith("|")


def parse_table_rows(text: str) -> list[str]:
    """Parse markdown table into rows."""
    return [l.strip() for l in text.strip().split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]


def count_table_cols(row: str) -> int:
    """Count columns in a markdown table row."""
    return len([c for c in row.split("|") if c.strip() != ""])


def has_header_row(rows: list[str]) -> bool:
    """Check if table has a separator row (|---|---|)."""
    for r in rows[:3]:
        if re.match(r"^\|[\s\-|:]+\|$", r):
            return True
    return False


# ══════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details: list[str] = []

    def ok(self, msg: str):
        self.passed += 1

    def fail(self, msg: str):
        self.failed += 1
        self.details.append(f"  FAIL: {msg}")

    def warn(self, msg: str):
        self.warnings += 1
        self.details.append(f"  WARN: {msg}")

    def summary(self) -> str:
        total = self.passed + self.failed
        status = "PASS" if self.failed == 0 else "FAIL"
        s = f"[{status}] {self.name}: {self.passed}/{total} passed, {self.warnings} warnings"
        if self.details:
            s += "\n" + "\n".join(self.details[:20])
            if len(self.details) > 20:
                s += f"\n  ... and {len(self.details) - 20} more"
        return s


def test_d1_semantic_completeness(chunks: list[dict]) -> TestResult:
    """D1: Semantic completeness — no orphaned table rows, no empty chunks."""
    r = TestResult("D1: Semantic Completeness")

    # Check no empty text
    for c in chunks:
        text = c.get("text", "").strip()
        if not text:
            r.fail(f"Empty chunk: {c.get('id', '?')}")
        else:
            r.ok("Non-empty chunk")

    # Check table rows have header context
    table_chunks = [c for c in chunks if is_table_chunk(c)]
    orphaned = 0
    for c in table_chunks:
        text = c.get("text", "")
        lines = text.strip().split("\n")
        # A well-formed table chunk should have header + separator + data row(s)
        if not has_header_row(lines):
            # Check if first line looks like header (has column names, not just data)
            first_data = [l for l in lines if not re.match(r"^\|[\s\-|:]+\|$", l)]
            if first_data:
                cols = first_data[0].split("|")
                # If first line has only numbers/dashes, likely orphaned data row
                non_numeric = [c for c in cols if c.strip() and not re.match(r"^[\d\.\-]+$", c.strip())]
                if len(non_numeric) <= 1:
                    orphaned += 1
                    if orphaned <= 5:
                        r.fail(f"Orphaned table row (no header context): {c.get('id', '?')}")
    if orphaned > 5:
        r.fail(f"Total orphaned table rows: {orphaned}")

    # Check token count reasonable
    for c in chunks:
        tok = count_tokens(c.get("text", ""))
        if tok < TABLE_ROW_MIN_TOKENS and is_table_chunk(c):
            r.warn(f"Very short table chunk ({tok} tokens): {c.get('id', '?')}")
        elif tok > HARD_MAX_TOKENS:
            r.fail(f"Chunk exceeds hard cap {HARD_MAX_TOKENS} ({tok} tokens): {c.get('id', '?')}")
        elif tok > MAX_CHUNK_TOKENS:
            r.warn(f"Very large chunk ({tok} tokens): {c.get('id', '?')}")
        else:
            r.ok("Token count OK")

    return r


def test_d3_table_extraction(chunks: list[dict]) -> TestResult:
    """D3: Table extraction — wide-table bypass, program grouping, row explosion check."""
    r = TestResult("D3: Table Extraction Quality")

    table_chunks = [c for c in chunks if is_table_chunk(c)]

    # Count table chunks vs text chunks
    r.details.append(f"  Table chunks: {len(table_chunks)}, Text chunks: {len(chunks) - len(table_chunks)}")

    # Check for row explosion: each table row = 1 chunk is the KNOWN BUG
    single_row_tables = 0
    multi_row_tables = 0
    wide_tables_chunked = 0

    for c in table_chunks:
        text = c.get("text", "")
        rows = parse_table_rows(text)
        # Count actual data rows (exclude separator and header)
        data_rows_all = [row for row in rows if not re.match(r"^\|[\s\-|:]+\|$", row)]
        # Determine if first non-separator row is header or data
        # A header row contains column names (alphabetic, not numeric codes)
        # A data row typically starts with a number or mã ngành code
        actual_data = data_rows_all
        if len(data_rows_all) >= 2:
            first_cells = [c.strip() for c in data_rows_all[0].split("|") if c.strip()]
            # If first row starts with a number or mã ngành pattern, it's data not header
            first_cell = first_cells[0] if first_cells else ""
            if first_cell and re.match(r"^\d", first_cell):
                # First row is data, not header
                actual_data = data_rows_all
            else:
                # First row is likely header
                actual_data = data_rows_all[1:]

        if len(actual_data) <= 1:  # only 0-1 actual data rows
            chunk_type = c.get("metadata", {}).get("chunk_type", "")
            # Wide tables with 1 data row are CORRECT per Phase 3 (single-chunk bypass)
            # Only flag if chunk_type is NOT wide_table (meaning it was misclassified)
            if len(actual_data) >= 1:
                cols = count_table_cols(actual_data[0])
                if cols >= WIDE_TABLE_COLS and chunk_type != "wide_table":
                    wide_tables_chunked += 1
                    if wide_tables_chunked <= 3:
                        r.fail(f"Wide table ({cols} cols) misclassified as {chunk_type} (should be wide_table): {c.get('id', '?')}")
            single_row_tables += 1
        else:
            multi_row_tables += 1
            r.ok("Multi-row table chunk")

    r.details.append(f"  Single-row table chunks: {single_row_tables}")
    r.details.append(f"  Multi-row table chunks: {multi_row_tables}")

    if single_row_tables > 100:
        r.fail(f"Row explosion detected: {single_row_tables} single-row table chunks (Phase 3 target: wide tables as single unit)")

    # Check for header repetition in multi-chunk tables
    headers_seen = Counter()
    for c in table_chunks:
        text = c.get("text", "")
        lines = text.strip().split("\n")
        if lines:
            headers_seen[lines[0]] += 1

    repeated_headers = sum(1 for h, cnt in headers_seen.items() if cnt > 1)
    if repeated_headers > 0:
        r.ok(f"Header repetition detected in {repeated_headers} header types (good for context)")
    else:
        r.warn("No header repetition — table chunks may lack header context")

    return r


def test_d5_section_boundaries(chunks: list[dict]) -> TestResult:
    """D5: Section boundaries — no mid-sentence splits, legal structure preserved."""
    r = TestResult("D5: Section Boundaries")

    # Check for mid-sentence splits (text chunks ending without punctuation)
    bad_endings = 0
    for c in chunks:
        if is_table_chunk(c):
            continue
        text = c.get("text", "").strip()
        if not text:
            continue
        # Vietnamese text often ends mid-phrase naturally (list items, headings)
        # Only flag if text is long (>200 chars) AND ends with letter AND no punctuation in last 30 chars
        last_char = text[-1] if text else ""
        last_30 = text[-30:] if len(text) > 30 else text
        has_punct = any(p in last_30 for p in ".!?;:…–—")
        if last_char.isalpha() and len(text) > 200 and not has_punct:
            bad_endings += 1
            if bad_endings <= 3:
                r.warn(f"Chunk likely mid-sentence split (no punct in last 30 chars): {c.get('id', '?')}")

    if bad_endings > 100:
        r.fail(f"Many chunks ({bad_endings}) appear to be mid-sentence splits")

    # Check legal structure (Điều/Khoản) preserved in policy chunks
    policy_chunks = [c for c in chunks if c.get("metadata", {}).get("info_type") == "van_ban_phap_ly"]
    for c in policy_chunks[:10]:
        text = c.get("text", "")
        if re.search(r"^(Điều|Khoản|CHƯƠNG|PHẦN)\s+", text, re.MULTILINE):
            r.ok("Legal structure marker preserved")
        else:
            r.warn(f"Policy chunk lacks legal structure markers: {c.get('id', '?')}")

    return r


def test_d8_metadata_preservation(chunks: list[dict]) -> TestResult:
    """D8: Metadata — all required fields present and valid."""
    r = TestResult("D8: Metadata Preservation")

    for c in chunks[:100]:  # Sample first 100
        meta = c.get("metadata", {})

        # Check required fields
        missing = REQUIRED_META_FIELDS - set(meta.keys())
        if missing:
            r.fail(f"Missing metadata fields {missing}: {c.get('id', '?')}")
        else:
            r.ok("All required metadata fields present")

        # Check year validity
        year = meta.get("year", "")
        data_year = meta.get("data_year", "")
        if year and not re.match(r"^\d{4}$", str(year)):
            r.fail(f"Invalid year format: {year}")
        if data_year and not isinstance(data_year, int):
            r.warn(f"data_year not int: {data_year}")

        # Check source not empty
        source = meta.get("source", "")
        if not source:
            r.fail(f"Empty source: {c.get('id', '?')}")

        # Check chunk_type for table chunks
        if is_table_chunk(c):
            chunk_type = meta.get("chunk_type", "")
            if not chunk_type:
                r.warn(f"Table chunk missing chunk_type: {c.get('id', '?')}")

    return r


def test_overlap_integrity(chunks: list[dict]) -> TestResult:
    """Check overlap between consecutive chunks from same doc."""
    r = TestResult("D: Overlap Integrity")

    # Group by doc_id
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        doc_id = c.get("doc_id", "unknown")
        by_doc.setdefault(doc_id, []).append(c)

    checked = 0
    no_overlap = 0
    for doc_id, doc_chunks in by_doc.items():
        # Sort by chunk_id
        doc_chunks.sort(key=lambda c: c.get("chunk_id", 0))
        for i in range(len(doc_chunks) - 1):
            curr = doc_chunks[i].get("text", "")
            nxt = doc_chunks[i + 1].get("text", "")
            if not curr or not nxt:
                continue

            # Check if there's some content overlap (shared text fragments)
            # Simple check: last 50 chars of curr vs first 50 chars of nxt
            curr_tail = curr[-50:] if len(curr) > 50 else curr
            nxt_head = nxt[:50] if len(nxt) > 50 else nxt

            # For text chunks, expect some overlap
            if not is_table_chunk(doc_chunks[i]) and not is_table_chunk(doc_chunks[i + 1]):
                # Check if they share at least a few words
                curr_words = set(re.findall(r"\b\w+\b", curr_tail.lower()))
                nxt_words = set(re.findall(r"\b\w+\b", nxt_head.lower()))
                shared = curr_words & nxt_words
                if len(shared) >= 2:
                    r.ok("Overlap detected between consecutive chunks")
                else:
                    no_overlap += 1
                    if no_overlap <= 3:
                        r.warn(f"No overlap between consecutive chunks in {doc_id}")
                checked += 1

    r.details.append(f"  Checked {checked} consecutive chunk pairs, {no_overlap} with no overlap")
    if checked > 0 and no_overlap > checked * 0.5:
        r.warn(f"More than 50% of consecutive chunks lack overlap ({no_overlap}/{checked})")

    return r


def test_encoding_safety(chunks: list[dict]) -> TestResult:
    """Check for encoding issues, broken Vietnamese diacritics."""
    r = TestResult("D: Encoding Safety")

    broken_diacritics = 0
    for c in chunks[:200]:
        text = c.get("text", "")
        # Check for common encoding artifacts
        if re.search(r"[aouAOU]̀|́|̃|̉|̣", text):
            broken_diacritics += 1
            if broken_diacritics <= 2:
                r.fail(f"Broken Vietnamese diacritics (combining marks): {c.get('id', '?')}")
        # Check for replacement characters
        if "�" in text:
            r.fail(f"Unicode replacement character found: {c.get('id', '?')}")

    if broken_diacritics == 0:
        r.ok("No broken Vietnamese diacritics detected")

    return r


def test_v4_acceptance_criteria(chunks: list[dict]) -> TestResult:
    """Check against v4 Phase 3 acceptance criteria."""
    r = TestResult("v4 Acceptance Criteria")

    total = len(chunks)
    r.details.append(f"  Total chunks: {total}")

    # AC: Total chunk count 250-400 (from Phase 3) OR 1200+ (user request)
    # User explicitly wants 1200+ chunks
    if total >= 1200:
        r.ok(f"Chunk count meets user target: {total} >= 1200")
    else:
        r.warn(f"Chunk count {total} below user target of 1200+")

    # AC: Fee table should be coherent (check for fee-related table chunks)
    fee_chunks = [c for c in chunks if is_table_chunk(c) and ("hoc_phi" in c.get("text", "").lower() or "học phí" in c.get("text", "").lower() or "tín chỉ" in c.get("text", "").lower())]
    r.details.append(f"  Fee-related table chunks: {len(fee_chunks)}")

    # AC: Zero chunks with fill<0.2 AND row_occupancy<3 (FIX-013)
    sparse_junk = 0
    for c in chunks:
        if is_table_chunk(c):
            text = c.get("text", "")
            rows = parse_table_rows(text)
            data_rows = [row for row in rows if not re.match(r"^\|[\s\-|:]+\|$", row)]
            if len(data_rows) < 3:
                # Check fill ratio
                total_cells = sum(count_table_cols(row) for row in data_rows)
                filled_cells = sum(1 for row in data_rows for cell in row.split("|") if cell.strip())
                if total_cells > 0 and filled_cells / total_cells < 0.2:
                    sparse_junk += 1
    if sparse_junk > 0:
        r.warn(f"Found {sparse_junk} sparse table chunks (FIX-013 guard)")
    else:
        r.ok("No sparse junk table chunks (FIX-013 maintained)")

    # AC: Each chunk has metadata with source, year, document_type, section, school
    for c in chunks[:50]:
        meta = c.get("metadata", {})
        if not meta.get("source"):
            r.fail(f"Chunk missing source: {c.get('id', '?')}")
        if not meta.get("year"):
            r.warn(f"Chunk missing year: {c.get('id', '?')}")

    # AC: data_year field should exist
    has_data_year = sum(1 for c in chunks if "data_year" in c.get("metadata", {}))
    r.details.append(f"  Chunks with data_year: {has_data_year}/{total}")

    return r


def test_row_explosion_deep(chunks: list[dict]) -> TestResult:
    """Deep test for the specific row-explosion bug identified in v4 Phase 2/3.

    The v4 plans explicitly state:
    - _split_table_aware() treating each row as atomic = WRONG
    - Fee table (p54): 17 rows → should NOT be 17 chunks
    - Wide tables (≥8 cols, ≥3 rows) → should be single chunk
    - Program-grouped tables → 1 chunk per program family
    """
    r = TestResult("v4 Row Explosion Deep Check")

    table_chunks = [c for c in chunks if is_table_chunk(c)]

    # Analyze each table chunk's dimensions
    for c in table_chunks[:30]:  # Sample
        text = c.get("text", "")
        rows = parse_table_rows(text)
        data_rows = [row for row in rows if not re.match(r"^\|[\s\-|:]+\|$", row)]
        header_rows = [row for row in rows[:3] if not re.match(r"^\|[\s\-|:]+\|$", row)]

        if not data_rows:
            continue

        max_cols = max(count_table_cols(row) for row in data_rows) if data_rows else 0

        # Phase 3 Rule: Wide table (≥8 cols, ≥3 data rows) → should be 1 chunk
        if max_cols >= WIDE_TABLE_COLS and len(data_rows) >= WIDE_TABLE_MIN_ROWS:
            # This SHOULD be a single chunk — if we find it split, that's the bug
            r.fail(
                f"WIDE TABLE SPLIT (Phase 3 violation): {max_cols} cols, {len(data_rows)} data rows "
                f"→ should be single chunk, but chunked separately: {c.get('id', '?')}"
            )
        elif len(data_rows) == 1 and max_cols >= 6:
            # Single-row table chunk with many columns = row explosion artifact
            r.fail(
                f"ROW EXPLOSION: Single row ({max_cols} cols) chunked alone "
                f"(Phase 3: should group with header + related rows): {c.get('id', '?')}"
            )
        else:
            r.ok("Table chunking reasonable")

    # Count how many table chunks are single-data-row
    single_row_count = 0
    wide_single_row = 0
    for c in table_chunks:
        text = c.get("text", "")
        rows = parse_table_rows(text)
        data_rows = [row for row in rows if not re.match(r"^\|[\s\-|:]+\|$", row)]
        if len(data_rows) == 1:
            single_row_count += 1
            cols = count_table_cols(data_rows[0])
            if cols >= 6:
                wide_single_row += 1

    r.details.append(f"  Single-data-row table chunks: {single_row_count}/{len(table_chunks)}")
    r.details.append(f"  Wide (≥6 cols) single-row chunks: {wide_single_row}")

    if wide_single_row > 10:
        r.fail(
            f"SEVERE ROW EXPLOSION: {wide_single_row} wide table rows chunked individually. "
            f"v4 Phase 3 requires wide-table bypass (≥8 cols → single chunk) and program grouping."
        )

    return r


def main():
    print("=" * 70)
    print("  ULTRAQA — CHUNK QUALITY DEEP TEST")
    print("  Based on v4 plans: Phase 2 (content audit) + Phase 3 (chunking redesign)")
    print("=" * 70)

    chunks = load_all_chunks()
    print(f"\n  Loaded {len(chunks)} chunks from {CHUNKED_DIR}")

    if not chunks:
        print("  ERROR: No chunks found!")
        return

    # Run all tests
    tests = [
        test_d1_semantic_completeness(chunks),
        test_d3_table_extraction(chunks),
        test_d5_section_boundaries(chunks),
        test_d8_metadata_preservation(chunks),
        test_overlap_integrity(chunks),
        test_encoding_safety(chunks),
        test_v4_acceptance_criteria(chunks),
        test_row_explosion_deep(chunks),
    ]

    print()
    total_pass = 0
    total_fail = 0
    total_warn = 0

    for t in tests:
        print(t.summary())
        total_pass += t.passed
        total_fail += t.failed
        total_warn += t.warnings

    print()
    print("=" * 70)
    print(f"  ULTRAQA SUMMARY: {total_pass} passed, {total_fail} failed, {total_warn} warnings")
    if total_fail == 0:
        print("  VERDICT: PASS — Chunks ready for LanceDB ingest + NER graph build")
    else:
        print(f"  VERDICT: FAIL — {total_fail} issues must be fixed before ingest")
    print("=" * 70)


if __name__ == "__main__":
    main()
