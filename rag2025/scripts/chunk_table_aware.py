"""
Table-aware chunker: splits markdown tables row-by-row for granular RAG retrieval.
Each table row becomes its own chunk with column headers as context.
Non-table text uses smaller chunk sizes (200 tokens) for higher granularity.

Target: 1200+ chunks from all staging data.

Usage:
    cd rag2025
    $env:PYTHONPATH="src;."; python scripts/chunk_table_aware.py
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

from chunker import Chunker, ChunkConfig, RAGChunk
from config.settings import RAGSettings
from services.temporal_authority import get_current_admission_year

STAGING_HTML = Path(__file__).parent.parent / "data" / "staging" / "html"
STAGING_PDF = Path(__file__).parent.parent / "data" / "staging" / "pdf"
CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"
CHUNK_PROFILES = Path(__file__).parent.parent / "config" / "chunk_profiles.yaml"

ENCODING = "cl100k_base"
TABLE_CHUNK_SIZE = 180
TEXT_CHUNK_SIZE = 200
OVERLAP = 40
MIN_TOKENS = 30


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


def extract_table_headers(table_lines: list[str]) -> str:
    """Extract header row and separator from markdown table."""
    for line in table_lines[:3]:
        if "|" in line and re.match(r"^\|[\s\-|:]+\|$", line.strip()):
            continue
        if "|" in line:
            return line.strip()
    return ""


def chunk_table_rows(table_lines: list[str], headers: str, base_meta: dict, doc_id: str, enc, start_idx: int) -> list[dict]:
    """Each table data row becomes its own chunk, with header context."""
    chunks = []
    for i, line in enumerate(table_lines):
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        # Skip separator line
        if re.match(r"^\|[\s\-|:]+\|$", stripped):
            continue
        # Skip header line (already captured)
        if headers and stripped == headers:
            continue

        # Build chunk text with header context
        chunk_text = f"{headers}\n{stripped}"
        tok_count = count_tokens(chunk_text, enc)

        # If single row is too big, split it anyway
        if tok_count >= MIN_TOKENS:
            chunk = {
                "id": f"{doc_id}_tchunk_{start_idx + i}",
                "doc_id": doc_id,
                "chunk_id": start_idx + i,
                "text": chunk_text,
                "text_plain": chunk_text,
                "metadata": {
                    **base_meta,
                    "chunk_type": "table_row",
                    "row_index": i,
                },
                "breadcrumbs": [base_meta.get("source", "")],
                "prev_chunk_id": "",
                "next_chunk_id": "",
                "sparse_terms": extract_sparse_terms(chunk_text),
            }
            chunks.append(chunk)

    return chunks


def chunk_text_sections(text: str, base_meta: dict, doc_id: str, enc, start_idx: int) -> list[dict]:
    """Split non-table text into smaller chunks with overlap."""
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Further split long paragraphs by single newlines or sentences
    parts = []
    for p in paragraphs:
        if count_tokens(p, enc) > TEXT_CHUNK_SIZE:
            # Split by lines
            lines = p.split("\n")
            sub_parts = []
            current = []
            current_tok = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                lt = count_tokens(line, enc)
                if current_tok + lt > TEXT_CHUNK_SIZE and current:
                    parts.append("\n".join(current))
                    current = [line]
                    current_tok = lt
                else:
                    current.append(line)
                    current_tok += lt
            if current:
                parts.append("\n".join(current))
        else:
            parts.append(p)

    chunks = []
    for i, part in enumerate(parts):
        tok = count_tokens(part, enc)
        if tok < MIN_TOKENS:
            continue

        chunk = {
            "id": f"{doc_id}_xchunk_{start_idx + i}",
            "doc_id": doc_id,
            "chunk_id": start_idx + i,
            "text": part,
            "text_plain": part,
            "metadata": {
                **base_meta,
                "chunk_type": "text_section",
                "token_count": tok,
            },
            "breadcrumbs": [base_meta.get("source", "")],
            "prev_chunk_id": "",
            "next_chunk_id": "",
            "sparse_terms": extract_sparse_terms(part),
        }
        chunks.append(chunk)

    return chunks


STOPWORDS = {
    "và", "các", "của", "có", "được", "cho", "trong", "là", "một", "này",
    "để", "với", "theo", "từ", "đã", "sẽ", "không", "khi", "bằng",
}


def extract_sparse_terms(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2][:50]


def process_file(filepath: Path, enc, file_idx: int) -> list[dict]:
    """Process a single file with table-aware chunking."""
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

    # Separate table blocks from text blocks
    lines = text.split("\n")
    table_block: list[str] = []
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
        nonlocal chunk_idx, table_block
        if table_block:
            headers = extract_table_headers(table_block)
            chunks = chunk_table_rows(table_block, headers, base_meta, doc_id, enc, chunk_idx)
            all_chunks.extend(chunks)
            chunk_idx += len(chunks)
            table_block = []

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            if not in_table:
                flush_text()
                in_table = True
            table_block.append(stripped)
        else:
            if in_table:
                flush_table()
                in_table = False
            text_buffer.append(line)

    flush_text()
    flush_table()

    logger.info(f"  → {filepath.name}: {len(all_chunks)} chunks (year={year}, type={info_type})")
    return all_chunks


def main():
    logger.info("=== TABLE-AWARE CHUNKING ===")

    # Collect staging files
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

    # Delete old chunked files from previous runs (12+)
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

    by_type = {}
    for c in all_chunks:
        t = c.get("metadata", {}).get("info_type", "unknown")
        ct = c.get("metadata", {}).get("chunk_type", "unknown")
        key = f"{t}/{ct}"
        by_type[key] = by_type.get(key, 0) + 1
    logger.info("By info_type/chunk_type:")
    for k, v in sorted(by_type.items()):
        logger.info(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"  TABLE-AWARE CHUNKING COMPLETE")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
