"""
Chunk all raw markdown from staging/ into JSONL files for LanceDB + NER graph.

Pipeline:
1. Read all .md files from data/staging/html/ and data/staging/pdf/
2. Auto-detect metadata (source, year, info_type) from content
3. Chunk using system Chunker with profile support
4. Save to data/chunked/chunked_N.jsonl
5. Re-ingest into LanceDB
6. Rebuild NER graph

Usage:
    cd rag2025
    PYTHONPATH=src:. python scripts/chunk_all_staging.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunker import Chunker, ChunkConfig
from config.settings import RAGSettings
from services.temporal_authority import get_current_admission_year

STAGING_HTML = Path(__file__).parent.parent / "data" / "staging" / "html"
STAGING_PDF = Path(__file__).parent.parent / "data" / "staging" / "pdf"
CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"
CHUNK_PROFILES = Path(__file__).parent.parent / "config" / "chunk_profiles.yaml"

# Track next file number
def _next_chunk_index() -> int:
    existing = sorted(CHUNKED_DIR.glob("chunked_*.jsonl"))
    existing = [f for f in existing if "enhanced" not in f.name]
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.search(r"chunked_(\d+)", f.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def detect_year(text: str) -> str:
    """Detect admission year from content."""
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
    """Auto-detect info_type from content."""
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
    """Detect if file came from web crawl or PDF."""
    if "thongbao_" in filepath.name:
        return "web_crawl"
    if "pdf_" in filepath.name:
        return "pdf_extract"
    return "unknown"


def extract_title_from_husc_web(text: str) -> str:
    """Extract title from HUSC thongbao content."""
    lines = text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip().lstrip("#").strip()
        if line and len(line) > 5:
            return line[:100]
    return "HUSC Thông báo tuyển sinh"


def process_file(filepath: Path, chunker: Chunker, file_idx: int) -> list[dict]:
    """Process a single markdown file into chunks."""
    logger.info(f"Processing: {filepath.name}")

    text = filepath.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning(f"Empty file: {filepath}")
        return []

    year = detect_year(text)
    source_type = detect_source_type(filepath)

    if source_type == "web_crawl":
        m = re.search(r"thongbao_(\d+)", filepath.name)
        source = f"tuyensinh.husc.edu.vn/thongbao.php?id={m.group(1)}" if m else filepath.stem
        info_type = detect_info_type(text, source)
    else:
        # PDF source - extract from first heading or filename
        source = filepath.stem
        info_type = detect_info_type(text, source)

    doc = {
        "id": f"{source_type}_{filepath.stem}_{year}",
        "doc_id": f"chunked_{file_idx}",
        "text": text,
        "text_plain": text,
        "metadata": {
            "source": source,
            "source_type": source_type,
            "year": year,
            "data_year": int(year),
            "info_type": info_type,
            "school": "HUSC",
            "audience": "thi_sinh",
            "unit": "HUSC",
        },
    }

    # Auto-detect profile and chunk
    profile_name = "auto"
    if info_type == "van_ban_phap_ly":
        profile_name = "policy"
    elif "faq" in info_type or "hỏi đáp" in text.lower():
        profile_name = "faq"

    chunks = chunker.chunk_document(doc, profile_name=profile_name)
    logger.info(f"  → {filepath.name}: {len(chunks)} chunks (year={year}, type={info_type}, profile={profile_name})")
    return [c.model_dump() for c in chunks]


def main():
    logger.info("=== CHUNK ALL STAGING DATA ===")

    # Collect all staging files
    staging_files = []
    if STAGING_HTML.exists():
        staging_files.extend(sorted(STAGING_HTML.glob("*.md")))
    if STAGING_PDF.exists():
        staging_files.extend(sorted(STAGING_PDF.glob("*.md")))

    if not staging_files:
        logger.error("No staging files found! Run crawl and PDF conversion first.")
        return

    logger.info(f"Found {len(staging_files)} staging files")

    # Preview content for chunking decisions
    logger.info("=== STAGING CONTENT PREVIEW ===")
    for f in staging_files:
        preview = f.read_text(encoding="utf-8")[:300]
        year = detect_year(f.read_text(encoding="utf-8"))
        logger.info(f"  {f.name} ({year}): {len(f.read_text(encoding='utf-8'))} chars | {preview[:100]}...")

    # Initialize chunker
    config = ChunkConfig(CHUNK_PROFILES)
    chunker = Chunker(config)

    # Process each file
    file_idx = _next_chunk_index()
    all_chunks = []

    for filepath in staging_files:
        chunks = process_file(filepath, chunker, file_idx)
        if chunks:
            # Save individual JSONL
            out_path = CHUNKED_DIR / f"chunked_{file_idx}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            logger.info(f"  → Saved {len(chunks)} chunks to {out_path.name}")
            all_chunks.extend(chunks)
            file_idx += 1

    logger.info(f"=== CHUNKING COMPLETE ===")
    logger.info(f"Total: {len(all_chunks)} chunks from {len(staging_files)} files")
    logger.info(f"Files saved: chunked_{_next_chunk_index() - len(staging_files)} to chunked_{file_idx - 1}")

    # Summary by info_type
    by_type = {}
    for c in all_chunks:
        t = c.get("metadata", {}).get("info_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    logger.info("By info_type:")
    for t, cnt in sorted(by_type.items()):
        logger.info(f"  {t}: {cnt}")

    # Print next steps
    print("\n" + "=" * 60)
    print("  CHUNKING COMPLETE - Next steps:")
    print("=" * 60)
    print(f"  1. Re-ingest LanceDB:")
    print(f"     python scripts/ingest_lancedb.py")
    print(f"  2. Build NER graph:")
    print(f"     python scripts/build_graph.py --incremental")
    print(f"  3. Run 86Q eval:")
    print(f"     python scripts/eval_86q.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
