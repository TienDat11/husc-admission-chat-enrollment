"""Build husc_v2026_clean LanceDB table — Option C non-destructive flip.

Sources (in order):
  1. data/chunked/husc_2026_canonical_v2.jsonl   — 3 canonical Q&A chunks
  2. data/chunked/nganh_detail_2026.jsonl         — 28 ngành detail chunks
  3. data/chunked/chunked_22.jsonl                — web_crawl 2026 (TB73)
  4. data/chunked/chunked_25.jsonl                — web_crawl 2026

Inclusion rules:
  - metadata.data_year == 2026
  - text does NOT contain any quarantine pattern

Quarantine patterns (skip if any match):
  - bản giấy
  - Quản trị và phân tích dữ liệu
  - 7340405
  - 7480107TD
  - điểm trung bình lớp 12

Idempotent: drops + recreates husc_v2026_clean if it already exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import lancedb
from dotenv import load_dotenv
from loguru import logger

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

load_dotenv(ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_TABLE = "husc_v2026_clean"
EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B

SOURCES = [
    ROOT / "data" / "chunked" / "husc_2026_canonical_v2.jsonl",
    ROOT / "data" / "chunked" / "husc_2026_canonical_v3.jsonl",
    ROOT / "data" / "chunked" / "nganh_detail_2026.jsonl",
    ROOT / "data" / "chunked" / "thongbao_2026_v2.jsonl",
    ROOT / "data" / "chunked" / "baiviet_2026.jsonl",
    ROOT / "data" / "chunked" / "husc_2026_vision_ocr.jsonl",
    ROOT / "data" / "chunked" / "chunked_22.jsonl",
    ROOT / "data" / "chunked" / "chunked_25.jsonl",
]

QUARANTINE_PATTERNS = [
    "bản giấy",
    "Quản trị và phân tích dữ liệu",
    "7340405",
    "7480107TD",
    "điểm trung bình lớp 12",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_quarantined(text: str) -> bool:
    for pat in QUARANTINE_PATTERNS:
        if pat in text:
            return True
    return False


def load_chunks(path: Path) -> tuple[list[dict], int]:
    """Load JSONL, apply year filter (2026 + 2025 reference) + quarantine filter.

    Returns (kept_chunks, skipped_count).
    """
    # thongbao_2026_v2.jsonl carries both 2026 official + 2025 reference
    # (id=40,49,50,53 are 2025 references intentionally kept for fallback questions
    #  about điểm sàn 2025, chỉ tiêu 2025, nộp minh chứng 2025, ĐK KTX).
    # 2025 reference docs legitimately mention 7340405 (Quản trị PT dữ liệu) — that
    # ngành DID exist in 2025 — so we bypass the quarantine pattern filter for them.
    # baiviet_2026.jsonl is intro/index/thongbao_list snapshots — also allow 2025 refs
    # because those landing pages enumerate both years.
    # husc_2026_vision_ocr.jsonl is OCR'd from thông báo images, some of which are 2025
    # reference (điểm chuẩn 2024/2025, công thức cũ); allow those too.
    allow_2025_ref = path.name in (
        "thongbao_2026_v2.jsonl",
        "baiviet_2026.jsonl",
        "husc_2026_vision_ocr.jsonl",
    )

    kept: list[dict] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            year = meta.get("data_year")
            text = obj.get("text", "")

            # Year filter — only 2026 unless this file is whitelisted for 2025 ref
            is_2025_ref = bool(allow_2025_ref and year is not None and int(year) == 2025)
            if year is not None and int(year) != 2026 and not is_2025_ref:
                skipped += 1
                logger.debug(f"SKIP year={year} id={obj.get('chunk_id','?')}")
                continue

            # 2025 reference docs legitimately contain 7340405 / "Quản trị và phân tích
            # dữ liệu" — that ngành existed in 2025. Skip the quarantine pattern check
            # for those reference docs only.
            if not is_2025_ref and _is_quarantined(text):
                skipped += 1
                logger.warning(f"SKIP quarantine id={obj.get('chunk_id','?')} text[:80]={text[:80]!r}")
                continue

            kept.append(obj)

    return kept, skipped


def build_record(chunk: dict, vector: list[float]) -> dict:
    """Convert a raw chunk dict + embedding vector into a LanceDB record."""
    meta = chunk.get("metadata", {})
    chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
    return {
        "id": str(chunk_id),
        "text": chunk.get("text", ""),
        "summary": chunk.get("summary", None),
        "embedding": vector,
        "metadata": meta,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    settings = RAGSettings()
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL} dim={settings.EMBEDDING_DIM}")
    logger.info(f"LanceDB URI: {settings.LANCEDB_URI}")
    logger.info(f"Target table: {TARGET_TABLE}")

    # ── Step 1: Load + filter all chunks ─────────────────────────────────────
    all_chunks: list[dict] = []
    total_skipped = 0
    source_counts: dict[str, int] = {}

    for src in SOURCES:
        if not src.exists():
            logger.error(f"Source file not found: {src}")
            return 1
        kept, skipped = load_chunks(src)
        source_counts[src.name] = len(kept)
        total_skipped += skipped
        all_chunks.extend(kept)
        logger.info(f"  {src.name}: kept={len(kept)} skipped={skipped}")

    logger.info(f"Total chunks to ingest: {len(all_chunks)} | total skipped: {total_skipped}")

    if not all_chunks:
        logger.error("No chunks to ingest — aborting.")
        return 1

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    logger.info("Loading embedding model (first call may take 10-30s)…")
    # Override batch size to 32 for faster CPU encoding (benchmark: ~13s for 410 docs)
    settings.EMBEDDING_BATCH_SIZE = 32
    embedding_service = EmbeddingService(settings)
    texts = [c["text"] for c in all_chunks]
    logger.info(f"Encoding {len(texts)} documents (batch_size=32)…")
    embeddings = embedding_service.encode_documents(texts)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # ── Step 3: Build records ─────────────────────────────────────────────────
    records = [
        build_record(chunk, embeddings[i].tolist())
        for i, chunk in enumerate(all_chunks)
    ]

    # ── Step 4: Create / recreate table ──────────────────────────────────────
    db_uri = str(ROOT / "data" / "lancedb")
    db = lancedb.connect(db_uri)
    existing = list(db.table_names())

    if TARGET_TABLE in existing:
        logger.warning(f"Table '{TARGET_TABLE}' already exists — dropping for idempotent rebuild.")
        db.drop_table(TARGET_TABLE)

    logger.info(f"Creating table '{TARGET_TABLE}'…")
    table = db.create_table(TARGET_TABLE, data=records)
    row_count = table.count_rows()
    logger.info(f"Table '{TARGET_TABLE}' created with {row_count} rows.")

    # ── Step 5: Verify ────────────────────────────────────────────────────────
    logger.info("=== VERIFICATION ===")
    logger.info(f"Row count: {row_count}")
    logger.info("Source breakdown:")
    for name, cnt in source_counts.items():
        logger.info(f"  {name}: {cnt}")
    logger.info(f"Total skipped (quarantine/year filter): {total_skipped}")

    # Sample 5 rows
    df = table.to_pandas()
    logger.info("Top 5 sample chunks:")
    for _, row in df.head(5).iterrows():
        text_preview = str(row["text"])[:100]
        logger.info(f"  id={row['id']} | {text_preview!r}")

    # Print summary to stdout for easy capture
    print("\n=== INGEST SUMMARY ===")
    print(f"Table: {TARGET_TABLE}")
    print(f"Total rows ingested: {row_count}")
    print(f"Total skipped: {total_skipped}")
    print("\nSource breakdown:")
    for name, cnt in source_counts.items():
        print(f"  {name}: {cnt}")
    print("\nTop 5 sample chunks:")
    for _, row in df.head(5).iterrows():
        text_preview = str(row["text"])[:100]
        print(f"  [{row['id']}] {text_preview}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
