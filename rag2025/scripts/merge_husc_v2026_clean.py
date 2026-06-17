"""Incrementally embed and merge new chunks into husc_v2026_clean LanceDB table.

Avoids full re-embed when only new chunks are added (fast: 13 thongbao + 7 v3 + 5 baiviet
= ~25 new chunks vs 410 existing).

Pipeline:
  1. Load all chunks from SOURCES (same logic as build_husc_v2026_clean.py)
  2. Open existing husc_v2026_clean table — get existing chunk_ids
  3. Compute diff: new_chunks = those not in table
  4. Embed only new_chunks
  5. Add embeddings as new rows (preserves existing 410+ rows)
  6. Verify count + sample

Falls back to full rebuild if table doesn't exist.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Force log to file so the harness can monitor progress
LOG_PATH = ROOT / "logs" / "merge_husc_v2026_clean.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
import lancedb
from dotenv import load_dotenv
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(str(LOG_PATH), level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService
from scripts.build_husc_v2026_clean import (
    SOURCES, QUARANTINE_PATTERNS, TARGET_TABLE,
    load_chunks, build_record,
)

load_dotenv(ROOT / ".env")


def main() -> int:
    settings = RAGSettings()
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL} dim={settings.EMBEDDING_DIM}")
    logger.info(f"LanceDB URI: {settings.LANCEDB_URI}")
    logger.info(f"Target table: {TARGET_TABLE}")

    # Load all chunks (with the same year + quarantine filters)
    all_chunks: list[dict] = []
    for src in SOURCES:
        if not src.exists():
            logger.warning(f"Source file not found, skipping: {src.name}")
            continue
        kept, skipped = load_chunks(src)
        logger.info(f"  {src.name}: kept={len(kept)} skipped={skipped}")
        all_chunks.extend(kept)
    logger.info(f"Total candidate chunks: {len(all_chunks)}")

    db_uri = str(ROOT / "data" / "lancedb")
    db = lancedb.connect(db_uri)
    existing_tables = list(db.table_names())

    if TARGET_TABLE not in existing_tables:
        logger.warning(f"Table '{TARGET_TABLE}' missing — falling back to full build.")
        from scripts.build_husc_v2026_clean import main as full_build
        return full_build()

    table = db.open_table(TARGET_TABLE)
    existing_ids: set[str] = set()
    try:
        df = table.to_pandas()
        existing_ids = set(df["id"].astype(str).tolist())
        logger.info(f"Existing rows: {len(existing_ids)}")
    except Exception as e:
        logger.error(f"Failed to read existing table: {e}")
        return 1

    new_chunks = []
    for c in all_chunks:
        cid = str(c.get("chunk_id") or c.get("id") or "")
        if not cid:
            continue
        if cid not in existing_ids:
            new_chunks.append(c)

    logger.info(f"NEW chunks to embed: {len(new_chunks)}")
    if not new_chunks:
        logger.info("Nothing new to add. Table is up to date.")
        # Print summary for harness
        print(f"\n=== INCREMENTAL MERGE ===")
        print(f"Table: {TARGET_TABLE}")
        print(f"Existing rows: {len(existing_ids)}")
        print(f"New chunks added: 0")
        print(f"Total rows after merge: {len(existing_ids)}")
        return 0

    settings.EMBEDDING_BATCH_SIZE = 16  # smaller batch for stability
    embedding_service = EmbeddingService(settings)
    texts = [c["text"] for c in new_chunks]
    logger.info(f"Encoding {len(texts)} new docs (batch_size={settings.EMBEDDING_BATCH_SIZE})…")
    embeddings = embedding_service.encode_documents(texts)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    records = [
        build_record(chunk, embeddings[i].tolist())
        for i, chunk in enumerate(new_chunks)
    ]

    # Append to existing table
    table.add(records)

    new_count = table.count_rows()
    logger.info(f"Table '{TARGET_TABLE}' now has {new_count} rows.")

    # Sample 5 new rows
    df_after = table.to_pandas()
    new_added = df_after[df_after["id"].isin([r["id"] for r in records])]
    print(f"\n=== INCREMENTAL MERGE ===")
    print(f"Table: {TARGET_TABLE}")
    print(f"Existing rows before: {len(existing_ids)}")
    print(f"New chunks added: {len(records)}")
    print(f"Total rows after: {new_count}")
    print(f"\nNew chunk IDs added:")
    for cid in new_added["id"].head(20).tolist():
        print(f"  - {cid}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
