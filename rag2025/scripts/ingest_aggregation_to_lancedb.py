"""Ingest aggregation chunks (husc_2026_aggregation.jsonl) into LanceDB."""
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

load_dotenv()
CHUNKS_FILE = ROOT / "data" / "chunked" / "husc_2026_aggregation.jsonl"


def main():
    settings = RAGSettings()
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        return 1
    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} aggregation chunks")
    embedding_service = EmbeddingService(settings)
    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.encode_documents(texts)
    records = []
    for chunk, vector in zip(chunks, embeddings):
        meta = chunk.get("metadata", {})
        records.append({
            "id": chunk["chunk_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk.get("source", "tuyensinh.husc.edu.vn"),
            "faq_type": chunk.get("faq_type", ""),
            "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
            "vector": vector.tolist(),
        })
    db = lancedb.connect(settings.LANCEDB_URI)
    table_list = db.list_tables()
    # list_tables() returns ListTablesResponse object with .tables attribute
    actual_tables = getattr(table_list, 'tables', []) or []
    if not actual_tables:
        # fallback: try iterating as legacy format
        try:
            for item in table_list:
                if isinstance(item, tuple) and item[0] == 'tables':
                    actual_tables = item[1]
                    break
        except Exception:
            pass
    if settings.LANCEDB_TABLE not in actual_tables:
        logger.warning(f"Table {settings.LANCEDB_TABLE} not in {actual_tables}, trying open_table directly")
    table = db.open_table(settings.LANCEDB_TABLE)
    df = table.to_pandas()
    existing_ids = set(str(v) for v in df["id"].tolist())
    new_ids = [r["id"] for r in records]
    overlapping = [i for i in new_ids if i in existing_ids]
    if overlapping:
        logger.info(f"Deleting {len(overlapping)} existing rows that will be replaced")
        ids_str = ", ".join(f"'{i}'" for i in overlapping)
        table.delete(f"id IN ({ids_str})")
    table.add(records)
    new_count = table.count_rows()
    logger.info(f"Ingested {len(records)} chunks. Total rows now: {new_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
