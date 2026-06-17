"""Ingest operational Q-A chunks (24 total, including 6 new công thức điểm) into LanceDB.

Self-contained: sets cwd to rag2025 root so LANCEDB_URI='./data/lancedb' resolves correctly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import lancedb  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402

load_dotenv()
CHUNKS_FILE = ROOT / "data" / "chunked" / "husc_2026_operational_qa.jsonl"


def main() -> int:
    settings = RAGSettings()
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        return 1
    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} operational Q-A chunks")
    embedding_service = EmbeddingService(settings)
    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.encode_documents(texts)
    records = []
    for chunk, vector in zip(chunks, embeddings):
        meta = chunk.get("metadata", {})
        records.append(
            {
                "id": chunk["chunk_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk.get("source", "tuyensinh.husc.edu.vn"),
                "faq_type": chunk.get("faq_type", "operational_qa"),
                "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
                "vector": vector.tolist(),
            }
        )
    db = lancedb.connect(settings.LANCEDB_URI)
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
