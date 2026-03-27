"""
Ingest all chunked JSONL files into LanceDB using Qwen3-Embedding.

Model is configured via settings.QWEN_EMBEDDING_MODEL (default: Qwen3-Embedding-4B).
Uses EmbeddingService for consistent Qwen3 instruction-aware encoding.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import lancedb
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

load_dotenv()


def _load_chunks(chunks_dir: Path) -> List[Dict]:
    all_chunks: List[Dict] = []
    jsonl_files = sorted(chunks_dir.glob("chunked_*.jsonl"))
    jsonl_files = [f for f in jsonl_files if "enhanced" not in f.name]

    for file in jsonl_files:
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(obj, dict):
                    continue

                text = obj.get("text") or ""
                if not text:
                    continue

                chunk_id = str(obj.get("id", obj.get("chunk_id", f"chunk_{len(all_chunks)}")))
                all_chunks.append(
                    {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "text": text,
                        "source": obj.get("metadata", {}).get("source", file.name),
                        "faq_type": obj.get("faq_type", obj.get("metadata", {}).get("faq_type", "")),
                        "metadata": obj.get("metadata", {}),
                    }
                )

    return [c for c in all_chunks if c.get("text")]


def ingest_lancedb() -> bool:
    settings = RAGSettings()
    chunks_dir = Path(__file__).parent.parent / "data" / "chunked"

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return False

    logger.info("Loading chunks from JSONL files...")
    chunks = _load_chunks(chunks_dir)
    if not chunks:
        logger.error("No chunks found to ingest")
        return False

    logger.info(f"Loaded {len(chunks)} chunks")
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})")

    # Use EmbeddingService for consistent Qwen3 instruction-aware encoding
    # Documents are encoded WITHOUT instruction prefix (Qwen3 recommendation)
    embedding_service = EmbeddingService(settings)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.encode_documents(texts)

    records = []
    for chunk, vector in zip(chunks, embeddings):
        raw_meta = chunk.get("metadata", {})
        records.append(
            {
                "id": chunk["id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "faq_type": chunk.get("faq_type", ""),
                "metadata_json": json.dumps(raw_meta, ensure_ascii=False, default=str),
                "vector": vector.tolist(),
            }
        )

    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.create_table(settings.LANCEDB_TABLE, data=records, mode="overwrite")
    logger.info(
        f"Ingest complete: table={settings.LANCEDB_TABLE}, rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
    )
    return True


if __name__ == "__main__":
    ok = ingest_lancedb()
    raise SystemExit(0 if ok else 1)
