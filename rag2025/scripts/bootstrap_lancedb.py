"""
Bootstrap LanceDB table if missing.

Priority:
1) If LanceDB table exists -> done.
2) If legacy index/vector_store.npz exists -> migrate vectors to LanceDB.
3) Else -> instruct user to run scripts/ingest_lancedb.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import lancedb
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings


def _to_record(doc_id: str, metadata: dict, vector: np.ndarray) -> dict:
    chunk_id = str(
        metadata.get("chunk_id")
        or metadata.get("id")
        or doc_id
    )
    text = metadata.get("text") or metadata.get("text_plain") or metadata.get("text_raw") or ""
    source = metadata.get("source") or metadata.get("metadata", {}).get("source", "legacy_npz")
    faq_type = metadata.get("faq_type", "")
    raw_meta = metadata.get("metadata", metadata if isinstance(metadata, dict) else {})

    return {
        "id": str(doc_id),
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "faq_type": faq_type,
        "metadata_json": json.dumps(raw_meta, ensure_ascii=False, default=str),
        "vector": vector.tolist(),
    }


def bootstrap_lancedb() -> int:
    settings = RAGSettings()

    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE in db.list_tables():
        logger.info(f"LanceDB table exists: {settings.LANCEDB_TABLE}")
        return 0

    legacy_npz = Path(settings.INDEX_DIR) / "vector_store.npz"
    if not legacy_npz.exists():
        logger.error(
            f"Missing LanceDB table '{settings.LANCEDB_TABLE}' and no legacy npz found at {legacy_npz}. "
            "Run: python scripts/ingest_lancedb.py"
        )
        return 1

    logger.warning(f"LanceDB table missing. Migrating from legacy npz: {legacy_npz}")

    data = np.load(legacy_npz, allow_pickle=True)
    vectors = data["vectors"]
    ids = data["ids"].tolist()
    metadatas = data["metadatas"].tolist()

    if len(ids) != len(vectors) or len(metadatas) != len(vectors):
        logger.error("Legacy npz corrupted: ids/metadatas/vectors length mismatch")
        return 1

    records = []
    for doc_id, metadata, vector in zip(ids, metadatas, vectors):
        md = metadata if isinstance(metadata, dict) else {}
        records.append(_to_record(str(doc_id), md, vector))

    db.create_table(settings.LANCEDB_TABLE, data=records)
    logger.info(
        f"Bootstrap complete: created LanceDB table '{settings.LANCEDB_TABLE}' with {len(records)} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(bootstrap_lancedb())
