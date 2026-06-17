"""One-off ingest of the timeline-framed học-bạ chunk (S17.2).

Reuses ``ingest_files`` from ``ingest_missing_2026.py`` (the dedup-by-chunk_id
+ ``table.add`` pattern). Idempotent — if ``qa_lich_xet_hoc_ba_2026_v1`` is
already in the live ``rag2025`` table it will be skipped.

Run from the project root:

    python scripts/ingest_timeline_hocba_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``config`` and ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb
from loguru import logger

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService
from scripts.ingest_missing_2026 import ingest_files


CHUNK_FILE_NAME = "husc_2026_timeline_hocba_v1.jsonl"


def main() -> None:
    settings = RAGSettings()
    chunks_dir = Path(__file__).resolve().parent.parent / "data" / "chunked"
    file_path = chunks_dir / CHUNK_FILE_NAME
    if not file_path.exists():
        raise FileNotFoundError(f"Missing chunk file: {file_path}")

    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)

    embedder = EmbeddingService(settings)
    summary = ingest_files(files=[file_path], table=table, embedder=embedder)
    for fname, counts in summary.items():
        logger.info(f"{fname}: added={counts['added']} skipped={counts['skipped']}")
    total_added = sum(c["added"] for c in summary.values())
    total_skipped = sum(c["skipped"] for c in summary.values())
    print(f"INGEST TIMELINE HOCBA: added={total_added} skipped={total_skipped}")


if __name__ == "__main__":
    main()
