"""
Ingest chunked JSONL files into LanceDB using Qwen3-Embedding.

Default mode performs full rebuild (overwrite table).
Incremental mode (--incremental) ingests only changed chunk files and appends
new chunk IDs that are not already present in LanceDB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

import lancedb
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

load_dotenv()

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "chunked" / ".ingest_manifest.json"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _list_chunk_files(chunks_dir: Path) -> List[Path]:
    files = sorted(chunks_dir.glob("chunked_*.jsonl"))
    return [f for f in files if "enhanced" not in f.name]


def _load_manifest() -> Dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Ingest manifest unreadable, rebuilding from scratch")
        return {}


def _save_manifest(manifest: Dict[str, str]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _refresh_manifest_for_all_files(chunks_dir: Path) -> None:
    manifest = {file.name: _file_digest(file) for file in _list_chunk_files(chunks_dir)}
    _save_manifest(manifest)


def _resolve_target_files(chunks_dir: Path, incremental: bool) -> List[Path]:
    files = _list_chunk_files(chunks_dir)
    if not incremental:
        return files

    old_manifest = _load_manifest()
    changed: List[Path] = []
    for file in files:
        digest = _file_digest(file)
        if old_manifest.get(file.name) != digest:
            changed.append(file)
    return changed


def _load_chunks(files: List[Path]) -> List[Dict]:
    all_chunks: List[Dict] = []

    for file in files:
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


def _load_existing_ids(table) -> Set[str]:
    try:
        df = table.to_pandas(columns=["id"])
        return {str(v) for v in df["id"].tolist()}
    except Exception as e:
        logger.warning(f"Failed to read existing IDs from LanceDB table: {e}")
        return set()


def ingest_lancedb(incremental: bool = False) -> bool:
    settings = RAGSettings()
    chunks_dir = Path(__file__).parent.parent / "data" / "chunked"

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return False

    target_files = _resolve_target_files(chunks_dir, incremental=incremental)
    if incremental and not target_files:
        logger.info("Incremental ingest: no changed chunk files detected")
        return True

    logger.info(f"Loading chunks from {len(target_files)} JSONL file(s)...")
    chunks = _load_chunks(target_files)
    if not chunks:
        if incremental:
            logger.info("Incremental ingest: no valid chunk rows found in changed files")
            _refresh_manifest_for_all_files(chunks_dir)
            return True
        logger.error("No chunks found to ingest")
        return False

    logger.info(f"Loaded {len(chunks)} chunks")
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})")

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
    table_exists = settings.LANCEDB_TABLE in db.table_names()

    if incremental and table_exists:
        table = db.open_table(settings.LANCEDB_TABLE)
        existing_ids = _load_existing_ids(table)
        new_records = [r for r in records if r["id"] not in existing_ids]
        if not new_records:
            logger.info("Incremental ingest: no new chunk IDs to append")
        else:
            table.add(new_records)
            logger.info(
                f"Incremental ingest complete: appended={len(new_records)}, total_rows={table.count_rows()}, "
                f"table={settings.LANCEDB_TABLE}, uri={settings.LANCEDB_URI}"
            )
    else:
        if table_exists:
            table = db.create_table(settings.LANCEDB_TABLE, data=records, mode="overwrite")
            logger.info(
                f"Full ingest complete (overwrite): table={settings.LANCEDB_TABLE}, "
                f"rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
            )
        else:
            table = db.create_table(settings.LANCEDB_TABLE, data=records)
            logger.info(
                f"Initial ingest complete: table={settings.LANCEDB_TABLE}, "
                f"rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
            )

    _refresh_manifest_for_all_files(chunks_dir)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest chunked JSONL files into LanceDB")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only ingest changed chunk files and append new chunk IDs",
    )
    args = parser.parse_args()

    ok = ingest_lancedb(incremental=args.incremental)
    raise SystemExit(0 if ok else 1)
