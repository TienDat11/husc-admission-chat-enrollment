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
import re
import subprocess
from datetime import datetime
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

# M3-N2b: ingest target is unified husc_chunks (not legacy rag2025)
import os
TARGET_TABLE = os.environ.get("LANCEDB_UNIFIED_TABLE", "husc_chunks")


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
    # T1.5-bl / G4: the canonical corpus spans MANY sources, not just the
    # legacy `chunked_*.jsonl` set. Include every `*.jsonl` in data/chunked/
    # except stubs/noise:
    #   - chunked_thongbao_* / batch_thongbao_* / crawl_thongbao_*: 556-byte
    #     crawl stubs + defensive (T1.3 G1)
    #   - *_enhanced: derived/duplicate chunks (never canonical)
    #   - chunked_test: fixture, not corpus
    #   - husc_yearly_{year}: literal template with braces, not a real file
    #   - .* hidden files (manifest etc.)
    files = sorted(chunks_dir.glob("*.jsonl"))
    _stub_prefixes = ("chunked_thongbao_", "batch_thongbao_", "crawl_thongbao_")
    _exact_names = {"chunked_test.jsonl", "husc_yearly_{year}.jsonl"}
    return [
        f for f in files
        if not f.name.startswith(_stub_prefixes)
        and "enhanced" not in f.name
        and f.name not in _exact_names
        and not f.name.startswith(".")
    ]


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


def _normalize_for_hash(text: str) -> str:
    """Normalize text for content hashing: strip whitespace + remove date patterns."""
    t = re.sub(r"\d{2}/\d{2}/\d{4}", "", text)
    return " ".join(t.split())


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def _text_dedup_hash(text: str) -> str:
    """Normalized text hash for cross-file dedup (case/whitespace-insensitive).

    T1.2: unlike `_content_hash` (re-embed guardrail, date-pattern-insensitive),
    this is the *insert* guard. Two JSONL files carrying the same text under
    different ids must be stored once. Lowercase + collapsed whitespace means
    "Nội dung X" and "  nội DUNG X  " resolve to the same hash.
    """
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _unique_id(source: str, ingest_file: str, chunk_id: str) -> str:
    """Globally-unique row id (T1.5/G9).

    Bare chunk ids such as '0','13','61' collide across documents — several
    ingest scripts emit identical numeric chunk_ids, so `id = chunk_id` lets the
    last writer silently overwrite the others. Hashing source|ingest_file|chunk_id
    yields a stable, unique id per (document, file, chunk) triple. `doc_id` keeps
    the original chunk_id so document-level grouping is preserved.
    """
    raw = f"{source}|{ingest_file}|{chunk_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
                source = obj.get("metadata", {}).get("source", file.name)
                all_chunks.append(
                    {
                        # T1.5 (G9): globally-unique row id — bare chunk ids
                        # ('0','13','61') collide across documents. `chunk_id`
                        # below keeps the original value for doc_id grouping.
                        "id": _unique_id(source, file.name, chunk_id),
                        "chunk_id": chunk_id,
                        "text": text,
                        "source": source,
                        # File provenance for orphan scoping. Distinct from
                        # `source` (often a URL, shared across several files).
                        "ingest_file": file.name,
                        "faq_type": obj.get("faq_type", obj.get("metadata", {}).get("faq_type", "")),
                        "metadata": obj.get("metadata", {}),
                        "content_hash": _content_hash(text),
                    }
                )

    all_chunks = [c for c in all_chunks if c.get("text")]

    # T1.2: cross-file text dedup — keep one canonical chunk (first by
    # ingest_file + chunk_id) when two files carry the same normalized text.
    seen_text_hashes: Set[str] = set()
    deduped: List[Dict] = []
    for c in sorted(all_chunks, key=lambda x: (x["ingest_file"], x["chunk_id"])):
        th = _text_dedup_hash(c["text"])
        if th in seen_text_hashes:
            continue
        seen_text_hashes.add(th)
        deduped.append(c)

    return deduped


def _load_existing_ids(table) -> Set[str]:
    try:
        rows = table.to_arrow().select(["id"]).to_pylist()
        return {str(r["id"]) for r in rows}
    except Exception as e:
        logger.warning(f"Failed to read existing IDs from LanceDB table: {e}")
        return set()


def _esc(raw: str) -> str:
    """Escape single quotes for SQL IN-clause values."""
    return raw.replace("'", "''")


def _orphan_ids(table, batch_files: Set[str], batch_ids: Set[str]) -> List[str]:
    """Rows ingested from batch_files but absent from batch_ids.

    husc_chunks (CHUNK_SCHEMA) has no `source` column, so provenance lives in
    metadata_json under `ingest_file` (the JSONL filename). Rows with no
    recorded ingest_file are never orphans — they came from another ingest
    script (581 of 2068 production rows) and must not be touched.

    Scoping by `source` is WRONG: it is usually a URL shared by several files,
    so it matches rows this batch does not own.
    """
    if not batch_files:
        return []
    try:
        rows = table.to_arrow().select(["id", "metadata_json"]).to_pylist()
    except Exception as e:
        logger.warning(f"Orphan scan failed, skipping orphan delete: {e}")
        return []

    orphans: List[str] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in batch_ids:
            continue
        try:
            meta = json.loads(row.get("metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if meta.get("ingest_file") in batch_files:
            orphans.append(row_id)
    return orphans


def _trigger_graph_sync(new_ids: List[str]) -> None:
    """M3-N3a: Post-ingest graph sync hook (fail-safe, non-blocking).

    Writes new chunk IDs to data/graph/.new_chunk_ids.json and triggers
    build_graph.py --incremental via subprocess. Failures are logged but
    do NOT propagate — eventual consistency per R3 direction A.
    """
    repo_root = Path(__file__).parent.parent
    graph_dir = repo_root / "data" / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Write new chunk IDs for incremental build
    new_ids_path = graph_dir / ".new_chunk_ids.json"
    try:
        new_ids_path.write_text(json.dumps(new_ids, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write .new_chunk_ids.json: {e}")
        return

    # Trigger incremental graph build (non-blocking, fail-safe)
    try:
        subprocess.run(
            [sys.executable, "scripts/build_graph.py", "--incremental"],
            cwd=str(repo_root),
            timeout=300,  # 5 min timeout
            capture_output=True,
            text=True,
        )
        logger.info(f"Graph sync triggered for {len(new_ids)} new chunk(s)")
    except subprocess.TimeoutExpired:
        logger.warning("Graph sync timed out after 300s (non-fatal)")
    except Exception as e:
        logger.warning(f"Graph sync failed (non-fatal): {e}")


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

    # M3-N1b: content-hash guardrail — skip re-embed when text unchanged
    db = lancedb.connect(settings.LANCEDB_URI)
    target = TARGET_TABLE
    table_exists = target in db.table_names()

    existing_hashes: Dict[str, str] = {}
    if incremental and table_exists:
        try:
            table_tmp = db.open_table(target)
            rows = table_tmp.to_arrow().select(["id", "metadata_json"]).to_pylist()
            for r in rows:
                try:
                    meta = json.loads(r.get("metadata_json") or "{}")
                    h = meta.get("content_hash")
                    if h:
                        existing_hashes[r["id"]] = h
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception as e:
            logger.warning(f"Failed to load existing hashes: {e}")

    chunks_to_embed = [
        c for c in chunks
        if c["content_hash"] != existing_hashes.get(c["id"])
    ]
    chunks_unchanged = [
        c for c in chunks
        if c["content_hash"] == existing_hashes.get(c["id"])
    ]
    skipped = len(chunks_unchanged)
    if skipped:
        logger.info(f"Content-hash guardrail: skipping {skipped} unchanged chunk(s)")

    # Embed only changed chunks
    embeddings_map: Dict[str, list] = {}
    if chunks_to_embed:
        logger.info(f"Embedding {len(chunks_to_embed)} chunk(s) (skipped {skipped} unchanged)")
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})")
        embedding_service = EmbeddingService(settings)
        texts = [chunk["text"] for chunk in chunks_to_embed]
        embeddings = embedding_service.encode_documents(texts)
        for chunk, vector in zip(chunks_to_embed, embeddings):
            embeddings_map[chunk["id"]] = vector.tolist()

    # For unchanged chunks, reuse existing vectors from table
    if chunks_unchanged and table_exists:
        try:
            table_tmp2 = db.open_table(target)
            unchanged_ids = [c["id"] for c in chunks_unchanged]
            escaped_ids = ", ".join(f"'{_esc(i)}'" for i in unchanged_ids)
            rows = table_tmp2.search().where(f"id IN ({escaped_ids})").limit(len(unchanged_ids)).to_arrow().to_pylist()
            for r in rows:
                embeddings_map[r["id"]] = r["vector"]
        except Exception as e:
            logger.warning(f"Failed to load existing vectors for unchanged chunks: {e}")
            # Fallback: embed them anyway
            embedding_service = EmbeddingService(settings)
            for chunk in chunks_unchanged:
                if chunk["id"] not in embeddings_map:
                    embeddings_map[chunk["id"]] = embedding_service.encode_documents([chunk["text"]])[0].tolist()

    records = []
    for chunk in chunks:
        vector = embeddings_map.get(chunk["id"])
        if vector is None:
            continue  # safety: skip if no vector available
        raw_meta = chunk.get("metadata", {})
        # M3-N2b: extract data_year (pattern from migrate_schema_v2.py:32-41)
        data_year = 0
        needs_review = True
        for key in ("data_year", "year"):
            val = raw_meta.get(key)
            if isinstance(val, int) and val > 0:
                data_year, needs_review = val, False
                break
            if isinstance(val, str) and val.isdigit():
                data_year, needs_review = int(val), False
                break

        faq_type = chunk.get("faq_type", "")
        # Provenance: husc_chunks has no `source` column, so record it in
        # metadata_json — orphan detection needs it to scope deletes by file.
        meta_out = {
            **raw_meta,
            "ingest_file": chunk["ingest_file"],
            "needs_review": needs_review,
            "content_hash": chunk["content_hash"],
        }
        if faq_type:
            meta_out["source_faq_type"] = faq_type

        records.append(
            {
                "id": chunk["id"],
                "doc_id": chunk.get("chunk_id") or chunk["id"],
                "release_id": "legacy",
                "text": chunk["text"],
                "summary": None,
                "vector": vector,
                "data_year": data_year,
                "scope": "admissions",
                "doc_type": faq_type or "chunk",
                "status": "current",
                "effective_from": 0,
                "effective_to": 99991231,
                "is_tombstone": False,
                "superseded_by": None,
                "metadata_json": json.dumps(meta_out, ensure_ascii=False, default=str),
                "is_reference": bool(raw_meta.get("is_reference")),
                "crawl_date": raw_meta.get("crawl_date"),
            }
        )

    if incremental and table_exists:
        table = db.open_table(target)

        # Pre-ingest tag for rollback safety
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            from src.infrastructure.lancedb_versioning import tag_release
            tag_release(table, f"pre-ingest-{ts}")
        except Exception:
            logger.warning(f"tag_release failed (non-fatal): {ts}")

        # Orphan delete only when table has "source" column (legacy rag2025).
        # husc_chunks (CHUNK_SCHEMA) has no "source" column — skip.
        where_clause = None
        if "source" in table.schema.names:
            sources_in_batch = sorted({c.get("source", "unknown") for c in chunks})
            if sources_in_batch:
                escaped = ", ".join(f"'{_esc(s)}'" for s in sources_in_batch)
                where_clause = f"source IN ({escaped})"

        job = (
            table.merge_insert("id")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
        )
        if where_clause:
            job.when_not_matched_by_source_delete(condition=where_clause)
        else:
            # husc_chunks has no `source` column, so orphans are resolved from
            # metadata_json provenance instead. An UNCONDITIONAL delete here
            # would wipe every row outside the batch: in --incremental mode the
            # batch is only the CHANGED files, and 581 of 2068 production rows
            # come from other ingest scripts entirely.
            orphans = _orphan_ids(
                table,
                {c["ingest_file"] for c in chunks},
                # From `chunks`, NOT `records`: records drops any chunk whose
                # vector went missing (:309-311), and that loss is silent. A
                # chunk still present in the JSONL must never count as an orphan.
                {c["id"] for c in chunks},
            )
            if orphans:
                escaped_orphans = ", ".join(f"'{_esc(o)}'" for o in orphans)
                job.when_not_matched_by_source_delete(
                    condition=f"id IN ({escaped_orphans})"
                )
                logger.info(f"Orphan delete: {len(orphans)} row(s) from batch files")
        job.execute(records)

        logger.info(
            f"Incremental ingest complete: merged={len(records)}, total_rows={table.count_rows()}, "
            f"table={target}, uri={settings.LANCEDB_URI}"
        )
    else:
        if table_exists:
            # T1.4 (G4): pre-ingest tag BEFORE overwrite so the previous table
            # state is recoverable. The old code overwrote first, so the
            # pre-overwrite version was never tagged (a mistaken default run
            # dropped 581/2068 rows with no rollback point).
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                from src.infrastructure.lancedb_versioning import tag_release
                table_tmp = db.open_table(target)
                tag_release(table_tmp, f"pre-ingest-{ts}")
            except Exception:
                logger.warning(f"tag_release failed (non-fatal): {ts}")
            table = db.create_table(target, data=records, mode="overwrite")
            logger.info(
                f"Full ingest complete (overwrite): table={target}, "
                f"rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
            )
        else:
            table = db.create_table(target, data=records)
            logger.info(
                f"Initial ingest complete: table={target}, "
                f"rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
            )

    _refresh_manifest_for_all_files(chunks_dir)

    # M3-N3a: trigger graph sync for new chunks
    new_chunk_ids = [r["id"] for r in records]
    _trigger_graph_sync(new_chunk_ids)

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
