#!/usr/bin/env python3
# @milestone(M3-B1) one-shot migration rag2025+husc_v2026_clean → husc_chunks
# Backup path: {uri}_backup_pre_b1_20260729
# Rollback: drop husc_chunks table after execute
"""One-shot migration script for M3-B1 schema unification."""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import lancedb
import pyarrow as pa

# Import schema from src
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
from infrastructure.lancedb_schema import CHUNK_SCHEMA, EMBEDDING_DIM


def _parse_metadata_json(raw: str) -> dict:
    """Parse metadata_json string to dict, return {} on failure."""
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_data_year(meta: dict) -> tuple:
    """Extract data_year from metadata dict. Returns (year_int, needs_review_flag)."""
    for key in ("data_year", "year"):
        if key in meta:
            val = meta[key]
            if isinstance(val, int) and val > 0:
                return val, False
            if isinstance(val, str) and val.isdigit():
                return int(val), False
    return 0, True  # missing → 0 + needs_review


def _map_rag2025_row(row: dict) -> dict:
    """Map a rag2025 source row → CHUNK_SCHEMA record."""
    meta_raw = row.get("metadata_json") or "{}"
    meta = _parse_metadata_json(meta_raw)
    data_year, needs_review = _extract_data_year(meta)

    # Merge faq_type into metadata if present
    extra_meta = {"needs_review": needs_review}
    faq_type = row.get("faq_type")
    if faq_type:
        extra_meta["source_faq_type"] = faq_type

    metadata_out = {**meta, **extra_meta}

    return {
        "id": row["id"],
        "doc_id": row.get("chunk_id") or row["id"],
        "release_id": "legacy",
        "text": row.get("text", ""),
        "summary": None,
        "vector": row.get("vector", [0.0] * EMBEDDING_DIM)[:EMBEDDING_DIM],
        "data_year": data_year,
        "scope": "admissions",
        "doc_type": row.get("faq_type") or "chunk",
        "status": "current",
        "effective_from": 0,
        "effective_to": 99991231,
        "is_tombstone": False,
        "superseded_by": None,
        "metadata_json": json.dumps(metadata_out, ensure_ascii=False),
    }


def _map_husc_v2026_row(row: dict) -> dict:
    """Map a husc_v2026_clean source row → CHUNK_SCHEMA record."""
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    data_year, needs_review = _extract_data_year(meta)
    chunk_type = meta.get("chunk_type") or "chunk"

    metadata_out = {"needs_review": needs_review, "source_url": row.get("source_url")}

    vec = row.get("embedding") or []
    if len(vec) < EMBEDDING_DIM:
        vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
    vec = vec[:EMBEDDING_DIM]

    return {
        "id": row["id"],
        "doc_id": row["id"],
        "release_id": "legacy",
        "text": row.get("text", ""),
        "summary": row.get("summary"),
        "vector": vec,
        "data_year": data_year,
        "scope": "admissions",
        "doc_type": chunk_type,
        "status": "current",
        "effective_from": 0,
        "effective_to": 99991231,
        "is_tombstone": False,
        "superseded_by": None,
        "metadata_json": json.dumps(metadata_out, ensure_ascii=False),
    }


def _run_migration(uri: str, target: str, dry_run: bool):
    """Core migration logic."""
    db = lancedb.connect(uri)
    existing_tables = db.table_names()

    # Read source tables
    rag_rows = []
    husc_rows = []

    if "rag2025" in existing_tables:
        rag_tbl = db.open_table("rag2025")
        rag_rows = list(rag_tbl.to_arrow().to_pylist())
        print(f"[INFO] rag2025: {len(rag_rows)} rows loaded")
    else:
        print("[WARN] Table 'rag2025' not found — skipping")

    if "husc_v2026_clean" in existing_tables:
        husc_tbl = db.open_table("husc_v2026_clean")
        husc_rows = list(husc_tbl.to_arrow().to_pylist())
        print(f"[INFO] husc_v2026_clean: {len(husc_rows)} rows loaded")
    else:
        print("[WARN] Table 'husc_v2026_clean' not found — skipping")

    # Map rows
    rag_mapped = [_map_rag2025_row(r) for r in rag_rows]
    husc_mapped = [_map_husc_v2026_row(r) for r in husc_rows]

    # Detect duplicates between sources
    rag_ids = {r["id"] for r in rag_mapped}
    husc_ids = {r["id"] for r in husc_mapped}
    dup_ids = rag_ids & husc_ids
    dup_count = len(dup_ids)

    if dup_ids:
        # Prefix duplicate IDs
        for r in rag_mapped:
            if r["id"] in dup_ids:
                r["id"] = f"rag2025:{r['id']}"
        for r in husc_mapped:
            if r["id"] in dup_ids:
                r["id"] = f"husc:{r['id']}"
        print(f"[WARN] {dup_count} duplicate IDs detected — prefixed with source prefix")

    all_records = rag_mapped + husc_mapped
    needs_review_count = sum(
        1 for r in all_records
        if r.get("metadata_json") and json.loads(r["metadata_json"]).get("needs_review")
    )

    if dry_run:
        print(f"\n=== DRY-RUN SUMMARY ===")
        print(f"Expected rows: {len(all_records)}")
        print(f"  rag2025 mapped: {len(rag_mapped)}")
        print(f"  husc_v2026_clean mapped: {len(husc_mapped)}")
        print(f"Duplicates prefixed: {dup_count}")
        print(f"Needs review (data_year=0): {needs_review_count}")
        print("=======================")
        return

    # --- EXECUTE MODE ---
    backup_path = str(str(Path(uri)) + "_backup_pre_b1_20260729")
    if Path(uri).exists():
        if Path(backup_path).exists():
            print(f"[WARN] Backup dir already exists at {backup_path} — skipping backup (idempotent)")
        else:
            print(f"[INFO] Creating backup at {backup_path} ...")
            shutil.copytree(uri, backup_path)
            print("[INFO] Backup complete")

    print(f"[INFO] Creating table '{target}' with {len(all_records)} records ...")
    db.create_table(target, data=all_records, schema=CHUNK_SCHEMA, mode="overwrite")

    final_tbl = db.open_table(target)
    print(f"[INFO] Rows written: {final_tbl.count_rows()}")
    indices = final_tbl.list_indices()
    print(f"[INFO] Indices: {[idx['name'] for idx in indices]}")
    print(f"[LOG] total_rows={len(all_records)}, duplicates_prefixed={dup_count}, needs_review={needs_review_count}")


def main():
    parser = argparse.ArgumentParser(description="M3-B1 schema migration to husc_chunks")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview only — do not write (default)")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Actually perform the migration")
    parser.add_argument("--uri", default="./data/lancedb",
                        help="LanceDB URI (default: ./data/lancedb)")
    parser.add_argument("--target", default="husc_chunks",
                        help="Target table name (default: husc_chunks)")
    args = parser.parse_args()

    if args.execute:
        _run_migration(args.uri, args.target, dry_run=False)
    else:
        _run_migration(args.uri, args.target, dry_run=True)


if __name__ == "__main__":
    main()
