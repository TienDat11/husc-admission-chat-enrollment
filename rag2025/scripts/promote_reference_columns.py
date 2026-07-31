#!/usr/bin/env python3
"""M3-N4a: Promote is_reference + crawl_date from metadata_json to flat columns.

Adds BITMAP index on is_reference for filter pushdown.
Idempotent: skips columns/index that already exist.

Usage:
    python scripts/promote_reference_columns.py --uri ./data/lancedb --table husc_chunks --dry-run
    python scripts/promote_reference_columns.py --uri ./data/lancedb --table husc_chunks --execute
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from loguru import logger


def promote(uri: str, table_name: str, execute: bool = False) -> None:
    db = lancedb.connect(uri)
    if table_name not in db.table_names():
        logger.error(f"Table '{table_name}' not found in {uri}")
        sys.exit(1)

    table = db.open_table(table_name)
    schema_names = table.schema.names
    row_count = table.count_rows()

    need_is_reference = "is_reference" not in schema_names
    need_crawl_date = "crawl_date" not in schema_names

    logger.info(f"Table: {table_name}, rows: {row_count}")
    logger.info(f"  is_reference column: {'MISSING' if need_is_reference else 'exists'}")
    logger.info(f"  crawl_date column: {'MISSING' if need_crawl_date else 'exists'}")

    if not need_is_reference and not need_crawl_date:
        logger.info("Both columns already exist. Checking index...")
        _ensure_index(table, execute)
        return

    if not execute:
        logger.info("[DRY-RUN] Would add columns and backfill. Use --execute to apply.")
        return

    # Step 1: add columns
    add_map = {}
    if need_is_reference:
        add_map["is_reference"] = "CAST(NULL AS BOOLEAN)"
    if need_crawl_date:
        add_map["crawl_date"] = "CAST(NULL AS STRING)"
    table.add_columns(add_map)
    logger.info(f"Added columns: {list(add_map.keys())}")

    # Step 2: backfill from metadata_json
    rows = table.to_arrow().select(["id", "metadata_json"]).to_pylist()
    true_ids = []
    false_ids = []
    date_map: dict[str, str] = {}

    for r in rows:
        rid = r["id"]
        try:
            meta = json.loads(r.get("metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}

        is_ref = bool(meta.get("is_reference"))
        if is_ref:
            true_ids.append(rid)
        else:
            false_ids.append(rid)

        cd = meta.get("crawl_date")
        if cd and isinstance(cd, str):
            date_map[rid] = cd

    # Batch update by group (avoids 2119 individual commits)
    if true_ids:
        escaped = ", ".join(f"'{i.replace(chr(39), chr(39)*2)}'" for i in true_ids)
        table.update(where=f"id IN ({escaped})", values={"is_reference": True})
        logger.info(f"  is_reference=True: {len(true_ids)} rows")

    if false_ids:
        escaped = ", ".join(f"'{i.replace(chr(39), chr(39)*2)}'" for i in false_ids)
        table.update(where=f"id IN ({escaped})", values={"is_reference": False})
        logger.info(f"  is_reference=False: {len(false_ids)} rows")

    # crawl_date: batch by value groups
    date_groups: dict[str, list[str]] = {}
    for rid, cd in date_map.items():
        date_groups.setdefault(cd, []).append(rid)
    for cd, ids in date_groups.items():
        escaped = ", ".join(f"'{i.replace(chr(39), chr(39)*2)}'" for i in ids)
        table.update(where=f"id IN ({escaped})", values={"crawl_date": cd})
    if date_groups:
        logger.info(f"  crawl_date backfilled: {len(date_map)} rows in {len(date_groups)} groups")

    # Step 3: BITMAP index
    _ensure_index(table, execute=True)

    logger.info(f"Done. Final row count: {table.count_rows()}")


def _ensure_index(table, execute: bool) -> None:
    existing_indices = [i.name for i in table.list_indices()]
    if "is_reference_idx" in existing_indices:
        logger.info("  BITMAP index 'is_reference_idx' already exists")
        return
    if not execute:
        logger.info("  [DRY-RUN] Would create BITMAP index on is_reference")
        return
    table.create_scalar_index("is_reference", index_type="BITMAP")
    logger.info("  Created BITMAP index on is_reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote is_reference/crawl_date to flat columns")
    parser.add_argument("--uri", default="./data/lancedb", help="LanceDB URI")
    parser.add_argument("--table", default="husc_chunks", help="Table name")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview without writing")
    group.add_argument("--execute", action="store_true", help="Apply changes")
    args = parser.parse_args()

    promote(args.uri, args.table, execute=args.execute)
