"""Create scalar indices on LanceDB tables — schema-aware, idempotent.

Milestone M3-B4: data_lifecycle phase B4 (scalar index creation).
Note: ≥8 scalar indices per table only achievable after M3-B1 produces
a `husc_chunks` table with metadata(struct{data_year,status,...}).
On existing tables (rag2025_bge), only column `id` matches target columns.
"""

import argparse
import logging
import sys
from pathlib import Path

import lancedb

logger = logging.getLogger(__name__)

# Target columns grouped by index type
BTREE_TARGETS = ["id", "data_year", "effective_from", "effective_to", "superseded_by"]
BITMAP_TARGETS = ["status", "scope", "doc_type", "is_tombstone"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scalar indices on LanceDB tables")
    parser.add_argument("--table", required=True, help="Table to index (e.g. rag2025_bge, husc_v2026_clean)")
    parser.add_argument("--uri", default="./data/lancedb", help="LanceDB URI")
    args = parser.parse_args()

    db = lancedb.connect(args.uri)
    try:
        table = db.open_table(args.table)
    except ValueError as exc:
        logger.error("Table '%s' not found in %s: %s", args.table, args.uri, exc)
        sys.exit(1)

    # Build set of top-level field names present in schema
    schema_fields = {f.name for f in table.schema}
    all_targets = BTREE_TARGETS + BITMAP_TARGETS

    created = 0
    skipped_existing = 0
    skipped_missing = 0

    existing_index_names = {idx["name"] for idx in table.list_indices()}

    for col in all_targets:
        if col not in schema_fields:
            logger.warning("skip %s: not in schema", col)
            skipped_missing += 1
            continue

        # Derive expected index name: "col_<type>" or just check by column presence
        # list_indices returns dicts like {"name": "...", "column": "...", "index_type": "..."}
        already_indexed = any(col in idx.columns for idx in table.list_indices())
        if already_indexed:
            logger.info("skip %s: exists", col)
            skipped_existing += 1
            continue

        # Determine index type
        idx_type = "BITMAP" if col in BITMAP_TARGETS else "BTREE"
        try:
            table.create_scalar_index(column=col, index_type=idx_type)
            logger.info("created %s (%s)", col, idx_type)
            created += 1
        except Exception as exc:
            logger.error("failed %s (%s): %s", col, idx_type, exc)
            skipped_missing += 1

    print(f"\nSummary: created={created} skipped_existing={skipped_existing} skipped_missing={skipped_missing}")
    print(f"Indices on '{args.table}': {table.list_indices()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
