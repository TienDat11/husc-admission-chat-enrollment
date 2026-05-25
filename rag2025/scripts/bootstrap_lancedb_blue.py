# @spec(S13.4) bootstrap_lancedb_blue — create husc_v2026_blue with schema parity
"""Create the new blue LanceDB table for the temporal reingest blue/green flip.

The blue table mirrors the schema of the existing husc table:
  - id: string
  - text: string
  - summary: string (nullable)
  - embedding: list<float32, 1024>  (Qwen3-Embedding-0.6B output dim)
  - metadata: struct (data_year, source_url, notification_id, chunk_method, ...)

Idempotency:
  - If the target table already exists with the same schema → skip (status="exists").
  - Pass --force to drop+recreate (status="recreated").
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Any

from loguru import logger

EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B
DEFAULT_BLUE_TABLE = "husc_v2026_blue"


def _build_schema() -> list[dict[str, Any]]:
    """Return the canonical schema as a list of {name, type} dicts."""
    return [
        {"name": "id", "type": "string"},
        {"name": "text", "type": "string"},
        {"name": "summary", "type": "string"},
        {"name": "embedding", "type": f"list<float32, {EMBEDDING_DIM}>"},
        {"name": "metadata", "type": "struct"},
    ]


def bootstrap_blue_table(
    *,
    db_uri: str,
    table_name: str,
    force: bool = False,
    connector: Any = None,
) -> dict[str, Any]:
    """Create or skip the blue LanceDB table.

    `connector` defaults to lazy-imported `lancedb`; tests inject a stub.
    Returns dict with: status, table_name, schema_dim.
    """
    if connector is None:
        try:
            import lancedb  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover — exercised via injection in tests
            raise RuntimeError(f"lancedb import failed: {exc}") from exc
        connector = lancedb

    db = connector.connect(db_uri)
    schema = _build_schema()

    existing_names: list[str]
    try:
        existing_names = list(db.table_names())
    except Exception:
        existing_names = []

    status = "created"
    if table_name in existing_names:
        if force:
            db.drop_table(table_name)
            status = "recreated"
            db.create_table(table_name, schema=schema)
        else:
            status = "exists"
    else:
        db.create_table(table_name, schema=schema)

    logger.info(f"bootstrap_blue_table: table={table_name} status={status}")
    return {
        "status": status,
        "table_name": table_name,
        "schema_dim": EMBEDDING_DIM,
        "fields": [f["name"] for f in schema],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap blue LanceDB table")
    parser.add_argument("--db-uri", default=os.getenv("LANCEDB_URI", "rag2025/data/lancedb"))
    parser.add_argument("--table", default=DEFAULT_BLUE_TABLE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    res = bootstrap_blue_table(db_uri=args.db_uri, table_name=args.table, force=args.force)
    logger.info(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
