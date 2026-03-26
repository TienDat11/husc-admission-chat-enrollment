"""Quick LanceDB table health check."""
from __future__ import annotations

from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings
from src.infrastructure.lancedb_adapter import LanceDBAdapter


def main() -> int:
    settings = RAGSettings()
    adapter = LanceDBAdapter(uri=settings.LANCEDB_URI, table_name=settings.LANCEDB_TABLE)

    if not adapter.table_exists():
        logger.error(f"Table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'")
        return 1

    count = adapter.count()
    logger.info(f"LanceDB OK: table={settings.LANCEDB_TABLE}, rows={count}, uri={settings.LANCEDB_URI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
