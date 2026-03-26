"""Infrastructure adapter for LanceDB operations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
from loguru import logger


class LanceDBAdapter:
    """Thin infrastructure wrapper around LanceDB client and table."""

    def __init__(self, uri: str, table_name: str) -> None:
        self._uri = uri
        self._table_name = table_name
        self._db = lancedb.connect(uri)
        self._table = None

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def uri(self) -> str:
        return self._uri

    def table_exists(self) -> bool:
        return self._table_name in self._db.table_names()

    def ensure_table(self) -> None:
        if not self.table_exists():
            raise FileNotFoundError(
                f"LanceDB table '{self._table_name}' not found at '{self._uri}'. "
                "Run ingestion first."
            )
        self._table = self._db.open_table(self._table_name)

    def count(self) -> int:
        if self._table is None:
            self.ensure_table()
        return int(self._table.count_rows())

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        where_clause: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self._table is None:
            self.ensure_table()

        query = self._table.search(query_vector).limit(top_k)
        if where_clause:
            query = query.where(where_clause)

        rows = query.to_list()
        logger.debug(f"LanceDB search returned {len(rows)} rows")
        return rows
