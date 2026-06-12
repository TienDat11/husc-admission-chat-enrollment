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

    def fetch_rows_by_ids(self, ids: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Fetch rows whose `id` column is in the given set, server-side.

        Avoids a full `to_pandas()` materialization of the whole table.
        Built on `self._table.search().where(pred).limit(...).to_list()`,
        which lancedb 0.29.2 supports with a NO-vector search-builder
        (the underlying query plan is a filtered table scan over the id
        index, not a vector search).

        The predicate uses SQL string escaping: each id is wrapped in
        single quotes and internal `'` are doubled to `''`. Non-string
        ids are skipped.

        Returns a list of dicts (one per matching row, columns as stored).
        Returns `[]` for None/empty input or when all ids are filtered
        out by the escaping step. Never raises into the caller — caller
        is expected to wrap in try/except as needed.
        """
        if not ids:
            return []
        if self._table is None:
            self.ensure_table()

        safe_ids: List[str] = []
        for raw in ids:
            if not isinstance(raw, str):
                continue
            # SQL string escaping: double any embedded single quote.
            safe_ids.append("'" + raw.replace("'", "''") + "'")
        if not safe_ids:
            return []

        predicate = "id IN (" + ", ".join(safe_ids) + ")"
        logger.debug(f"fetch_rows_by_ids predicate: {predicate}")
        # No-vector search-builder + where + limit. lancedb 0.29.x
        # accepts this and pushes the IN-clause down to the dataset
        # (no full-table materialization in Python).
        return self._table.search().where(predicate).limit(len(safe_ids)).to_list()
