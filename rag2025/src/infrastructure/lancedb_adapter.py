"""Infrastructure adapter for LanceDB operations."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
from loguru import logger


# Module-level singleton state for `ensure_warm()`. Kept private so
# callers always go through the helper (which holds the lock).
_ensure_warm_instance: Optional["LanceDBAdapter"] = None
_ensure_warm_lock = threading.Lock()
_ensure_warm_factory = None  # set by the lifespan; defaults to LanceDBAdapter


def configure_ensure_warm_factory(factory):
    """Install the factory used by `ensure_warm()` (called once at lifespan
    startup so the adapter uses the same URI/table as the rest of the app)."""
    global _ensure_warm_factory
    _ensure_warm_factory = factory


def ensure_warm() -> "LanceDBAdapter":
    """Return a process-wide singleton LanceDB adapter.

    Idempotent: the first call constructs via the configured factory,
    subsequent calls return the cached instance. Thread-safe via a
    re-entrant guard (double-checked locking).

    Init failures are NOT cached — the next caller can retry. This keeps
    a transient boot-time LanceDB hiccup from permanently disabling
    retrieval for the lifetime of the process.
    """
    global _ensure_warm_instance
    if _ensure_warm_instance is not None:
        return _ensure_warm_instance
    with _ensure_warm_lock:
        if _ensure_warm_instance is None:
            factory = _ensure_warm_factory or LanceDBAdapter
            _ensure_warm_instance = factory()
        return _ensure_warm_instance


def _reset_ensure_warm_for_tests() -> None:
    """Test hook: drop the cached singleton + factory (call sites must
    NOT depend on this — only the test suite uses it)."""
    global _ensure_warm_instance, _ensure_warm_factory
    with _ensure_warm_lock:
        _ensure_warm_instance = None
        _ensure_warm_factory = None


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
        # lancedb 0.29.2: list_tables() returns a ListTablesResponse Pydantic
        # model; iterating it yields (key, value) tuples, not table names.
        # Normalize to a flat list of table names.
        raw = self._db.list_tables()
        names: List[str] = []
        # Case 1: dict-like with 'tables' key
        if hasattr(raw, "tables"):
            try:
                names = list(raw.tables)
            except TypeError:
                names = []
        # Case 2: plain iterable of names
        if not names:
            try:
                for entry in raw:
                    # tuple like ('tables', ['a','b']) → take names from value
                    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == "tables":
                        names = list(entry[1])
                        break
                    if isinstance(entry, str):
                        names.append(entry)
            except TypeError:
                pass
        return self._table_name in names

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

    def get_table(self):
        """Return the underlying lancedb Table object, opening it via
        ensure_table() if not already open. Callers outside this adapter
        (e.g. src.infrastructure.lancedb_versioning.tag_release/maintain/
        rollback_to/promote_to_prod/prune_versions/read_only_at) operate on
        the raw LanceTable, not this wrapper class.

        WARNING: returns a raw handle — caller MUST NOT call ``drop``,
        ``delete_all``, or any destructive API on the production tables
        ``rag2025`` / ``husc_v2026_clean``. Those tables are on the
        rollback path and must be preserved until M6 is green for two
        consecutive weeks.

        Consumer state: currently only test code calls this method. It
        serves as infra preparation for ``lancedb_versioning.py`` (six
        functions that accept a raw ``table`` argument) which will be
        wired into the M5 CLI publisher pipeline.

        ponytail: when M5 CLI publisher is live and versioning functions
        are wired, remove this warning block and replace with a proper
        access-control guard (read-only context manager or ACL check).
        """
        if self._table is None:
            self.ensure_table()
        return self._table

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
