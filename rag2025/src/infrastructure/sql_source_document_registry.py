"""PostgreSQL-backed source-document registry (the ``source_documents`` table).

:class:`SqlSourceDocumentRegistry` keeps the file registry's interface and state
machine: it inherits ``create``/``update``/``retire``/``is_registered`` and the
path rules, and replaces only how an entry is stored. Under
``STORE_BACKEND=postgres`` this table is the source of truth, because the record
store refuses a record whose ``source_document_id`` has no row here
(``canonical_records`` foreign key, and :class:`SqlRecordStore`).

Column mapping (``alembic/versions/0001_baseline.py``):

- ``document_id`` -> ``document_id``; ``sha256`` (hex) -> ``sha256`` bytea. The
  column is NOT NULL, so an entry with no hash (file absent) cannot be stored.
- ``kind`` -> ``kind``; the table's CHECK admits pdf, docx and xlsx only.
- ``REGISTERED`` -> ``status`` 'uploaded' (kept when the row is already past
  it); ``RETIRED`` -> 'archived'.
- ``registered_at``/``registered_by`` -> ``created_at``/``uploaded_by``;
  ``updated_at`` -> ``updated_at``.
- ``storage_uri`` = the store-relative document id; ``original_name`` = its basename.

The table has no column for ``revision``, ``notes`` or ``updated_by``, so they
are not persisted: a stored entry reads back at revision 1 with no notes. The
file registry continues its revision counter across a revival; this one does
not, until the table gains those columns.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from src.domain.source_document import SourceDocumentState
from src.infrastructure.source_document_registry import (
    RegistryPathError,
    RegistryStateError,
    SourceDocumentEntry,
    SourceDocumentRegistry,
    _validate_document_id,
)
from src.infrastructure.sql_record_store import RecordStoreUnavailableError, _conninfo

_KINDS = frozenset({"pdf", "docx", "xlsx"})

_SELECT = (
    "SELECT document_id, sha256, status, uploaded_by, created_at, updated_at "
    "FROM source_documents"
)

_UPSERT = """
INSERT INTO source_documents
    (document_id, sha256, kind, storage_uri, original_name, status,
     uploaded_by, created_at, updated_at)
VALUES
    (%(document_id)s, %(sha256)s, %(kind)s, %(document_id)s, %(original_name)s,
     %(status)s, %(uploaded_by)s, %(created_at)s, %(updated_at)s)
ON CONFLICT (document_id) DO UPDATE SET
    sha256 = EXCLUDED.sha256,
    status = CASE
        WHEN EXCLUDED.status = 'archived' THEN 'archived'
        WHEN source_documents.status = 'archived' THEN 'uploaded'
        ELSE source_documents.status
    END,
    uploaded_by = EXCLUDED.uploaded_by,
    updated_at = EXCLUDED.updated_at
"""


class SqlSourceDocumentRegistry(SourceDocumentRegistry):
    """The file registry's state machine over the SQL ``source_documents`` table.

    One short-lived connection per call, as in :class:`SqlRecordStore`. The
    blob bytes stay under ``root/source_documents/``; only registry rows move.
    """

    def __init__(self, root: str | Path, repo_root: str | Path, dsn: str) -> None:
        super().__init__(root, repo_root)
        # The shared normalizer: SQLAlchemy-style URLs (+asyncpg, ssl=) -> libpq conninfo.
        self._conninfo = _conninfo(dsn)

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        # psycopg's own message can echo the conninfo, password included, so only
        # the exception class name leaves this method.
        try:
            conn = psycopg.connect(self._conninfo, row_factory=dict_row, connect_timeout=10)
        except psycopg.Error as exc:
            raise RuntimeError(
                f"source_documents registry connection failed: {type(exc).__name__}"
            ) from None
        with conn:
            yield conn

    def put(self, entry: SourceDocumentEntry) -> SourceDocumentEntry:
        _validate_document_id(entry.document_id)
        if entry.sha256 is None:
            raise RegistryStateError(
                f"{entry.document_id!r} has no content hash; the SQL registry "
                "stores only hashed documents"
            )
        kind = Path(entry.document_id).suffix.lower().lstrip(".")
        if kind not in _KINDS:
            raise RegistryPathError(
                f"{entry.document_id!r}: {kind!r} is not a kind the source_documents table holds"
            )
        status = "archived" if entry.state is SourceDocumentState.RETIRED else "uploaded"
        with self._connect() as conn:
            conn.execute(
                _UPSERT,
                {
                    "document_id": entry.document_id,
                    "sha256": bytes.fromhex(entry.sha256),
                    "kind": kind,
                    "original_name": Path(entry.document_id).name,
                    "status": status,
                    "uploaded_by": entry.registered_by,
                    "created_at": entry.registered_at,
                    "updated_at": entry.updated_at,
                },
            )
        return entry

    def get(self, document_id: str) -> SourceDocumentEntry | None:
        try:
            _validate_document_id(document_id)
        except RegistryPathError:
            return None
        with self._connect() as conn:
            row = conn.execute(f"{_SELECT} WHERE document_id = %s", (document_id,)).fetchone()
        return None if row is None else self._entry(row)

    def list_entries(self, *, include_retired: bool = False) -> list[SourceDocumentEntry]:
        with self._connect() as conn:
            rows = conn.execute(f"{_SELECT} ORDER BY document_id").fetchall()
        entries = [self._entry(row) for row in rows]
        return [entry for entry in entries if include_retired or entry.is_registered]

    def _entry(self, row: dict) -> SourceDocumentEntry:
        document_id = row["document_id"]
        state = (
            SourceDocumentState.RETIRED
            if row["status"] == "archived"
            else SourceDocumentState.REGISTERED
        )
        return self._build(
            document_id,
            sha256=bytes(row["sha256"]).hex(),
            file_present=self._file_present(document_id),
            state=state,
            revision=1,
            registered_at=row["created_at"],
            registered_by=row["uploaded_by"],
            updated_at=row["updated_at"],
            updated_by=row["uploaded_by"],
            notes=None,
        )

    def _file_present(self, document_id: str) -> bool:
        try:
            return self.resolve_path(document_id).is_file()
        except RegistryPathError:
            return False
