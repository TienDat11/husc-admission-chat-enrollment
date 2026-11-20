"""PostgreSQL-backed record store with the :class:`CanonicalRecordStore` interface.

A sync drop-in for ``canonical_record_store.CanonicalRecordStore`` over the
``canonical_records`` and ``supersession_retractions`` tables of the PL-4
baseline (``alembic/versions/0001_baseline.py``). The schema is authoritative;
where the SQL model is stricter than the file store, this module enforces the
SQL rules and the differences are listed below.

Exceptions and dataclasses are reused from ``canonical_record_store``.

Connections: one short-lived connection per public call, no pool, no
import-time or constructor-time database access.

Documented divergences from the file store (the SQL model enforces these)
------------------------------------------------------------------------
- Publishing archives the previous published version of the same ``record_id``
  in the same transaction (``status='archived'``, ``archive_reason='superseded'``).
  ``uq_current_coordinate`` allows one published version per coordinate, so the
  file store's "every version stays published" is not representable.
- The workflow appends one version per transition, so every write archives the
  earlier open (draft / review / approved) versions of its ``record_id`` in the
  same transaction (``status='archived'``, ``archive_reason='superseded'``). One
  open version per record is then kept, as ``uq_one_open_draft`` requires. A
  draft written on top of a published version leaves the published one as it is.
  A duplicate version number still raises :class:`RecordVersionConflictError`.
- ``source_document_id`` must exist in ``source_documents`` (FK); otherwise
  :class:`RecordStoreError` is raised instead of a raw psycopg error.
- Any unique violation on write (primary key, open-draft, coordinate) raises
  :class:`RecordVersionConflictError`. ``coordinate_key`` equals ``record_id``
  (DDL CHECK), so a coordinate clash with a different record id cannot occur.
- ``retract_supersession`` needs a version-bearing claim (``record_id@version``)
  whose issuing version is stored, and a reason of at least 10 characters (DDL
  CHECK). ``at`` is stored as ``revoked_at``. ``superseded_version`` is the
  version named by the claim's ``supersedes`` (``target@N``), else the target's
  latest stored version. Retraction ids are ``uuid5(target|claim)``, the same
  scheme the migrator uses, so a repeat retraction is a no-op returning False.

Column mapping (CanonicalRecord field -> storage)
-------------------------------------------------
Columns: ``record_id``, ``version``, ``record_type``, ``admission_year``,
``admission_round``, ``status``, ``authority_level``, ``effective_from``,
``effective_to``, ``text``, ``major_codes``, ``admission_methods``,
``source_document_id``, ``source_url``, ``source_page``, ``supersedes``,
``authored_by``, ``published_at``. ``coordinate_key`` = ``record_id``;
``scope`` / ``subject`` are ``''`` (the model has no such fields);
``archive_reason`` is derived from status.

Fields with no column, and the sentinel-backed NULLs, live in
``structured_payload`` (JSONB):
- ``schema_version``: the model's schema version.
- ``extra_fields``: the model's extra (unknown) keys, restored verbatim.
- ``null_fields``: names of ``admission_round`` / ``authored_by`` that were
  ``None``. Their columns are NOT NULL, so they hold ``'all'`` / ``''`` and this
  list restores ``None`` on read.
- ``published_at_iso``: the exact ISO-8601 published timestamp, so naive and
  aware datetimes round-trip unchanged.

Migrated rows (``scripts/migrate_file_stores.py``) keep their own payload shape,
which is read as the exact inverse: ``legacy_extra`` keys are restored as
top-level fields, and ``values`` is restored only when non-empty (the migrator
writes ``{}`` as a default). ``legacy_record_id``, ``blockers`` and
``legacy_source_document_id`` are migration markers and are not restored.
Migrator defaults that are not invertible are not restored either:
``admission_round=None`` reads back as ``'all'`` and ``authored_by=None`` as
``'legacy-migration'``.
"""
from __future__ import annotations

import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.domain.canonical import CanonicalRecord, RecordStatus
from src.infrastructure.canonical_record_store import (
    RecordNotFoundError,
    RecordProvenance,
    RecordStoreError,
    RecordVersionChain,
    RecordVersionConflictError,
    _named_record_id,
)

_COLUMNS = (
    "record_id, version, record_type, admission_year, admission_round, status, "
    "authority_level, effective_from, effective_to, text, structured_payload, "
    "major_codes, admission_methods, source_document_id, source_url, source_page, "
    "supersedes, authored_by, published_at"
)

#: Every stored version, grouped by record_id in the same codepoint order as record_ids().
_ALL_VERSIONS_SQL = (
    "SELECT " + _COLUMNS + ' FROM canonical_records ORDER BY record_id COLLATE "C", version'
)

_INSERT = """
INSERT INTO canonical_records (
    record_id, version, coordinate_key, record_type, scope, subject, admission_year,
    admission_round, status, archive_reason, authority_level, effective_from,
    effective_to, text, structured_payload, major_codes, admission_methods,
    source_document_id, source_url, source_page, supersedes, authored_by, published_at
) VALUES (
    %(record_id)s, %(version)s, %(record_id)s, %(record_type)s, '', '', %(admission_year)s,
    %(admission_round)s, %(status)s, %(archive_reason)s, %(authority_level)s, %(effective_from)s,
    %(effective_to)s, %(text)s, %(structured_payload)s, %(major_codes)s, %(admission_methods)s,
    %(source_document_id)s, %(source_url)s, %(source_page)s, %(supersedes)s, %(authored_by)s,
    %(published_at)s
)
"""

_ARCHIVE_PREVIOUS = (
    "UPDATE canonical_records SET status = 'archived', archive_reason = 'superseded', "
    "updated_at = now() WHERE record_id = %s AND status = 'published'"
)
_ARCHIVE_OPEN = (
    "UPDATE canonical_records SET status = 'archived', archive_reason = 'superseded', "
    "updated_at = now() WHERE record_id = %s AND status IN ('draft', 'review', 'approved')"
)

_RETRACT = """
INSERT INTO supersession_retractions (
    retraction_id, target_record_id, superseded_version, superseding_record_id,
    superseding_version, revoked_by, revoked_at, reason
) VALUES (
    %(retraction_id)s, %(target)s, %(superseded_version)s, %(superseding_id)s,
    %(superseding_version)s, %(revoked_by)s, %(revoked_at)s, %(reason)s
) ON CONFLICT (retraction_id) DO NOTHING
"""


def _conninfo(url: str) -> str:
    """SQLAlchemy-style URL (``postgresql+asyncpg://`` / ``+psycopg``) -> libpq conninfo.

    asyncpg's ``ssl=`` query parameter is libpq's ``sslmode=``; psycopg rejects ``ssl``.
    """
    url = re.sub(r"^(postgresql)\+\w+://", lambda m: m.group(1) + "://", url)
    return re.sub(r"([?&])ssl=", r"\1sslmode=", url)

#: Client-side bounds on every connection the SQL stores open. These are libpq
#: options kept out of the startup packet, so a transaction-mode pooler (PgBouncer)
#: accepts them. TCP keepalive makes a silently dead socket fail instead of hanging
#: the thread in a read.
CONNECT_KWARGS: dict[str, int] = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}

#: Server-side bound on each statement, lock waits included. It is applied with
#: SET LOCAL inside the caller's transaction, never as a startup ``options``
#: parameter, which a transaction-mode pooler may refuse.
STATEMENT_TIMEOUT_MS = 30_000


def apply_statement_timeout(conn: psycopg.Connection) -> None:
    """Bound the statements of the transaction ``conn`` is about to run."""
    conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'")


def _version_numbers(conn, record_id: str) -> tuple[int, ...]:
    rows = conn.execute(
        "SELECT version FROM canonical_records WHERE record_id = %s ORDER BY version",
        (record_id,),
    ).fetchall()
    return tuple(row["version"] for row in rows)


def _all_records(conn, record_id: str) -> list[CanonicalRecord]:
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM canonical_records WHERE record_id = %s ORDER BY version",
        (record_id,),
    ).fetchall()
    return [_from_row(row) for row in rows]


def _published(conn) -> list[CanonicalRecord]:
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM canonical_records WHERE status = 'published'"
    ).fetchall()
    return [_from_row(row) for row in rows]


def _retracted_keys(conn, target: str) -> frozenset[str]:
    rows = conn.execute(
        "SELECT superseding_record_id, superseding_version FROM supersession_retractions "
        "WHERE target_record_id = %s",
        (target,),
    ).fetchall()
    return frozenset(f"{row['superseding_record_id']}@{row['superseding_version']}" for row in rows)


def _require_record(conn, record_id: str) -> None:
    if not _version_numbers(conn, record_id):
        raise RecordNotFoundError(f"no versions stored for record_id {record_id!r}")


def _from_row(row: dict) -> CanonicalRecord:
    payload = dict(row["structured_payload"] or {})
    nulls = set(payload.get("null_fields") or [])
    data: dict = dict(payload.get("legacy_extra") or {})  # migrated rows: the migrator's inverse
    if payload.get("values"):
        data["values"] = payload["values"]
    data.update(payload.get("extra_fields") or {})
    if "schema_version" in payload:
        data["schema_version"] = payload["schema_version"]
    published = row["published_at"]
    if "published_at_iso" in payload:
        published = datetime.fromisoformat(payload["published_at_iso"])
    data.update({
        "record_id": row["record_id"],
        "record_type": row["record_type"],
        "version": row["version"],
        "admission_year": row["admission_year"],
        "admission_round": None if "admission_round" in nulls else row["admission_round"],
        "status": row["status"],
        "authority_level": row["authority_level"],
        "effective_from": row["effective_from"],
        "effective_to": row["effective_to"],
        "text": row["text"],
        "major_codes": row["major_codes"],
        "admission_methods": row["admission_methods"],
        "source_document_id": row["source_document_id"],
        "source_url": row["source_url"],
        "source_page": row["source_page"],
        "supersedes": row["supersedes"],
        "authored_by": None if "authored_by" in nulls else row["authored_by"],
        "published_at": published,
    })
    return CanonicalRecord.model_validate(data)


def _to_params(record: CanonicalRecord) -> dict:
    payload: dict = {
        "schema_version": record.schema_version,
        "extra_fields": dict(record.model_extra or {}),
    }
    nulls = [name for name in ("admission_round", "authored_by") if getattr(record, name) is None]
    if nulls:
        payload["null_fields"] = nulls
    if record.published_at is not None:
        payload["published_at_iso"] = record.published_at.isoformat()
    return {
        "record_id": record.record_id,
        "version": record.version,
        "record_type": record.record_type.value,
        "admission_year": record.admission_year,
        "admission_round": record.admission_round if record.admission_round is not None else "all",
        "status": record.status.value,
        "archive_reason": "archived" if record.status is RecordStatus.ARCHIVED else None,
        "authority_level": record.authority_level.value,
        "effective_from": record.effective_from,
        "effective_to": record.effective_to,
        "text": record.text,
        "structured_payload": Jsonb(payload),
        "major_codes": Jsonb(list(record.major_codes)),
        "admission_methods": Jsonb(list(record.admission_methods)),
        "source_document_id": record.source_document_id,
        "source_url": record.source_url,
        "source_page": record.source_page,
        "supersedes": record.supersedes,
        "authored_by": record.authored_by if record.authored_by is not None else "",
        "published_at": record.published_at,
    }


def _claim_parts(key: str) -> tuple[str, int]:
    head, sep, tail = key.rpartition("@")
    if not sep or not head or not tail.isdigit():
        raise ValueError(f"not a record version key: {key!r} (expected 'record_id@version')")
    return head, int(tail)


class SourceDocumentNotRegisteredError(RecordStoreError):
    """The record cites a source document that the SQL source_documents table does not hold."""


class RecordStoreUnavailableError(RecordStoreError):
    """The PostgreSQL record store could not be reached."""


class SqlRecordStore:
    """PostgreSQL-backed :class:`CanonicalRecordStore` (same method surface)."""

    def __init__(self, database_url: str | None = None) -> None:
        url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        if not url:
            raise ValueError("SqlRecordStore needs a database_url or DATABASE_URL")
        self._conninfo = _conninfo(url)

    @contextmanager
    def _connect(self):
        # Errors carry no host, user, password or SQL text: the caller never sees the URL.
        try:
            conn = psycopg.connect(self._conninfo, row_factory=dict_row, **CONNECT_KWARGS)
        except psycopg.OperationalError:
            raise RecordStoreUnavailableError("record store database unavailable") from None
        with conn:
            apply_statement_timeout(conn)
            yield conn

    def write_record(self, record: CanonicalRecord) -> CanonicalRecord:
        """Append ``record`` as the next version of its ``record_id``; returns it.

        Earlier open versions of the record are archived as superseded, and
        publishing also archives the previous published version, all in the same
        transaction. Any failure rolls the whole write back.
        """
        with self._connect() as conn:
            existing = _version_numbers(conn, record.record_id)
            expected = existing[-1] + 1 if existing else 1
            if record.version != expected:
                raise RecordVersionConflictError(
                    f"cannot write {record.record_id}@{record.version}: "
                    f"version {expected} is next (stored versions: {existing or 'none'}) "
                    "— a stored version is never overwritten"
                )
            if record.source_document_id is not None:
                found = conn.execute(
                    "SELECT 1 FROM source_documents WHERE document_id = %s",
                    (record.source_document_id,),
                ).fetchone()
                if found is None:
                    raise SourceDocumentNotRegisteredError(
                        f"cannot write {record.record_id}@{record.version}: "
                        "source_document_id is not registered in source_documents"
                    )
            try:
                conn.execute(_ARCHIVE_OPEN, (record.record_id,))
                if record.status is RecordStatus.PUBLISHED:
                    conn.execute(_ARCHIVE_PREVIOUS, (record.record_id,))
                conn.execute(_INSERT, _to_params(record))
            except psycopg.errors.UniqueViolation as exc:
                raise RecordVersionConflictError(
                    f"cannot write {record.record_id}@{record.version}: "
                    f"unique constraint {exc.diag.constraint_name or '-'} (sqlstate {exc.diag.sqlstate or '-'})"
                ) from None
            except psycopg.OperationalError:
                raise RecordStoreUnavailableError("record store database unavailable") from None
            except psycopg.Error as exc:
                raise RecordStoreError(
                    f"cannot write {record.record_id}@{record.version}: rejected by PostgreSQL "
                    f"(sqlstate {exc.diag.sqlstate or '-'}, constraint {exc.diag.constraint_name or '-'})"
                ) from None
        return record

    def edit_record(self, record_id: str, **changes: object) -> CanonicalRecord:
        """Append a new version of ``record_id`` with ``changes`` applied."""
        if "record_id" in changes:
            raise ValueError("edit_record cannot change record_id")
        latest = self.get_record(record_id).latest
        payload = latest.model_dump()
        payload.update(changes)
        payload["version"] = self.next_version(record_id)
        return self.write_record(CanonicalRecord.model_validate(payload))

    def next_version(self, record_id: str) -> int:
        """Version a new write would take: one past the newest, else 1."""
        existing = self.versions(record_id)
        return existing[-1] + 1 if existing else 1

    def versions(self, record_id: str) -> tuple[int, ...]:
        """Stored version numbers for ``record_id``, ascending (empty if none)."""
        with self._connect() as conn:
            return _version_numbers(conn, record_id)

    def get_version(self, record_id: str, version: int) -> CanonicalRecord:
        """Read one stored version."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM canonical_records WHERE record_id = %s AND version = %s",
                (record_id, version),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"no such record version: {record_id}@{version}")
        return _from_row(row)

    def stored_history(self) -> dict[str, dict[int, CanonicalRecord]]:
        """Every stored version of every record, read in one query on one connection.

        Keyed by ``record_id`` in :meth:`record_ids` order; each value maps version to
        record in ascending version order. The materializer reads this once rather than
        opening a connection per record.
        """
        with self._connect() as conn:
            rows = conn.execute(_ALL_VERSIONS_SQL).fetchall()
        history: dict[str, dict[int, CanonicalRecord]] = {}
        for row in rows:
            history.setdefault(row["record_id"], {})[row["version"]] = _from_row(row)
        return history

    def get_record(self, record_id: str) -> RecordVersionChain:
        """The latest version of ``record_id`` plus every version before it."""
        with self._connect() as conn:
            records = _all_records(conn, record_id)
        if not records:
            raise RecordNotFoundError(f"no versions stored for record_id {record_id!r}")
        return RecordVersionChain(
            record_id=record_id,
            latest=records[-1],
            prior=tuple(reversed(records[:-1])),
        )

    def record_ids(self) -> tuple[str, ...]:
        """Every ``record_id`` with at least one version, in Python (codepoint) order."""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT record_id FROM canonical_records GROUP BY record_id '
                'ORDER BY record_id COLLATE "C"'
            ).fetchall()
        return tuple(row["record_id"] for row in rows)

    def superseding_record(self, record_id: str) -> CanonicalRecord | None:
        """The published version whose ``supersedes`` names ``record_id``, or None."""
        with self._connect() as conn:
            _require_record(conn, record_id)
            return _supersession_index(conn).get(record_id)

    def is_superseded(self, record_id: str) -> bool:
        return self.superseding_record(record_id) is not None

    def supersession_index(self) -> dict[str, CanonicalRecord]:
        """Whole-store reverse index: named ``record_id`` -> superseding version."""
        with self._connect() as conn:
            return _supersession_index(conn)

    def superseding_versions(self, record_id: str) -> tuple[CanonicalRecord, ...]:
        """Every active published version naming ``record_id``, winner-first."""
        with self._connect() as conn:
            _require_record(conn, record_id)
            retracted = _retracted_keys(conn, record_id)
            candidates = [
                candidate
                for candidate in _published(conn)
                if _named_record_id(candidate.supersedes) == record_id
                and candidate.record_id != record_id
                and candidate.record_version_key not in retracted
            ]
        candidates.sort(key=lambda r: (r.version, r.record_id), reverse=True)
        return tuple(candidates)

    def retract_supersession(
        self,
        target_record_id: str,
        superseding_version_key: str,
        *,
        reason: str,
        retracted_by: str,
        at: str,
    ) -> bool:
        """Mark one supersession claim as revoked. Returns False if already marked."""
        claim_id, claim_version = _claim_parts(superseding_version_key)
        if len(reason) < 10:
            raise RecordStoreError("retraction reason must be at least 10 characters")
        revoked_at = datetime.fromisoformat(at)
        if revoked_at.tzinfo is None:
            revoked_at = revoked_at.replace(tzinfo=timezone.utc)
        with self._connect() as conn:
            issuing = conn.execute(
                "SELECT supersedes FROM canonical_records WHERE record_id = %s AND version = %s",
                (claim_id, claim_version),
            ).fetchone()
            if issuing is None:
                raise RecordNotFoundError(f"no such record version: {superseding_version_key}")
            named = _claim_parts(issuing["supersedes"]) if _has_version(issuing["supersedes"]) else None
            if named is not None:
                superseded_version = named[1]
            else:
                stored = _version_numbers(conn, target_record_id)
                if not stored:
                    raise RecordNotFoundError(f"no versions stored for record_id {target_record_id!r}")
                superseded_version = stored[-1]
            result = conn.execute(_RETRACT, {
                "retraction_id": uuid.uuid5(uuid.NAMESPACE_URL, f"{target_record_id}|{superseding_version_key}"),
                "target": target_record_id,
                "superseded_version": superseded_version,
                "superseding_id": claim_id,
                "superseding_version": claim_version,
                "revoked_by": retracted_by,
                "revoked_at": revoked_at,
                "reason": reason,
            })
        return result.rowcount == 1

    def retracted_supersession_keys(self, target_record_id: str) -> frozenset[str]:
        """Claim keys (``record_id@version``) retracted against ``target_record_id``."""
        with self._connect() as conn:
            return _retracted_keys(conn, target_record_id)

    def resolve_provenance(self, record_version_key: str) -> RecordProvenance:
        """Source document and page of one stored version."""
        record_id, sep, version_text = record_version_key.rpartition("@")
        if not sep or not record_id or not version_text.isdigit():
            raise ValueError(
                f"not a record version key: {record_version_key!r} "
                "(expected 'record_id@version')"
            )
        record = self.get_version(record_id, int(version_text))
        return RecordProvenance(
            record_id=record_id,
            version=record.version,
            source_document_id=record.source_document_id,
            source_url=record.source_url,
            source_page=record.source_page,
        )


def _has_version(reference: str | None) -> bool:
    if not reference:
        return False
    head, sep, tail = reference.rpartition("@")
    return bool(sep and head and tail.isdigit())


def _supersession_index(conn) -> dict[str, CanonicalRecord]:
    """The file store's winner rule over published rows (see CanonicalRecordStore.supersession_index)."""
    index: dict[str, CanonicalRecord] = {}
    retracted: dict[str, frozenset[str]] = {}
    for candidate in _published(conn):
        target = _named_record_id(candidate.supersedes)
        if target is None or target == candidate.record_id:
            continue
        if target not in retracted:
            retracted[target] = _retracted_keys(conn, target)
        if candidate.record_version_key in retracted[target]:
            continue
        current = index.get(target)
        if current is None or (candidate.version, candidate.record_id) > (
            current.version,
            current.record_id,
        ):
            index[target] = candidate
    return index
