"""Sync PostgreSQL audit ledger over ``audit_events``, with the AuditLedger surface.

Same public methods as :class:`src.infrastructure.audit_ledger.AuditLedger`:
``append``, ``record_permission_change``, ``entries``, ``get``, ``find`` and
``verify``. Writes use the chain format of ``sql_audit_log.append_audit_event``,
so rows from the async writer and from this class interleave in one chain.

Column mapping (AuditEvent -> audit_events)
-------------------------------------------
- ``target_kind`` -> ``aggregate_type``, ``target`` -> ``aggregate_id``,
  ``timestamp`` -> ``created_at`` (naive is taken as UTC), ``action`` -> ``action``,
  ``event_id`` -> ``event_id`` (uuid), ``release_id``, ``reason``.
- ``role`` None -> ``'unknown'`` (the column is NOT NULL).
- ``before_version`` / ``after_version``: digit-only strings (no leading zero, int4
  range) become the int columns; any other version string stays NULL in the column
  and is kept in ``after`` as ``{"audit_extras": {...}}``.
- Fields with no column (``entity``, ``source_ip``, ``review_id``,
  ``schema_version``, non-numeric versions) are kept in that same ``audit_extras``
  object, only when they differ from the default. The object is in ``after``, so
  the entry hash covers it.
- ``before`` and ``after`` are stored as JSON (``null`` when empty), exactly as
  ``append_audit_event`` hashes them.

Rows migrated from the file ledger carry ``request_id == LEGACY_REQUEST_ID``. Their
hash format differs and some event fields were lost in migration, so :meth:`verify`
checks only their linkage and ordering. Legacy rows must form a leading run.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.domain.audit import SCHEMA_VERSION, AuditAction, AuditEvent
from src.domain.canonical.lifecycle import ENTITY_NAME
from src.infrastructure.audit_ledger import (
    GENESIS_HASH,
    AuditEntry,
    AuditLedger,
    AuditLedgerError,
    ChainVerification,
)
from src.infrastructure.sql_audit_log import canonical_payload, compute_entry_hash
from src.infrastructure.sql_record_store import (
    CONNECT_KWARGS,
    _conninfo,
    apply_statement_timeout,
)

# Marker the migrator writes on every row it loads from the file ledger.
LEGACY_REQUEST_ID = "legacy:file-ledger"

_INT4_MAX = 2_147_483_647
_INT_VERSION = re.compile(r"0|[1-9][0-9]*")

_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext('audit_chain'))"
_TAIL_SQL = "SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
_INSERT_SQL = """
INSERT INTO audit_events (event_id, aggregate_type, aggregate_id, actor_id, role, action, before, after,
    reason, before_version, after_version, release_id, request_id, created_at, prev_hash, entry_hash)
VALUES (%(event_id)s, %(aggregate_type)s, %(aggregate_id)s, %(actor_id)s, %(role)s, %(action)s,
    CAST(%(before)s AS jsonb), CAST(%(after)s AS jsonb), %(reason)s, %(before_version)s, %(after_version)s,
    %(release_id)s, %(request_id)s, %(created_at)s, %(prev_hash)s, %(entry_hash)s)
RETURNING seq
"""
_SELECT_SQL = (
    "SELECT seq, event_id, aggregate_type, aggregate_id, actor_id, role, action, before, after, "
    "reason, before_version, after_version, release_id, request_id, created_at, prev_hash, entry_hash "
    "FROM audit_events ORDER BY seq"
)


def _int_version(value: str | None) -> int | None:
    if value is None or not _INT_VERSION.fullmatch(value):
        return None
    number = int(value)
    return number if number <= _INT4_MAX else None


def _extras(event: AuditEvent) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if event.entity != ENTITY_NAME:
        extras["entity"] = event.entity
    if event.source_ip is not None:
        extras["source_ip"] = event.source_ip
    if event.review_id is not None:
        extras["review_id"] = event.review_id
    if event.schema_version != SCHEMA_VERSION:
        extras["schema_version"] = event.schema_version
    if event.before_version is not None and _int_version(event.before_version) is None:
        extras["before_version"] = event.before_version
    if event.after_version is not None and _int_version(event.after_version) is None:
        extras["after_version"] = event.after_version
    return extras


def _columns(event: AuditEvent) -> dict[str, Any]:
    """The audit_events column values for ``event``, before hashing or insert."""
    timestamp = event.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    extras = _extras(event)
    return {
        "event_id": uuid.UUID(event.event_id),
        "aggregate_type": event.target_kind.value,
        "aggregate_id": event.target,
        "actor_id": event.actor_id,
        "role": event.role or "unknown",
        "action": event.action.value,
        "before": None,
        "after": {"audit_extras": extras} if extras else None,
        "reason": event.reason,
        "before_version": _int_version(event.before_version),
        "after_version": _int_version(event.after_version),
        "release_id": event.release_id,
        "request_id": None,
        "created_at": timestamp,
    }


def _version(column: int | None, blob: Any, extras: dict[str, Any], side: str) -> str | None:
    """Invert the version split: int column, then extras, then a migrated ``legacy_version``."""
    if column is not None:
        return str(column)
    if f"{side}_version" in extras:
        return str(extras[f"{side}_version"])
    if isinstance(blob, dict) and "legacy_version" in blob:
        return str(blob["legacy_version"])
    return None


def _event_from_row(row: dict[str, Any]) -> AuditEvent:
    after = row["after"]
    extras = after.get("audit_extras") if isinstance(after, dict) else None
    if not isinstance(extras, dict):
        extras = {}
    return AuditEvent(
        action=row["action"],
        actor_id=row["actor_id"],
        target=row["aggregate_id"],
        timestamp=row["created_at"].astimezone(timezone.utc),
        before_version=_version(row["before_version"], row["before"], extras, "before"),
        after_version=_version(row["after_version"], after, extras, "after"),
        reason=row["reason"],
        role=row["role"],
        target_kind=row["aggregate_type"],
        entity=extras.get("entity", ENTITY_NAME),
        source_ip=extras.get("source_ip"),
        review_id=extras.get("review_id"),
        release_id=row["release_id"],
        event_id=row["event_id"].hex,
        schema_version=extras.get("schema_version", SCHEMA_VERSION),
    )


def _entry_from_row(row: dict[str, Any]) -> AuditEntry:
    try:
        event = _event_from_row(row)
    except (ValueError, TypeError):
        raise AuditLedgerError(f"audit row {row['seq']} holds an invalid event") from None
    prev = row["prev_hash"]
    return AuditEntry(
        seq=int(row["seq"]),
        prev_hash=GENESIS_HASH if prev is None else bytes(prev).hex(),
        entry_hash=bytes(row["entry_hash"]).hex(),
        event=event,
    )


def _broken(
    index: int,
    checked: int,
    legacy_checked: int,
    reason: str,
    last_hash: str | None,
    legacy_last_hash: str | None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "checked": checked,
        "legacy_checked": legacy_checked,
        "broken_index": index,
        "reason": reason,
        "last_hash": last_hash,
        "legacy_last_hash": legacy_last_hash,
    }


def _is_skipped(row: dict[str, Any]) -> bool:
    """A legacy row :meth:`SqlAuditLedger.entries` cannot rebuild as an event."""
    if row["request_id"] != LEGACY_REQUEST_ID:
        return False
    try:
        _entry_from_row(row)
    except AuditLedgerError:
        return True
    return False


def _driver_error(exc: psycopg.Error) -> AuditLedgerError:
    """A driver error as an AuditLedgerError: SQLSTATE and constraint name only, never row content."""
    if isinstance(exc, psycopg.errors.LockNotAvailable):
        return AuditLedgerError("audit ledger busy")
    if isinstance(exc, psycopg.OperationalError):
        return AuditLedgerError("audit ledger database unavailable")
    state = exc.sqlstate or "none"
    if isinstance(exc, psycopg.errors.UniqueViolation):
        constraint = exc.diag.constraint_name or "unknown"
        return AuditLedgerError(f"audit row rejected by constraint {constraint} (sqlstate {state})")
    return AuditLedgerError(f"audit ledger database error (sqlstate {state})")


def append_on(conn: psycopg.Connection, event: AuditEvent) -> AuditEntry:
    """Extend the chain with ``event`` inside the caller's transaction, under ``audit_chain``.

    The caller owns the transaction, so a write and its audit row commit or roll back together.
    """
    cols = _columns(event)
    conn.execute(_LOCK_SQL)
    tail = conn.execute(_TAIL_SQL).fetchone()
    prev_hash = bytes(tail["entry_hash"]) if tail is not None else None
    entry_hash = compute_entry_hash(prev_hash, canonical_payload(cols))
    params = dict(
        cols,
        before=json.dumps(cols["before"]),
        after=json.dumps(cols["after"]),
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )
    seq = conn.execute(_INSERT_SQL, params).fetchone()["seq"]
    return AuditEntry(
        seq=int(seq),
        prev_hash=GENESIS_HASH if prev_hash is None else prev_hash.hex(),
        entry_hash=entry_hash.hex(),
        event=event,
    )


class SqlAuditLedger:
    """Audit ledger over the ``audit_events`` table. The constructor never connects.

    ``lock_timeout_ms`` bounds how long an append waits for the ``audit_chain``
    advisory lock; a longer wait raises :class:`AuditLedgerError` ("audit ledger busy").
    """

    def __init__(self, database_url: str | None = None, *, lock_timeout_ms: int = 10_000) -> None:
        url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        if not url:
            raise ValueError("SqlAuditLedger needs a database_url or DATABASE_URL")
        if not isinstance(lock_timeout_ms, int) or isinstance(lock_timeout_ms, bool) or lock_timeout_ms <= 0:
            raise ValueError("lock_timeout_ms must be a positive int number of milliseconds")
        self._conninfo = _conninfo(url)
        self._lock_timeout_ms = lock_timeout_ms

    @contextmanager
    def _connect(self):
        # Errors carry no host, user, password or SQL text: the caller never sees the URL.
        try:
            conn = psycopg.connect(self._conninfo, row_factory=dict_row, **CONNECT_KWARGS)
        except psycopg.Error:
            raise AuditLedgerError("audit ledger database unavailable") from None
        try:
            with conn:
                apply_statement_timeout(conn)
                yield conn
        except psycopg.Error as exc:
            raise _driver_error(exc) from None

    # -- writing -------------------------------------------------------------

    def append(self, event: AuditEvent) -> AuditEntry:
        """Extend the chain with ``event`` in one transaction under the ``audit_chain`` lock."""
        with self._connect() as conn:
            conn.execute(f"SET LOCAL lock_timeout = '{self._lock_timeout_ms}ms'")
            return append_on(conn, event)

    # Same emitter as the file ledger: it touches only ``self.append``.
    record_permission_change = AuditLedger.record_permission_change

    # -- reading -------------------------------------------------------------

    def _fetch(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return conn.execute(_SELECT_SQL).fetchall()

    def _entries_and_skips(self) -> tuple[list[AuditEntry], int]:
        entries: list[AuditEntry] = []
        skipped = 0
        for row in self._fetch():
            try:
                entries.append(_entry_from_row(row))
            except AuditLedgerError:
                if row["request_id"] != LEGACY_REQUEST_ID:
                    raise
                skipped += 1
        return entries, skipped

    def entries(self) -> tuple[AuditEntry, ...]:
        """Every entry, oldest first, in seq order.

        A migrated legacy row whose action or target kind is a migrator placeholder
        (``'legacy-migration'``) cannot be rebuilt as an :class:`AuditEvent`. It is
        skipped here, and in :meth:`get` and :meth:`find`, and counted by
        :meth:`skipped_legacy_rows`. It is not replaced with a guessed enum value.
        Non-legacy rows with an invalid enum raise :class:`AuditLedgerError`.
        """
        return tuple(self._entries_and_skips()[0])

    def skipped_legacy_rows(self) -> int:
        """How many legacy rows :meth:`entries` skips because they cannot be rebuilt."""
        return self._entries_and_skips()[1]

    def get(self, event_id: str) -> AuditEntry | None:
        """The entry whose event has ``event_id``, or ``None``."""
        for entry in self.entries():
            if entry.event.event_id == event_id:
                return entry
        return None

    def find(
        self,
        *,
        target: str | None = None,
        actor_id: str | None = None,
        action: AuditAction | str | None = None,
    ) -> tuple[AuditEntry, ...]:
        """Entries matching every supplied filter, oldest first. An unknown action matches nothing."""
        wanted = None
        if action is not None:
            try:
                wanted = AuditAction(action)
            except (ValueError, TypeError):
                return ()
        found: list[AuditEntry] = []
        for entry in self.entries():
            event = entry.event
            if target is not None and event.target != target:
                continue
            if actor_id is not None and event.actor_id != actor_id:
                continue
            if wanted is not None and event.action is not wanted:
                continue
            found.append(entry)
        return tuple(found)

    # -- verification --------------------------------------------------------

    def verify_detailed(self) -> dict[str, Any]:
        """Walk the chain in seq order and report the first break.

        Returns ``ok``, ``checked``, ``legacy_checked``, ``broken_index``
        (0-based, in seq order), ``reason``, ``last_hash``, ``legacy_last_hash``
        (the entry hash of the last legacy row walked, or ``None``) and
        ``skipped_legacy_rows``. Never builds an :class:`AuditEvent`, so a row
        with an unknown enum value is a break, not a crash.
        """
        rows = self._fetch()
        result = self._walk(rows)
        result["skipped_legacy_rows"] = sum(1 for row in rows if _is_skipped(row))
        return result

    def _walk(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        checked = 0
        legacy_checked = 0
        prev: bytes | None = None
        prev_seq: int | None = None
        seen_sql_row = False
        last_hash: str | None = None
        legacy_last_hash: str | None = None

        for index, row in enumerate(rows):
            legacy = row["request_id"] == LEGACY_REQUEST_ID
            if legacy and seen_sql_row:
                return _broken(index, checked, legacy_checked, "legacy row after a non-legacy row", last_hash, legacy_last_hash)
            if not legacy:
                seen_sql_row = True
            seq = int(row["seq"])
            if prev_seq is not None and seq <= prev_seq:
                return _broken(index, checked, legacy_checked, "seq is not strictly increasing", last_hash, legacy_last_hash)
            stored_prev = bytes(row["prev_hash"]) if row["prev_hash"] is not None else None
            if (stored_prev or b"") != (prev or b""):
                return _broken(index, checked, legacy_checked, "prev_hash does not match the preceding entry", last_hash, legacy_last_hash)
            entry_hash = bytes(row["entry_hash"])
            if legacy:
                legacy_checked += 1
                legacy_last_hash = entry_hash.hex()
            else:
                try:
                    expected = compute_entry_hash(prev, canonical_payload(dict(row)))
                except (KeyError, TypeError, ValueError, AttributeError):
                    return _broken(index, checked, legacy_checked, "row content cannot be hashed", last_hash, legacy_last_hash)
                if entry_hash != expected:
                    return _broken(index, checked, legacy_checked, "entry_hash does not match row content", last_hash, legacy_last_hash)
            checked += 1
            prev = entry_hash
            prev_seq = seq
            last_hash = entry_hash.hex()

        return {
            "ok": True,
            "checked": checked,
            "legacy_checked": legacy_checked,
            "broken_index": None,
            "reason": None,
            "last_hash": last_hash,
            "legacy_last_hash": legacy_last_hash,
        }

    def verify(self) -> ChainVerification:
        """The :class:`ChainVerification` subset of :meth:`verify_detailed`."""
        result = self.verify_detailed()
        return ChainVerification(
            ok=result["ok"],
            checked=result["checked"],
            broken_index=result["broken_index"],
            reason=result["reason"],
            last_hash=result["last_hash"],
        )
