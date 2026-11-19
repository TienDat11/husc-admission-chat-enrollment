"""PostgreSQL-backed admin identities and sessions (U10, PL-4 tables).

Tables (migration 0001 baseline): ``admin_identities`` and ``admin_sessions``.

Rules this module enforces, each in the database transaction that needs it:

- A header token is stored only as ``sha256(token)``; the plaintext is returned
  once, by the caller that created the identity, and never read back.
- A session cookie carries a random token. The row is keyed by a UUID derived
  from ``sha256(token)``, so a database read never yields a usable cookie.
- Five consecutive wrong passwords lock a known actor for fifteen minutes. The
  counter and the lock live in the row, so a restart does not reset them.
- Changing a role or disabling an identity revokes its sessions in the same
  transaction as the change.
- The last active admin cannot be demoted or disabled. The admin set is row-locked
  in a fixed order before the target, so two concurrent demotions serialize.

Connections: one short-lived connection per call, no pool, no import-time or
constructor-time database access (same convention as ``sql_record_store``).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import secrets
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Literal

import psycopg
from psycopg.rows import dict_row

from src.domain.admin_identity import Role
from src.infrastructure.identity_store import (
    IdentityConflictError,
    IdentityEntry,
    IdentityNotFoundError,
    IdentityStoreError,
    hash_password,
    verify_password,
)
from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.infrastructure.audit_ledger import permission_change_event
from src.infrastructure.sql_audit_ledger import append_on

__all__ = [
    "AccountDisabledError",
    "AccountLockedError",
    "IdentityRuleError",
    "LastAdminError",
    "LOGIN_FAIL_LIMIT",
    "LOGIN_LOCK_SECONDS",
    "LoginResult",
    "SESSION_TTL_SECONDS",
    "SessionExpiredError",
    "SqlIdentityRepository",
    "header_token_digest",
    "issue_header_token",
    "session_id_for",
]

LOGIN_FAIL_LIMIT = 5
LOGIN_LOCK_SECONDS = 900
SESSION_TTL_SECONDS = 8 * 3600

_COLUMNS = "subject_id, actor_id, role, disabled, password_hash, last_login_at"


class AccountDisabledError(Exception):
    """The identity is disabled; every credential it holds is refused."""


class AccountLockedError(Exception):
    """The identity is inside its lockout window; ``retry_after`` is in seconds."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"account is locked; try again in {retry_after}s")
        self.retry_after = retry_after


class SessionExpiredError(Exception):
    """The session cookie is unknown, revoked, or past its expiry."""


class IdentityRuleError(Exception):
    """The request breaks an identity rule (no credential left, malformed subject_id)."""


class LastAdminError(Exception):
    """The change would leave the system with no active admin."""


LoginStatus = Literal["ok", "unknown", "disabled", "locked", "bad_password"]


@dataclass(frozen=True)
class LoginResult:
    status: LoginStatus
    entry: IdentityEntry | None = None
    session_token: str | None = None
    retry_after: int = 0


def issue_header_token() -> str:
    """A fresh header token (256 bits, URL-safe). Shown once, then only its digest is kept."""
    return secrets.token_urlsafe(32)


def header_token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def session_id_for(token: str) -> uuid.UUID:
    """The ``admin_sessions`` key for a cookie token: 128 bits of its SHA-256."""
    return uuid.UUID(bytes=hashlib.sha256(token.encode("utf-8")).digest()[:16])


_CONN_ERROR = "identity database unavailable"


def _conninfo(url: str) -> str:
    """SQLAlchemy-style URL (``postgresql+asyncpg://``) -> libpq conninfo."""
    return re.sub(r"^(postgresql)\+\w+://", lambda m: m.group(1) + "://", url)


def _uuid_or_none(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value).strip())
    except ValueError:
        return None


def _entry(row: dict) -> IdentityEntry:
    return IdentityEntry(
        subject_id=str(row["subject_id"]),
        actor_id=row["actor_id"],
        role=Role(row["role"]),
        password_hash=row["password_hash"],
        disabled=row["disabled"],
        last_login_at=row["last_login_at"],
    )


def _remaining(row: dict) -> int:
    until = row["locked_until"]
    if until is None:
        return 0
    left = (until - row["db_now"]).total_seconds()
    return math.ceil(left) if left > 0 else 0


@dataclass(frozen=True)
class PermissionAudit:
    """Who changes an identity and why. The repository writes the audit row in the same transaction."""

    actor_id: str
    actor_role: str | None
    reason: str
    four_eyes: bool | None = None


class SqlIdentityRepository:
    """PostgreSQL-backed identity registry with the same read surface as ``IdentityStore``."""

    def __init__(self, database_url: str | None = None) -> None:
        url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        if not url:
            raise ValueError("SqlIdentityRepository needs a database_url or DATABASE_URL")
        self._conninfo = _conninfo(url)

    @contextmanager
    def _txn(self) -> Iterator[psycopg.Connection]:
        # Errors carry no host, user, password or SQL text: the caller never sees the URL.
        try:
            conn = psycopg.connect(self._conninfo, row_factory=dict_row, connect_timeout=10)
        except psycopg.OperationalError:
            raise IdentityStoreError(_CONN_ERROR) from None
        try:
            with conn:
                yield conn
        except psycopg.OperationalError:
            raise IdentityStoreError(_CONN_ERROR) from None
        finally:
            conn.close()

    # -- reads ---------------------------------------------------------------

    def entries(self) -> list[IdentityEntry]:
        with self._txn() as conn:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM admin_identities ORDER BY actor_id").fetchall()
        return [_entry(row) for row in rows]

    def get(self, subject_id: str) -> IdentityEntry | None:
        key = _uuid_or_none(subject_id)
        if key is None:
            return None
        with self._txn() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM admin_identities WHERE subject_id = %s", (key,)
            ).fetchone()
        return _entry(row) if row else None

    def find_by_actor(self, actor_id: str) -> IdentityEntry | None:
        with self._txn() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM admin_identities WHERE actor_id = %s", (actor_id,)
            ).fetchone()
        return _entry(row) if row else None

    def find(self, credential: str) -> IdentityEntry | None:
        """The identity holding this header token, or None.

        Raises :class:`AccountDisabledError` or :class:`AccountLockedError` when the
        identity is refused; the caller maps those to 403 and 423.
        """
        with self._txn() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS}, locked_until, now() AS db_now FROM admin_identities "
                "WHERE header_token_hash = %s",
                (header_token_digest(credential),),
            ).fetchone()
        if row is None:
            return None
        _refuse_if_inactive(row)
        return _entry(row)

    def resolve_session(self, token: str) -> IdentityEntry:
        """The identity behind a session cookie. Raises when the session or identity is refused."""
        with self._txn() as conn:
            row = conn.execute(
                "SELECT i.subject_id, i.actor_id, i.role, i.disabled, i.password_hash, "
                "i.last_login_at, i.locked_until, s.revoked_at, s.expires_at, now() AS db_now "
                "FROM admin_sessions s JOIN admin_identities i ON i.subject_id = s.subject_id "
                "WHERE s.session_id = %s",
                (session_id_for(token),),
            ).fetchone()
        if row is None or row["revoked_at"] is not None or row["expires_at"] <= row["db_now"]:
            raise SessionExpiredError("invalid or expired session")
        _refuse_if_inactive(row)
        return _entry(row)

    # -- writes --------------------------------------------------------------

    def add(
        self, entry: IdentityEntry, *, created_by: str, audit: PermissionAudit
    ) -> IdentityEntry:
        """Insert one identity. ``entry.credential`` (the header token) is stored as its digest only."""
        if entry.password_hash is None and entry.credential is None:
            raise IdentityRuleError("identity needs a password or a header token")
        digest = header_token_digest(entry.credential) if entry.credential else None
        try:
            with self._txn() as conn:
                conn.execute(
                    "INSERT INTO admin_identities (subject_id, actor_id, role, password_hash, "
                    "header_token_hash, disabled, created_by) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        _required_uuid(entry.subject_id),
                        entry.actor_id,
                        entry.role.value,
                        entry.password_hash,
                        digest,
                        entry.disabled,
                        created_by,
                    ),
                )
                append_on(
                    conn,
                    permission_change_event(
                        actor_id=audit.actor_id,
                        actor_role=audit.actor_role,
                        subject_id=entry.subject_id,
                        before_role=None,
                        after_role=entry.role.value,
                        reason=audit.reason,
                        four_eyes=audit.four_eyes,
                    ),
                )
        except psycopg.errors.UniqueViolation:
            raise IdentityConflictError(
                f"identity {entry.actor_id!r} or subject {entry.subject_id!r} already exists"
            ) from None
        return entry

    def seed(self, entries: Iterable[IdentityEntry]) -> None:
        """Insert bootstrap identities from configuration; existing rows are left alone."""
        with self._txn() as conn:
            for entry in entries:
                conn.execute(
                    "INSERT INTO admin_identities (subject_id, actor_id, role, password_hash, "
                    "header_token_hash, disabled, created_by) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 'env-bootstrap') ON CONFLICT DO NOTHING",
                    (
                        _subject_key(entry.subject_id),
                        entry.actor_id,
                        entry.role.value,
                        entry.password_hash,
                        header_token_digest(entry.credential) if entry.credential else None,
                        entry.disabled,
                    ),
                )

    def update(
        self,
        subject_id: str,
        *,
        role: Role | None = None,
        disabled: bool | None = None,
        password: str | None = None,
        audit: PermissionAudit,
    ) -> IdentityEntry:
        """Change role, disabled, or password. ``password=""`` clears it.

        Revokes the identity's sessions in the same transaction when the role
        changes or the identity is disabled. Raises :class:`LastAdminError` when
        the change would remove the last active admin.
        """
        key = _uuid_or_none(subject_id)
        if key is None:
            raise IdentityNotFoundError(f"unknown subject_id: {subject_id}")
        set_password = password is not None
        new_hash = hash_password(password) if password else None
        with self._txn() as conn:
            # Lock the whole active-admin set in a fixed order, then the target.
            admins = conn.execute(
                "SELECT subject_id FROM admin_identities "
                "WHERE role = 'admin' AND NOT disabled ORDER BY subject_id FOR UPDATE"
            ).fetchall()
            current = conn.execute(
                "SELECT role, disabled, header_token_hash FROM admin_identities "
                "WHERE subject_id = %s FOR UPDATE",
                (key,),
            ).fetchone()
            if current is None:
                raise IdentityNotFoundError(f"unknown subject_id: {subject_id}")

            cur_role = current["role"]
            cur_disabled = current["disabled"]
            new_role = role.value if role is not None else cur_role
            new_disabled = cur_disabled if disabled is None else disabled
            was_active_admin = cur_role == Role.ADMIN.value and not cur_disabled
            still_active_admin = new_role == Role.ADMIN.value and not new_disabled
            if was_active_admin and not still_active_admin and len(admins) <= 1:
                raise LastAdminError("cannot demote or disable the last active admin")
            if password == "" and current["header_token_hash"] is None:
                raise IdentityRuleError("identity needs a password or a header token")

            row = conn.execute(
                "UPDATE admin_identities SET role = %s, disabled = %s, "
                "password_hash = CASE WHEN %s THEN %s ELSE password_hash END, updated_at = now() "
                f"WHERE subject_id = %s RETURNING {_COLUMNS}",
                (new_role, new_disabled, set_password, new_hash, key),
            ).fetchone()
            if new_role != cur_role or (new_disabled and not cur_disabled):
                conn.execute(
                    "UPDATE admin_sessions SET revoked_at = now() "
                    "WHERE subject_id = %s AND revoked_at IS NULL",
                    (key,),
                )
            entry = _entry(row)
            append_on(
                conn,
                permission_change_event(
                    actor_id=audit.actor_id,
                    actor_role=audit.actor_role,
                    subject_id=entry.subject_id,
                    before_role=cur_role,
                    after_role=None if entry.disabled else entry.role.value,
                    reason=audit.reason,
                    four_eyes=audit.four_eyes,
                ),
            )
        return entry

    def login(self, actor_id: str, password: str, *, user_agent: str | None = None) -> LoginResult:
        """Check a password; on success open a session. Counts failures and locks the row."""
        with self._txn() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS}, password_hash, failed_login_count, locked_until, "
                "now() AS db_now FROM admin_identities WHERE actor_id = %s FOR UPDATE",
                (actor_id,),
            ).fetchone()
            if row is None:
                return LoginResult("unknown")
            if row["disabled"]:
                return LoginResult("disabled")
            remaining = _remaining(row)
            if remaining > 0:
                return LoginResult("locked", retry_after=remaining)

            if verify_password(password, row["password_hash"] or ""):
                token = issue_header_token()
                conn.execute(
                    "UPDATE admin_identities SET failed_login_count = 0, locked_until = NULL, "
                    "last_login_at = now() WHERE subject_id = %s",
                    (row["subject_id"],),
                )
                conn.execute(
                    "INSERT INTO admin_sessions (session_id, subject_id, actor_id, role, "
                    "created_at, expires_at, user_agent) "
                    "VALUES (%s, %s, %s, %s, now(), now() + make_interval(secs => %s), %s)",
                    (
                        session_id_for(token),
                        row["subject_id"],
                        row["actor_id"],
                        row["role"],
                        SESSION_TTL_SECONDS,
                        user_agent,
                    ),
                )
                entry = replace(_entry(row), last_login_at=row["db_now"])
                return LoginResult("ok", entry=entry, session_token=token)

            failures = row["failed_login_count"] + 1
            if failures >= LOGIN_FAIL_LIMIT:
                conn.execute(
                    "UPDATE admin_identities SET failed_login_count = 0, "
                    "locked_until = now() + make_interval(secs => %s) WHERE subject_id = %s",
                    (LOGIN_LOCK_SECONDS, row["subject_id"]),
                )
                append_on(
                    conn,
                    AuditEvent(
                        action=AuditAction.ACCOUNT_LOCKED,
                        actor_id=row["actor_id"],
                        role=row["role"],
                        target=str(row["subject_id"]),
                        target_kind=TargetKind.SUBJECT,
                        timestamp=datetime.now(timezone.utc),
                        reason=f"{LOGIN_FAIL_LIMIT} wrong passwords; locked for {LOGIN_LOCK_SECONDS}s",
                    ),
                )
            else:
                conn.execute(
                    "UPDATE admin_identities SET failed_login_count = %s WHERE subject_id = %s",
                    (failures, row["subject_id"]),
                )
            return LoginResult("bad_password")

    def unlock(self, actor_id: str, *, audit: AuditEvent) -> bool:
        """Clear one actor's lockout and record the decision in the same transaction."""
        with self._txn() as conn:
            row = conn.execute(
                "UPDATE admin_identities SET failed_login_count = 0, locked_until = NULL, "
                "updated_at = now() WHERE actor_id = %s RETURNING subject_id",
                (actor_id,),
            ).fetchone()
            if row is None:
                return False
            append_on(conn, audit)
        return True

    def revoke_session(self, token: str) -> None:
        with self._txn() as conn:
            conn.execute(
                "UPDATE admin_sessions SET revoked_at = now() "
                "WHERE session_id = %s AND revoked_at IS NULL",
                (session_id_for(token),),
            )


def _required_uuid(value: str) -> uuid.UUID:
    key = _uuid_or_none(value)
    if key is None:
        raise IdentityRuleError(f"subject_id must be a UUID: {value!r}")
    return key


_LEGACY_SUBJECT_NAMESPACE = "husc-identity:"


def _subject_key(value: str) -> uuid.UUID:
    """A UUID subject as given; any other id gets the uuid5 the file migrator assigns it."""
    key = _uuid_or_none(value)
    return key if key is not None else uuid.uuid5(uuid.NAMESPACE_URL, _LEGACY_SUBJECT_NAMESPACE + value)


def _refuse_if_inactive(row: dict) -> None:
    if row["disabled"]:
        raise AccountDisabledError("admin identity is disabled")
    remaining = _remaining(row)
    if remaining > 0:
        raise AccountLockedError(remaining)
