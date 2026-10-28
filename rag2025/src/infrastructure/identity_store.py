"""Persistent admin identity store - ticket 37.

The registry ticket 13 kept in ``ADMIN_IDENTITIES`` was configuration: an
actor existed only while the environment said so, and changing a role meant
editing the env and restarting, which left the audit ledger knowing nothing.
This module moves the same data (credential -> actor + role) into a durable
file under the canonical store root, so a role grant outlives the process that
made it and can be audited.

Shape on disk (``<root>/identity/identities.json``)::

    {"schema_version": "1.0",
     "entries": [{"subject_id": "...", "actor_id": "...", "role": "editor",
                  "credential": "...", "disabled": false}]}

Design notes:

- **One JSON document, written atomically.** The registry is small (an office
  of human actors), and the write path is a mutation through
  ``roles.manage`` - not request traffic. The atomic ``.tmp`` +
  ``os.replace`` idiom is the same one
  ``src.infrastructure.canonical_record_store._write_atomic`` uses, and the
  cross-process lock is the ledger's (``_file_lock`` / ``_process_lock``), so
  two writers cannot interleave a read-modify-write. A lock timeout from the
  ledger helper is re-raised as :class:`IdentityStoreError` so every failure
  this store can produce is one the HTTP seam already maps.
- **``subject_id`` is the key; ``actor_id`` is an attribute.** A subject is
  the person a permission change is about - stable across role edits, and
  what the audit event names as its target. It defaults to ``actor_id``.
- **``find`` never returns early.** Credential comparison mirrors
  :func:`src.config.admin_identity.resolve_identity`: every entry is compared
  with :func:`hmac.compare_digest` and the last match wins, so an unknown
  credential costs the same work regardless of how much of it matches a real
  one.
- **No process-local cache.** ``entries()`` reads the file. A disable must
  take effect on the next request without a restart, so a cache here would
  be a correctness bug, not an optimization.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

from src.domain.admin_identity import Role
from src.infrastructure import audit_ledger

__all__ = [
    "IDENTITIES_DIRNAME",
    "IDENTITIES_FILENAME",
    "SCHEMA_VERSION",
    "IdentityConflictError",
    "IdentityEntry",
    "IdentityNotFoundError",
    "IdentityStore",
    "IdentityStoreError",
    "hash_password",
    "verify_password",
]

SCHEMA_VERSION = "1.0"

IDENTITIES_DIRNAME = "identity"
IDENTITIES_FILENAME = "identities.json"

# scrypt parameters: N=2^14, r=8, p=1. The salt is embedded in the stored
# string so verification needs no side data. maxmem is set explicitly so the
# call does not depend on the OpenSSL build's default.
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_KDFLEN = 64
_SCRYPT_MAXMEM = 32 * 1024 * 1024  # 32 MiB, sufficient for N=16384


def _scrypt(password: str, salt: bytes | None = None) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt or os.urandom(16),
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_KDFLEN,
        maxmem=_SCRYPT_MAXMEM,
    )


def hash_password(password: str) -> str:
    """Hash ``password`` with scrypt; returns ``scrypt:<salt_hex>:<hash_hex>``."""
    salt = os.urandom(16)
    derived = _scrypt(password, salt=salt)
    return f"scrypt:{salt.hex()}:{derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check ``password`` against a stored scrypt hash. Returns False on mismatch."""
    if not stored:
        return False
    parts = stored.split(":", 2)
    if len(parts) != 3 or parts[0] != "scrypt":
        return False
    try:
        salt = bytes.fromhex(parts[1])
        expected = bytes.fromhex(parts[2])
    except ValueError:
        return False
    derived = _scrypt(password, salt=salt)
    return hmac.compare_digest(derived, expected)




class IdentityStoreError(Exception):
    """Base failure for the identity store (corrupt file, bad schema, lock)."""


class IdentityConflictError(IdentityStoreError):
    """A new entry would duplicate an existing subject, actor, or credential."""


class IdentityNotFoundError(IdentityStoreError, KeyError):
    """``update`` named a subject the store does not have."""


@dataclass(frozen=True)
class IdentityEntry:
    """One stored identity: who, under what role, with what credential."""

    subject_id: str
    actor_id: str
    role: Role
    credential: str
    password_hash: str | None = None
    disabled: bool = False


    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "actor_id": self.actor_id,
            "role": self.role.value,
            "credential": self.credential,
            "password_hash": self.password_hash,
            "disabled": self.disabled,
        }

    @classmethod
    def from_dict(cls, data: object) -> "IdentityEntry":
        if not isinstance(data, dict):
            raise IdentityStoreError(
                f"identity entry must be an object, got {type(data).__name__}"
            )
        try:
            role = Role(str(data.get("role")))
        except ValueError as exc:
            raise IdentityStoreError(
                f"unknown role {data.get('role')!r} in identity store"
            ) from exc
        password_hash = data.get("password_hash")
        password_hash = str(password_hash) if password_hash else None
        subject_id = str(data.get("subject_id") or "").strip()
        actor_id = str(data.get("actor_id") or "").strip()
        credential = str(data.get("credential") or "")
        if not subject_id or not actor_id or not credential:
            raise IdentityStoreError(
                "identity entries require subject_id, actor_id and credential"
            )
        return cls(
            subject_id=subject_id,
            actor_id=actor_id,
            role=role,
            credential=credential,
            disabled=bool(data.get("disabled", False)),
            password_hash=password_hash,
        )


class IdentityStore:
    """The file-backed registry of admin identities."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def path(self) -> Path:
        """The JSON document. Two instances over one root see one store."""
        return self._root / IDENTITIES_DIRNAME / IDENTITIES_FILENAME

    # -- reading -------------------------------------------------------------

    def entries(self) -> tuple[IdentityEntry, ...]:
        """Every stored entry, or empty when the file does not exist yet.

        A corrupt file raises rather than reading as empty: an empty read
        would let the bootstrap seed overwrite evidence of tampering with
        whatever the environment currently says.
        """
        if not self.path.exists():
            return ()
        raw = self.path.read_bytes()
        if not raw.strip():
            return ()
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise IdentityStoreError(
                f"{self.path} is not readable JSON: {exc}"
            ) from exc
        if not isinstance(doc, dict):
            raise IdentityStoreError(f"{self.path} must hold a JSON object")
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise IdentityStoreError(
                "unsupported identity schema_version "
                f"{doc.get('schema_version')!r} "
                f"(this build writes {SCHEMA_VERSION!r})"
            )
        raw_entries = doc.get("entries")
        if not isinstance(raw_entries, list):
            raise IdentityStoreError(f"{self.path} requires an 'entries' list")
        return tuple(IdentityEntry.from_dict(item) for item in raw_entries)

    def find(self, credential: str) -> IdentityEntry | None:
        """The entry ``credential`` opens, or ``None``.

        Timing-safe and non-short-circuiting, for the reason
        :func:`src.config.admin_identity.resolve_identity` gives: an early
        return on a match would make "this credential is real" observable in
        the comparison count.
        """
        matched: IdentityEntry | None = None
        for entry in self.entries():
            if hmac.compare_digest(entry.credential, credential):
                matched = entry
        return matched

    def get(self, subject_id: str) -> IdentityEntry | None:
        """The entry whose ``subject_id`` is this, or ``None``."""
        for entry in self.entries():
            if entry.subject_id == subject_id:
                return entry

    def find_by_actor(self, actor_id: str) -> IdentityEntry | None:
        """The entry whose ``actor_id`` is this, or ``None``."""
        for entry in self.entries():
            if entry.actor_id == actor_id:
                return entry
        return None


    def seed(self, entries: Iterable[IdentityEntry]) -> None:
        """Write ``entries`` as the whole store (bootstrap from the env).

        Unconditional replace: the "only when empty" decision belongs to the
        caller (:func:`src.routers.admin_auth.ensure_seeded`), so the guard
        that keeps a restarted environment from overwriting API-granted
        permissions lives in exactly one place.
        """
        with self._locked():
            self._write(entries)

    def add(self, entry: IdentityEntry) -> IdentityEntry:
        """Append one entry; raises on a duplicate subject, actor, or credential."""
        with self._locked():
            current = self.entries()
            for existing in current:
                if existing.subject_id == entry.subject_id:
                    raise IdentityConflictError(
                        f"subject_id {entry.subject_id!r} already exists"
                    )
                if existing.actor_id == entry.actor_id:
                    raise IdentityConflictError(
                        f"actor_id {entry.actor_id!r} already has an identity"
                    )
                if hmac.compare_digest(existing.credential, entry.credential):
                    raise IdentityConflictError(
                        "credential is already registered to another subject"
                    )
            self._write((*current, entry))
        return entry

    def update(
        self,
        subject_id: str,
        *,
        role: Role | None = None,
        disabled: bool | None = None,
        password: str | None = None,
    ) -> IdentityEntry:
        """Apply a partial mutation to one subject and return the new entry.

        ``password`` sets or clears the entry's password hash (``None`` leaves
        it unchanged; an empty string clears it).
        """
        with self._locked():
            current = self.entries()
            index = next(
                (i for i, e in enumerate(current) if e.subject_id == subject_id),
                None,
            )
            if index is None:
                raise IdentityNotFoundError(f"unknown subject_id {subject_id!r}")
            old = current[index]
            new_hash = old.password_hash
            if password is not None:
                new_hash = "" if password == "" else hash_password(password)
            new = IdentityEntry(
                subject_id=old.subject_id,
                actor_id=old.actor_id,
                role=role if role is not None else old.role,
                credential=old.credential,
                disabled=disabled if disabled is not None else old.disabled,
                password_hash=new_hash,
            )
            updated = list(current)
            updated[index] = new
            self._write(updated)
        return new

    # -- internals -------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> "IdentityStore":
        """Serialize a read-modify-write against threads and processes.

        The ledger's helpers, wrapped so their timeout error
        (``AuditLedgerError``) surfaces as this module's own
        :class:`IdentityStoreError` - the HTTP seam maps that to 503 and must
        never see an unmapped exception escape a store call.
        """
        path = self.path
        try:
            with audit_ledger._process_lock(path), audit_ledger._file_lock(path):
                yield self
        except audit_ledger.AuditLedgerError as exc:
            raise IdentityStoreError(str(exc)) from exc

    def _write(self, entries: Iterable[IdentityEntry]) -> None:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "entries": [e.to_dict() for e in entries],
        }
        data = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self.path)
