"""Append-only, hash-chained audit ledger — ticket 14.

The ledger is a JSONL file: one entry per line, each line the canonical JSON of
::

    {"seq": n, "prev_hash": "...", "entry_hash": "...", "event": {...}}

and each entry's ``entry_hash`` is the SHA-256 of the canonical JSON of
``{"seq", "prev_hash", "event"}``. Two consequences, and they are the whole
design:

- **Every entry is bound to its predecessor.** ``prev_hash`` is the previous
  entry's ``entry_hash`` (and :data:`GENESIS_HASH` for the first), and it is
  inside the hashed payload rather than beside it — so editing an entry changes
  its hash, which breaks the next entry's ``prev_hash``, which breaks the one
  after that. :meth:`AuditLedger.verify` walks the file and reports the first
  break by index.
- **There is no update and no delete.** Not a guarded one, not a private one:
  *absent*. The ticket's acceptance is "không tồn tại endpoint xóa hoặc sửa
  event", and the strongest form of that is a class with no method that could
  rewrite a line. A caller who wants to undo a change appends the undo — see
  :meth:`append`, which only ever extends the file.

- **Appends are serialized, not merely fast.** The tail read, the hash, and the
  write are one critical section under a per-path process lock *and* an OS file
  lock, because two writers that both read the same tail append two entries
  claiming the same ``seq`` — and the failure then surfaces as ``verify``
  reporting the ledger tampered with, which is the worst possible lie for an
  audit log to tell. Concurrent publishes therefore queue instead of forking.
  Readers are lock-free and see either the file before a write or the file after
  it, since a write is one ``write()`` of one complete line.

What the chain can and cannot detect, stated plainly because a chain that
overstates itself is worse than none:

- *Detected*: an edited field in any entry (its own ``entry_hash`` no longer
  matches), an edited ``entry_hash``/``prev_hash``/``seq`` (the same, since all
  three are inside the hash), a deleted or inserted middle entry (the link from
  its neighbour no longer matches), a truncated file that lost its tail
  *mid-chain* (the following entry's ``prev_hash`` dangles).
- *Not detected*: deleting the final entry (or entries) — the remaining chain is
  still internally consistent. Detecting that needs an external anchor (a
  signed head published elsewhere), which is out of scope here and is why the
  module exposes :attr:`ChainVerification.last_hash`: a caller that records the
  head hash elsewhere can prove the tail was not dropped.

Reads are strict and diagnostics are lenient, deliberately: :meth:`entries`
raises on a line it cannot parse, because a listing that silently skips a
corrupt line would hide exactly the tampering this file exists to expose, while
:meth:`verify` returns a :class:`ChainVerification` describing the break instead
of raising, because its caller is asking "is this intact?" and needs the answer,
not an exception.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from src.domain.audit import AuditAction, AuditEvent, TargetKind

if os.name == "nt":  # pragma: no cover - platform switch
    import msvcrt
else:  # pragma: no cover - platform switch
    import fcntl

__all__ = [
    "AuditEntry",
    "AuditLedger",
    "AuditLedgerCorruptError",
    "AuditLedgerError",
    "ChainVerification",
    "GENESIS_HASH",
    "LEDGER_FILENAME",
]

# The ``prev_hash`` of the first entry: 64 zeros, the same width as a SHA-256
# hex digest, so the field is always a hash-shaped string.
GENESIS_HASH = "0" * 64

LEDGER_FILENAME = "ledger.jsonl"

# How long an append waits for another writer before giving up, and how often it
# retries. The critical section is one tail read plus one line write, so the
# realistic wait is microseconds; the timeout exists so a stuck holder surfaces
# as an error rather than as a request that never returns.
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 0.005

# One process lock per ledger path, shared by every ``AuditLedger`` over it. A
# lock owned by the instance would not serialize two instances over one file,
# and the router and a CLI both construct their own.
_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()

def _process_lock(path: Path) -> threading.Lock:
    """The process-wide lock for ``path``, created on first use.

    Keyed by the resolved path, so ``audit/ledger.jsonl`` and
    ``./audit/../audit/ledger.jsonl`` share one lock instead of forking the
    chain while each holds what it believes is the only lock.
    """
    key = os.path.normcase(str(Path(path).resolve()))
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock

@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """An exclusive lock another *process* also respects.

    Taken on a sibling ``<name>.lock`` file rather than on the ledger itself:
    the append handle is opened ``ab`` and Windows byte-range locking wants a
    read-write handle, so locking the ledger would mean holding a second handle
    to the file being written.

    Advisory on POSIX (``flock``) and mandatory-but-cooperative on Windows
    (``LockFile`` via ``msvcrt.locking``); both only bind writers that ask for
    the lock, which is every writer this module has.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        _acquire(handle, lock_path)
        try:
            yield
        finally:
            _release(handle)
    finally:
        os.close(handle)

def _acquire(handle: int, lock_path: Path) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            if os.name == "nt":
                os.lseek(handle, 0, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise AuditLedgerError(
                    f"could not acquire the ledger lock {lock_path} within "
                    f"{_LOCK_TIMEOUT_SECONDS:g}s"
                ) from None
            time.sleep(_LOCK_RETRY_SECONDS)

def _release(handle: int) -> None:
    if os.name == "nt":
        os.lseek(handle, 0, os.SEEK_SET)
        msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)


class AuditLedgerError(Exception):
    """Base class for audit ledger failures."""


class AuditLedgerCorruptError(AuditLedgerError):
    """A ledger line could not be read as an entry.

    Raised by the strict readers (:meth:`AuditLedger.entries`,
    :meth:`AuditLedger.append`) rather than skipped: appending to a file whose
    tail is unreadable would extend a chain that no longer means anything.
    """


@dataclass(frozen=True)
class AuditEntry:
    """One event plus its position in the chain."""

    seq: int
    prev_hash: str
    entry_hash: str
    event: AuditEvent

    def to_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "event": self.event.to_dict(),
        }

    def computed_hash(self) -> str:
        """The hash this entry's content implies — equals ``entry_hash`` unless
        the entry (or the hash itself) was edited on disk."""
        return _entry_hash(self.seq, self.prev_hash, self.event)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AuditEntry":
        try:
            seq = int(data["seq"])  # type: ignore[arg-type]
            prev_hash = str(data["prev_hash"])
            entry_hash = str(data["entry_hash"])
            raw_event = data["event"]
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditLedgerCorruptError(f"malformed ledger entry: {exc}") from exc
        if not isinstance(raw_event, Mapping):
            raise AuditLedgerCorruptError("ledger entry 'event' is not an object")
        try:
            event = AuditEvent.from_dict(raw_event)
        except ValueError as exc:
            raise AuditLedgerCorruptError(f"ledger entry holds an invalid event: {exc}") from exc
        return cls(seq=seq, prev_hash=prev_hash, entry_hash=entry_hash, event=event)


@dataclass(frozen=True)
class ChainVerification:
    """The outcome of walking the chain.

    ``broken_index`` is the 0-based line index of the first entry that does not
    verify, or ``None`` when the whole file does. ``checked`` counts the entries
    that verified before the break, so a caller can tell "intact" from "broke at
    the very first line".
    """

    ok: bool
    checked: int
    broken_index: int | None = None
    reason: str | None = None
    last_hash: str | None = None


class AuditLedger:
    """Append-only hash-chained ledger over one file.

    ``path`` is the ledger file (``ledger.jsonl`` by convention); it is created
    on first append, and its parent directory with it. There is no update and no
    delete method — see the module docstring for why absence beats a guard.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The ledger file. Two instances over it see the same entries."""
        return self._path

    # -- writing -------------------------------------------------------------

    def append(self, event: AuditEvent) -> AuditEntry:
        """Extend the chain with ``event`` and return the stored entry.

        The read of the tail, the hash, and the write are **one critical
        section** — held under a process lock and an OS file lock (see
        :func:`_file_lock`). Without it two writers read the same tail and both
        append the same ``seq``, forking the chain into two entries claiming one
        position; ``verify`` then reports the ledger as broken, so the failure
        mode of a concurrent publish is "the audit log says it was tampered
        with". The lock is what makes the append-only guarantee hold under the
        concurrent publish/rollback workload, not just under one worker.

        The write itself is one ``write()`` of one complete line, then
        ``fsync``: a crash leaves either the previous file or the extended one,
        never a half-written entry for the next ``verify`` to trip over. Nothing
        already on disk is touched — an undo is another append, never a rewrite.
        """
        with _process_lock(self._path), _file_lock(self._path):
            seq, prev_hash = self._tail()
            seq += 1
            entry = AuditEntry(
                seq=seq,
                prev_hash=prev_hash,
                entry_hash=_entry_hash(seq, prev_hash, event),
                event=event,
            )
            line = _canonical_json(entry.to_dict()) + b"\n"
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def record_permission_change(
        self,
        *,
        actor_id: str,
        actor_role: str | None,
        subject_id: str,
        before_role: str | None,
        after_role: str | None,
        reason: str,
        source_ip: str | None = None,
        four_eyes: bool | None = None,
    ) -> AuditEntry:
        """Append a ``permission_change`` event for one subject's role move.

        This is the emitter seam for ticket 37's identity mutations: the route
        that grants, edits, or revokes a role calls it and cannot record the
        change without a reason — ``PERMISSION_CHANGE`` is in
        :data:`src.domain.audit.REASON_REQUIRED`, so a blank reason raises
        before anything touches the file.

        The before/after fields carry the subject's **role state**, encoded
        ``subject_id@role`` (``subject_id@none`` for a first grant,
        ``subject_id@disabled`` for a revocation). Record version keys mean
        nothing for a target that is a person, not a record; the same two
        fields answer "từ gì sang gì" for permissions as they do for data.

        Returns the stored entry, so the caller can report the ``seq`` and
        ``event_id`` it recorded.
        """
        return self.append(
            AuditEvent(
                action=AuditAction.PERMISSION_CHANGE,
                actor_id=actor_id,
                role=actor_role,
                target=subject_id,
                target_kind=TargetKind.SUBJECT,
                timestamp=datetime.now(timezone.utc),
                before_version=f"{subject_id}@{before_role or 'none'}",
                after_version=f"{subject_id}@{after_role or 'disabled'}",
                reason=reason,
                source_ip=source_ip,
                review_id=None if four_eyes is None else f"four_eyes={'on' if four_eyes else 'off'}",
            )
        )

    # -- reading -------------------------------------------------------------
    def entries(self) -> tuple[AuditEntry, ...]:
        """Every entry, oldest first, in chain order.

        Strict: a line that is not a valid entry raises
        :class:`AuditLedgerCorruptError`. A listing that quietly dropped one
        would be the place tampering went unnoticed.
        """
        return tuple(self._parse(line, index) for index, line in enumerate(self._lines()))

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
        """Entries matching every supplied filter, oldest first.

        Filters combine with AND, and an unknown ``action`` matches nothing
        rather than raising: this is a read path, and "no such action" and "no
        events of that action" are the same answer to the question asked.
        """
        wanted = None
        if action is not None:
            try:
                wanted = AuditAction(action)
            except (ValueError, TypeError):
                return ()
        found = []
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

    def verify(self) -> ChainVerification:
        """Walk the chain and report the first break.

        Checks, in order, per line: the line parses as an entry, ``seq`` is the
        line's own position, ``prev_hash`` is the predecessor's ``entry_hash``,
        and ``entry_hash`` is the hash the entry's content implies. The first
        line that fails any of those stops the walk and is reported by index —
        a later break is not evidence about an earlier one, so the walk does not
        continue past it.
        """
        expected_prev = GENESIS_HASH
        last_hash: str | None = None
        checked = 0

        for index, line in enumerate(self._lines()):
            if not line.strip():
                return _broken(index, checked, "blank line", last_hash)
            try:
                entry = self._parse(line, index)
            except AuditLedgerCorruptError as exc:
                return _broken(index, checked, str(exc), last_hash)
            if entry.seq != index + 1:
                return _broken(
                    index,
                    checked,
                    f"entry {index + 1} carries seq {entry.seq}",
                    last_hash,
                )
            if entry.prev_hash != expected_prev:
                return _broken(
                    index,
                    checked,
                    "prev_hash does not match the preceding entry",
                    last_hash,
                )
            computed = entry.computed_hash()
            if entry.entry_hash != computed:
                return _broken(
                    index,
                    checked,
                    "entry_hash does not match the entry's content",
                    last_hash,
                )
            expected_prev = entry.entry_hash
            last_hash = entry.entry_hash
            checked += 1

        return ChainVerification(ok=True, checked=checked, last_hash=last_hash)

    # -- internals -----------------------------------------------------------

    def _lines(self) -> tuple[bytes, ...]:
        if not self._path.is_file():
            return ()
        return tuple(self._path.read_bytes().splitlines())

    def _parse(self, line: bytes, index: int) -> AuditEntry:
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditLedgerCorruptError(
                f"ledger line {index + 1} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise AuditLedgerCorruptError(f"ledger line {index + 1} is not a JSON object")
        return AuditEntry.from_dict(payload)

    def _tail(self) -> tuple[int, str]:
        """``(seq, entry_hash)`` of the last stored entry, or the genesis pair.

        Read from the file rather than cached in memory: the file is the ledger,
        and a cache would fork the chain if a second process ever appended.
        """
        raw = _read_last_line(self._path)
        if raw is None:
            return 0, GENESIS_HASH
        try:
            entry = self._parse(raw, -1)
        except AuditLedgerCorruptError as exc:
            raise AuditLedgerCorruptError(
                f"cannot append: the ledger's last line is unreadable ({exc})"
            ) from exc
        return entry.seq, entry.entry_hash


def _broken(
    index: int, checked: int, reason: str, last_hash: str | None
) -> ChainVerification:
    return ChainVerification(
        ok=False,
        checked=checked,
        broken_index=index,
        reason=reason,
        last_hash=last_hash,
    )


def _entry_hash(seq: int, prev_hash: str, event: AuditEvent) -> str:
    """SHA-256 over the canonical JSON of one entry's content.

    ``prev_hash`` and ``seq`` are inside the hashed payload, not merely stored
    next to it: that is what makes the chain a chain (editing a predecessor
    changes this entry's hash) and what makes a rewritten ``seq`` detectable.
    """
    payload = _canonical_json(
        {"seq": seq, "prev_hash": prev_hash, "event": event.to_dict()}
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    """Deterministic UTF-8 bytes: sorted keys, no spaces, no ``\\uXXXX`` escapes.

    Same encoding rules as ``to_canonical_json`` in the canonical record schema,
    so a hash is reproducible across processes and runs, and Vietnamese text is
    hashed as the bytes it is written as.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_last_line(path: Path, *, initial_window: int = 65536) -> bytes | None:
    """The file's last non-empty line, read without parsing the whole file.

    ``None`` for a missing or empty file. The window doubles until it reaches a
    line boundary or the start of the file, so an entry longer than the initial
    window is still read whole rather than truncated into a parse error.
    """
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return None
        window = initial_window
        while True:
            start = max(0, size - window)
            handle.seek(start)
            data = handle.read(size - start).rstrip(b"\r\n")
            if not data:
                if start == 0:
                    return None
                window *= 2
                continue
            index = data.rfind(b"\n")
            if index >= 0:
                return data[index + 1 :]
            if start == 0:
                return data
            window *= 2
