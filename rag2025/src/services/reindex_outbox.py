"""Published-version re-index outbox — ticket 15.

What an outbox is for, here
---------------------------
Publishing a version changes retrieval for a *bounded* set of canonical ids:
the record itself, whatever it supersedes, and whatever supersedes it. The
outbox is the durable record of those publications — appended at publish time,
read by the re-index job — so "what changed since the last re-index" is answered
by a file, not by re-scanning the whole store and diffing it.

The second trigger, and why the stamp is on every entry
-------------------------------------------------------
The id set above is a **record-level** trigger. It cannot see a parser or
chunker change: ticket 32/33 altered ``scripts/chunk_table_aware_v2.py`` so that
chunk *metadata* changes for records whose canonical content did not, and the
ids of those chunks do not move. A record-level trigger never fires for that,
and the built corpus drifts away from the code that is running.

So every entry also carries the corpus stamp — ``version_stamp()`` from ticket
06, the single definition of chunking logic — read at the moment of the
publication, never a constant invented here. Ticket 15's re-index job compares
that stamp (and the stamp the corpus itself was built with) against the current
code and forces a full rebuild on a mismatch, with no affected ids needed at
all. An entry that cannot carry a stamp is refused rather than written: a
stamp-less entry is precisely the record that cannot answer "is this corpus
still what the code would produce?".

Storage
-------
Append-only JSONL plus a cursor file. One entry per (release, record version):
``entry_id`` is derived, so re-publishing the same version appends nothing new
— the outbox is idempotent by construction, not by a caller remembering to
check.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from src.domain.canonical.record import CanonicalRecord
from src.infrastructure.canonical_record_store import (
    CanonicalRecordStore,
    _named_record_id,
)
from src.services.chunk_identity import version_stamp

OUTBOX_FILENAME = "reindex_outbox.jsonl"
CURSOR_FILENAME = "reindex_outbox.cursor.json"

Clock = Callable[[], datetime]

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class OutboxError(Exception):
    """Base class for outbox failures."""

class OutboxFormatError(OutboxError):
    """A line in the outbox is not an entry this module wrote.

    Loud on purpose. A skipped entry is an unpublished change nobody re-indexes;
    a skipped *stamp* is the silent divergence this ticket exists to close.
    """

class OutboxAppendError(OutboxError):
    """An entry could not be written. The publish it describes is unaffected.

    Raised, never swallowed. The publish that produced the entry has already
    committed by the time this runs, so the failure is not "the publish failed"
    — it is "the publish succeeded and its re-index trigger is missing". That is
    the one state this ticket exists to make impossible, and the only honest
    response is to say so where the caller can see it.
    """

def affected_record_ids_for(
    record: CanonicalRecord, store: CanonicalRecordStore | None = None
) -> tuple[str, ...]:
    """The canonical ids whose served data a publication of ``record`` changes.

    Three sources, unioned and sorted:

    * the published record's own ``record_id`` — its chunks are re-cut;
    * the record its ``supersedes`` names, spelled as a bare id or a
      ``record_id@version`` key (the naming rule lives in one place,
      ``canonical_record_store._named_record_id``) — that record's served
      chunks stop being the live answer;
    * when a store is given, whatever *published* version supersedes this
      record's id — the relation that decides which of the two is live just
      changed for it as well.

    One hop, not a transitive closure: the supersession relation is a single
    naming step by construction (a record names its replacement directly), and
    a closure over it would let one edit drag an unbounded slice of the store
    into a "scoped" rebuild.
    """
    ids = {record.record_id}
    superseded = _named_record_id(record.supersedes)
    if superseded:
        ids.add(superseded)
    if store is not None:
        superseder = store.supersession_index().get(record.record_id)
        if superseder is not None:
            ids.add(superseder.record_id)
    return tuple(sorted(ids))

@dataclass(frozen=True)
class OutboxEntry:
    """One published version, and everything the re-index job needs from it."""

    entry_id: str
    record_id: str
    version: int
    record_version_key: str
    affected_record_ids: tuple[str, ...]
    stamp: dict
    appended_at: str
    release_id: str | None = None
    action: str = "publish"

    def __post_init__(self) -> None:
        if not self.stamp:
            raise OutboxFormatError(
                f"outbox entry {self.entry_id!r} carries no corpus stamp — an "
                "entry that cannot name the chunking logic it was published "
                "under cannot trigger the staleness re-index"
            )
        if self.record_id not in self.affected_record_ids:
            raise OutboxFormatError(
                f"outbox entry {self.entry_id!r} does not list its own record "
                f"{self.record_id!r} among the affected ids "
                f"{list(self.affected_record_ids)}"
            )

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "release_id": self.release_id,
            "record_id": self.record_id,
            "version": self.version,
            "record_version_key": self.record_version_key,
            "affected_record_ids": list(self.affected_record_ids),
            "stamp": dict(self.stamp),
            "appended_at": self.appended_at,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OutboxEntry":
        """Read back one line written by :meth:`to_dict`."""
        try:
            affected = data["affected_record_ids"]
            stamp = data["stamp"]
            if not isinstance(affected, (list, tuple)) or not isinstance(stamp, dict):
                raise TypeError("affected_record_ids must be a list, stamp a mapping")
            return cls(
                entry_id=str(data["entry_id"]),
                release_id=data.get("release_id"),  # type: ignore[arg-type]
                record_id=str(data["record_id"]),
                version=int(data["version"]),  # type: ignore[arg-type]
                record_version_key=str(data["record_version_key"]),
                affected_record_ids=tuple(str(i) for i in affected),
                stamp={str(k): v for k, v in stamp.items()},
                appended_at=str(data["appended_at"]),
                action=str(data.get("action", "publish")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OutboxFormatError(f"malformed outbox entry: {exc}") from exc

@dataclass(frozen=True)
class OutboxBatch:
    """One drain: the entries consumed and the id set they add up to."""

    entries: tuple[OutboxEntry, ...]
    affected_record_ids: tuple[str, ...]
    cursor: int

    @property
    def stamps(self) -> tuple[dict, ...]:
        """The corpus stamp of each consumed entry, in order."""
        return tuple(entry.stamp for entry in self.entries)

class ReindexOutbox:
    """Append-only outbox of published versions, plus its read cursor.

    ``path`` is the JSONL file itself; the cursor lives beside it. Two instances
    over one path see the same outbox, and :meth:`append` is idempotent on
    ``entry_id`` — so a retried publish cannot queue the same version twice.
    """

    def __init__(self, path: str | Path, *, clock: Clock = _utcnow) -> None:
        self._path = Path(path)
        self._clock = clock

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cursor_path(self) -> Path:
        return self._path.with_name(CURSOR_FILENAME)

    # ── write ──────────────────────────────────────────────────────────────

    def entry(self, record: CanonicalRecord, *, release_id: str | None = None,
              store: CanonicalRecordStore | None = None) -> OutboxEntry:
        """The entry a publication of ``record`` appends — without appending it."""
        return OutboxEntry(
            entry_id=_entry_id(release_id, record.record_version_key),
            release_id=release_id,
            record_id=record.record_id,
            version=record.version,
            record_version_key=record.record_version_key,
            affected_record_ids=affected_record_ids_for(record, store),
            stamp=dict(version_stamp()),
            appended_at=self._clock().astimezone(timezone.utc).isoformat(),
        )

    def append(self, entry: OutboxEntry) -> OutboxEntry:
        """Append ``entry`` unless its id is already present. Returns the stored entry.

        Idempotent on ``entry_id``: publishing the same version into the same
        release twice is one entry, so a re-index never sees the same change
        twice and never rebuilds the same record twice in one drain.

        A write failure raises :class:`OutboxAppendError` naming the entry and
        the ids it would have queued — never a bare ``OSError`` and never a
        swallowed one. The caller (the publish path) must let it propagate: a
        publish whose trigger was not queued is a divergence that only surfaces
        as a stale answer months later.
        """
        existing = {e.entry_id for e in self.entries()}
        if entry.entry_id in existing:
            return entry
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(_dump(entry.to_dict()) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise OutboxAppendError(
                f"could not append outbox entry {entry.entry_id!r} to {self._path}: "
                f"{exc}. The publish it describes is already committed; what is "
                f"missing is its re-index trigger, so {list(entry.affected_record_ids)} "
                "will not be re-indexed until this entry is written. Re-appending the "
                "same version is idempotent, so retrying is safe."
            ) from exc
        return entry

    def record_publication(
        self,
        record: CanonicalRecord,
        *,
        release_id: str | None = None,
        store: CanonicalRecordStore | None = None,
    ) -> OutboxEntry:
        """Append the entry for a just-published version. The publish-path call."""
        return self.append(self.entry(record, release_id=release_id, store=store))

    # ── read ───────────────────────────────────────────────────────────────

    def entries(self) -> tuple[OutboxEntry, ...]:
        """Every entry, oldest first."""
        if not self._path.is_file():
            return ()
        out: list[OutboxEntry] = []
        for lineno, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OutboxFormatError(f"{self._path}:{lineno}: {exc}") from exc
            if not isinstance(payload, dict):
                raise OutboxFormatError(f"{self._path}:{lineno}: not a JSON object")
            try:
                out.append(OutboxEntry.from_dict(payload))
            except OutboxFormatError as exc:
                raise OutboxFormatError(f"{self._path}:{lineno}: {exc}") from exc
        return tuple(out)

    def cursor(self) -> int:
        """How many entries have already been drained."""
        if not self.cursor_path.is_file():
            return 0
        try:
            payload = json.loads(self.cursor_path.read_text(encoding="utf-8"))
            return int(payload.get("cursor", 0))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return 0

    def pending(self) -> tuple[OutboxEntry, ...]:
        """Entries appended since the last drain."""
        return self.entries()[self.cursor() :]

    def affected_record_ids(self) -> tuple[str, ...]:
        """Union of the affected ids over every entry, drained or not."""
        return _union(e.affected_record_ids for e in self.entries())

    def pending_affected_record_ids(self) -> tuple[str, ...]:
        """Union of the affected ids over the entries a drain would consume."""
        return _union(e.affected_record_ids for e in self.pending())

    def drain(self) -> OutboxBatch:
        """Consume every pending entry and advance the cursor past it.

        The cursor moves only after the batch is returned, and it counts entries
        rather than ids, so a drain is repeatable in the sense that matters:
        draining twice returns an empty batch, never the same work twice.
        """
        entries = self.entries()
        start = self.cursor()
        batch = OutboxBatch(
            entries=entries[start:],
            affected_record_ids=_union(e.affected_record_ids for e in entries[start:]),
            cursor=len(entries),
        )
        if batch.cursor != start:
            _write_cursor(self.cursor_path, batch.cursor)
        return batch

def _entry_id(release_id: str | None, record_version_key: str) -> str:
    """``release|record_id@version`` — the same version in the same release is one entry."""
    return f"{release_id or '-'}|{record_version_key}"

def _union(groups) -> tuple[str, ...]:
    merged: set[str] = set()
    for group in groups:
        merged.update(group)
    return tuple(sorted(merged))

def _dump(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

def _write_cursor(path: Path, cursor: int) -> None:
    """Write via a ``.tmp`` sibling + ``os.replace``, as the rest of the repo does.

    A failure raises rather than being swallowed, but it loses no work: the
    entries are already in the outbox and the cursor simply did not move, so the
    next drain sees them again. Re-indexing them twice is a no-op — chunk ids
    are content-addressed — which is why "the cursor is behind" is safe while
    "the cursor ran ahead" would not be.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps({"cursor": cursor}, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except OSError as exc:
        raise OutboxError(
            f"could not advance the re-index cursor to {cursor} at {path}: {exc}. "
            "No work is lost — the entries are still pending and the next drain "
            "returns them again — but the cursor is behind."
        ) from exc
