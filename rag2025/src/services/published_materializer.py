"""Published-only materializer — ticket 05.

The single path from a canonical release to chunking/index input. Everything
that is not a live, published admissions fact stops here, so no draft,
in-review, approved, archived or superseded record can reach a student answer.

The filter chain, in order (each step is one gate in ``_exclusion_reason``):

1. **Live version.** The version of a ``record_id`` that is live is its newest
   *settled* version. A newer ``draft``/``review``/``approved`` version is an
   edit in progress — it does not un-publish the fact it edits, so the published
   version underneath it stays live.
2. **Status.** ``draft`` / ``review`` / ``approved`` are refused. ``approved``
   is refused on its own status, never on the assumption that some later step
   would catch it: it is the status immediately before ``published``, which is
   exactly why relying on a downstream gate for it is a bug waiting to happen.
3. **Temporal policy.** ``archived`` is the terminal state, reached through
   ``supersede``. A current-facts materialization (the default) refuses it; a
   materialization answering a question about a past date admits it, but only
   when the caller names that date and only when the record's effective window
   covers it.
4. **Supersession.** A record named in a *published* record's ``supersedes`` is
   refused. This is a relation derived from the store, not a status: the named
   record is still ``published`` itself until somebody runs the ``supersede``
   action on it, so the relation is the only thing that closes that window —
   and an unpublished version that names (or stops naming) a record changes
   nothing.

Cost: ``CanonicalRecordStore.supersession_index()`` is built **once per call**.
``is_superseded()`` answers the same question by rescanning every stored version
(the index is derived per call), so invoking it per record would make a
materialization O(n^2) in the size of the store. Ticket 05 calls this out
explicitly; ``tests/services/test_published_materializer.py`` pins it by call
count and by a scaling property, not by the shape of this code.

Output is byte-deterministic: two runs over the same store and release produce
byte-identical artifacts, which is what makes "re-materializing creates no
duplicate entries" checkable by comparison rather than by inspection. Nothing in
the manifest is a timestamp, so there is no field that could differ between two
otherwise identical runs.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

from src.domain.canonical import CanonicalRecord, RecordStatus, to_canonical_json
from src.infrastructure.canonical_record_store import CanonicalRecordStore

RECORDS_FILENAME = "materialized_records.jsonl"
MANIFEST_FILENAME = "materialization_manifest.json"
MANIFEST_VERSION = 1

# Fixed reason vocabulary, so the manifest's counts always carry the same keys:
# a consumer can tell "this release had no such record" from "this kind of
# exclusion is not reported at all".
EXCLUSION_REASONS = ("draft", "review", "approved", "archived", "superseded")

# Statuses an edit passes through on its way to published. A version in one of
# them has not replaced anything yet.
_IN_FLIGHT = frozenset(
    {RecordStatus.DRAFT, RecordStatus.REVIEW, RecordStatus.APPROVED}
)

class MaterializationState(str, Enum):
    """What a materialization actually produced.

    ``record_count`` alone cannot tell "the store held nothing to materialize"
    apart from "the release materialized"; an empty JSONL plus a zero is exactly
    what a downstream indexer ingests without complaining. The state names which
    of the three cases it was, so an empty store can never be read as a
    successful materialization of real data.
    """

    MATERIALIZED = "materialized"
    """At least one published record reached the output."""

    NOTHING_TO_MATERIALIZE = "nothing_to_materialize"
    """The store holds no record version at all."""

    NO_PUBLISHED_RECORDS = "no_published_records"
    """Records are stored, but none was admitted — nothing reaches students."""

@dataclass(frozen=True)
class TemporalPolicy:
    """The point in time the materialization is answering questions about.

    ``archived`` is not merely "old": it is the terminal lifecycle state, and a
    record reaches it through ``supersede``. It is therefore withheld by default
    — a release that serves today's applicants must not carry retracted facts.

    A question about a past admission year is the one thing that changes the
    answer, so the policy names that point in time. ``as_of`` set admits an
    archived record whose effective window covers that date — the record was the
    live statement then, which is exactly what the question asks about — and
    still withholds one whose window does not.

    ``None`` (the default) means "the facts that are live now", and there is no
    value of the policy that admits an archived record without naming a date:
    the default cannot drift into serving retracted data.
    """

    as_of: date | None = None

    def to_dict(self) -> dict:
        return {"as_of": self.as_of.isoformat() if self.as_of is not None else None}

    def admits_archived(self, record: CanonicalRecord) -> bool:
        """Whether ``record`` was the live statement at this policy's date."""
        if self.as_of is None:
            return False
        if self.as_of < record.effective_from:
            return False
        return record.effective_to is None or self.as_of <= record.effective_to

@dataclass(frozen=True)
class ExcludedRecord:
    """One stored record withheld from the output, and the gate that stopped it."""

    record_id: str
    version: int
    status: RecordStatus
    reason: str

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "version": self.version,
            "status": self.status.value,
            "reason": self.reason,
        }

@dataclass(frozen=True)
class MaterializationResult:
    """The artifacts one ``materialize`` call wrote, plus what it withheld."""

    release_id: str
    state: MaterializationState
    output_dir: Path
    records_path: Path
    manifest_path: Path
    manifest: dict
    admitted: tuple[CanonicalRecord, ...]
    excluded: tuple[ExcludedRecord, ...]

    @property
    def is_materialized(self) -> bool:
        """Whether any published record reached the output.

        A caller that must not act on an empty release checks this instead of
        reading ``record_count``, which is zero in two different situations.
        """
        return self.state is MaterializationState.MATERIALIZED

class PublishedMaterializer:
    """Turns a canonical release into the records the chunker may index."""

    def __init__(
        self,
        store: CanonicalRecordStore,
        *,
        policy: TemporalPolicy | None = None,
    ) -> None:
        self._store = store
        self._policy = policy if policy is not None else TemporalPolicy()

    @property
    def store(self) -> CanonicalRecordStore:
        return self._store

    def materialize(
        self, release_id: str, output_dir: str | Path
    ) -> MaterializationResult:
        """Write the release's live published records to ``output_dir``.

        ``release_id`` is recorded in the output manifest so the artifacts are
        traceable back to the release they were built from. It is required and
        non-blank: an untraceable materialization is the thing ticket 22's
        rollback would have to guess at.
        """
        release_id = (release_id or "").strip()
        if not release_id:
            raise ValueError(
                "materialize requires a release_id — output that cannot be traced "
                "to a release cannot be rolled back"
            )
        output = Path(output_dir)

        # ONE pass over the stored versions for the whole supersession relation.
        # Per-record `is_superseded()` would rebuild this index for every record.
        supersession = self._store.supersession_index()

        record_ids = self._store.record_ids()
        admitted: list[CanonicalRecord] = []
        excluded: list[ExcludedRecord] = []
        stored = 0
        for record_id in record_ids:
            versions = self._store.versions(record_id)
            if not versions:
                # A record directory can outlive a crash between its creation and
                # the version write. It holds no fact, so it states nothing to
                # materialize — and it must not abort the whole release.
                continue
            stored += 1
            record = self._live_version(record_id, versions)
            reason = self._exclusion_reason(record, supersession)
            if reason is None:
                admitted.append(record)
            else:
                excluded.append(
                    ExcludedRecord(
                        record_id=record.record_id,
                        version=record.version,
                        status=record.status,
                        reason=reason,
                    )
                )

        records_bytes = b"".join(
            to_canonical_json(record) + b"\n" for record in admitted
        )
        manifest = self._manifest(
            release_id=release_id,
            records_bytes=records_bytes,
            admitted=admitted,
            excluded=excluded,
            stored_record_count=stored,
        )

        records_path = output / RECORDS_FILENAME
        manifest_path = output / MANIFEST_FILENAME
        _write_atomic(records_path, records_bytes)
        _write_atomic(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        return MaterializationResult(
            release_id=release_id,
            state=MaterializationState(manifest["materialization_state"]),
            output_dir=output,
            records_path=records_path,
            manifest_path=manifest_path,
            manifest=manifest,
            admitted=tuple(admitted),
            excluded=tuple(excluded),
        )

    # -- the filter chain ----------------------------------------------------

    def _live_version(
        self, record_id: str, versions: tuple[int, ...]
    ) -> CanonicalRecord:
        """The stored version of ``record_id`` that is live.

        The newest *settled* version: ``published`` or ``archived``. A version
        still in flight (``draft``/``review``/``approved``) is an edit that has
        not replaced anything, so it is skipped and the settled version beneath
        it is the record's state. A record that never left flight is judged by
        its newest version — it is a draft, and the status gate says so.
        """
        for version in reversed(versions):
            record = self._store.get_version(record_id, version)
            if record.status not in _IN_FLIGHT:
                return record
        return self._store.get_version(record_id, versions[-1])

    def _exclusion_reason(
        self, record: CanonicalRecord, supersession: dict[str, CanonicalRecord]
    ) -> str | None:
        """Why ``record`` must not be materialized, or None to admit it."""
        status = record.status
        if status is RecordStatus.ARCHIVED:
            if not self._policy.admits_archived(record):
                return RecordStatus.ARCHIVED.value
        elif status is not RecordStatus.PUBLISHED:
            # draft / review / approved — including approved, which is one
            # lifecycle action away from published and must never ride in on
            # that proximity.
            return status.value
        if record.record_id in supersession:
            return "superseded"
        return None

    # -- output --------------------------------------------------------------

    def _manifest(
        self,
        *,
        release_id: str,
        records_bytes: bytes,
        admitted: list[CanonicalRecord],
        excluded: list[ExcludedRecord],
        stored_record_count: int,
    ) -> dict:
        counts = {reason: 0 for reason in EXCLUSION_REASONS}
        for item in excluded:
            counts[item.reason] += 1
        state, warning = self._state(len(admitted), stored_record_count)
        return {
            "manifest_version": MANIFEST_VERSION,
            "release_id": release_id,
            "materialization_state": state.value,
            "warning": warning,
            "temporal_policy": self._policy.to_dict(),
            "record_count": len(admitted),
            "stored_record_count": stored_record_count,
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "records": [record.record_version_key for record in admitted],
            "excluded_counts": counts,
            "excluded": [
                item.to_dict()
                for item in sorted(
                    excluded, key=lambda item: (item.record_id, item.version)
                )
            ],
        }

    @staticmethod
    def _state(admitted: int, stored: int) -> tuple[MaterializationState, str | None]:
        if admitted:
            return MaterializationState.MATERIALIZED, None
        if not stored:
            return (
                MaterializationState.NOTHING_TO_MATERIALIZE,
                "the store holds no record version — nothing was materialized",
            )
        return (
            MaterializationState.NO_PUBLISHED_RECORDS,
            "the store holds records but none is a live published record — "
            "nothing was materialized",
        )

def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` at ``path`` via a ``.tmp`` sibling + ``os.replace``.

    Same idiom as ``canonical_store.publish_pointer``: a crash between the two
    steps leaves the destination absent rather than half-written, and a second
    run replaces the whole file instead of appending to it — which is what keeps
    re-materializing one release from duplicating anything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
