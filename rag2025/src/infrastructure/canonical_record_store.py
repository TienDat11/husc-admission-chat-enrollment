"""Versioned canonical record store — one admissions fact per version file,
with provenance readback. Ticket 03, see
``.scratch/husc-rag-admin-upgrade/issues/03-canonical-store-provenance.md``.

This is the layer *below* the release layer: ``src/domain/release.py`` +
``src/infrastructure/canonical_store.py`` already own the release manifest and
the atomic ``active_release.json`` pointer, and this module adds the business
data those releases carry. It extends the ticket-02 schema package
(``src/domain/canonical``) — it never redefines a field of it.

Layout::

    <root>/
      records/
        rec-DIEM_CHUAN%3ANGANH%3Acntt%3A2026_r/   <- percent-encoded record_id
          v0001.json                              <- CanonicalRecord, raw UTF-8
          v0002.json

Guarantees
----------
- **Append-only versions.** A version file is written once and never rewritten:
  ``write_record`` refuses any version that is not the next one, so an edit
  appends a new version instead of mutating history (``edit_record`` is the
  append-shaped edit path). Prior versions stay readable forever.
- **Published versions are immutable twice over.** The store refuses to write
  over an existing version, and writing a ``published`` version also makes the
  record's version files read-only on disk by reusing
  ``mark_release_immutable`` — a stray direct write fails at the filesystem,
  not only at the API.
- **Byte-faithful Vietnamese.** Files hold exactly the bytes
  ``to_canonical_json`` produces (``ensure_ascii=False``, no ``\\uXXXX``
  escapes) and are read back with ``from_canonical_json``. Nothing normalizes
  Unicode, so an NFC and an NFD spelling of the same word stay distinct bytes.
- **Provenance is in the record, not beside it.** ``source_document_id`` /
  ``source_url`` / ``source_page`` live in the version file itself, so
  ``resolve_provenance`` reads the same bytes the fact was written with.
- **Supersession is a relation, not a status.** ``RecordStatus`` stays five
  values; a record is superseded iff a *published* version of another record
  names it in ``supersedes``. Only the store can answer that reverse question,
  so ``superseding_record`` / ``is_superseded`` live here. The relation is the
  union over published versions, and an unpublished version never changes the
  answer — in either direction (naming or un-naming). See the method docstrings
  for the determinism and termination rules.
- **A supersession claim can be retracted.** Because published bytes are
  immutable, un-superseding (ticket 36's ``revoke_supersession``) cannot delete
  or rewrite the claim; it appends a retraction marker beside it, and
  ``supersession_index`` skips claimed pairs the marker covers. A retraction is
  bound to the exact version that issued the claim, so a *later* version making
  the same claim is counted again — retracting yesterday's decision never
  silently vetoes tomorrow's.

Single writer per store: the publisher pipeline is single-writer
(``data_lifecycle_plan_v2.md`` §0.2), so two processes racing the same
``record_id`` are out of scope. Lifecycle *transition rules* are ticket 12's;
this module stores whatever state it is handed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from src.domain.canonical import (
    CanonicalRecord,
    RecordStatus,
    from_canonical_json,
    to_canonical_json,
)
from src.infrastructure.canonical_store import mark_release_immutable

RECORDS_DIRNAME = "records"

# Sidecar directory for retracted supersession claims (ticket 36). Kept outside
# ``records/`` on purpose: a record dir is made read-only once a published
# version lands, and a retraction is an *append* of new metadata, never an edit
# of published bytes.
RETRACTIONS_DIRNAME = "supersession_retractions"

# Directory name for a record_id: prefixed so that even a pathological id such
# as ".." cannot escape the records directory, and percent-encoded so "/", "\"
# and ":" are legal on every filesystem. Reversible via ``unquote``.
#
# The trailing suffix is load-bearing on Windows: ``quote`` leaves "." literal,
# and Win32 silently strips a *trailing* dot from a path component, so a
# record_id ending in "." would otherwise land in the same directory as the id
# without it — the collision that silently overwrites history. Ending every
# component with a non-dot terminator makes the encoding injective.
_RECORD_DIR_PREFIX = "rec-"
_RECORD_DIR_SUFFIX = "_r"
_VERSION_FILE_RE = re.compile(r"^v(\d+)\.json$")

class RecordStoreError(Exception):
    """Base class for record-store failures."""

class RecordNotFoundError(RecordStoreError):
    """No stored version matches the requested record_id (+ version)."""

class RecordVersionConflictError(RecordStoreError):
    """A version that is not the next one was requested — history is
    append-only, so an existing version is never overwritten."""

@dataclass(frozen=True)
class RecordVersionChain:
    """One ``record_id`` and its whole version history.

    ``prior`` is newest-first: the version that ``latest`` succeeded comes
    first. A record with a single stored version has an empty ``prior``.
    """

    record_id: str
    latest: CanonicalRecord
    prior: tuple[CanonicalRecord, ...]

@dataclass(frozen=True)
class RecordProvenance:
    """Where a stored record version came from.

    ``source_document_id`` is guaranteed non-empty for a ``published`` version
    (the schema rejects the alternative); it is ``None`` for a draft authored
    before its source document was uploaded.
    """

    record_id: str
    version: int
    source_document_id: str | None
    source_url: str | None
    source_page: int | None

class CanonicalRecordStore:
    """Disk-backed store of :class:`CanonicalRecord` versions.

    ``root`` is the store directory (created on first write). Pass an explicit
    path — nothing here defaults to a production location.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The store directory. Two instances sharing it see the same records."""
        return self._root

    def write_record(self, record: CanonicalRecord) -> CanonicalRecord:
        """Append ``record`` as the next version of its ``record_id``.

        The version must be exactly one past the newest stored one, so a
        re-write of any existing version — published or not — raises
        :class:`RecordVersionConflictError` instead of destroying history.
        Returns the stored record.
        """
        existing = self.versions(record.record_id)
        expected = existing[-1] + 1 if existing else 1
        path = self.version_path(record.record_id, record.version)
        if record.version != expected or path.exists():
            raise RecordVersionConflictError(
                f"cannot write {record.record_id}@{record.version}: "
                f"version {expected} is next (stored versions: {existing or 'none'}) "
                "— a stored version is never overwritten"
            )

        _write_atomic(path, to_canonical_json(record))
        if record.status is RecordStatus.PUBLISHED:
            mark_release_immutable(path.parent)
        return record

    def edit_record(self, record_id: str, **changes: object) -> CanonicalRecord:
        """Append a new version of ``record_id`` with ``changes`` applied.

        Reads the latest version, applies the changes, bumps ``version`` to the
        next one and writes that as a new file. The previous version — including
        a published one — is left byte-identical on disk.

        ``version`` is owned by the store and cannot be set here; ``record_id``
        cannot be changed (that would write a different record).
        """
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
        directory = self._record_dir(record_id)
        if not directory.is_dir():
            return ()
        found = [
            int(match.group(1))
            for match in (_VERSION_FILE_RE.match(p.name) for p in directory.iterdir())
            if match is not None
        ]
        return tuple(sorted(found))

    def get_version(self, record_id: str, version: int) -> CanonicalRecord:
        """Read one stored version back from disk."""
        path = self.version_path(record_id, version)
        if not path.is_file():
            raise RecordNotFoundError(f"no such record version: {record_id}@{version}")
        return from_canonical_json(path.read_bytes())

    def get_record(self, record_id: str) -> RecordVersionChain:
        """The latest version of ``record_id`` plus every version before it."""
        stored = self.versions(record_id)
        if not stored:
            raise RecordNotFoundError(f"no versions stored for record_id {record_id!r}")
        return RecordVersionChain(
            record_id=record_id,
            latest=self.get_version(record_id, stored[-1]),
            prior=tuple(self.get_version(record_id, v) for v in reversed(stored[:-1])),
        )

    def record_ids(self) -> tuple[str, ...]:
        """Every ``record_id`` this store holds at least one version of."""
        directory = self._root / RECORDS_DIRNAME
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                unquote(child.name[len(_RECORD_DIR_PREFIX) : -len(_RECORD_DIR_SUFFIX)])
                for child in directory.iterdir()
                if child.is_dir()
                and child.name.startswith(_RECORD_DIR_PREFIX)
                and child.name.endswith(_RECORD_DIR_SUFFIX)
            )
        )

    def superseding_record(self, record_id: str) -> CanonicalRecord | None:
        """The published version whose ``supersedes`` names ``record_id``.

        A *published* version of another record is the only thing that can
        supersede — a draft, review or approved version naming the old record
        does not retract live admissions data, so it is not a candidate. The
        relation is the union over published versions: an unpublished version
        never changes the answer, whether it adds or removes a reference.

        The name may be spelled either as a bare ``record_id`` or as a full
        ``record_id@version`` key; both name the same record.

        When several versions name it, the highest ``(version, record_id)``
        wins — version first, record_id only to break a tie between two
        different records. The winner's ``record_version_key`` is the answer to
        "which version superseded it".

        A record never supersedes itself (a self-reference is ignored), and
        nothing is traversed transitively: one pass over stored versions, so a
        supersession cycle (A names B, B names A) cannot hang and yields the
        same answer every call. Raises :class:`RecordNotFoundError` if
        ``record_id`` has no stored version — the same strict contract as the
        other lookups in this module, so a typo is never silently read as
        "nothing supersedes this".
        """
        self.get_record(record_id)  # existence check, module-wide convention
        return self.supersession_index().get(record_id)

    def is_superseded(self, record_id: str) -> bool:
        """Whether a published version of another record names ``record_id``.

        The predicate ticket 05 filters on: exactly the records that must not be
        materialized. Same rules and same failure mode as
        :meth:`superseding_record`.
        """
        return self.superseding_record(record_id) is not None

    def supersession_index(self) -> dict[str, CanonicalRecord]:
        """Whole-store reverse index: named ``record_id`` -> superseding version.

        The batch form of :meth:`superseding_record`, and the one place the
        winner rule is implemented. A caller gating a whole release (ticket 05)
        builds this once and looks records up in it, instead of rescanning every
        stored version once per record. Records nobody names are absent.

        Retracted claims (ticket 36's ``retract_supersession``) do not name
        anyone: a published version whose ``record_version_key`` has a
        retraction marker beside it is skipped, so the record it had been
        superseding reappears as live. A *later* version making the same claim
        is not skipped — markers bind to the issuing version, never to the
        target-and-claimant pair.
        """
        index: dict[str, CanonicalRecord] = {}
        retracted: dict[str, frozenset[str]] = {}
        for candidate in self._published_versions():
            target = _named_record_id(candidate.supersedes)
            if target is None or target == candidate.record_id:
                continue
            if target not in retracted:
                retracted[target] = self.retracted_supersession_keys(target)
            if candidate.record_version_key in retracted[target]:
                continue
            current = index.get(target)
            if current is None or (candidate.version, candidate.record_id) > (
                current.version,
                current.record_id,
            ):
                index[target] = candidate
        return index

    def superseding_versions(self, record_id: str) -> tuple[CanonicalRecord, ...]:
        """Every *active* published version naming ``record_id``, winner-first.

        The plural of :meth:`superseding_record`: where the index keeps only
        the winner, this returns every candidate the index would consider —
        published, naming ``record_id``, not itself, and not retracted. Ordered
        exactly as the index's winner rule ranks candidates (version, then
        record_id, descending), so a caller that retracts "all current claims
        against X" retracts in the same deterministic order on every run.
        Raises :class:`RecordNotFoundError` if ``record_id`` has no stored
        version, matching the other reverse-lookup methods.
        """
        self.get_record(record_id)  # existence check, module-wide convention
        retracted = self.retracted_supersession_keys(record_id)
        candidates = [
            candidate
            for candidate in self._published_versions()
            if _named_record_id(candidate.supersedes) == record_id
            and candidate.record_id != record_id
            and candidate.record_version_key not in retracted
        ]
        candidates.sort(
            key=lambda r: (r.version, r.record_id), reverse=True
        )
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
        """Mark one supersession claim as revoked. Returns False if already marked.

        Published bytes are immutable, so un-superseding cannot delete B@2's
        ``supersedes: A`` — it appends a marker beside it instead, under
        ``supersession_retractions/<target>/<claim-key>.json``. Idempotent by
        design: a second revoke of the same claim is a no-op returning False,
        never a rewrite of the first marker's reason or timestamp.

        ``at`` is an ISO-8601 string supplied by the caller (the service's
        clock), keeping this store layer free of the clock the domain owns.
        """
        path = self._retraction_path(target_record_id, superseding_version_key)
        if path.exists():
            return False
        _write_atomic(
            path,
            json.dumps(
                {
                    "target_record_id": target_record_id,
                    "superseding_version_key": superseding_version_key,
                    "reason": reason,
                    "retracted_by": retracted_by,
                    "at": at,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
        )
        return True

    def retracted_supersession_keys(self, target_record_id: str) -> frozenset[str]:
        """Claim keys retracted against ``target_record_id``.

        One file per retracted claim, named after the *issuing version key*
        (percent-encoded, so ``@`` and ``:`` are filesystem-legal). Scanning
        the directory — not a registry file — keeps a retraction append-only
        like everything else in this store.
        """
        directory = self._retraction_dir(target_record_id)
        if not directory.is_dir():
            return frozenset()
        return frozenset(
            unquote(path.stem)
            for path in directory.glob("*.json")
        )

    def _retraction_dir(self, target_record_id: str) -> Path:
        safe_name = _RECORD_DIR_PREFIX + quote(target_record_id, safe="") + _RECORD_DIR_SUFFIX
        return self._root / RETRACTIONS_DIRNAME / safe_name

    def _retraction_path(self, target_record_id: str, claim_key: str) -> Path:
        return self._retraction_dir(target_record_id) / f"{quote(claim_key, safe='')}.json"

    def _published_versions(self) -> tuple[CanonicalRecord, ...]:
        """Every stored version whose status is ``published``, in store order."""
        return tuple(
            record
            for record_id in self.record_ids()
            for version in self.versions(record_id)
            for record in (self.get_version(record_id, version),)
            if record.status is RecordStatus.PUBLISHED
        )

    def resolve_provenance(self, record_version_key: str) -> RecordProvenance:
        """Source document and page of one stored version.

        ``record_version_key`` is the schema's ``record_id@version`` identity
        (``CanonicalRecord.record_version_key``). Raises
        :class:`RecordNotFoundError` if that version is not stored.
        """
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

    def version_path(self, record_id: str, version: int) -> Path:
        """Path of one version file (its bytes are the canonical JSON bytes)."""
        return self._record_dir(record_id) / f"v{version:04d}.json"

    def _record_dir(self, record_id: str) -> Path:
        safe_name = _RECORD_DIR_PREFIX + quote(record_id, safe="") + _RECORD_DIR_SUFFIX
        return self._root / RECORDS_DIRNAME / safe_name

def _named_record_id(reference: str | None) -> str | None:
    """The ``record_id`` a ``supersedes`` value names, or None if it names none.

    Two spellings are accepted, version key first: a full ``record_id@version``
    key names the record before the ``@``; anything else names itself as a bare
    ``record_id``. This is the single naming rule — ``supersession_index`` and
    ``superseding_record`` cannot disagree about what a reference points at.

    Consequence, by design: a record_id whose own text ends in ``@<digits>`` is
    not reachable through this relation, because that spelling is read as the
    version-key form. Canonical ids are slugs (``DIEM_CHUAN:NGANH:cntt:2026``),
    so this cannot bite a real record.
    """
    if not reference:
        return None
    head, sep, tail = reference.rpartition("@")
    if sep and head and tail.isdigit():
        return head
    return reference

def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` at ``path`` via a ``.tmp`` sibling + ``os.replace``.

    Same idiom as ``canonical_store.publish_pointer``: a crash between the two
    steps leaves the destination absent rather than half-written, and an
    existing file is replaced whole.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
