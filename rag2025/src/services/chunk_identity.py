"""Deterministic chunk identity and lineage — ticket 06.

See ``.scratch/husc-rag-admin-upgrade/issues/06-deterministic-chunk-identity.md``.
The unit *definition* lives in ``src/domain/canonical/semantic_units.py``; this
module turns a unit into an id and records where the id came from.

The hash construction
---------------------
::

    chunk_id = sha256(canonical_json({
        "record_version_key": "<record_id>@<version>",
        "unit":               "<record_type>|<field>|<slot>",
        "kind":               "atom" | "text" | "table",
        "text":               " ".join(unit.value.split()),
    })).hexdigest()

Properties this buys, and why each is deliberate:

* **Same input, same id, in any process.** Every input is a string the caller
  passed in; ``json.dumps(..., sort_keys=True, ensure_ascii=False)`` fixes the
  byte encoding of Vietnamese text (no ``\\uXXXX`` escapes, no Unicode
  normalization) and no dict ordering, ``PYTHONHASHSEED``, clock or filesystem
  value can reach the digest.
* **Scoped change.** The four inputs are the version key and *one* unit — never
  a sibling unit, never the corpus. Editing one field of one record therefore
  changes the ids of the units that field feeds, and nothing else.
* **The digest is not truncated.** 64 hex characters; a shorter id would buy
  nothing and only add a collision surface.
* **``CHUNKER_VERSION`` is NOT in the digest.** The ticket defines the id as
  ``hash(canonical_record_version + semantic_unit)``, and that is exactly what
  is hashed. The chunking-logic version travels beside the id, in lineage and in
  the manifest, because it answers a different question: *is this corpus still
  what the current code would produce?* Folding it into the id would make a
  parser change rewrite every id in the corpus, which is the one thing the
  stamp exists to describe rather than to cause. Ticket 15 reads the stamp to
  declare a corpus stale; see :data:`CHUNKER_VERSION`.

The version stamp
-----------------
:data:`CHUNKER_VERSION` identifies the **chunking logic** — the splitter rules in
``semantic_units``, the table-chunker wiring, the id construction above. It is a
hand-written semantic version:

* stable across runs on the same code — it is a literal, not a timestamp and not
  a hash of the output;
* bumped when the logic changes, so a parser change invalidates a corpus whose
  canonical records are unchanged (ticket 15's re-index trigger is keyed on
  affected canonical ids and would otherwise never fire);
* **not** bumped when record content changes — content changes are visible in
  ``record_version_key``, which is already in the id.

:func:`validate_version_stamp` enforces the first of those structurally: a
timestamp or a content digest is rejected at the point it would be written.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping

from src.domain.canonical.record import CanonicalRecord
from src.domain.canonical.semantic_units import UNIT_RULES_VERSION, SemanticUnit

#: Chunking-LOGIC version. Bump on any change to how chunks are produced:
#: the unit rules in ``semantic_units``, the table-chunker reuse in
#: ``canonical_chunk_builder``, or the hash construction in this module.
#:
#: 1.0 — ticket 06 initial: structural units, atom-safe splitting,
#:       sha256(record_version_key + semantic_unit) identity, ticket-32/33
#:       table chunker reused for table content.
CHUNKER_VERSION = "1.0"

#: A hand-written version. Refuses the two shapes the stamp must never take:
#: a date/time ("2026-09-13", "20260913T101500") and a hex content digest.
_VERSION_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")

CHUNK_MANIFEST_VERSION = 1


def chunker_version() -> str:
    """The current chunking-logic version.

    Accessor rather than a bare import so ticket 15 and ticket 25 read the stamp
    through one name, and so a future change (a settings override, a per-run
    pin) has a single seam.
    """
    return CHUNKER_VERSION


def version_stamp() -> dict:
    """The stamp block written into every manifest and every lineage record."""
    return {
        "chunker_version": CHUNKER_VERSION,
        "unit_rules_version": UNIT_RULES_VERSION,
        "chunk_manifest_version": CHUNK_MANIFEST_VERSION,
    }


def validate_version_stamp(value: str) -> str:
    """Return ``value`` if it is a hand-written version, else raise.

    Called on the constant at import time (see the bottom of this module), so a
    stamp that is a timestamp or a content hash fails loudly instead of silently
    making every corpus look stale.
    """
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ValueError(
            f"chunker version {value!r} is not a semantic version — a timestamp "
            "or a content digest cannot answer 'did the chunking logic change?'"
        )
    return value


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace only.

    Deliberately not case-folding and not Unicode-normalizing: two records that
    differ in case or in NFC/NFD spelling are different facts and must keep
    different ids. Whitespace is collapsed because re-wrapping a line is not an
    edit to the fact.
    """
    return " ".join(text.split())


def chunk_id_hash_input(record_version_key: str, unit: SemanticUnit) -> bytes:
    """The exact bytes hashed into ``chunk_id``. Public so it can be asserted on."""
    payload = {
        "record_version_key": record_version_key,
        "unit": unit.key,
        "kind": unit.kind.value,
        "text": normalize_for_hash(unit.value),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def chunk_id_for(record_version_key: str, unit: SemanticUnit) -> str:
    """``sha256(record_version_key + semantic_unit)`` as 64 lowercase hex chars."""
    return hashlib.sha256(chunk_id_hash_input(record_version_key, unit)).hexdigest()


@dataclass(frozen=True)
class ChunkLineage:
    """Where one chunk came from: the record version, the unit, and the page.

    ``record_id`` and ``record_version_key`` are both carried — the key is what
    the id was hashed from, the id is what an admin UI groups by — and
    ``source_page`` is the provenance ticket 10's citation needs. ``source_page``
    is ``None`` for a corpus-derived record: its ``source_document_id`` names a
    JSONL line, not a page, and inventing a page there would be a fabricated
    citation.
    """

    chunk_id: str
    record_id: str
    record_version_key: str
    record_type: str
    unit_key: str
    unit_field: str
    unit_slot: str
    unit_kind: str
    source_document_id: str | None = None
    source_page: int | None = None
    source_url: str | None = None
    chunker_version: str = CHUNKER_VERSION

    @classmethod
    def of(
        cls,
        record: CanonicalRecord,
        unit: SemanticUnit,
        *,
        chunk_id: str | None = None,
    ) -> "ChunkLineage":
        """Lineage for ``unit`` read from ``record``; the id is derived, not passed."""
        key = record.record_version_key
        return cls(
            chunk_id=chunk_id if chunk_id is not None else chunk_id_for(key, unit),
            record_id=record.record_id,
            record_version_key=key,
            record_type=record.record_type.value,
            unit_key=unit.key,
            unit_field=unit.field,
            unit_slot=unit.slot,
            unit_kind=unit.kind.value,
            source_document_id=record.source_document_id,
            source_page=record.source_page,
            source_url=record.source_url,
            chunker_version=CHUNKER_VERSION,
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "record_version_key": self.record_version_key,
            "record_type": self.record_type,
            "unit_key": self.unit_key,
            "unit_field": self.unit_field,
            "unit_slot": self.unit_slot,
            "unit_kind": self.unit_kind,
            "source_document_id": self.source_document_id,
            "source_page": self.source_page,
            "source_url": self.source_url,
            "chunker_version": self.chunker_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ChunkLineage":
        """Read back a lineage written by :meth:`to_dict` (JSONL round-trip)."""
        return cls(
            chunk_id=str(data["chunk_id"]),
            record_id=str(data["record_id"]),
            record_version_key=str(data["record_version_key"]),
            record_type=str(data["record_type"]),
            unit_key=str(data["unit_key"]),
            unit_field=str(data["unit_field"]),
            unit_slot=str(data["unit_slot"]),
            unit_kind=str(data["unit_kind"]),
            source_document_id=data.get("source_document_id"),  # type: ignore[arg-type]
            source_page=data.get("source_page"),  # type: ignore[arg-type]
            source_url=data.get("source_url"),  # type: ignore[arg-type]
            chunker_version=str(data.get("chunker_version", CHUNKER_VERSION)),
        )


def resolve_record(
    lineage: ChunkLineage, records: Mapping[str, CanonicalRecord]
) -> CanonicalRecord:
    """The record version a chunk claims to come from — the lineage round-trip.

    ``records`` is keyed by ``record_version_key`` (``canonical_chunk_builder``
    exposes ``records_by_key`` for that). A missing key is an error, never a
    ``None``: a chunk whose lineage cannot be resolved is exactly the
    untraceable chunk this ticket exists to make impossible.
    """
    try:
        return records[lineage.record_version_key]
    except KeyError as exc:
        raise KeyError(
            f"chunk {lineage.chunk_id} names {lineage.record_version_key!r}, "
            "which is not among the records it was built from"
        ) from exc


validate_version_stamp(CHUNKER_VERSION)
