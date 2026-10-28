"""Canonical records → chunks — ticket 06.

See ``.scratch/husc-rag-admin-upgrade/issues/06-deterministic-chunk-identity.md``.

Ticket 05's ``PublishedMaterializer`` writes the records a release may serve; it
writes no chunks. This module is that missing step, and it is the public seam
ticket 25's admin preview indexer calls.

Division of labour
------------------
* **Admission** is not this module's to invent: only ``published`` record
  versions are chunked, and a caller that passes anything else is refused
  (:func:`_reject_unpublishable`). ``PublishedMaterializer`` remains the gate
  that decides a release's contents; this is the independent second gate at the
  boundary the chunker owns.
* **Text and structured content** is cut by
  ``src/domain/canonical/semantic_units`` into atom-safe units, one chunk per
  unit. A major code, a subject combination and a timeline milestone therefore
  cannot be split, because the splitter refuses those boundaries and the audit
  (:func:`chunk_spans_split_atoms`) can prove it.
* **Table content** is handed to ``scripts/chunk_table_aware_v2.process_file``
  — the chunker tickets 32/33 hardened — with the record's table blocks and
  nothing else. This module never re-implements table parsing, row grouping or
  header repetition; it consumes the chunker's chunks and gives them
  deterministic ids.
* **Identity** is ``src/services/chunk_identity``: ``chunk_id =
  sha256(record_version_key + semantic_unit)``, plus lineage back to the record
  version and source page.

Table chunks and the token budget
---------------------------------
The chunker merges consecutive ``small_table`` chunks **only when they carry the
same header**, so a merged chunk still reads as one table with one header — that
is context preservation, not row-merging to hit a token target. Rows from
different tables are never combined here: a table chunk is assigned to the table
whose header it carries, and the emitted ids never depend on the merge.

Output
------
``<output_root>/chunks.jsonl`` — one JSON object per chunk, in the shape
``scripts/ingest_lancedb.py`` already reads (``id``, ``chunk_id``, ``text``,
``metadata``, ``breadcrumbs``, ``faq_type``), so this corpus reaches LanceDB
through the existing ingest path without an adapter.
``<output_root>/chunk_manifest.json`` — counts, the version stamp, and the
release id the chunks were built from.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping, Sequence

from loguru import logger

from src.domain.canonical.record import CanonicalRecord, RecordStatus, from_canonical_json
from src.domain.canonical.semantic_units import (
    SemanticUnit,
    UnitKind,
    find_atomic_spans,
    split_field_units,
    split_text_units,
)
from src.services.chunk_identity import (
    CHUNKER_VERSION,
    ChunkLineage,
    chunk_id_for,
    chunker_version,
    normalize_for_hash,
    resolve_record,
    version_stamp,
)

CHUNKS_FILENAME = "chunks.jsonl"
CHUNK_MANIFEST_FILENAME = "chunk_manifest.json"
#: The only status a record version may hold when it is turned into an
#: indexable chunk. ``PublishedMaterializer`` (ticket 05) is the primary gate;
#: this is the second one, enforced at the boundary this module owns, so a
#: caller that reaches the chunker directly — ticket 15's re-index job accepts
#: records as an argument, ticket 26 wires it into production — fails loudly
#: instead of silently indexing a draft.
INDEXABLE_STATUS = RecordStatus.PUBLISHED
#: The name the ingest layer records in ``metadata.ingest_file``; orphan
#: scoping (``scripts/ingest_lancedb.py:_orphan_ids``) matches on it.
INGEST_FILE = CHUNKS_FILENAME

#: ``process_file`` reads this off the record's own text when present.
MATERIALIZED_RECORDS_FILENAME = "materialized_records.jsonl"

_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-|:]+\|$")
_CHUNKER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunk_table_aware_v2.py"

#: chunk_type values the table chunker emits. Anything else it returns came from
#: its own text splitter, which this module does not use (the semantic-unit
#: splitter is atom-safe where the chunker's is not).
_TABLE_CHUNK_TYPES = frozenset(
    {"wide_table", "small_table", "program_group", "packed_table"}
)

_chunker_module: ModuleType | None = None
_encoder = None


# ── the public entry point ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkIndexResult:
    """The artifacts one ``build_chunk_index`` call wrote, and what is in them."""

    release_id: str
    output_root: Path
    chunks_path: Path
    manifest_path: Path
    chunks: tuple[dict, ...]
    manifest: dict
    records: tuple[CanonicalRecord, ...]

    @property
    def chunk_ids(self) -> frozenset[str]:
        """The chunk id SET — what idempotency is asserted on, not the count."""
        return frozenset(chunk["id"] for chunk in self.chunks)

    @property
    def lineage(self) -> tuple[ChunkLineage, ...]:
        return tuple(ChunkLineage.from_dict(c["lineage"]) for c in self.chunks)

    def chunks_of(self, record_id: str) -> tuple[dict, ...]:
        """Every chunk of one record, in emission order."""
        return tuple(c for c in self.chunks if c["metadata"]["record_id"] == record_id)

    def resolve(self, chunk_id: str) -> CanonicalRecord:
        """The record version the chunk with ``chunk_id`` came from."""
        lineage = {c["id"]: ChunkLineage.from_dict(c["lineage"]) for c in self.chunks}
        return resolve_record(lineage[chunk_id], records_by_key(self.records))


def records_by_key(
    records: Iterable[CanonicalRecord],
) -> dict[str, CanonicalRecord]:
    """Index records by ``record_version_key`` — the lineage round-trip lookup."""
    return {record.record_version_key: record for record in records}


def build_chunk_index(
    records: Iterable[CanonicalRecord] | str | Path,
    output_root: str | Path,
    release_id: str,
) -> ChunkIndexResult:
    """Turn a release's materialized records into a chunk corpus.

    ``records`` is either the materialized records themselves or the path of a
    ticket-05 artifact: a ``materialized_records.jsonl`` file, or a directory
    holding one.

    ``output_root`` receives ``chunks.jsonl`` and ``chunk_manifest.json``.
    ``release_id`` is stamped into every chunk's lineage and into the manifest,
    so a corpus can always be traced to the release it was built from — the same
    rule ``PublishedMaterializer`` applies to its own output.

    Deterministic by construction: the same records produce byte-identical
    ``chunks.jsonl`` regardless of the order they are passed in, the process
    they are read in, or how many times the call is repeated.
    """
    release_id = (release_id or "").strip()
    if not release_id:
        raise ValueError(
            "build_chunk_index requires a release_id — chunks that cannot be "
            "traced to a release cannot be re-indexed or rolled back"
        )
    # Loaded and checked BEFORE the output directory is created: a refused call
    # must leave no artifact behind, not an empty directory a later reader could
    # mistake for a built corpus.
    loaded = _load_records(records)
    _reject_unpublishable(loaded)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    # One unit set per record, then one chunk per unit. Sorted at the end, so
    # the caller's iteration order cannot reach the file.
    chunks: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="chunk_staging_", dir=output) as staging:
        staging_dir = Path(staging)
        for record in loaded:
            chunks.extend(_record_chunks(record, release_id, staging_dir))

    chunks.sort(key=lambda c: (c["metadata"]["record_version_key"], c["metadata"]["unit_key"]))

    manifest = _manifest(
        release_id=release_id,
        records=loaded,
        chunks=chunks,
        records_source=_describe_source(records),
    )

    chunks_path = output / CHUNKS_FILENAME
    manifest_path = output / CHUNK_MANIFEST_FILENAME
    _write_atomic(
        chunks_path,
        b"".join(
            json.dumps(chunk, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            for chunk in chunks
        ),
    )
    _write_atomic(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    logger.info(
        f"Chunk index: {len(chunks)} chunk(s) from {len(loaded)} record(s) "
        f"→ {chunks_path.name} (chunker {CHUNKER_VERSION}, release {release_id})"
    )
    return ChunkIndexResult(
        release_id=release_id,
        output_root=output,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        chunks=tuple(chunks),
        manifest=manifest,
        records=loaded,
    )


# ── input ──────────────────────────────────────────────────────────────────


def _reject_unpublishable(records: Sequence[CanonicalRecord]) -> None:
    """Refuse to chunk any record version a release is not allowed to serve.

    The second gate, independent of ``PublishedMaterializer``: the materializer
    decides what a release *contains*, and this decides what the chunker will
    *accept*, from the records themselves rather than from the caller's word.
    A caller that hands over a store's raw versions — draft, review, approved,
    archived — is a bug at the call site, and it must fail here, before a single
    chunk id is computed, instead of putting unreviewed text in front of a
    student.

    Only ``published`` passes. An ``archived`` version is a retracted fact, and
    serving it is an as-of-time query the release policy owns — not something
    this boundary may infer from an argument. A caller that needs that capability
    must admit it explicitly (a materializer run with that policy), so the
    decision stays visible in one place instead of being re-derived here.
    """
    offenders = [record for record in records if record.status is not INDEXABLE_STATUS]
    if not offenders:
        return
    named = ", ".join(
        f"{record.record_version_key} ({_status_label(record.status)})"
        for record in offenders[:5]
    )
    if len(offenders) > 5:
        named += f", and {len(offenders) - 5} more"
    raise ValueError(
        f"build_chunk_index refuses {len(offenders)} record version(s) that are not "
        f"{INDEXABLE_STATUS.value}: {named}. A draft, review or approved version is "
        "an edit that has not replaced anything, and an archived version is a "
        "retracted fact; neither may reach the index. Pass ticket 05's "
        "PublishedMaterializer output (materialized_records.jsonl), which is the "
        "gate that admits records to a release."
    )


def _status_label(status: object) -> str:
    """A status as text for an error message — never raising while reporting one.

    The check above is identity against the enum member, so a caller can reach
    this with something that is not a ``RecordStatus`` at all (a raw string from
    a hand-built record). A message that raised ``AttributeError`` on the way out
    would hide the cause it is supposed to name.
    """
    value = getattr(status, "value", status)
    return str(value)


def _load_records(
    records: Iterable[CanonicalRecord] | str | Path,
) -> tuple[CanonicalRecord, ...]:
    """Materialized records from a path or from the caller's own sequence."""
    if isinstance(records, (str, Path)):
        return load_materialized_records(records)
    if isinstance(records, Mapping):
        return tuple(records.values())
    return tuple(records)


def load_materialized_records(path: str | Path) -> tuple[CanonicalRecord, ...]:
    """Read a ticket-05 ``materialized_records.jsonl`` (or the directory holding one).

    The file is one canonical JSON object per line, written by
    ``to_canonical_json`` — so it is read back with ``from_canonical_json``, not
    with a hand-rolled parser that could silently reinterpret Vietnamese bytes.
    """
    target = Path(path)
    if target.is_dir():
        target = target / MATERIALIZED_RECORDS_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"no materialized records at {target} — ticket 05's materializer must "
            "run before a chunk index can be built"
        )
    records: list[CanonicalRecord] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(from_canonical_json(line))
        except Exception as exc:  # noqa: BLE001 - re-raised with the line number
            raise ValueError(f"{target}:{lineno} is not a canonical record: {exc}") from exc
    return tuple(records)


def _describe_source(records: Iterable[CanonicalRecord] | str | Path) -> dict:
    if isinstance(records, (str, Path)):
        return {"kind": "materialized_records", "path": str(Path(records).resolve())}
    return {"kind": "in_memory"}


# ── one record → its chunks ────────────────────────────────────────────────


def _record_chunks(
    record: CanonicalRecord, release_id: str, staging_dir: Path
) -> list[dict]:
    """Every chunk of one record version: structured atoms, text units, tables."""
    text_body, table_blocks = _partition_blocks(record.text)
    # Structured units first: the record's declared assertions, one unit each.
    units: list[SemanticUnit] = list(split_field_units(record))
    # Text units come from the non-table body only: the table's rows belong to
    # the table chunker below, and letting the prose splitter see them would
    # index every row twice — once as a table chunk, once inside a text unit.
    if text_body.strip():
        units.extend(
            split_text_units(
                text_body,
                record_id=record.record_id,
                record_type=record.record_type.value,
                enc=_get_encoder(),
            )
        )

    chunks = [_chunk_for(record, unit, release_id) for unit in units]
    if table_blocks:
        chunks.extend(_table_chunks(record, table_blocks, release_id, staging_dir))
    return chunks


def _chunk_for(record: CanonicalRecord, unit: SemanticUnit, release_id: str) -> dict:
    """One chunk for one unit, with its lineage."""
    record_version_key = record.record_version_key
    chunk_id = chunk_id_for(record_version_key, unit)
    lineage = ChunkLineage.of(record, unit, chunk_id=chunk_id)
    metadata = {
        "source": record.record_id,
        "source_type": "canonical_record",
        "ingest_file": INGEST_FILE,
        "release_id": release_id,
        "record_id": record.record_id,
        "record_version_key": record_version_key,
        "record_type": record.record_type.value,
        "admission_year": record.admission_year,
        "data_year": record.admission_year,
        "status": record.status.value,
        "authority_level": record.authority_level.value,
        "source_document_id": record.source_document_id,
        "source_page": record.source_page,
        "source_url": record.source_url,
        "chunk_type": unit.kind.value,
        "unit_key": unit.key,
        "unit_field": unit.field,
        "unit_slot": unit.slot,
        "unit_atomic": unit.atomic,
        "chunker_version": CHUNKER_VERSION,
    }
    if unit.source_span is not None:
        # Carried so atomicity is checkable against the record's own text: an
        # atom is whole iff one chunk's span contains it.
        metadata["unit_source_span"] = [unit.source_span[0], unit.source_span[1]]
    # DOCX heading metadata for the citation reader: take the closest heading
    # that precedes the unit's start offset inside the source document.
    headings = getattr(record, "docx_headings", None)
    if headings is not None and unit.source_span is not None:
        before = [h for h in headings if h["offset"] < unit.source_span[0]]
        if before:
            metadata["section"] = before[-1]["text"]
    return _chunk_dict(chunk_id, unit.value, record.record_id, metadata, lineage)


def _table_chunks(
    record: CanonicalRecord,
    table_blocks: Sequence[str],
    release_id: str,
    staging_dir: Path,
) -> list[dict]:
    """Chunks for the record's tables, produced by the ticket-32/33 chunker.

    The chunker is handed the table blocks alone, so every chunk it returns is a
    table chunk and the record's prose is never re-split by a second splitter.
    Table identity comes from the header the chunk carries, which is also what
    the chunker uses to decide that two blocks are the same table.
    """
    chunker = _load_table_chunker()
    staging_path = staging_dir / f"{_safe_name(record.record_id)}__v{record.version}.md"
    staging_path.write_text("\n\n".join(table_blocks), encoding="utf-8")

    produced = chunker.process_file(staging_path, _get_encoder(), 0)
    table_chunks = [
        c
        for c in produced
        if (c.get("metadata") or {}).get("chunk_type") in _TABLE_CHUNK_TYPES
    ]

    chunks: list[dict] = []
    table_index = -1
    previous_header: str | None = None
    piece_slot = 0
    for raw in table_chunks:
        header = _header_of(raw.get("text", ""))
        if header != previous_header:
            table_index += 1
            piece_slot = 0
            previous_header = header
        unit = SemanticUnit(
            record_id=record.record_id,
            record_type=record.record_type.value,
            field=f"table:{table_index}",
            # Slot is the piece's ordinal inside its table. The row range is
            # deliberately NOT in the key: a token-budget shift must not renumber
            # a piece whose own text did not change.
            slot=str(piece_slot),
            value=raw.get("text", ""),
            kind=UnitKind.TABLE,
        )
        piece_slot += 1
        chunk = _chunk_for(record, unit, release_id)
        # The chunker's own view of the table, kept for the size/type gates the
        # ingest layer reads.
        chunk["metadata"]["table_chunker_type"] = raw["metadata"]["chunk_type"]
        chunk["metadata"]["table_dims"] = raw["metadata"].get("table_dims", "")
        for key in ("rowspan_attribution", "rowspan_unresolved"):
            if raw["metadata"].get(key):
                chunk["metadata"][key] = raw["metadata"][key]
        chunks.append(chunk)
    return chunks


def _chunk_dict(
    chunk_id: str,
    text: str,
    doc_id: str,
    metadata: dict,
    lineage: ChunkLineage,
) -> dict:
    """The on-disk chunk shape ``scripts/ingest_lancedb.py`` already reads."""
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "metadata": metadata,
        "breadcrumbs": [metadata.get("source_document_id") or metadata["record_id"]],
        "faq_type": "",
        "content_hash": hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest(),
        "lineage": lineage.to_dict(),
    }


# ── block partitioning ─────────────────────────────────────────────────────


def _is_table_line(line: str) -> bool:
    """The chunker's own table-line predicate, so both agree where a table starts."""
    stripped = line.strip()
    return (
        stripped.startswith("|")
        and stripped.endswith("|")
        and "|" in stripped[1:-1]
    )


def _partition_blocks(text: str) -> tuple[str, list[str]]:
    """``(non-table text, table markdown blocks)``.

    A line scan, not table parsing: the blocks are handed to the chunker
    verbatim, and the chunker decides columns, rows, grouping and header
    repetition. Splitting them out is what keeps a row from being indexed twice
    (once as a table chunk, once inside a text unit).
    """
    body: list[str] = []
    blocks: list[str] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        if _is_table_line(line):
            buffer.append(line.strip())
            continue
        if buffer:
            blocks.append("\n".join(buffer))
            buffer = []
        body.append(line)
    if buffer:
        blocks.append("\n".join(buffer))
    return "\n".join(body), blocks


def _header_of(chunk_text: str) -> str:
    """The table header a chunk carries — its first non-separator line."""
    for line in chunk_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _TABLE_SEPARATOR_RE.match(stripped):
            continue
        return stripped
    return ""


def _safe_name(record_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", record_id)[:80] or "record"


# ── atomicity audit, at chunk level ────────────────────────────────────────


def chunk_spans_split_atoms(
    record: CanonicalRecord, chunks: Sequence[Mapping]
) -> tuple[str, ...]:
    """Atoms in the record's *prose* that no single text chunk holds whole.

    The chunk-level form of the atomicity rule: every major code, subject
    combination and timeline milestone the semantic-unit splitter was given must
    sit inside one chunk's ``unit_source_span``. A chunk whose span cuts one —
    because the splitter ignored the rule — is reported here.

    Scoped to the prose body, which is the region the splitter owns. A table's
    rows are the table chunker's to keep whole (its pipe-row guard) and are not
    unit spans at all; a row cut across chunks is reported by
    :func:`table_rows_not_whole` instead, so neither region can be silently
    unaudited.
    """
    body, _ = _partition_blocks(record.text)
    spans = [
        tuple(chunk["metadata"]["unit_source_span"])
        for chunk in chunks
        if chunk["metadata"].get("unit_source_span") is not None
    ]
    cut: list[str] = []
    for atom in find_atomic_spans(body):
        whole = any(start <= atom.start and atom.end <= end for start, end in spans)
        if not whole:
            cut.append(atom.value)
    return tuple(cut)


def table_rows_not_whole(
    record: CanonicalRecord, chunks: Sequence[Mapping]
) -> tuple[tuple[str, ...], ...]:
    """Table rows of the record that no single chunk carries whole.

    Cells are compared stripped and split on ``|``, because the chunker re-renders
    a table with padded columns — byte equality with the source line would fail
    on formatting the chunker is entitled to change, while cell equality is the
    property that matters. A row missing from every chunk, or present only as
    fragments of two chunks, shows up here.
    """
    _, blocks = _partition_blocks(record.text)
    chunk_rows: list[tuple[str, ...]] = []
    for chunk in chunks:
        for line in str(chunk.get("text", "")).split("\n"):
            if _is_table_line(line) and not _TABLE_SEPARATOR_RE.match(line.strip()):
                chunk_rows.append(_cells(line))

    missing: list[tuple[str, ...]] = []
    for block in blocks:
        for line in block.split("\n"):
            if _TABLE_SEPARATOR_RE.match(line.strip()):
                continue
            row = _cells(line)
            if row and row not in chunk_rows:
                missing.append(row)
    return tuple(missing)


def _cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


# ── manifest ───────────────────────────────────────────────────────────────


def _manifest(
    *,
    release_id: str,
    records: Sequence[CanonicalRecord],
    chunks: Sequence[dict],
    records_source: dict,
) -> dict:
    by_chunk_type = Counter(c["metadata"]["chunk_type"] for c in chunks)
    by_record_type = Counter(r.record_type.value for r in records)
    per_record = Counter(c["metadata"]["record_id"] for c in chunks)
    pages = sum(1 for c in chunks if c["metadata"].get("source_page") is not None)
    return {
        "chunk_manifest_version": version_stamp()["chunk_manifest_version"],
        # The stamp ticket 15 compares to decide a corpus is stale. A literal,
        # identical across runs on the same code, and independent of content.
        "chunker_version": chunker_version(),
        "unit_rules_version": version_stamp()["unit_rules_version"],
        "release_id": release_id,
        "records_source": records_source,
        "chunk_count": len(chunks),
        "record_count": len(records),
        "chunks_with_source_page": pages,
        "by_chunk_type": dict(sorted(by_chunk_type.items())),
        "by_record_type": dict(sorted(by_record_type.items())),
        "chunks_per_record": dict(sorted(per_record.items())),
        # Every record version this corpus speaks for, sorted — the set ticket 15
        # intersects with the canonical ids a parser change affected.
        "record_version_keys": sorted(r.record_version_key for r in records),
        "chunks_filename": CHUNKS_FILENAME,
    }


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a ``.tmp`` sibling + ``os.replace``, as ticket 05 does."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ── the reused table chunker ───────────────────────────────────────────────


def _load_table_chunker() -> ModuleType:
    """Import ``scripts/chunk_table_aware_v2.py`` without forking it.

    Loaded by path rather than by module name: ``scripts/`` is not a package, and
    adding it to ``sys.path`` would let a bare ``import config`` elsewhere
    resolve against whichever directory happened to be first.
    """
    global _chunker_module
    if _chunker_module is None:
        spec = importlib.util.spec_from_file_location(
            "chunk_table_aware_v2_ticket06", _CHUNKER_PATH
        )
        if spec is None or spec.loader is None:  # pragma: no cover - path is fixed
            raise ImportError(f"cannot load the table chunker at {_CHUNKER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _chunker_module = module
    return _chunker_module


def _get_encoder():
    """cl100k_base, the encoding the chunker and the ingest layer both use."""
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder
