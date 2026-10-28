"""
Ingest chunked JSONL files into LanceDB using Qwen3-Embedding.

Default mode performs full rebuild (overwrite table).
Incremental mode (--incremental) ingests only changed chunk files and appends
new chunk IDs that are not already present in LanceDB.

Canonical mode (--canonical-corpus [DIR]) ingests the corpus built by the
ticket-15 re-index job (chunks.jsonl + vectors.jsonl + chunk_manifest.json),
merging on row id and sweeping only canonical rows. It refuses a corpus whose
stamp (ticket 06) is not the stamp the running code produces, and it works from
an empty canonical footprint — the served table may hold legacy rows only, or
not exist at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import lancedb
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings
from src.infrastructure.lancedb_schema import EMBEDDING_DIM
from src.services.embedding import EmbeddingService
from src.services.reindex_job import (
    CorpusStamp,
    read_corpus_stamp,
    stamp_is_stale,
)

load_dotenv()

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "chunked" / ".ingest_manifest.json"

# M3-N2b: ingest target is unified husc_chunks (not legacy rag2025)
import os
TARGET_TABLE = os.environ.get("LANCEDB_UNIFIED_TABLE", "husc_chunks")

# STEP 8b: ingest-time size-gate guardrail — a chunk may NEVER be silently
# dropped for size; it is LOG-AND-FLAG (warning log + metadata marker) and the
# count is surfaced at the end of the run. Knobs mirror config.settings defaults
# (512 / 50) and resolve from settings when the caller provides them.
HARD_MAX_TOKENS = 512
SOFT_MIN_TOKENS = 50
class IngestCorpusError(Exception):
    """A canonical corpus cannot be served as-is.

    Refusals that used to be ``return False``: the publish hook needs the
    difference between "ingest refused" and "ingest succeeded", because a
    refused ingest must fail the job (and keep the outbox entry), while
    ``False`` told the hook the publish was fine.
    """

_ENCODING = None


def _count_tokens(text: str) -> int:
    """Token count via tiktoken cl100k_base (same encoding as chunker.py).

    Lazy-loaded and cached at module level so ingest paths that never touch the
    size gate (and tests) pay no import cost.
    """
    global _ENCODING
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODING.encode(text))


def _size_flag_for(
    text: str,
    hard_max: int = HARD_MAX_TOKENS,
    soft_min: int = SOFT_MIN_TOKENS,
) -> str:
    """Classify a chunk against the size gates (STEP 8b).

    Returns one of: 'oversized' (> hard_max), 'undersized' (< soft_min), 'ok'.
    Pure classification — callers decide to log/flag, never to drop.
    """
    tokens = _count_tokens(text)
    if tokens > hard_max:
        return "oversized"
    if tokens < soft_min:
        return "undersized"
    return "ok"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _list_chunk_files(chunks_dir: Path) -> List[Path]:
    # T1.5-bl / G4: the canonical corpus spans MANY sources, not just the
    # legacy `chunked_*.jsonl` set. Include every `*.jsonl` in data/chunked/
    # except stubs/noise:
    #   - chunked_thongbao_* / batch_thongbao_* / crawl_thongbao_*: 556-byte
    #     crawl stubs + boilerplate navigation rác (batch_thongbao_* carries
    #     only "Trang chủ Thông báo Tuyển sinh..." chrome, not corpus) — T1.3 G1
    #   - chunked_test: fixture, not corpus
    #   - husc_yearly_{year}: literal template with braces, not a real file
    #   - .* hidden files (manifest etc.)
    #   - *_enhanced: T1.5-bl now INCLUDES these — chunked_10_enhanced.jsonl is
    #     real 2026 subject data (38 rows); T1.2 text-dedup collapses its
    #     overlap with the base file at insert.
    files = sorted(chunks_dir.glob("*.jsonl"))
    _stub_prefixes = ("chunked_thongbao_", "batch_thongbao_", "crawl_thongbao_")
    _exact_names = {"chunked_test.jsonl", "husc_yearly_{year}.jsonl"}
    return [
        f for f in files
        if not f.name.startswith(_stub_prefixes)
        and f.name not in _exact_names
        and not f.name.startswith(".")
    ]


def _load_manifest() -> Dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Ingest manifest unreadable, rebuilding from scratch")
        return {}


def _save_manifest(manifest: Dict[str, str]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _refresh_manifest_for_all_files(chunks_dir: Path) -> None:
    manifest = {file.name: _file_digest(file) for file in _list_chunk_files(chunks_dir)}
    _save_manifest(manifest)


def _resolve_target_files(chunks_dir: Path, incremental: bool) -> List[Path]:
    files = _list_chunk_files(chunks_dir)
    if not incremental:
        return files

    old_manifest = _load_manifest()
    changed: List[Path] = []
    for file in files:
        digest = _file_digest(file)
        if old_manifest.get(file.name) != digest:
            changed.append(file)
    return changed


def _normalize_for_hash(text: str) -> str:
    """Normalize text for content hashing: collapse whitespace only.

    T7: date tokens (\\d{2}/\\d{2}/\\d{4}) are PRESERVED in the hashed text so
    the content→vector reuse key is date-sensitive — two texts differing ONLY by
    a date must not reuse each other's embedding. Other boilerplate
    normalization (whitespace collapse) is unchanged.
    """
    return " ".join(text.split())


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def _text_dedup_hash(text: str) -> str:
    """Normalized text hash for cross-file dedup (case/whitespace-insensitive).

    T1.2: unlike `_content_hash` (re-embed guardrail, date-sensitive since T7),
    this is the *insert* guard. Two JSONL files carrying the same text under
    different ids must be stored once. Lowercase + collapsed whitespace means
    "Nội dung X" and "  nội DUNG X  " resolve to the same hash.
    """
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _unique_id(source: str, ingest_file: str, chunk_id: str) -> str:
    """Globally-unique row id (T1.5/G9).

    Bare chunk ids such as '0','13','61' collide across documents — several
    ingest scripts emit identical numeric chunk_ids, so `id = chunk_id` lets the
    last writer silently overwrite the others. Hashing source|ingest_file|chunk_id
    yields a stable, unique id per (document, file, chunk) triple. `doc_id` keeps
    the original chunk_id so document-level grouping is preserved.
    """
    raw = f"{source}|{ingest_file}|{chunk_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_chunks(files: List[Path]) -> List[Dict]:
    all_chunks: List[Dict] = []
    malformed_lines = 0

    for file in files:
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue

                if not isinstance(obj, dict):
                    continue

                text = obj.get("text") or ""
                if not text:
                    continue

                # Metadata may be malformed (None/list/str) in a hand-authored
                # JSONL — treat as empty rather than crash the whole ingest.
                meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

                # `id: null` must not stringify to "None": two such rows would
                # collide on _unique_id and silently overwrite in a rebuild.
                chunk_id = str(obj.get("id") or obj.get("chunk_id") or f"chunk_{len(all_chunks)}")
                source = meta.get("source", file.name)

                # STEP 8a: inherit the chunker/normalize breadcrumbs (source
                # label + section markers) so the persisted record keeps them
                # in metadata_json. Read top-level first, then metadata.
                breadcrumbs = obj.get("breadcrumbs")
                if not isinstance(breadcrumbs, list):
                    breadcrumbs = meta.get("breadcrumbs")
                if not isinstance(breadcrumbs, list):
                    breadcrumbs = []
                breadcrumbs = [str(b) for b in breadcrumbs if str(b).strip()]

                all_chunks.append(
                    {
                        # T1.5 (G9): globally-unique row id — bare chunk ids
                        # ('0','13','61') collide across documents. `chunk_id`
                        # below keeps the original value for doc_id grouping.
                        "id": _unique_id(source, file.name, chunk_id),
                        "chunk_id": chunk_id,
                        "text": text,
                        "source": source,
                        # File provenance for orphan scoping. Distinct from
                        # `source` (often a URL, shared across several files).
                        "ingest_file": file.name,
                        "faq_type": obj.get("faq_type", meta.get("faq_type", "")),
                        "metadata": meta,
                        "breadcrumbs": breadcrumbs,
                        "content_hash": _content_hash(text),
                    }
                )

    if malformed_lines:
        logger.warning(f"Skipped {malformed_lines} malformed JSON line(s) in {len(files)} file(s)")

    all_chunks = [c for c in all_chunks if c.get("text")]

    # T1.2: cross-file text dedup — keep one chunk (first by ingest_file +
    # chunk_id) when two files carry the same normalized text. Scoped per
    # canonical record AND per unit: two DIFFERENT records may legitimately
    # carry identical text (e.g. "mã ngành: 7420201" in every record naming
    # that major), and collapsing them would delete one record's chunk_id +
    # lineage, leaving it uncitable. The same reasoning applies one level down:
    # a record with two majors, or two years, holds several distinct facts that
    # legitimately repeat a value, and each is separately citable — so the
    # record's own unit fields are part of the identity of the text. Only
    # repetition of the same text *within one unit* is duplication. Legacy
    # chunks have no `record_version_key` → `rvk == ""` → they keep the
    # original whole-corpus dedup, same winner as before.
    seen_dedup_keys: Set[tuple] = set()
    deduped: List[Dict] = []
    for c in sorted(all_chunks, key=lambda x: (x["ingest_file"], x["chunk_id"])):
        rvk = str(c["metadata"].get("record_version_key") or "")
        unit = str(c["metadata"].get("unit_key") or "")
        key = (rvk, unit, _text_dedup_hash(c["text"]))
        if key in seen_dedup_keys:
            continue
        seen_dedup_keys.add(key)
        deduped.append(c)

    return deduped


def _load_existing_ids(table) -> Set[str]:
    try:
        rows = table.to_arrow().select(["id"]).to_pylist()
        return {str(r["id"]) for r in rows}
    except Exception as e:
        logger.warning(f"Failed to read existing IDs from LanceDB table: {e}")
        return set()


def _esc(raw: str) -> str:
    """Escape single quotes for SQL IN-clause values."""
    return raw.replace("'", "''")


def _orphan_ids(table, batch_files: Set[str], batch_ids: Set[str]) -> List[str]:
    """Rows ingested from batch_files but absent from batch_ids.

    husc_chunks (CHUNK_SCHEMA) has no `source` column, so provenance lives in
    metadata_json under `ingest_file` (the JSONL filename). Rows with no
    recorded ingest_file are never orphans — they came from another ingest
    script (581 of 2068 production rows) and must not be touched.

    Scoping by `source` is WRONG: it is usually a URL shared by several files,
    so it matches rows this batch does not own.
    """
    if not batch_files:
        return []
    try:
        rows = table.to_arrow().select(["id", "metadata_json"]).to_pylist()
    except Exception as e:
        logger.warning(f"Orphan scan failed, skipping orphan delete: {e}")
        return []

    orphans: List[str] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in batch_ids:
            continue
        try:
            meta = json.loads(row.get("metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if meta.get("ingest_file") in batch_files:
            orphans.append(row_id)
    return orphans


def _trigger_graph_sync(new_ids: List[str]) -> None:
    """M3-N3a: Post-ingest graph sync hook (fail-safe, blocking up to 300s).

    Writes new chunk IDs to data/graph/.new_chunk_ids.json and triggers
    build_graph.py --incremental via subprocess. Failures — including a
    non-zero exit or a 300s timeout — are logged but do NOT propagate:
    eventual consistency per R3 direction A.
    """
    repo_root = Path(__file__).parent.parent
    graph_dir = repo_root / "data" / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Write new chunk IDs for incremental build
    new_ids_path = graph_dir / ".new_chunk_ids.json"
    try:
        new_ids_path.write_text(json.dumps(new_ids, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write .new_chunk_ids.json: {e}")
        return

    # Trigger incremental graph build (blocking, fail-safe)
    try:
        result = subprocess.run(
            [sys.executable, "scripts/build_graph.py", "--incremental"],
            cwd=str(repo_root),
            timeout=300,  # 5 min timeout
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Graph sync exited {result.returncode} (non-fatal)")
        else:
            logger.info(f"Graph sync triggered for {len(new_ids)} new chunk(s)")
    except subprocess.TimeoutExpired:
        logger.warning("Graph sync timed out after 300s (non-fatal)")
    except Exception as e:
        logger.warning(f"Graph sync failed (non-fatal): {e}")


def _table_exists(db, target: str) -> bool:
    """Whether ``target`` exists, on any lancedb client version.

    lancedb 0.29 ``list_tables()`` returns a ``ListTablesResponse`` (with a
    ``.tables`` field), not a plain list — ``target in response`` is always
    False. ``table_names()`` (deprecated) returns a plain list.
    """
    listed = db.list_tables()
    names = listed.tables if hasattr(listed, "tables") else listed
    return target in names

# ── canonical corpus ingestion (ticket 15) ─────────────────────────────────
#
# The canonical track (tickets 02/03/05/06/28) reaches the served table through
# here. Two things this path must get right that the legacy path does not have
# to:
#
#   * it works from an EMPTY canonical footprint. The served table holds 1679
#     legacy rows and zero canonical ones, so the first run is an initial
#     ingestion, not a delta — every row is new and there is no prior canonical
#     row to diff against.
#   * it refuses a STALE corpus. The corpus carries the stamp (ticket 06) the
#     code that built it produced; if that is not the stamp the running code
#     would produce, the corpus describes a chunker this process does not have,
#     and serving it would put chunk output in the index that no current code
#     path can reproduce. Refused, loudly, with the fix named.

CANONICAL_CORPUS_DIR = Path(__file__).parent.parent / "data" / "canonical_corpus"
CANONICAL_CHUNKS_FILENAME = "chunks.jsonl"
CANONICAL_VECTORS_FILENAME = "vectors.jsonl"
#: What a canonical chunk records as its own ingest_file
#: (``canonical_chunk_builder.INGEST_FILE``). Orphan scoping matches on it, so
#: a canonical sweep can never delete a row another ingest script wrote.
CANONICAL_INGEST_FILE = "chunks.jsonl"
#: The only record status a served canonical chunk may carry
#: (``canonical_chunk_builder.INDEXABLE_STATUS``). Checked per chunk at ingest,
#: so a corpus that reached disk without going through the chunker's gate — a
#: hand-assembled one, one built by older code, or one edited in between — is
#: still refused before a single row is written.
CANONICAL_PUBLISHED_STATUS = "published"

def _read_corpus_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"no {path.name} at {path} — the re-index job (ticket 15) must build "
            "the corpus before it can be ingested"
        )
    rows: List[Dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{lineno} is not a JSON object")
        rows.append(payload)
    return rows

def canonical_corpus_rows(
    corpus_dir: Path, *, embedding_dim: int, stamp
) -> List[Dict]:
    """Rows for a built corpus, with both release-id views forced to agree.

    ``husc_chunks`` records the release in two places: the top-level
    ``release_id`` column (what the ingest stamps on the row) and
    ``metadata_json.release_id`` (what the chunk carries). Both are written from
    one value here, so a canonical row cannot be read as belonging to two
    releases — the divergence ``HybridSearchService`` reports as an error.

    Every chunk must have its vector, matching on ``chunk_id`` AND
    ``content_hash``: a vector belonging to different text would be indexed
    silently and only show up as a wrong answer. A chunk without one aborts the
    run rather than being dropped — a canonical chunk that reaches the corpus
    must reach the index.

    The second gate, independent of ``build_chunk_index``: every chunk must
    carry ``metadata.status == "published"``. The chunker refuses to *build* a
    corpus out of unpublished records; this refuses to *serve* one that reached
    the corpus some other way — a hand-assembled corpus, a corpus left by older
    code, or a file edited between the two steps. Both read the status from the
    artifact itself, so neither depends on the other having run.
    """
    chunks = _read_corpus_jsonl(Path(corpus_dir) / CANONICAL_CHUNKS_FILENAME)
    vectors = {
        str(entry.get("chunk_id")): entry
        for entry in _read_corpus_jsonl(Path(corpus_dir) / CANONICAL_VECTORS_FILENAME)
    }

    rows: List[Dict] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        release_id = str(meta.get("release_id") or "")
        status = str(meta.get("status") or "")
        if status != CANONICAL_PUBLISHED_STATUS:
            raise ValueError(
                f"canonical chunk {chunk_id or '<no id>'} is not "
                f"{CANONICAL_PUBLISHED_STATUS} (status={status or '<missing>'}) — "
                "only published record versions may reach the served table: a "
                "draft, review or approved version is an edit that has not "
                "replaced anything, and an archived one is a retracted fact. "
                "Rebuild the corpus with build_chunk_index from ticket 05's "
                "materialized_records.jsonl."
            )
        if not chunk_id or not release_id:
            raise ValueError(
                f"canonical chunk {chunk_id or '<no id>'} carries no release_id — "
                "a chunk that cannot name its release cannot be rolled back"
            )
        entry = vectors.get(chunk_id)
        if entry is None:
            raise ValueError(f"canonical chunk {chunk_id} has no vector in the corpus")
        if entry.get("content_hash") != chunk.get("content_hash"):
            raise ValueError(
                f"canonical chunk {chunk_id}: vector content_hash "
                f"{entry.get('content_hash')!r} does not match the chunk's "
                f"{chunk.get('content_hash')!r}"
            )
        vector = [float(x) for x in entry.get("vector") or []]
        if len(vector) != embedding_dim:
            raise ValueError(
                f"canonical chunk {chunk_id}: vector has {len(vector)} dims, "
                f"table schema expects {embedding_dim}"
            )

        meta_out = {
            **meta,
            # Both views, one value.
            "release_id": release_id,
            "corpus_stamp": stamp.to_dict(),
            "ingest_file": str(meta.get("ingest_file") or CANONICAL_INGEST_FILE),
            "content_hash": chunk.get("content_hash"),
        }
        admission_year = meta.get("admission_year")
        rows.append(
            {
                "id": chunk_id,
                "doc_id": str(chunk.get("chunk_id") or chunk_id),
                "release_id": release_id,
                "text": chunk.get("text") or "",
                "summary": None,
                "vector": vector,
                "data_year": int(admission_year) if isinstance(admission_year, int) else 0,
                "scope": "admissions",
                "doc_type": str(meta.get("record_type") or "chunk"),
                "status": "current",
                "effective_from": 0,
                "effective_to": 99991231,
                "is_tombstone": False,
                "superseded_by": None,
                "metadata_json": json.dumps(meta_out, ensure_ascii=False, default=str),
                "is_reference": False,
                "crawl_date": None,
            }
        )
    return rows

def served_corpus_stamp(table) -> "CorpusStamp | None":
    """The stamp the served table's rows agree on, or ``None``.

    ``None`` covers both "no row carries a stamp" — the served table today, 1679
    legacy rows — and "rows disagree". Both mean the table cannot be shown to
    have been built by the running code, and the re-index treats that as stale.
    """
    rows = table.to_arrow().select(["metadata_json"]).to_pylist()
    return CorpusStamp.from_rows(rows)

def ingest_canonical_corpus(corpus_dir: str | Path | None = None) -> bool:
    """Ingest a built canonical corpus into the served table. Ticket 15's write path.

    Runs from an empty canonical footprint (the table may hold only legacy rows,
    or not exist at all) and from a populated one: the write is a
    ``merge_insert`` on ``id``, so re-running the same corpus updates rows in
    place instead of duplicating them, and canonical rows the corpus no longer
    contains are swept — scoped to canonical rows by ``ingest_file``, so no
    legacy row is ever touched.

    Raises:
        IngestCorpusError: the corpus is stale (built by other chunking logic
            than the running code) or holds no chunk. The publish hook that
            calls this turns the raise into a FAILED job and keeps the outbox
            entry, so the previous corpus keeps serving and the next tick
            retries — which ``return False`` could not express, because the
            hook would have treated ``False`` as success and created a
            release for a corpus the table never received.
    """
    settings = RAGSettings()
    corpus = Path(corpus_dir) if corpus_dir else CANONICAL_CORPUS_DIR

    stamp = read_corpus_stamp(corpus)
    if stamp_is_stale(stamp):
        raise IngestCorpusError(
            f"Canonical corpus at {corpus} is STALE "
            f"(built by {stamp.to_dict() if stamp else 'nothing — no manifest stamp'}; "
            f"code now {CorpusStamp.current().to_dict()}) — refusing to serve it. "
            "Re-run the ticket-15 re-index job to rebuild the corpus."
        )

    embedding_dim = int(getattr(settings, "EMBEDDING_DIM", EMBEDDING_DIM))
    rows = canonical_corpus_rows(corpus, embedding_dim=embedding_dim, stamp=stamp)
    if not rows:
        raise IngestCorpusError(
            f"Canonical corpus at {corpus} holds no chunk — aborting"
        )

    db = lancedb.connect(settings.LANCEDB_URI)
    target = TARGET_TABLE
    ids = {row["id"] for row in rows}

    if not _table_exists(db, target):
        table = db.create_table(target, data=rows)
        logger.info(
            f"Initial canonical ingest: table={target} created with "
            f"{table.count_rows()} row(s), release(s) "
            f"{sorted({r['release_id'] for r in rows})}"
        )
    else:
        table = db.open_table(target)
        orphans = _orphan_ids(table, {CANONICAL_INGEST_FILE}, ids)
        job = (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
        )
        if orphans:
            escaped = ", ".join(f"'{_esc(o)}'" for o in orphans)
            job.when_not_matched_by_source_delete(condition=f"id IN ({escaped})")
        job.execute(rows)
        logger.info(
            f"Canonical ingest: merged={len(rows)}, swept={len(orphans)}, "
            f"total_rows={table.count_rows()}, table={target}"
        )

    _warn_on_release_view_mismatch(db, target)
    return True

def _warn_on_release_view_mismatch(db, target: str) -> None:
    """Report rows whose two release-id views disagree, without touching them.

    Canonical rows written here always agree (one value, two places). Legacy
    rows carry a top-level ``release_id`` of ``"legacy"`` and no
    ``metadata_json.release_id`` at all, so a table holding both spans two
    release sets — the divergence ``HybridSearchService`` logs. Named here, at
    the ingest that produced the mix, so it is diagnosable without reading the
    BM25 path.
    """
    try:
        table = db.open_table(target)
        rows = table.to_arrow().select(["release_id", "metadata_json"]).to_pylist()
    except Exception as exc:  # noqa: BLE001 - reporting only, never fatal
        logger.warning(f"Release-view audit skipped: {exc}")
        return
    mismatched = 0
    for row in rows:
        meta = row.get("metadata_json")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if (row.get("release_id") or None) != (meta.get("release_id") or None):
            mismatched += 1
    if mismatched:
        logger.warning(
            f"Release-id views disagree on {mismatched}/{len(rows)} row(s) in "
            f"{target}: the top-level column and metadata_json.release_id do not "
            "match. Canonical rows are written agreeing; these are rows from "
            "another ingest path."
        )

def ingest_lancedb(incremental: bool = False) -> bool:
    settings = RAGSettings()
    chunks_dir = Path(__file__).parent.parent / "data" / "chunked"

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return False

    target_files = _resolve_target_files(chunks_dir, incremental=incremental)
    if incremental and not target_files:
        logger.info("Incremental ingest: no changed chunk files detected")
        return True

    logger.info(f"Loading chunks from {len(target_files)} JSONL file(s)...")
    chunks = _load_chunks(target_files)
    if not chunks:
        if incremental:
            logger.info("Incremental ingest: no valid chunk rows found in changed files")
            _refresh_manifest_for_all_files(chunks_dir)
            return True
        logger.error("No chunks found to ingest")
        return False

    logger.info(f"Loaded {len(chunks)} chunks")

    # M3-N1b: content-hash guardrail — skip re-embed when text unchanged
    db = lancedb.connect(settings.LANCEDB_URI)
    target = TARGET_TABLE
    table_exists = _table_exists(db, target)

    existing_hashes: Dict[str, str] = {}
    if incremental and table_exists:
        try:
            table_tmp = db.open_table(target)
            # T7: recompute the id-keyed hash from the immutable `text` column
            # (same rationale as the reuse map) so rows persisted under the
            # old date-insensitive scheme still count as unchanged in
            # incremental mode — otherwise an unchanged date-bearing chunk
            # would be treated as changed and re-embedded.
            rows = table_tmp.to_arrow().select(["id", "text"]).to_pylist()
            for r in rows:
                h = _content_hash(r.get("text") or "")
                if h:
                    existing_hashes[r["id"]] = h
        except Exception as e:
            logger.warning(f"Failed to load existing hashes: {e}")

    # T1.5-bl vector-reuse: a full-overwrite rebuild re-hashes every row id
    # (G9), so the id-keyed guardrail above can never match. Build a
    # content-hash → vector map from the CURRENT table and reuse any embedding
    # whose content is unchanged, instead of re-embedding the whole corpus
    # (~2h CPU at ~0.22 chunks/s on this machine).
    content_to_vector: Dict[str, list] = {}
    if table_exists:
        try:
            table_old = db.open_table(target)
            old_rows = table_old.to_arrow().select(["metadata_json", "vector", "text"]).to_pylist()
            for r in old_rows:
                try:
                    # T7: ALWAYS recompute the key from the immutable `text`
                    # column instead of trusting the persisted content_hash.
                    # Persisted hashes may predate the date-sensitive scheme
                    # (T7) or a different script's `_content_hash`; keying on
                    # them would orphan already-embedded date-bearing rows and
                    # force a re-embed. content_hash depends only on text, and
                    # the reuse map is rebuilt per run, so recomputing keeps
                    # keys consistent on both sides of a scheme change. Guard
                    # against embedding-model drift by requiring matching dim.
                    h = _content_hash(r.get("text") or "")
                    vec = r.get("vector")
                    if h and vec is not None and len(vec) == settings.EMBEDDING_DIM:
                        content_to_vector[h] = vec
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception as e:
            logger.warning(f"Failed to build content→vector reuse map: {e}")

    chunks_to_embed = [
        c for c in chunks
        if c["content_hash"] != existing_hashes.get(c["id"])
    ]
    chunks_unchanged = [
        c for c in chunks
        if c["content_hash"] == existing_hashes.get(c["id"])
    ]
    skipped = len(chunks_unchanged)
    if skipped:
        logger.info(f"Content-hash guardrail: skipping {skipped} unchanged chunk(s)")

    # Embed only changed chunks
    embeddings_map: Dict[str, list] = {}
    if chunks_to_embed:
        # T1.5-bl: in an overwrite rebuild every id is re-hashed (G9), so first
        # try to reuse a vector whose content-hash already exists in the current
        # table; only freshly embed the remainder.
        to_embed: list = []
        reused = 0
        for chunk in chunks_to_embed:
            vec = content_to_vector.get(chunk["content_hash"])
            if vec is not None:
                embeddings_map[chunk["id"]] = vec
                reused += 1
            else:
                to_embed.append(chunk)
        if reused:
            logger.info(f"Vector-reuse: reused {reused} chunk(s) from current table")
        chunks_to_embed = to_embed
        if chunks_to_embed:
            logger.info(f"Embedding {len(chunks_to_embed)} chunk(s) (skipped {skipped} unchanged, {reused} reused)")
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})")
            embedding_service = EmbeddingService(settings)
            texts = [chunk["text"] for chunk in chunks_to_embed]
            embeddings = embedding_service.encode_documents(texts)
            for chunk, vector in zip(chunks_to_embed, embeddings):
                embeddings_map[chunk["id"]] = vector.tolist()

    # For unchanged chunks, reuse existing vectors from table
    if chunks_unchanged and table_exists:
        try:
            table_tmp2 = db.open_table(target)
            unchanged_ids = [c["id"] for c in chunks_unchanged]
            escaped_ids = ", ".join(f"'{_esc(i)}'" for i in unchanged_ids)
            rows = table_tmp2.search().where(f"id IN ({escaped_ids})").limit(len(unchanged_ids)).to_arrow().to_pylist()
            for r in rows:
                embeddings_map[r["id"]] = r["vector"]
        except Exception as e:
            logger.warning(f"Failed to load existing vectors for unchanged chunks: {e}")
            # Fallback: embed them anyway
            embedding_service = EmbeddingService(settings)
            for chunk in chunks_unchanged:
                if chunk["id"] not in embeddings_map:
                    embeddings_map[chunk["id"]] = embedding_service.encode_documents([chunk["text"]])[0].tolist()

    records = []
    skipped_no_vector = 0
    # STEP 8b: size-gate counters, surfaced as a warning summary at the end.
    size_flags = {"oversized": 0, "undersized": 0}
    for chunk in chunks:
        vector = embeddings_map.get(chunk["id"])
        if vector is None:
            # T1.5 (python-reviewer): never skip silently — in overwrite mode a
            # chunk dropped here would vanish from the rebuilt table.
            skipped_no_vector += 1
            continue
        raw_meta = chunk.get("metadata", {})
        # M3-N2b: extract data_year (pattern from migrate_schema_v2.py:32-41)
        data_year = 0
        needs_review = True
        for key in ("data_year", "year"):
            val = raw_meta.get(key)
            if isinstance(val, int) and val > 0:
                data_year, needs_review = val, False
                break
            if isinstance(val, str) and val.isdigit():
                data_year, needs_review = int(val), False
                break

        faq_type = chunk.get("faq_type", "")
        # Provenance: husc_chunks has no `source` column, so record it in
        # metadata_json — orphan detection needs it to scope deletes by file.
        meta_out = {
            **raw_meta,
            "ingest_file": chunk["ingest_file"],
            "needs_review": needs_review,
            "content_hash": chunk["content_hash"],
        }
        if faq_type:
            meta_out["source_faq_type"] = faq_type

        # STEP 8a: inherit heading/title + breadcrumbs into metadata_json.
        # Metadata-only variant: `text` stays byte-identical so content hashes
        # (and existing embeddings) are preserved — injecting `### {heading}`
        # into text would re-embed the whole corpus (~2h CPU).
        heading = raw_meta.get("name") or raw_meta.get("title") or faq_type
        if heading:
            meta_out["title"] = str(heading)
        if chunk.get("breadcrumbs"):
            meta_out["breadcrumbs"] = chunk["breadcrumbs"]

        # STEP 8b: LOG-AND-FLAG size gate — never drop a chunk for size, never
        # silently accept an out-of-range one. Knobs fall back to module
        # defaults when the caller's settings object omits them (tests).
        size_flag = _size_flag_for(
            chunk["text"],
            getattr(settings, "HARD_MAX_TOKENS", HARD_MAX_TOKENS),
            getattr(settings, "SOFT_MIN_TOKENS", SOFT_MIN_TOKENS),
        )
        meta_out["size_flag"] = size_flag
        if size_flag != "ok":
            size_flags[size_flag] += 1
            logger.warning(
                f"Size-gate: chunk {chunk['id']} flagged {size_flag} "
                f"({_count_tokens(chunk['text'])} tokens) — kept as-is"
            )

        records.append(
            {
                "id": chunk["id"],
                "doc_id": chunk.get("chunk_id") or chunk["id"],
                # Real release id when the chunk carries one (canonical
                # corpus); the top-level column must agree with
                # metadata_json.release_id, which spreads raw_meta below.
                "release_id": raw_meta.get("release_id") or "legacy",
                "text": chunk["text"],
                "summary": None,
                "vector": vector,
                "data_year": data_year,
                "scope": "admissions",
                "doc_type": faq_type or "chunk",
                "status": "current",
                "effective_from": 0,
                "effective_to": 99991231,
                "is_tombstone": False,
                "superseded_by": None,
                "metadata_json": json.dumps(meta_out, ensure_ascii=False, default=str),
                # bool("false") is True — parse strings defensively (python-reviewer LOW)
                "is_reference": raw_meta.get("is_reference")
                if isinstance(raw_meta.get("is_reference"), bool)
                else str(raw_meta.get("is_reference", "")).lower() in ("1", "true", "yes"),
                "crawl_date": raw_meta.get("crawl_date"),
            }
        )

    if skipped_no_vector:
        logger.warning(f"Skipped {skipped_no_vector} chunk(s) with no vector available")

    if size_flags["oversized"] or size_flags["undersized"]:
        logger.warning(
            f"Size-gate summary: {size_flags['oversized']} oversized (> max), "
            f"{size_flags['undersized']} undersized (< min) out of {len(records)} "
            f"written record(s) — all kept, flagged via metadata_json['size_flag']"
        )

    if not records:
        # T1.5 (python-reviewer HIGH): a full overwrite of an empty batch would
        # silently destroy the production table. Guard both write paths.
        logger.error("No records to write — aborting ingest, table preserved")
        return False

    if incremental and table_exists:
        table = db.open_table(target)

        # Pre-ingest tag for rollback safety
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            from src.infrastructure.lancedb_versioning import tag_release
            tag_release(table, f"pre-ingest-{ts}")
        except Exception:
            logger.warning(f"tag_release failed (non-fatal): {ts}")

        # Orphan delete only when table has "source" column (legacy rag2025).
        # husc_chunks (CHUNK_SCHEMA) has no "source" column — skip.
        where_clause = None
        if "source" in table.schema.names:
            sources_in_batch = sorted({c.get("source", "unknown") for c in chunks})
            if sources_in_batch:
                escaped = ", ".join(f"'{_esc(s)}'" for s in sources_in_batch)
                where_clause = f"source IN ({escaped})"

        job = (
            table.merge_insert("id")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
        )
        if where_clause:
            job.when_not_matched_by_source_delete(condition=where_clause)
        else:
            # husc_chunks has no `source` column, so orphans are resolved from
            # metadata_json provenance instead. An UNCONDITIONAL delete here
            # would wipe every row outside the batch: in --incremental mode the
            # batch is only the CHANGED files, and 581 of 2068 production rows
            # come from other ingest scripts entirely.
            orphans = _orphan_ids(
                table,
                {c["ingest_file"] for c in chunks},
                # From `chunks`, NOT `records`: records drops any chunk whose
                # vector went missing (:309-311), and that loss is silent. A
                # chunk still present in the JSONL must never count as an orphan.
                {c["id"] for c in chunks},
            )
            if orphans:
                escaped_orphans = ", ".join(f"'{_esc(o)}'" for o in orphans)
                job.when_not_matched_by_source_delete(
                    condition=f"id IN ({escaped_orphans})"
                )
                logger.info(f"Orphan delete: {len(orphans)} row(s) from batch files")
        job.execute(records)

        logger.info(
            f"Incremental ingest complete: merged={len(records)}, total_rows={table.count_rows()}, "
            f"table={target}, uri={settings.LANCEDB_URI}"
        )
    else:
        if table_exists:
            # T1.4 (G4): pre-ingest tag BEFORE overwrite so the previous table
            # state is recoverable. The old code overwrote first, so the
            # pre-overwrite version was never tagged (a mistaken default run
            # dropped 581/2068 rows with no rollback point).
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                from src.infrastructure.lancedb_versioning import tag_release
                table_tmp = db.open_table(target)
                tag_release(table_tmp, f"pre-ingest-{ts}")
            except Exception:
                logger.warning(f"tag_release failed (non-fatal): {ts}")
            table = db.create_table(target, data=records, mode="overwrite")
            logger.info(
                f"Full ingest complete (overwrite): table={target}, "
                f"rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
            )
        else:
            table = db.create_table(target, data=records)
            logger.info(
                f"Initial ingest complete: table={target}, "
                f"rows={table.count_rows()}, uri={settings.LANCEDB_URI}"
            )

    _refresh_manifest_for_all_files(chunks_dir)

    # M3-N3a: trigger graph sync for new chunks
    new_chunk_ids = [r["id"] for r in records]
    _trigger_graph_sync(new_chunk_ids)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest chunked JSONL files into LanceDB")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only ingest changed chunk files and append new chunk IDs",
    )
    parser.add_argument(
        "--canonical-corpus",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Ingest a built canonical corpus (ticket 15) instead of the legacy "
            "chunked JSONL set. DIR defaults to data/canonical_corpus. The "
            "corpus's stamp must match the running code or the run is refused."
        ),
    )
    args = parser.parse_args()

    if args.canonical_corpus is not None:
        try:
            ingest_canonical_corpus(args.canonical_corpus or None)
        except IngestCorpusError:
            raise SystemExit(1)
    else:
        ok = ingest_lancedb(incremental=args.incremental)
        raise SystemExit(0 if ok else 1)
