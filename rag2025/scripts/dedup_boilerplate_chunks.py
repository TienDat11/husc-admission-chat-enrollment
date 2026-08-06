"""Dedup exact-duplicate SHORT boilerplate chunks in LanceDB (S17.2).

TDD contract: see ``tests/scripts/test_dedup_boilerplate_chunks.py``.

The live ``rag2025`` table (1656 rows, dim=1024, Qwen3-Embedding-0.6B) contains
many EXACT-duplicate header / boilerplate chunks (len<80, count>=2 each),
crawled from multiple HUSC notifications. These duplicates flood the top-k for
short-keyword queries (e.g. ``lịch xét tuyển học bạ 2026``), pushing the real
timeline chunk out of top-3.

Strategy (LanceDB rows are immutable-ish — delete + re-insert with updated
metadata, mirroring ``scripts/supersede_generic_timeline.py``):

  1. Load all rows. Group by EXACT ``text`` (whitespace-stripped) where
     ``len(text.strip()) < SHORT_THRESHOLD`` (=80).
  2. For each group with count >= 2:
       * Sort the chunk_ids in the group with a STABLE order and keep the
         FIRST as the canonical.
       * For the REST: set ``metadata_json.is_superseded=True``,
         ``superseded_by = <kept_chunk_id>``,
         ``superseded_reason = "exact-duplicate boilerplate (dedup); canonical copy = <kept_chunk_id>"``.
       * ``table.delete(chunk_id=...)`` then ``table.add(replacement_row)``
         carrying the updated metadata_json + re-embedded vector.
  3. A SECOND pass (T1.6) handles LONGER boilerplate: group rows with
     ``len(text.strip()) >= LONG_THRESHOLD`` (=80, == SHORT_THRESHOLD) by their
     NORMALIZED text (lowercase + whitespace-collapse -> sha256) and supersede
     all-but-one per group. The floor is 80 because the LIVE letterhead
     fragments are 80-199 chars (the plan's original >=200 guess missed them).
  4. Single-occurrence short chunks are NEVER touched.
  5. Already-superseded rows are skipped (idempotent re-run = no-op).
  6. NEVER touch the kept canonical copy.

The table is NEVER dropped — only the targeted duplicate rows are
delete+re-added with the updated metadata flag.

Run (real, against the live ``rag2025`` table):

    python scripts/dedup_boilerplate_chunks.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# Make ``config`` and ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402


# ---------------------------------------------------------------------------
# TDD-locked constants
# ---------------------------------------------------------------------------
SHORT_THRESHOLD: int = 80
SUPERSEDED_REASON_PREFIX: str = "exact-duplicate boilerplate (dedup); canonical copy = "
# T1.6: long-text dedup floor (NORMALIZED match: lowercase + whitespace-collapse).
# NOTE: the plan assumed boilerplate >= 200 chars, but the LIVE table's letterhead
# fragments are 80-199 chars (166 excess rows there) — so the floor is 80 (==
# SHORT_THRESHOLD). This mirrors sweep_lancedb_noise._normalize_text so the
# sweep's active dup metric drops below the plan's <50 target.
LONG_THRESHOLD: int = 80
LONG_SUPERSEDED_REASON_PREFIX: str = "exact-duplicate long-text (dedup); canonical copy = "


# ---------------------------------------------------------------------------
# Protocols — keep this script testable with fakes
# ---------------------------------------------------------------------------
class _TableLike(Protocol):
    def add(self, rows: List[Dict[str, Any]]) -> Any: ...
    def delete(self, where_clause: str) -> Any: ...
    def update(self, where: str, values: Dict[str, Any]) -> Any: ...
    def to_pandas(self, columns: Optional[List[str]] = None): ...


class _EmbedderLike(Protocol):
    def encode_documents(self, texts: List[str]): ...
    def encode_single(self, text: str): ...


# ---------------------------------------------------------------------------
# Pure helpers (testable)
# ---------------------------------------------------------------------------
def _load_all_rows(table: _TableLike) -> List[Dict[str, Any]]:
    """Read the entire table as a list of row dicts (no vector column)."""
    df = table.to_pandas()
    if df is None or len(df) == 0:
        return []
    # Drop NaN-mapped None values; keep schema keys.
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(row.to_dict())
    return records


def _parse_meta(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _group_duplicates(
    rows: List[Dict[str, Any]],
    short_threshold: int = SHORT_THRESHOLD,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group rows by EXACT stripped text where len<short_threshold.

    Returns a mapping ``text -> [rows in source order]`` containing only
    short-text groups with >= 2 occurrences.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        text = (r.get("text") or "").strip()
        if len(text) >= short_threshold:
            continue
        if not text:
            continue
        # Use stripped text as group key. If the original has no extra
        # whitespace, this is the exact-match condition.
        buckets.setdefault(text, []).append(r)
    # Drop singletons — single-occurrence short chunks are NOT duplicates.
    return {t: rs for t, rs in buckets.items() if len(rs) >= 2}


def _normalize_long(text: str) -> str:
    """Return a stable sha256 key for long-text dedup.

    Mirrors ``sweep_lancedb_noise._normalize_text``: lowercase + collapse all
    whitespace runs to a single space + strip. The hexdigest is the grouping key
    so the raw (possibly very long) text is never used as a dict key.
    """
    norm = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _group_long_duplicates(
    rows: List[Dict[str, Any]],
    long_threshold: int = LONG_THRESHOLD,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group long rows (len >= long_threshold) by NORMALIZED text, >= 2 copies.

    Already-superseded rows are excluded so they can never be picked as the
    canonical copy. Returns ``hash -> [rows in source order]`` for groups that
    have at least 2 ACTIVE (non-superseded) copies.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        text = r.get("text") or ""
        stripped = text.strip()
        if len(stripped) < long_threshold or not stripped:
            continue
        meta = _parse_meta(r.get("metadata_json"))
        if bool(meta.get("is_superseded")):
            continue
        key = _normalize_long(text)
        buckets.setdefault(key, []).append(r)
    # Drop singletons — unique long chunks are NOT duplicates.
    return {k: rs for k, rs in buckets.items() if len(rs) >= 2}


def _embed_text(embedder: _EmbedderLike, text: str) -> List[float]:
    """Embed a single text; returns a plain list of floats."""
    if hasattr(embedder, "encode_single"):
        new_vec = embedder.encode_single(text)
    else:
        new_vec = embedder.encode_documents([text])[0]
    return new_vec.tolist() if hasattr(new_vec, "tolist") else list(new_vec)


def _find_duplicate_ids(rows: List[Dict[str, Any]]) -> set:
    """Return the set of id values that appear in MORE than one row.

    The live flat ``husc_chunks`` table has 61 non-unique ``id`` values (distinct
    content sharing an id). Deleting/updating by a duplicated id would corrupt
    the sibling, so these ids must never be targeted by a ``where id = ...``.
    """
    counts: Dict[str, int] = {}
    for r in rows:
        v = str(r.get("chunk_id") or r.get("id") or "")
        if v:
            counts[v] = counts.get(v, 0) + 1
    return {v for v, c in counts.items() if c > 1}


def _supersede_one(
    table: _TableLike,
    embedder: _EmbedderLike,
    row: Dict[str, Any],
    kept_chunk_id: str,
    reason_prefix: str = SUPERSEDED_REASON_PREFIX,
    dup_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Mark a single duplicate row as superseded. Idempotent.

    Schema-agnostic:
      * Legacy ``chunk_id`` schema → delete + re-add (re-embed, preserves every
        original column). Used by tests and the ``rag2025`` table.
      * Live flat ``husc_chunks`` schema (no ``chunk_id`` column) → in-place
        ``table.update``. Text is unchanged so the existing vector stays valid
        (no drift) and the row count is preserved.

    The live flat table has 61 non-unique ``id`` values; deleting/updating by
    ``id`` would corrupt the sibling row. Rows whose id is in ``dup_ids`` are
    SKIPPED (safe; leaves a few duplicates un-deduped rather than losing data).
    """
    # Prefer the chunk_id column if the row has one; else target by id.
    id_col = "chunk_id" if (row.get("chunk_id")) else "id"
    id_val = str(row.get(id_col) or row.get("id") or row.get("chunk_id") or "")
    if not id_val:
        return {
            "chunk_id": "",
            "already_superseded": False,
            "is_superseded_now": False,
        }

    meta = _parse_meta(row.get("metadata_json"))
    if bool(meta.get("is_superseded")):
        logger.info(f"{id_col}={id_val!r} already superseded; no-op")
        return {
            "chunk_id": id_val,
            "already_superseded": True,
            "is_superseded_now": False,
        }

    # Safety: never delete/update a row whose id is duplicated in the table.
    if dup_ids and id_val in dup_ids:
        logger.warning(
            f"{id_col}={id_val!r} id duplicated in table; skipping to avoid "
            f"corrupting the sibling row"
        )
        return {
            "chunk_id": id_val,
            "already_superseded": False,
            "is_superseded_now": False,
            "skipped_dup_id": True,
        }

    # Update metadata with the supersede flags.
    meta["is_superseded"] = True
    meta["superseded_by"] = kept_chunk_id
    meta["superseded_reason"] = f"{reason_prefix}{kept_chunk_id}"
    new_json = json.dumps(meta, ensure_ascii=False, default=str)

    if id_col == "chunk_id":
        # --- Legacy schema: delete + re-add with re-embed ---
        text = row.get("text", "")
        new_vec_list = _embed_text(embedder, text)

        # Preserve the FULL original row; only mutate metadata_json / vector /
        # the flat superseded_by (when it exists as a column).
        new_row = dict(row)
        new_row["id"] = row.get("id") or id_val
        new_row["chunk_id"] = new_row.get("chunk_id") or id_val
        new_row["metadata_json"] = new_json
        new_row["vector"] = new_vec_list
        if "superseded_by" in new_row:
            new_row["superseded_by"] = kept_chunk_id

        # delete + add (idempotent: existing superseded row just gets a fresh
        # flag write — but we short-circuit above, so only reached on first run).
        table.delete(f"chunk_id = '{id_val}'")
        table.add([new_row])
    else:
        # --- Live flat schema: in-place UPDATE (row count preserved) ---
        values: Dict[str, Any] = {"metadata_json": new_json}
        if "superseded_by" in row:
            values["superseded_by"] = kept_chunk_id
        table.update(f"id = '{id_val}'", values=values)

    logger.info(
        f"{id_col}={id_val!r} marked is_superseded=True by {kept_chunk_id!r}"
    )
    return {
        "chunk_id": id_val,
        "already_superseded": False,
        "is_superseded_now": True,
    }


def dedup_boilerplate(
    table: _TableLike,
    embedder: _EmbedderLike,
    short_threshold: int = SHORT_THRESHOLD,
    long_threshold: int = LONG_THRESHOLD,
) -> Dict[str, Any]:
    """Find exact-duplicate short AND long chunks, supersede all-but-one per group.

    Returns a summary dict:
      ``duplicate_groups``      → number of groups (short + long) with >= 2 rows
      ``chunks_superseded``     → total number of rows flipped to superseded
      ``groups``                → list of ``{text_snippet, kept_chunk_id,
                                  superseded_chunk_ids}`` records
      ``short_rows_considered`` → number of short rows that entered any short group
      ``long_rows_considered``  → number of long rows that entered any long group
      ``long_rows_skipped``     → number of rows with len>=short_threshold that
                                  were NOT superseded (untouched long content)
      ``singletons_skipped``    → number of short rows that appeared only once
                                  (untouched)
      ``long_duplicate_groups`` → number of LONG groups (normalized-text dups)
      ``long_chunks_superseded``→ number of long rows flipped to superseded
    """
    all_rows = _load_all_rows(table)
    dup_ids = _find_duplicate_ids(all_rows)

    duplicate_groups = _group_duplicates(all_rows, short_threshold=short_threshold)
    long_duplicate_groups = _group_long_duplicates(all_rows, long_threshold=long_threshold)

    # Count short singleton chunks: short rows that did NOT fall into any
    # duplicate group. We recompute by subtracting groups from the short pool.
    short_rows_in_groups = sum(len(rs) for rs in duplicate_groups.values())
    short_pool_size = sum(
        1 for r in all_rows
        if 0 < len((r.get("text") or "").strip()) < short_threshold
    )
    singletons_skipped = short_pool_size - short_rows_in_groups

    groups_summary: List[Dict[str, Any]] = []
    chunks_superseded = 0

    # --- SHORT pass (exact stripped-text groups) ---
    # Stable order: groups by first-seen text (the dict was built in
    # source-iteration order on Python 3.7+), then within a group the kept
    # row is whichever has the lowest chunk_id in stable sort.
    for text, group_rows in duplicate_groups.items():
        # Sort rows by chunk_id (string) to make the kept canonical DETERMINISTIC
        # and independent of insertion order. Tests can rely on this.
        sorted_rows = sorted(
            group_rows,
            key=lambda r: str(r.get("chunk_id") or r.get("id") or ""),
        )
        kept_row = sorted_rows[0]
        kept_chunk_id = str(kept_row.get("chunk_id") or kept_row.get("id") or "")
        superseded_ids: List[str] = []

        for r in sorted_rows[1:]:
            res = _supersede_one(
                table=table,
                embedder=embedder,
                row=r,
                kept_chunk_id=kept_chunk_id,
                dup_ids=dup_ids,
            )
            if res.get("is_superseded_now"):
                chunks_superseded += 1
                superseded_ids.append(res["chunk_id"])

        # Build a short text snippet for the report.
        snippet = text if len(text) <= 60 else text[:57] + "..."

        groups_summary.append({
            "text_snippet": snippet,
            "text_len": len(text),
            "group_size": len(group_rows),
            "kept_chunk_id": kept_chunk_id,
            "superseded_chunk_ids": superseded_ids,
        })

    # --- LONG pass (normalized-text groups, len >= long_threshold) ---
    long_chunks_superseded = 0
    long_rows_considered = 0
    for _key, group_rows in long_duplicate_groups.items():
        sorted_rows = sorted(
            group_rows,
            key=lambda r: str(r.get("chunk_id") or r.get("id") or ""),
        )
        kept_row = sorted_rows[0]
        kept_chunk_id = str(kept_row.get("chunk_id") or kept_row.get("id") or "")
        kept_text = (kept_row.get("text") or "").strip()
        superseded_ids: List[str] = []
        long_rows_considered += len(group_rows)

        for r in sorted_rows[1:]:
            res = _supersede_one(
                table=table,
                embedder=embedder,
                row=r,
                kept_chunk_id=kept_chunk_id,
                reason_prefix=LONG_SUPERSEDED_REASON_PREFIX,
                dup_ids=dup_ids,
            )
            if res.get("is_superseded_now"):
                chunks_superseded += 1
                long_chunks_superseded += 1
                superseded_ids.append(res["chunk_id"])

        snippet = kept_text if len(kept_text) <= 60 else kept_text[:57] + "..."
        groups_summary.append({
            "text_snippet": snippet,
            "text_len": len(kept_text),
            "group_size": len(group_rows),
            "kept_chunk_id": kept_chunk_id,
            "superseded_chunk_ids": superseded_ids,
        })

    # Long rows that were NOT superseded by the long pass (untouched content).
    long_rows_total = sum(
        1 for r in all_rows if len((r.get("text") or "").strip()) >= short_threshold
    )
    long_rows_skipped = long_rows_total - long_chunks_superseded

    return {
        "duplicate_groups": len(duplicate_groups) + len(long_duplicate_groups),
        "chunks_superseded": chunks_superseded,
        "groups": groups_summary,
        "short_rows_considered": short_rows_in_groups,
        "long_rows_considered": long_rows_considered,
        "long_rows_skipped": long_rows_skipped,
        "singletons_skipped": singletons_skipped,
        "long_duplicate_groups": len(long_duplicate_groups),
        "long_chunks_superseded": long_chunks_superseded,
    }


# ---------------------------------------------------------------------------
# Real-DB entry point
# ---------------------------------------------------------------------------
def run_against_live_db() -> Dict[str, Any]:
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at "
            f"'{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)
    return dedup_boilerplate(table=table, embedder=embedder)


def _print_summary(summary: Dict[str, Any]) -> None:
    print("Boilerplate dedup summary:")
    print(f"  duplicate_groups     : {summary['duplicate_groups']}")
    print(f"  chunks_superseded    : {summary['chunks_superseded']}")
    print(f"  short_rows_considered: {summary['short_rows_considered']}")
    print(f"  singletons_skipped   : {summary['singletons_skipped']}")
    print(f"  long_rows_skipped    : {summary['long_rows_skipped']}")
    print()
    print("Duplicate groups (text_snippet, kept_chunk_id, superseded_chunk_ids):")
    for g in summary["groups"]:
        print(
            f"  len={g['text_len']:>3}  size={g['group_size']:>2}  "
            f"kept={g['kept_chunk_id']!r}"
        )
        for sid in g["superseded_chunk_ids"]:
            print(f"      └─ superseded: {sid!r}")
        # Print first 60 chars of the snippet for the report
        snippet = g["text_snippet"].replace("\n", "\\n")
        print(f"      text: {snippet!r}")


if __name__ == "__main__":
    summary = run_against_live_db()
    _print_summary(summary)
