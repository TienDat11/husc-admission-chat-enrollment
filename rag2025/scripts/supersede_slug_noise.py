"""Supersede diacritic-stripped slug-header noise + incomplete needs_review rows (T1.7).

Context (corpus hygiene, 2026-08): the live ``husc_chunks`` table contains 128
chunks whose text begins with a ``### `` markdown header that is a
diacritic-stripped ASCII slug (e.g. ``### thi sinh bogddt tt 08 dieu 2 xet tuyen
thang 2025``) — navigation artifacts from an ingestion pass that stripped
Vietnamese diacritics in the chunk header. 20 of the 128 are curated ``husc_*``
rows (``husc_news_*``, ``husc_summary_*``, ``husc_hoc_phi_*``, ``husc_uoc_tinh_*``)
that are REAL content and are preserved. The remaining 108 diacritic-stripped slug
rows are superseded (soft delete: ``is_superseded=True``) so the retriever drops
them.

Additionally, 19 rows carry ``needs_review=True`` with near-empty metadata
(``thongbao_76/77/78/79_*``, no data_year, no doc_type). They are superseded too;
source jsonl exists under ``data/chunked/`` so a proper re-ingest can be done as a
follow-up.

Approach mirrors ``scripts/supersede_stale_aggregates.py`` (LanceDB rows are
immutable-ish — delete + re-insert with updated metadata, re-embedding the text).
Idempotent: already-superseded rows are a no-op. The table is NEVER dropped.

Run (real, against the live table):

    python scripts/supersede_slug_noise.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402


# ---------------------------------------------------------------------------
# TDD-locked constants
# ---------------------------------------------------------------------------
# A diacritic-stripped slug header prefix: "### " followed by ASCII lowercase /
# digits / spaces / underscores / dashes. NOT end-anchored — most slug rows put
# the Vietnamese body on the SAME line after two spaces (e.g.
# "### thi sinh Khoa Hoc ...  Trong tuyển sinh..."), so we only need the header
# PREFIX to be ASCII-stripped. The curated ``husc_*`` rows are excluded by
# KEEP_PREFIXES, so this regex + the prefix check selects exactly the noise.
SLUG_HEADER_RE = re.compile(r"^### [a-z0-9][a-z0-9 _-]*")

# Curated content prefixes that legitimately use a "### " header — NEVER superseded.
KEEP_PREFIXES: tuple = ("husc_",)

SLUG_REASON = "diacritic-stripped slug-header noise (T1.7 hygiene); not curated content"
NEEDS_REVIEW_REASON = "incomplete-metadata needs_review row (T1.7 hygiene); source jsonl available for re-ingest"


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
# Predicates (pure — testable)
# ---------------------------------------------------------------------------
def _is_slug_noise(text: str, chunk_id: Any) -> bool:
    """True for a diacritic-stripped ``### `` slug-header chunk.

    The first line must be an ASCII-only slug (lowercase / digits / space / - / _).
    Curated ``husc_*`` rows are excluded even though they also start with ``### ``.
    """
    st = (text or "").strip()
    if not st.startswith("### "):
        return False
    cid = str(chunk_id or "")
    if cid.startswith(KEEP_PREFIXES):
        return False
    first_line = st.split("\n", 1)[0]
    return bool(SLUG_HEADER_RE.match(first_line))


def _is_needs_review(metadata: Dict[str, Any]) -> bool:
    return bool(metadata.get("needs_review"))


# ---------------------------------------------------------------------------
# Core routine (pure — testable)
# ---------------------------------------------------------------------------
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


def _load_all_rows(table: _TableLike) -> List[Dict[str, Any]]:
    df = table.to_pandas()
    if df is None or len(df) == 0:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(row.to_dict())
    return records


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
    reason: str,
    dup_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Mark a single row as superseded. Idempotent.

    Schema-agnostic (see dedup_boilerplate_chunks._supersede_one): legacy
    ``chunk_id`` schema uses delete + re-add; live flat ``husc_chunks`` schema
    uses in-place ``table.update`` (vector unchanged, row count preserved). Rows
    whose id is duplicated in the table are SKIPPED to avoid corrupting the
    sibling.
    """
    id_col = "chunk_id" if (row.get("chunk_id")) else "id"
    id_val = str(row.get(id_col) or row.get("id") or row.get("chunk_id") or "")
    if not id_val:
        return {"chunk_id": "", "already_superseded": False, "is_superseded_now": False}

    meta = _parse_meta(row.get("metadata_json"))
    if bool(meta.get("is_superseded")):
        logger.info(f"{id_col}={id_val!r} already superseded; no-op")
        return {"chunk_id": id_val, "already_superseded": True, "is_superseded_now": False}

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

    meta["is_superseded"] = True
    meta["superseded_by"] = ""
    meta["superseded_reason"] = reason
    new_json = json.dumps(meta, ensure_ascii=False, default=str)

    if id_col == "chunk_id":
        # --- Legacy schema: delete + re-add with re-embed ---
        new_vec_list = _embed_text(embedder, row.get("text", ""))
        new_row = dict(row)
        new_row["id"] = row.get("id") or id_val
        new_row["chunk_id"] = new_row.get("chunk_id") or id_val
        new_row["metadata_json"] = new_json
        new_row["vector"] = new_vec_list
        if "superseded_by" in new_row:
            new_row["superseded_by"] = ""
        table.delete(f"chunk_id = '{id_val}'")
        table.add([new_row])
    else:
        # --- Live flat schema: in-place UPDATE (row count preserved) ---
        values: Dict[str, Any] = {"metadata_json": new_json}
        if "superseded_by" in row:
            values["superseded_by"] = ""
        table.update(f"id = '{id_val}'", values=values)

    logger.info(f"{id_col}={id_val!r} marked is_superseded=True ({reason})")
    return {"chunk_id": id_val, "already_superseded": False, "is_superseded_now": True}


def supersede_slug_noise(
    table: _TableLike,
    embedder: _EmbedderLike,
) -> Dict[str, Any]:
    """Supersede slug-noise + needs_review rows. Idempotent.

    Returns a summary dict with ``slug_superseded``, ``needs_review_superseded``
    and ``total_superseded`` counts plus ``items`` (per-row outcomes).
    """
    all_rows = _load_all_rows(table)
    dup_ids = _find_duplicate_ids(all_rows)
    slug_superseded = 0
    needs_review_superseded = 0
    items: List[Dict[str, Any]] = []

    for r in all_rows:
        meta = _parse_meta(r.get("metadata_json"))
        chunk_id = str(r.get("chunk_id") or r.get("id") or "")
        reason = None
        kind = None
        if _is_slug_noise(r.get("text"), chunk_id):
            reason = SLUG_REASON
            kind = "slug"
        elif _is_needs_review(meta):
            reason = NEEDS_REVIEW_REASON
            kind = "needs_review"

        if reason is None:
            continue

        res = _supersede_one(table=table, embedder=embedder, row=r, reason=reason, dup_ids=dup_ids)
        items.append({**res, "kind": kind})
        if res.get("is_superseded_now"):
            if kind == "slug":
                slug_superseded += 1
            else:
                needs_review_superseded += 1

    return {
        "slug_superseded": slug_superseded,
        "needs_review_superseded": needs_review_superseded,
        "total_superseded": slug_superseded + needs_review_superseded,
        "items": items,
    }


# ---------------------------------------------------------------------------
# Real-DB entry point
# ---------------------------------------------------------------------------
def run_against_live_db() -> Dict[str, Any]:
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)
    return supersede_slug_noise(table=table, embedder=embedder)


if __name__ == "__main__":
    summary = run_against_live_db()
    print("Supersede slug-noise summary:")
    print(f"  slug_superseded        : {summary['slug_superseded']}")
    print(f"  needs_review_superseded: {summary['needs_review_superseded']}")
    print(f"  total_superseded       : {summary['total_superseded']}")
