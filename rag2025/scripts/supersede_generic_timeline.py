"""Supersede the 2 generic timeline chunks with the new id=74 specific-dates chunk (S17.1).

TDD contract: see ``tests/scripts/test_supersede_generic_timeline.py``.

Approach (LanceDB rows are immutable-ish — delete + re-insert with updated metadata):
  1. For each target chunk_id, read the row's existing ``metadata_json``.
  2. If ``is_superseded`` is already True, no-op (idempotent re-run).
  3. Otherwise:
       a. Mutate the metadata dict to set
            ``is_superseded = True``
            ``superseded_by   = "qa_dang_ky_xet_tuyen_2026_v2"``
            ``superseded_reason = "generic placeholder timeline; specific id=74 dates ingested"``
       b. Delete the old row by chunk_id.
       c. Re-add a new row with the same id/chunk_id/text/source/faq_type, the
          updated metadata_json, and a re-embedded vector (the original vector
          is recoverable from the table; we always re-embed to keep semantics
          consistent and avoid stale-vector drift).

The table is NEVER dropped — only the 2 targeted rows are touched.

Run (real, against the live ``rag2025`` table):

    python scripts/supersede_generic_timeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

# Make ``config`` and ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402


# ---------------------------------------------------------------------------
# TDD-locked constants
# ---------------------------------------------------------------------------
TARGET_CHUNK_IDS: List[str] = [
    "qa_lich_tuyen_sinh_2026",
    "qa_xet_hoc_ba_quy_trinh_2026",
]

SUPERSEDED_BY: str = "qa_dang_ky_xet_tuyen_2026_v2"

SUPERSEDED_REASON: str = (
    "generic placeholder timeline; specific id=74 dates ingested"
)


# ---------------------------------------------------------------------------
# Protocols — keep this script testable with fakes
# ---------------------------------------------------------------------------
class _TableLike(Protocol):
    def add(self, rows: List[Dict[str, Any]]) -> Any: ...
    def delete(self, where_clause: str) -> Any: ...
    def to_pandas(self, columns: Optional[List[str]] = None): ...


class _EmbedderLike(Protocol):
    def encode_documents(self, texts: List[str]): ...
    def encode_single(self, text: str): ...


# ---------------------------------------------------------------------------
# Core routine (pure — testable)
# ---------------------------------------------------------------------------
def _fetch_row(table: _TableLike, chunk_id: str) -> Optional[Dict[str, Any]]:
    df = table.to_pandas()
    if "chunk_id" not in df.columns:
        return None
    match = df[df["chunk_id"] == chunk_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _build_superseded_row(old_row: Dict[str, Any], new_metadata: Dict[str, Any], vector: List[float]) -> Dict[str, Any]:
    return {
        "id": old_row.get("id") or old_row.get("chunk_id"),
        "chunk_id": old_row.get("chunk_id") or old_row.get("id"),
        "text": old_row.get("text", ""),
        "source": old_row.get("source", ""),
        "faq_type": old_row.get("faq_type", ""),
        "metadata_json": json.dumps(new_metadata, ensure_ascii=False, default=str),
        "vector": vector,
    }


def _supersede_one(
    table: _TableLike,
    embedder: _EmbedderLike,
    chunk_id: str,
) -> Dict[str, Any]:
    row = _fetch_row(table, chunk_id)
    if row is None:
        logger.warning(f"chunk_id={chunk_id!r} not present in table; skipping")
        return {
            "chunk_id": chunk_id,
            "found": False,
            "already_superseded": False,
            "is_superseded_now": False,
        }

    raw_meta = row.get("metadata_json") or "{}"
    meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})

    if bool(meta.get("is_superseded")):
        logger.info(f"chunk_id={chunk_id!r} already superseded; no-op")
        return {
            "chunk_id": chunk_id,
            "found": True,
            "already_superseded": True,
            "is_superseded_now": False,
        }

    # Mark superseded.
    meta["is_superseded"] = True
    meta["superseded_by"] = SUPERSEDED_BY
    meta["superseded_reason"] = SUPERSEDED_REASON

    # Re-embed the text (cheap; Qwen3 batch of 1).
    text = row.get("text", "")
    if hasattr(embedder, "encode_single"):
        new_vec = embedder.encode_single(text)
    else:
        new_vec = embedder.encode_documents([text])[0]
    new_vec_list = new_vec.tolist() if hasattr(new_vec, "tolist") else list(new_vec)

    new_row = _build_superseded_row(row, meta, new_vec_list)

    # delete + add (idempotent-ish: if delete fails or has 0 rows, that's ok)
    table.delete(f"chunk_id = '{chunk_id}'")
    table.add([new_row])

    logger.info(f"chunk_id={chunk_id!r} marked is_superseded=True by {SUPERSEDED_BY!r}")
    return {
        "chunk_id": chunk_id,
        "found": True,
        "already_superseded": False,
        "is_superseded_now": True,
    }


def supersede_generic_timeline(
    table: _TableLike,
    embedder: _EmbedderLike,
) -> Dict[str, Dict[str, Any]]:
    """Flip is_superseded=True on each target chunk_id. Idempotent."""
    summary: Dict[str, Dict[str, Any]] = {}
    for cid in TARGET_CHUNK_IDS:
        summary[cid] = _supersede_one(table=table, embedder=embedder, chunk_id=cid)
    return summary


# ---------------------------------------------------------------------------
# Real-DB entry point
# ---------------------------------------------------------------------------
def run_against_live_db() -> Dict[str, Dict[str, Any]]:
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)
    return supersede_generic_timeline(table=table, embedder=embedder)


if __name__ == "__main__":
    summary = run_against_live_db()
    print("Supersede summary:")
    for cid, info in summary.items():
        print(
            f"  {cid}: found={info.get('found')} "
            f"already_superseded={info.get('already_superseded')} "
            f"is_superseded_now={info.get('is_superseded_now')}"
        )
