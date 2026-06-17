"""Supersede 2 STALE aggregate chunks that conflict with their corrected *_v2 twins (S18).

Context (QA bug 2026-06-09): the compound question
"ngành nào nhiều chỉ tiêu nhất, ngành nào ít nhất, và tổng chỉ tiêu là bao nhiêu"
returned the WRONG total (1969 instead of 2079) and a phantom "3 ngành cùng 20
chỉ tiêu". Root cause: two stale aggregate chunks still live in LanceDB next to
their corrected ``*_v2`` twins (same ``data_year=2026``), so the retriever
surfaces both and the generator mixes them:

  * ``bang_xep_hang_chi_tieu_2026``  — 25+1 majors, total 1969, still lists the
    discontinued major ``7480107TD``. Corrected twin: ``..._v2`` (28 majors,
    total ~2079, dead major removed).
  * ``tuyensinh_overview_2026``      — also states total 1969. Corrected twin:
    ``tuyensinh_overview_2026_v2``.

Fix: flip ``is_superseded=True`` on the 2 stale chunks so the retriever's
temporal filter (``lancedb_retrieval.py`` Rule 2) drops them for non-historical
queries. The corrected ``*_v2`` chunks are NEVER touched.

TDD contract: ``tests/scripts/test_supersede_stale_aggregates.py``.

Approach (LanceDB rows are immutable-ish — delete + re-insert with updated
metadata, re-embedding the text to avoid stale-vector drift). Idempotent:
already-superseded rows are a no-op. The table is NEVER dropped — only the 2
targeted rows are touched.

Run (real, against the live ``rag2025`` table):

    python scripts/supersede_stale_aggregates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402


# ---------------------------------------------------------------------------
# TDD-locked constants — the 2 stale aggregates and their corrected twins.
# NB: every target MUST be a non-``_v2`` id; the test asserts we never
# supersede a corrected twin.
# ---------------------------------------------------------------------------
TARGET_CHUNK_IDS: List[str] = [
    "bang_xep_hang_chi_tieu_2026",
    "tuyensinh_overview_2026",
]

# Map each stale id → the corrected twin that replaces it (recorded in
# metadata.superseded_by for traceability).
SUPERSEDED_BY_MAP: Dict[str, str] = {
    "bang_xep_hang_chi_tieu_2026": "bang_xep_hang_chi_tieu_2026_v2",
    "tuyensinh_overview_2026": "tuyensinh_overview_2026_v2",
}

SUPERSEDED_REASON: str = (
    "stale aggregate (total 1969 + discontinued major 7480107TD); "
    "corrected *_v2 twin has total ~2079 and 28 current majors"
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


def _build_superseded_row(
    old_row: Dict[str, Any], new_metadata: Dict[str, Any], vector: List[float]
) -> Dict[str, Any]:
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

    meta["is_superseded"] = True
    meta["superseded_by"] = SUPERSEDED_BY_MAP.get(chunk_id, "")
    meta["superseded_reason"] = SUPERSEDED_REASON

    text = row.get("text", "")
    if hasattr(embedder, "encode_single"):
        new_vec = embedder.encode_single(text)
    else:
        new_vec = embedder.encode_documents([text])[0]
    new_vec_list = new_vec.tolist() if hasattr(new_vec, "tolist") else list(new_vec)

    new_row = _build_superseded_row(row, meta, new_vec_list)

    table.delete(f"chunk_id = '{chunk_id}'")
    table.add([new_row])

    logger.info(
        f"chunk_id={chunk_id!r} marked is_superseded=True "
        f"by {SUPERSEDED_BY_MAP.get(chunk_id)!r}"
    )
    return {
        "chunk_id": chunk_id,
        "found": True,
        "already_superseded": False,
        "is_superseded_now": True,
    }


def supersede_stale_aggregates(
    table: _TableLike,
    embedder: _EmbedderLike,
) -> Dict[str, Dict[str, Any]]:
    """Flip is_superseded=True on each stale aggregate chunk. Idempotent.

    Never touches the corrected ``*_v2`` twins (they are not in
    ``TARGET_CHUNK_IDS``).
    """
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
    return supersede_stale_aggregates(table=table, embedder=embedder)


if __name__ == "__main__":
    summary = run_against_live_db()
    print("Supersede stale-aggregates summary:")
    for cid, info in summary.items():
        print(
            f"  {cid}: found={info.get('found')} "
            f"already_superseded={info.get('already_superseded')} "
            f"is_superseded_now={info.get('is_superseded_now')}"
        )
