"""Remove EXACT-duplicate rows from LanceDB (S18 — keep 1 per chunk_id).

Context (2026-06-10): 4 chunk_ids exist as TWO byte-identical rows
(`bang_xep_hang_chi_tieu_2026_v2`, `liet_ke_28_nganh_2026_v2`,
`tuyensinh_overview_2026_v2`, `nganh_ngung_tuyen_2026`). A duplicate row wastes
a top-k slot and biases retrieval toward the duplicated content.

This script, for each chunk_id that appears MORE THAN ONCE:
  * If all copies are byte-identical (same `text` AND same `metadata_json`)
    → keep exactly ONE, delete the rest (delete-by-id then re-add one row).
  * If copies DIVERGE (different text/metadata) → REFUSE to touch it and flag
    `divergent: True` in the summary. Divergent copies need human review; blind
    deletion could drop the correct version.

Unique chunk_ids are never touched. Idempotent: a clean table is a no-op.
The table is NEVER dropped.

TDD contract: ``tests/scripts/test_dedup_exact_duplicate_ids.py``.

Run (real, against the live ``rag2025`` table):

    python scripts/dedup_exact_duplicate_ids.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402


class _TableLike(Protocol):
    def add(self, rows: List[Dict[str, Any]]) -> Any: ...
    def delete(self, where_clause: str) -> Any: ...
    def to_pandas(self): ...


def _fingerprint(row: Dict[str, Any]) -> str:
    """Identity of a row for dup-detection: text + metadata_json only.

    Vector/source/faq_type are derived/embedding artifacts; two rows with the
    same text+metadata are the same chunk for retrieval purposes.
    """
    return f"{row.get('text','')}\x00{row.get('metadata_json','')}"


def dedup_exact_duplicates(table: _TableLike) -> Dict[str, Dict[str, Any]]:
    """Collapse byte-identical duplicate rows to a single row per chunk_id.

    Returns a summary keyed by the DUPLICATED chunk_ids only:
        {chunk_id: {"kept": int, "removed": int, "divergent": bool}}
    Unique chunk_ids are absent from the summary.
    """
    df = table.to_pandas()
    if "chunk_id" not in df.columns:
        return {}

    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cid = str(r.get("chunk_id"))
        by_id.setdefault(cid, []).append(r)

    summary: Dict[str, Dict[str, Any]] = {}

    for cid, rows in by_id.items():
        if len(rows) <= 1:
            continue  # unique — never touch

        fingerprints = {_fingerprint(r) for r in rows}
        if len(fingerprints) > 1:
            # Divergent copies → refuse (human review).
            logger.warning(
                f"chunk_id={cid!r} has {len(rows)} DIVERGENT copies "
                f"({len(fingerprints)} distinct) — refusing to dedup; needs review"
            )
            summary[cid] = {"kept": len(rows), "removed": 0, "divergent": True}
            continue

        # All identical → keep one, drop the rest.
        keep = rows[0]
        removed = len(rows) - 1
        # delete ALL rows for this id, then re-add exactly one.
        table.delete(f"chunk_id = '{cid}'")
        # Strip pandas/np artifacts: re-add only the canonical columns.
        clean = {
            "id": keep.get("id") or cid,
            "chunk_id": cid,
            "text": keep.get("text", ""),
            "source": keep.get("source", ""),
            "faq_type": keep.get("faq_type", ""),
            "metadata_json": keep.get("metadata_json", "{}"),
            "vector": list(keep.get("vector")) if keep.get("vector") is not None else None,
        }
        table.add([clean])
        logger.info(f"chunk_id={cid!r}: kept 1, removed {removed} identical duplicate(s)")
        summary[cid] = {"kept": 1, "removed": removed, "divergent": False}

    return summary


def run_against_live_db() -> Dict[str, Dict[str, Any]]:
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)
    return dedup_exact_duplicates(table=table)


if __name__ == "__main__":
    summary = run_against_live_db()
    print("Dedup exact-duplicate summary:")
    if not summary:
        print("  (no duplicates found — clean)")
    for cid, info in summary.items():
        print(
            f"  {cid}: kept={info['kept']} removed={info['removed']} "
            f"divergent={info['divergent']}"
        )
    total_removed = sum(i["removed"] for i in summary.values())
    print(f"TOTAL removed: {total_removed}")
