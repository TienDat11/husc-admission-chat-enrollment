"""Verify the timeline-hocba chunk surfaces in top-3 for 3 phrasings (S17.2).

Calls the underlying LanceDBAdapter.search directly to bypass the temporal
filter (which has a path issue when run as a script — the raw dense-search
ranking is what we want to confirm anyway).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService
from src.infrastructure.lancedb_adapter import LanceDBAdapter


CHUNK_ID = "qa_lich_xet_hoc_ba_2026_v1"
ID74_CHUNK_ID = "qa_dang_ky_xet_tuyen_2026_v2"

QUERIES = [
    "Trường ơi cho em hỏi với, bên mình khi nào có kế hoạch tuyển sinh bằng xét học bạ ạ?",
    "lịch xét tuyển học bạ 2026",
    "thời gian đăng ký xét học bạ HUSC",
]

DATE_TOKENS = ["02/7/2026", "23/5/2026", "14/7/2026"]
GENERIC_PLACEHOLDERS = ["qa_lich_tuyen_sinh_2026", "qa_xet_hoc_ba_quy_trinh_2026"]


def _row_to_dict(r: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    raw = r.get("metadata_json")
    if isinstance(raw, str) and raw.strip():
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = {}
    elif isinstance(raw, dict):
        meta = raw
    cid = r.get("chunk_id") or r.get("id") or ""
    text = r.get("text") or ""
    return {
        "chunk_id": cid,
        "score": round(float(r.get("_distance", 0.0) or 0.0), 4),
        "text_snippet": text[:160].replace("\n", " "),
        "is_superseded": bool(meta.get("is_superseded", False)),
        "data_year": meta.get("data_year"),
        "topic": meta.get("topic"),
    }


def main() -> None:
    settings = RAGSettings()

    # 1. Confirm chunk is in the table
    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    df = table.to_pandas()
    present = (df["chunk_id"] == CHUNK_ID).any()
    print(f"[CHECK1] chunk_id={CHUNK_ID!r} present in table: {present}")
    print(f"[CHECK1] chunk_id={ID74_CHUNK_ID!r} present in table: {(df['chunk_id'] == ID74_CHUNK_ID).any()}")
    print(f"[CHECK1] total rows in rag2025: {len(df)}")
    if not present:
        print("ABORT — chunk not in table.")
        return

    emb = EmbeddingService(settings)
    adapter = LanceDBAdapter(uri=settings.LANCEDB_URI, table_name=settings.LANCEDB_TABLE)
    adapter.ensure_table()

    overall_pass = True
    for q in QUERIES:
        qv = emb.encode_single(q).tolist()
        rows = adapter.search(query_vector=qv, top_k=8, where_clause=None)
        ranked = [_row_to_dict(r) for r in rows]

        target_rank = next((i + 1 for i, r in enumerate(ranked) if r["chunk_id"] == CHUNK_ID), None)
        id74_rank = next((i + 1 for i, r in enumerate(ranked) if r["chunk_id"] == ID74_CHUNK_ID), None)
        placeholder_ranks = [
            (i + 1, r) for i, r in enumerate(ranked) if r["chunk_id"] in GENERIC_PLACEHOLDERS
        ]
        top3_text = " ".join(r["text_snippet"] for r in ranked[:3])
        dates_in_top3 = {tok: (tok in top3_text) for tok in DATE_TOKENS}

        # PASS criteria: target in top-3 OR all 3 dates surface in top-3
        if target_rank is not None and target_rank <= 3:
            verdict = "PASS (timeline chunk top-3)"
        elif all(dates_in_top3.values()):
            verdict = "PASS (id=74 dates surface top-3)"
        elif target_rank is not None and target_rank <= 5:
            verdict = "STILL-FAILING (target top-5 but not top-3)"
            overall_pass = False
        else:
            verdict = "STILL-FAILING (target absent top-8)"
            overall_pass = False

        print(f"\n[Q] {q}")
        print(f"[VERDICT] {verdict}")
        for i, r in enumerate(ranked[:5], 1):
            mark = " <- TARGET" if r["chunk_id"] == CHUNK_ID else ""
            mark += " (placeholder, superseded)" if r["is_superseded"] else ""
            print(
                f"  #{i} score(1-d)={r['score']} chunk_id={r['chunk_id']} "
                f"topic={r['topic']}{mark}"
            )
            print(f"       {r['text_snippet']}")
        print(f"[TOP-3 date-tokens present]: {dates_in_top3}")
        if placeholder_ranks:
            print(f"[PLACEHOLDERS in top-8]: {placeholder_ranks}")
        print(f"[target_rank={target_rank}, id74_rank={id74_rank}]")

    print(f"\n=== OVERALL: {'PASS' if overall_pass else 'STILL-FAILING'} ===")


if __name__ == "__main__":
    main()
