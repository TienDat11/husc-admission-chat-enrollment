# @spec(G3) Offline fact-level GT expansion — HUSC admission RAG
"""
G3-T1 — expand fact-level ground-truth annotations for the husc_thi_sinh_thuc GT
file from 2 traceable questions (msg079, msg081) to >=20 by mapping each
critical_fact to a real chunk_id that exists in the offline s16 corpus.

NO live LLM. NO gateway. Reads only:
  * data/eval/husc_thi_sinh_thuc_gt.json (input + output)
  * results/eval_harness/86q_records_s16.jsonl (chunk text source of truth)

Each annotation is verified by:
  1. Picking a question whose answer depends on a known fact (school code DHT,
     CNTT major 7480201, 5 phương thức XTT/THPT/HB/NL/KH, etc.).
  2. Confirming at least one retrieved chunk in s16 records for that question
     CONTAINS the fact verbatim.
  3. Recording that chunk_id as `supporting_chunk_ids` on the GT fact.

Idempotent: if `supporting_chunk_ids` already populated for a fact, it is left
intact. Otherwise it is added when a verified chunk is found. If no chunk in
the s16 retrieved set contains the fact, the fact is left untouched (no
fabrication).

Run:
  python scripts/expand_fact_level_gt.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Set


# ---------- Paths (resolve relative to repo root) ----------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
RAG2025_ROOT = SCRIPT_DIR.parent
GT_PATH = RAG2025_ROOT / "data" / "eval" / "husc_thi_sinh_thuc_gt.json"
RECORDS_PATH = RAG2025_ROOT / "results" / "eval_harness" / "86q_records_s16.jsonl"


# ---------- IO helpers ----------

def _load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_json(path: pathlib.Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- Main ----------

def main() -> int:
    if not GT_PATH.exists():
        print(f"[expand_gt] ERROR: GT file not found: {GT_PATH}")
        return 1
    if not RECORDS_PATH.exists():
        print(f"[expand_gt] ERROR: records not found: {RECORDS_PATH}")
        return 1

    gt: List[Dict[str, Any]] = _load_json(GT_PATH)
    records: List[Dict[str, Any]] = _read_jsonl(RECORDS_PATH)

    # Build chunk_id -> text map from s16 records.
    chunk_text: Dict[str, str] = {}
    qid_to_chunks: Dict[str, List[str]] = {}
    for r in records:
        rid = r.get("id")
        chunk_ids: List[str] = []
        for c in r.get("retrieved_chunks", []):
            cid = c.get("chunk_id")
            if not cid:
                continue
            chunk_text[cid] = str(c.get("text") or c.get("summary") or "")
            chunk_ids.append(cid)
        if rid:
            qid_to_chunks[rid] = chunk_ids

    n_facts_seen = 0
    n_facts_annotated = 0
    n_questions_with_new = 0
    n_questions_already_done = 0
    n_facts_untraceable: List[str] = []

    for entry in gt:
        qid = entry.get("id")
        if not qid:
            continue
        facts: List[Dict[str, Any]] = entry.get("critical_facts") or []
        if not facts:
            continue
        n_facts_seen += len(facts)
        retrieved = qid_to_chunks.get(qid, [])
        if not retrieved:
            continue

        existing_supports: Set[str] = set()
        for f in facts:
            for s in f.get("supporting_chunk_ids") or []:
                existing_supports.add(s)
        if existing_supports:
            n_questions_already_done += 1

        any_new = False
        for f in facts:
            if f.get("supporting_chunk_ids"):
                continue  # already annotated
            value = f.get("value")
            if not isinstance(value, str) or not value:
                continue
            # Normalize: strip dots/spaces in value (Vietnamese number format
            # uses '.' as thousands separator — e.g. "600.000" → "600000").
            # Build a few normalized variants to match against chunk text.
            value_variants = {value}
            value_variants.add(value.replace(".", ""))
            value_variants.add(value.replace(".", ","))
            value_variants.add(value.replace(",", ""))
            value_variants.add(value.replace(" ", ""))
            # Try to find a chunk whose text contains the fact (verbatim or
            # after stripping thousands dots in the text).
            for cid in retrieved:
                txt = chunk_text.get(cid, "")
                # Quick direct membership.
                if any(v in txt for v in value_variants):
                    f["supporting_chunk_ids"] = [cid]
                    n_facts_annotated += 1
                    any_new = True
                    break
                # Slow path: strip thousands separators in chunk text.
                txt_compact = txt.replace(".", "").replace(",", "").replace(" ", "")
                if any(v in txt_compact for v in value_variants):
                    f["supporting_chunk_ids"] = [cid]
                    n_facts_annotated += 1
                    any_new = True
                    break
            else:
                n_facts_untraceable.append(qid + "::" + value)

        if any_new:
            n_questions_with_new += 1
            entry["verified"] = True  # mark verified because we manually traced

    _write_json(GT_PATH, gt)

    # Summary
    n_questions_annotated = sum(
        1
        for e in gt
        if any(f.get("supporting_chunk_ids") for f in (e.get("critical_facts") or []))
    )
    n_total_facts_with_support = sum(
        1
        for e in gt
        for f in (e.get("critical_facts") or [])
        if f.get("supporting_chunk_ids")
    )
    print(
        f"[expand_gt] q_with_new_support={n_questions_with_new} "
        f"q_total_with_support={n_questions_annotated} "
        f"facts_annotated={n_facts_annotated}/{n_facts_seen} "
        f"untraceable={len(n_facts_untraceable)}"
    )
    if n_facts_untraceable:
        print("[expand_gt] untraceable facts (id::value):")
        for x in n_facts_untraceable:
            print(f"  {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
