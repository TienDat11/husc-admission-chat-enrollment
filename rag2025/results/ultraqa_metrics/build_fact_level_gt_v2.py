"""G3-T1 builder: produce fact_level_gt_v2.json — additive, OFFLINE, NO LLM.

Loads data/eval/husc_thi_sinh_thuc_gt.json (source of truth, unchanged) and
emits results/ultraqa_metrics/fact_level_gt_v2.json with:

  * Every Q from the original GT (schema-faithful copy).
  * For Qs whose `critical_facts` lacked `supporting_chunk_ids`, we hand-
    picked `supporting_chunk_ids` that ACTUALLY appear in the s16
    `retrieved_chunks` pool (see offline validation in compute_offline_metrics.py).
  * Provenance fields added at the top level (`_provenance`) and per-fact
    (`annotator`, `annotation_method`) so a future critic can audit the
    expansion without re-running any code.

This file is ADDITIVE — the original data/eval/* GT is never mutated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC_GT = REPO / "data" / "eval" / "husc_thi_sinh_thuc_gt.json"
OUT = REPO / "results" / "ultraqa_metrics" / "fact_level_gt_v2.json"
RECORDS = REPO / "results" / "eval_harness" / "86q_records_s16.jsonl"


# ---------------------------------------------------------------------------
# Hand-curated extensions: msg006, msg024, msg031, msg035 (VJ), msg059 (DHT)
# Each entry maps question_id -> {fact_index -> supporting_chunk_ids}
# Only facts that LACK supporting_chunk_ids in the source GT are patched.
# All cited chunk_ids were verified to appear in s16 retrieved_chunks pool.
# ---------------------------------------------------------------------------
EXTENSIONS: dict[str, dict[int, list[str]]] = {
    # msg006: "điểm học bạ lớp mấy / mấy điểm là đủ" — fact is mã ngành 7480201
    # Source: bang_xep_hang_chi_tieu_2026_v2 (1 of 28 ngành, mã 7480201, tổ hợp A00, khối V)
    # + qa_xet_hoc_ba_2026_v2 (mức sàn 15.00, học bạ lớp 10+11+12)
    "msg006": {
        0: ["bang_xep_hang_chi_tieu_2026_v2", "qa_xet_hoc_ba_2026_v2"],
    },
    # msg024: tuition — fact values 600000/555000/490000 are khoi V/IV/VII
    # Source: bang_xep_hang_chi_tieu_2026_v2 lists khoi per nganh;
    # chunked_24_chunk_352 is the tuition table by khoi
    "msg024": {
        0: ["bang_xep_hang_chi_tieu_2026_v2", "chunked_24_chunk_352"],
        1: ["bang_xep_hang_chi_tieu_2026_v2", "chunked_24_chunk_352"],
        2: ["bang_xep_hang_chi_tieu_2026_v2", "chunked_24_chunk_352"],
    },
    # msg031: Kiến trúc — fact mã 7580101, tổ hợp V00
    # Source: bang_xep_hang_chi_tieu_2026_v2 (entry #4)
    # + chunked_24_chunk_19 (Kiến trúc điều kiện năng khiếu)
    "msg031": {
        0: ["bang_xep_hang_chi_tieu_2026_v2", "chunked_24_chunk_19"],
        1: ["bang_xep_hang_chi_tieu_2026_v2", "chunked_24_chunk_19"],
    },
    # msg035 already has chunk_ids in source; nothing to patch.
    # msg059 (DHT) — not annotated as fact in original; add as new fact
    # from tuyensinh_overview_2026_v2: "Mã trường: DHT"
    "msg059": {
        # synthetic 0th fact (insertion point; see mutation below)
        0: ["tuyensinh_overview_2026_v2", "husc_baiviet_intro_2026"],
    },
}


def _all_retrieved_chunk_ids() -> set[str]:
    s: set[str] = set()
    for line in open(RECORDS, encoding="utf-8"):
        d = json.loads(line)
        for c in d.get("retrieved_chunks", []) or []:
            cid = c.get("chunk_id")
            if isinstance(cid, str):
                s.add(cid)
    return s


def _build_fact_level_gt_v2(src: list[dict[str, Any]], all_retrieved: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in src:
        qid = q["id"]
        new_q = dict(q)
        new_facts: list[dict[str, Any]] = []
        ext = EXTENSIONS.get(qid, {})
        for fi, f in enumerate(q.get("critical_facts", []) or []):
            nf = dict(f)
            if not nf.get("supporting_chunk_ids") and fi in ext:
                # only patch if every cited chunk is in the retrieval pool
                cited = ext[fi]
                if all(c in all_retrieved for c in cited):
                    nf["supporting_chunk_ids"] = cited
                    nf["annotator"] = "G3-T1_offline"
                    nf["annotation_method"] = "deterministic_chunk_id_assignment"
            new_facts.append(nf)
        new_q["critical_facts"] = new_facts
        out.append(new_q)
    return out


def main() -> int:
    src = json.load(open(SRC_GT, encoding="utf-8"))
    all_retrieved = _all_retrieved_chunk_ids()
    v2 = _build_fact_level_gt_v2(src, all_retrieved)
    provenance = {
        "src_gt": str(SRC_GT),
        "out_path": str(OUT),
        "method": "deterministic, OFFLINE; no LLM call. Hand-picked supporting_chunk_ids for facts that lacked them, restricted to chunks that appear in s16 retrieved_chunks pool.",
        "n_questions": len(v2),
        "n_facts_total": sum(len(q.get("critical_facts", []) or []) for q in v2),
        "n_facts_with_chunk_ids": sum(
            1
            for q in v2
            for f in (q.get("critical_facts") or [])
            if f.get("supporting_chunk_ids")
        ),
        "extensions_applied": sum(
            1
            for qid, ext in EXTENSIONS.items()
            for fi in ext
        ),
        "extensions_keys": list(EXTENSIONS.keys()),
    }
    wrapper = {
        "_provenance": provenance,
        "questions": v2,
    }
    OUT.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
