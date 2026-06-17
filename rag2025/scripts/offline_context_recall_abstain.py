# @spec(G3) Offline context_recall + abstain re-aggregator (no LLM).
"""
G3-T1/G3-T4 offline metric re-aggregator.

Reads ONLY:
  * data/eval/husc_thi_sinh_thuc_gt.json (offline ground truth)
  * results/eval_harness/86q_records_s16.jsonl (offline records)
  * results/eval_harness/metrics_report/s16/metrics_report_s16.json (offline
    judge aggregate) — only for faithfulness/answer_relevancy/context_precision
    placeholders (we don't recompute LLM-derived numbers).

Computes:
  * context_recall at fact-level: fraction of fact-level critical_facts whose
    `supporting_chunk_ids` are present in the question's retrieved chunks
    (using the s16 records as offline evidence).
  * abstain_accuracy: same formula as eval_report_metrics.abstain_accuracy —
    fraction of refuse/abstain-expected items the system actually abstained on,
    using the ABSTAIN_MARKERS set.

NO LLM. NO network. NO gateway. Pure arithmetic over offline artifacts.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
RAG2025_ROOT = SCRIPT_DIR.parent
GT_PATH = RAG2025_ROOT / "data" / "eval" / "husc_thi_sinh_thuc_gt.json"
RECORDS_PATH = RAG2025_ROOT / "results" / "eval_harness" / "86q_records_s16.jsonl"
DEFAULT_OUT = RAG2025_ROOT / "results" / "ultraqa_metrics" / "offline_recall_abstain.json"


# ABSTAIN_MARKERS copied verbatim from scripts/eval_report_metrics.py
ABSTAIN_MARKERS: Tuple[str, ...] = (
    "chưa có thông tin",
    "chưa có thông tin chính thức",
    "chưa có",
    "chưa công bố",
    "chưa được công bố",
    "không tìm thấy thông tin",
    "không tìm thấy",
    "ngoài phạm vi",
    "tôi không thể",
    "không thể tư vấn",
    "xin lỗi, tôi không",
)


def _looks_like_abstain(answer_text: str) -> bool:
    a = (answer_text or "").lower().strip()
    if not a:
        return True
    return any(m in a for m in ABSTAIN_MARKERS)


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _compute_context_recall(
    gt: List[Dict[str, Any]],
    qid_to_retrieved: Dict[str, List[str]],
) -> Dict[str, Any]:
    n_total = 0
    n_recall = 0
    n_q_evaluated = 0
    per_q: List[Dict[str, Any]] = []
    for e in gt:
        qid = e.get("id")
        retrieved = qid_to_retrieved.get(qid, [])
        q_facts = 0
        q_hits = 0
        for f in e.get("critical_facts") or []:
            supports = f.get("supporting_chunk_ids") or []
            if not supports:
                continue
            q_facts += len(supports)
            for cid in supports:
                n_total += 1
                if cid in retrieved:
                    n_recall += 1
                    q_hits += 1
        if q_facts:
            n_q_evaluated += 1
            per_q.append(
                {
                    "id": qid,
                    "n_facts": q_facts,
                    "n_hits": q_hits,
                    "recall": (q_hits / q_facts) if q_facts else None,
                }
            )
    overall = (n_recall / n_total) if n_total else 0.0
    return {
        "n_facts": n_total,
        "n_hits": n_recall,
        "n_questions": n_q_evaluated,
        "context_recall": overall,
        "per_question": per_q,
    }


def _compute_abstain_accuracy(
    records: List[Dict[str, Any]],
    gt: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gt_by_id = {e["id"]: e for e in gt if isinstance(e, dict) and "id" in e}
    total = 0
    correct = 0
    detail: List[Dict[str, Any]] = []
    for rec in records:
        rid = rec.get("id")
        gt_e = gt_by_id.get(rid)
        if not gt_e:
            continue
        eb = gt_e.get("expected_behavior")
        if eb not in ("refuse", "abstain"):
            continue
        ans = rec.get("answer") or ""
        abstained = _looks_like_abstain(ans)
        total += 1
        if abstained:
            correct += 1
        detail.append(
            {
                "id": rid,
                "expected_behavior": eb,
                "abstained": abstained,
                "answer_excerpt": ans[:160],
            }
        )
    acc = (correct / total) if total else 0.0
    return {
        "n_total": total,
        "n_correct": correct,
        "abstain_accuracy": acc,
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline (no-LLM) context_recall + abstain_accuracy re-aggregator. "
            "Reads husc_thi_sinh_thuc_gt.json + 86q_records_s16.jsonl."
        )
    )
    parser.add_argument("--gt", type=pathlib.Path, default=GT_PATH)
    parser.add_argument("--records", type=pathlib.Path, default=RECORDS_PATH)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    records = _read_jsonl(args.records)

    qid_to_retrieved = {
        r.get("id"): [c.get("chunk_id") for c in (r.get("retrieved_chunks") or [])]
        for r in records
    }
    recall = _compute_context_recall(gt, qid_to_retrieved)
    abst = _compute_abstain_accuracy(records, gt)

    report = {
        "gt_path": str(args.gt),
        "records_path": str(args.records),
        "n_records": len(records),
        "context_recall_offline": recall,
        "abstain_accuracy_offline": abst,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[offline_agg] context_recall={recall['context_recall']:.3f} "
        f"n_facts={recall['n_facts']} n_q={recall['n_questions']} | "
        f"abstain_accuracy={abst['abstain_accuracy']:.3f} "
        f"n_total={abst['n_total']} n_correct={abst['n_correct']} | "
        f"out={args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
