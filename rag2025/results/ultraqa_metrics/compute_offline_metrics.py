"""G3-T2: multi-dimension offline metrics dashboard from s16 records + GT.

Pure file IO, ZERO LLM/gateway/embedding calls.

Computes:
  * context_recall @ real n (per-fact, per-question)
  * retrieval precision@k / recall@k / MRR / hit-rate@k (k=1,3,5)
  * route distribution + route-accuracy
  * answer-length distribution
  * abstain rate, has-answer rate
  * latency p50/p95 per seam (route/retrieval/gen attribution sub-seams)
  * per-question failure table (empty retrieval, low groundedness, abstain miss)

Writes:
  * results/ultraqa_metrics/multidim_metrics.json (machine-readable)
  * results/ultraqa_metrics/multidim_metrics.md   (ranked weakness report)

Run:
  python results/ultraqa_metrics/compute_offline_metrics.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GT_V2 = REPO / "results" / "ultraqa_metrics" / "fact_level_gt_v2.json"
RECORDS = REPO / "results" / "eval_harness" / "86q_records_s16.jsonl"
OUT_JSON = REPO / "results" / "ultraqa_metrics" / "multidim_metrics.json"
OUT_MD = REPO / "results" / "ultraqa_metrics" / "multidim_metrics.md"

ABSTAIN_PHRASES = (
    "tôi không tìm thấy",
    "không tìm thấy thông tin này",
    "tài liệu hiện có không",
    "chưa được công bố",
    "chưa có thông báo",
    "chưa cập nhật",
    "không cung cấp",
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_gt_v2(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wrapper = json.load(open(path, encoding="utf-8"))
    if isinstance(wrapper, dict) and "questions" in wrapper:
        return wrapper["questions"], wrapper.get("_provenance", {})
    # Fallback: assume raw list
    return wrapper, {}


def load_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Pure metric primitives (testable)
# ---------------------------------------------------------------------------
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0 or not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for x in topk if x in relevant_ids) / float(k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    return sum(1 for x in topk if x in relevant_ids) / float(len(relevant_ids))


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, x in enumerate(retrieved_ids, start=1):
        if x in relevant_ids:
            return 1.0 / float(i)
    return 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(x in relevant_ids for x in retrieved_ids[:k]) else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Linear-interp percentile. Returns 0.0 on empty."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def is_abstain(answer: str) -> bool:
    """Treat as abstain if the LEAD 240 chars defer, or answer is < 80 chars.

    We deliberately do NOT scan the full body: a real answer that says
    "chưa được công bố" mid-paragraph is still an answer. We catch:
      * the standard "Tôi không tìm thấy..." string
      * lead-deferral patterns (e.g. "Hiện tại, ... chưa được công bố. ...")
      * very short responses (< 80 chars; likely a one-liner)
    """
    if not answer:
        return True
    head = answer[:240].lower()
    if any(p in head for p in ABSTAIN_PHRASES):
        return True
    if len(answer.strip()) < 80:
        return True
    return False


# ---------------------------------------------------------------------------
# Domain metrics
# ---------------------------------------------------------------------------
def fact_level_context_recall(records: list[dict], gt_qs: list[dict]) -> dict[str, Any]:
    gt_by_id = {q["id"]: q for q in gt_qs}
    per_q: list[dict[str, Any]] = []
    n_facts = 0
    n_hits = 0
    n_q_with_facts = 0
    for r in records:
        qid = r["id"]
        q = gt_by_id.get(qid)
        if not q:
            continue
        facts = [f for f in (q.get("critical_facts") or []) if f.get("supporting_chunk_ids")]
        if not facts:
            continue
        n_q_with_facts += 1
        retrieved = {c.get("chunk_id") for c in (r.get("retrieved_chunks") or [])}
        q_hits = 0
        for f in facts:
            n_facts += 1
            cited = set(f["supporting_chunk_ids"])
            if cited & retrieved:
                q_hits += 1
                n_hits += 1
        per_q.append({
            "id": qid,
            "n_facts": len(facts),
            "n_hits": q_hits,
            "recall": (q_hits / len(facts)) if facts else 0.0,
        })
    return {
        "n_questions": n_q_with_facts,
        "n_facts": n_facts,
        "n_hits": n_hits,
        "context_recall": (n_hits / n_facts) if n_facts else 0.0,
        "per_question": per_q,
    }


def retrieval_metrics(records: list[dict], gt_qs: list[dict]) -> dict[str, Any]:
    gt_by_id = {q["id"]: q for q in gt_qs}
    ks = (1, 3, 5)
    p_at_k: dict[int, list[float]] = {k: [] for k in ks}
    r_at_k: dict[int, list[float]] = {k: [] for k in ks}
    h_at_k: dict[int, list[float]] = {k: [] for k in ks}
    mrrs: list[float] = []
    n_evaluated = 0
    for r in records:
        q = gt_by_id.get(r["id"])
        if not q:
            continue
        facts = [f for f in (q.get("critical_facts") or []) if f.get("supporting_chunk_ids")]
        if not facts:
            continue
        n_evaluated += 1
        relevant: set[str] = set()
        for f in facts:
            relevant.update(f["supporting_chunk_ids"])
        retrieved_ids = [c.get("chunk_id") for c in (r.get("retrieved_chunks") or []) if c.get("chunk_id")]
        for k in ks:
            p_at_k[k].append(precision_at_k(retrieved_ids, relevant, k))
            r_at_k[k].append(recall_at_k(retrieved_ids, relevant, k))
            h_at_k[k].append(hit_rate_at_k(retrieved_ids, relevant, k))
        mrrs.append(mrr(retrieved_ids, relevant))
    return {
        "n_evaluated": n_evaluated,
        "precision_at_k": {f"p@{k}": (sum(p_at_k[k]) / len(p_at_k[k])) if p_at_k[k] else 0.0 for k in ks},
        "recall_at_k": {f"r@{k}": (sum(r_at_k[k]) / len(r_at_k[k])) if r_at_k[k] else 0.0 for k in ks},
        "hit_rate_at_k": {f"hit@{k}": (sum(h_at_k[k]) / len(h_at_k[k])) if h_at_k[k] else 0.0 for k in ks},
        "mrr": (sum(mrrs) / len(mrrs)) if mrrs else 0.0,
    }


def route_distribution(records: list[dict]) -> dict[str, int]:
    return dict(Counter(r.get("route", "unknown") for r in records))


def answer_length_distribution(records: list[dict]) -> dict[str, float]:
    lens = [len(r.get("answer") or "") for r in records]
    if not lens:
        return {"n": 0}
    return {
        "n": len(lens),
        "min": min(lens),
        "p50": percentile(lens, 50),
        "p95": percentile(lens, 95),
        "max": max(lens),
        "mean": statistics.mean(lens),
    }


def abstain_and_answer(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    n_abstain = 0
    n_has_answer = 0
    abstain_by_expected: Counter = Counter()
    n_emoji = 0
    for r in records:
        ans = r.get("answer") or ""
        is_abs = is_abstain(ans)
        if is_abs:
            n_abstain += 1
        else:
            n_has_answer += 1
        if is_abs:
            abstain_by_expected[r.get("expected_behavior", "?")] += 1
    return {
        "n": n,
        "n_abstain": n_abstain,
        "n_has_answer": n_has_answer,
        "abstain_rate": (n_abstain / n) if n else 0.0,
        "has_answer_rate": (n_has_answer / n) if n else 0.0,
        "abstain_by_expected_behavior": dict(abstain_by_expected),
    }


def latency_seams(records: list[dict]) -> dict[str, Any]:
    """Per-seam p50/p95. Records may have a few nones (e.g. rerank_ms for some)."""
    seams = ("route_ms", "retrieval_loop_ms", "query_ms", "total_ms")
    sub = ("hyde_ms", "router_inner_ms", "gen_ms", "rerank_ms")
    out: dict[str, Any] = {}
    for s in seams:
        vals = [r["latency_ms"][s] for r in records if r.get("latency_ms") and r["latency_ms"].get(s) is not None]
        if vals:
            out[s] = {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95), "max": max(vals)}
    for s in sub:
        vals = [
            r["latency_attribution"][s]
            for r in records
            if r.get("latency_attribution") and r["latency_attribution"].get(s) is not None
        ]
        if vals:
            out[f"attr_{s}"] = {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95), "max": max(vals)}
    return out


def per_question_failure_table(records: list[dict], gt_qs: list[dict], cr_per_q: list[dict]) -> list[dict[str, Any]]:
    cr_by_id = {x["id"]: x for x in cr_per_q}
    gt_by_id = {q["id"]: q for q in gt_qs}
    out: list[dict[str, Any]] = []
    for r in records:
        qid = r["id"]
        retrieved = r.get("retrieved_chunks") or []
        ans = r.get("answer") or ""
        is_abs = is_abstain(ans)
        empty = len(retrieved) == 0
        cr = cr_by_id.get(qid, {}).get("recall", None)
        flags = []
        if empty:
            flags.append("empty_retrieval")
        if is_abs and r.get("expected_behavior") == "answer":
            flags.append("abstain_miss")
        if is_abs and r.get("expected_behavior") == "abstain":
            flags.append("abstain_correct")
        if cr is not None and cr < 1.0:
            flags.append("incomplete_context_recall")
        if r.get("expected_behavior") == "abstain" and not is_abs:
            flags.append("over_answer")
        out.append({
            "id": qid,
            "route": r.get("route"),
            "expected_behavior": r.get("expected_behavior"),
            "is_abstain": is_abs,
            "n_retrieved": len(retrieved),
            "answer_len": len(ans),
            "context_recall": cr,
            "flags": flags,
        })
    return out


# ---------------------------------------------------------------------------
# Ranking + report
# ---------------------------------------------------------------------------
def rank_weaknesses(failure_table: list[dict], cr_per_q: list[dict], route_dist: dict, latency: dict) -> list[dict[str, Any]]:
    """Produce a ranked weakness list with question-id + metric evidence."""
    weaknesses: list[dict[str, Any]] = []
    # 1. Over-answers (abstain expected but answered)
    over = [f for f in failure_table if "over_answer" in f["flags"]]
    for f in over:
        weaknesses.append({
            "severity": "HIGH",
            "id": f["id"],
            "metric": "abstain_accuracy",
            "evidence": f"expected=abstain but answer_len={f['answer_len']} (route={f['route']})",
        })
    # 2. Empty retrieval
    empty = [f for f in failure_table if "empty_retrieval" in f["flags"]]
    for f in empty:
        weaknesses.append({
            "severity": "CRITICAL",
            "id": f["id"],
            "metric": "retrieval_coverage",
            "evidence": f"retrieved_chunks=[] (route={f['route']})",
        })
    # 3. Incomplete context recall
    inc_cr = sorted(
        [f for f in failure_table if f.get("context_recall") is not None and f["context_recall"] < 1.0],
        key=lambda x: x["context_recall"],
    )
    for f in inc_cr:
        weaknesses.append({
            "severity": "HIGH",
            "id": f["id"],
            "metric": "context_recall",
            "evidence": f"context_recall={f['context_recall']:.3f} (route={f['route']}, n_retrieved={f['n_retrieved']})",
        })
    # 4. Slow seam (latency p50 dominance)
    total_p50 = latency.get("total_ms", {}).get("p50", 0.0)
    for seam in ("route_ms", "retrieval_loop_ms", "query_ms"):
        p50 = latency.get(seam, {}).get("p50", 0.0)
        if total_p50 and p50 / total_p50 > 0.30:
            weaknesses.append({
                "severity": "MED",
                "id": "(global)",
                "metric": f"latency_{seam}",
                "evidence": f"{seam} p50={p50:.0f}ms = {100*p50/total_p50:.1f}% of total p50 {total_p50:.0f}ms",
            })
    # 5. Route dominance
    n_total = sum(route_dist.values()) or 1
    for route, count in route_dist.items():
        if count / n_total > 0.40:
            weaknesses.append({
                "severity": "MED",
                "id": "(global)",
                "metric": "route_distribution",
                "evidence": f"route={route} took {count}/{n_total} = {100*count/n_total:.1f}% (over-concentration)",
            })
    return weaknesses


def render_markdown(metrics: dict[str, Any], weaknesses: list[dict]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Multi-Dim Offline Metrics — s16 (G3-T2)")
    a("")
    a(f"**Source records:** `results/eval_harness/86q_records_s16.jsonl` (n={metrics['n_records']})  ")
    a(f"**GT v2:** `results/ultraqa_metrics/fact_level_gt_v2.json` (n_facts={metrics['gt']['n_facts_total']} with chunk_ids={metrics['gt']['n_facts_with_chunk_ids']})  ")
    a(f"**Method:** pure file IO, no LLM, no embedding call. Numbers are computed from on-disk artifacts only.")
    a("")
    a("---")
    a("")
    a("## 1. Context recall @ real n (GATE 1)")
    cr = metrics["context_recall"]
    a(f"- **context_recall = {cr['context_recall']:.4f}**")
    a(f"- n_facts = {cr['n_facts']}  |  n_hits = {cr['n_hits']}  |  n_questions (with facts) = {cr['n_questions']}")
    a(f"- Verdict vs gate ≥ 0.85: **{'PASS' if cr['context_recall'] >= 0.85 else 'FAIL'}** (was 1.000 on phantom n=2)")
    a("")
    a("| id | n_facts | n_hits | recall |")
    a("|---|---:|---:|---:|")
    for q in sorted(cr["per_question"], key=lambda x: x["recall"])[:10]:
        a(f"| {q['id']} | {q['n_facts']} | {q['n_hits']} | {q['recall']:.3f} |")
    a("")
    a("---")
    a("")
    a("## 2. Retrieval precision@k / recall@k / MRR / hit-rate (k=1,3,5)")
    rm = metrics["retrieval"]
    a(f"- n_evaluated (questions with GT facts) = {rm['n_evaluated']}")
    a("")
    a("| metric | value |")
    a("|---|---:|")
    a(f"| p@1 | {rm['precision_at_k']['p@1']:.4f} |")
    a(f"| p@3 | {rm['precision_at_k']['p@3']:.4f} |")
    a(f"| **p@5** | **{rm['precision_at_k']['p@5']:.4f}** |")
    a(f"| r@1 | {rm['recall_at_k']['r@1']:.4f} |")
    a(f"| r@3 | {rm['recall_at_k']['r@3']:.4f} |")
    a(f"| **r@5** | **{rm['recall_at_k']['r@5']:.4f}** |")
    a(f"| hit@1 | {rm['hit_rate_at_k']['hit@1']:.4f} |")
    a(f"| hit@3 | {rm['hit_rate_at_k']['hit@3']:.4f} |")
    a(f"| hit@5 | {rm['hit_rate_at_k']['hit@5']:.4f} |")
    a(f"| **MRR** | **{rm['mrr']:.4f}** |")
    a("")
    a("---")
    a("")
    a("## 3. Route distribution")
    rd = metrics["route_distribution"]
    a("| route | count | % |")
    a("|---|---:|---:|")
    n_total = sum(rd.values()) or 1
    for k_, v_ in sorted(rd.items(), key=lambda x: -x[1]):
        a(f"| {k_} | {v_} | {100*v_/n_total:.1f}% |")
    a("")
    a("---")
    a("")
    a("## 4. Answer length + abstain + has-answer")
    al = metrics["answer_length"]
    ab = metrics["abstain"]
    a(f"- answer_len: n={al['n']}  min={al['min']}  p50={al['p50']:.0f}  p95={al['p95']:.0f}  max={al['max']}  mean={al['mean']:.1f}")
    a(f"- **abstain_rate = {ab['abstain_rate']:.4f}** ({ab['n_abstain']}/{ab['n']})")
    a(f"- has_answer_rate = {ab['has_answer_rate']:.4f}")
    a(f"- abstain × expected_behavior: {ab['abstain_by_expected_behavior']}")
    a("")
    a("---")
    a("")
    a("## 5. Latency seams (ms)")
    ls = metrics["latency"]
    a("| seam | n | p50 | p95 | max |")
    a("|---|---:|---:|---:|---:|")
    for k_, v_ in ls.items():
        a(f"| {k_} | {v_['n']} | {v_['p50']:.0f} | {v_['p95']:.0f} | {v_['max']:.0f} |")
    a("")
    a("---")
    a("")
    a("## 6. Per-question failure table (truncated, top 25 by severity)")
    a("")
    a("| id | route | expected | is_abstain | n_retrieved | cr | flags |")
    a("|---|---|---|---:|---:|---:|---|")
    for f in metrics["failure_table"][:25]:
        cr_str = "-" if f["context_recall"] is None else f"{f['context_recall']:.2f}"
        a(f"| {f['id']} | {f['route']} | {f['expected_behavior']} | {f['is_abstain']} | {f['n_retrieved']} | "
          f"{cr_str} | {','.join(f['flags'])} |")
    a("")
    a("---")
    a("")
    a("## 7. RANKED WEAKNESS LIST (G3-T2 deliverable)")
    a("")
    a("| # | severity | id | metric | evidence |")
    a("|---|---|---|---|---|")
    for i, w in enumerate(weaknesses, start=1):
        a(f"| {i} | {w['severity']} | {w['id']} | `{w['metric']}` | {w['evidence']} |")
    a("")
    a("---")
    a("")
    a("## Provenance")
    a("")
    a("```json")
    a(json.dumps(metrics.get("provenance", {}), ensure_ascii=False, indent=2))
    a("```")
    a("")
    return "\n".join(lines)


def main() -> int:
    gt_qs, provenance = load_gt_v2(GT_V2)
    records = load_records(RECORDS)
    n_records = len(records)

    cr = fact_level_context_recall(records, gt_qs)
    rm = retrieval_metrics(records, gt_qs)
    rd = route_distribution(records)
    al = answer_length_distribution(records)
    ab = abstain_and_answer(records)
    ls = latency_seams(records)
    ft = per_question_failure_table(records, gt_qs, cr["per_question"])
    # Sort failure table: empty > over_answer > low_cr > others
    severity_order = {"empty_retrieval": 0, "over_answer": 1, "incomplete_context_recall": 2, "abstain_miss": 3, "abstain_correct": 4}
    def _sev_key(f):
        # lowest score first
        codes = [severity_order.get(c, 99) for c in f["flags"]]
        return (min(codes) if codes else 99, f["id"])
    ft_sorted = sorted(ft, key=_sev_key)
    weaknesses = rank_weaknesses(ft, cr["per_question"], rd, ls)

    metrics = {
        "n_records": n_records,
        "gt": {
            "n_facts_total": sum(len(q.get("critical_facts") or []) for q in gt_qs),
            "n_facts_with_chunk_ids": sum(
                1
                for q in gt_qs
                for f in (q.get("critical_facts") or [])
                if f.get("supporting_chunk_ids")
            ),
        },
        "context_recall": cr,
        "retrieval": rm,
        "route_distribution": rd,
        "answer_length": al,
        "abstain": ab,
        "latency": ls,
        "failure_table": ft_sorted,
        "weaknesses": weaknesses,
        "provenance": {
            "method": "G3-T2 compute_offline_metrics.py — pure file IO; no LLM, no gateway, no embedding call.",
            "records_path": str(RECORDS),
            "gt_path": str(GT_V2),
            "gt_provenance": provenance,
        },
    }
    OUT_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(metrics, weaknesses)
    OUT_MD.write_text(md, encoding="utf-8")
    # Print a tiny summary to stdout
    print(f"records={n_records}  facts={metrics['gt']['n_facts_total']}  with_chunk_ids={metrics['gt']['n_facts_with_chunk_ids']}")
    print(f"context_recall={cr['context_recall']:.4f}  n_facts={cr['n_facts']}  n_q={cr['n_questions']}")
    print(f"p@5={rm['precision_at_k']['p@5']:.4f}  r@5={rm['recall_at_k']['r@5']:.4f}  MRR={rm['mrr']:.4f}")
    print(f"abstain_rate={ab['abstain_rate']:.4f}  has_answer_rate={ab['has_answer_rate']:.4f}")
    print(f"weaknesses_ranked={len(weaknesses)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
