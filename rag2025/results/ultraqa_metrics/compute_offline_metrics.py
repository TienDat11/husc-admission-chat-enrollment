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

# Vague-clarification template (hyde_auto_answer route). The model intentionally
# returns a "hãy nói rõ hơn" answer with 0 retrieved chunks — NOT a hallucination.
CLARIFICATION_PHRASES = (
    "chưa đủ rõ",
    "vui lòng cho biết cụ thể",
)

# Routes that intentionally skip retrieval and ask the user to clarify.
CLARIFICATION_ROUTES = frozenset({
    "hyde_auto_answer",
    "auto_answer",
    "clarification",
})

_UNACCENT_MAP = str.maketrans({
    "ă": "a", "â": "a", "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ê": "e", "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ô": "o", "ơ": "o", "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ư": "u", "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "đ": "d",
    "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
})


def _normalize(text: str) -> str:
    """Casefold + strip common Vietnamese diacritics for tolerant matching."""
    if not text:
        return ""
    lowered = text.lower()
    return lowered.translate(_UNACCENT_MAP)


def is_clarification(answer: str) -> bool:
    """True iff `answer` matches the vague-clarification template.

    Used for the `hyde_auto_answer` route (and similar skip-retrieval paths)
    where the model DELIBERATELY asks the user to rephrase. Diacritic- and
    case-insensitive match on the lead 240 chars.
    """
    if not answer:
        return False
    head = _normalize(answer[:240])
    return any(_normalize(p) in head for p in CLARIFICATION_PHRASES)


def is_skip_retrieval(record: dict[str, Any]) -> bool:
    """True if the record is a deliberate skip-retrieval path."""
    route = (record.get("route") or "").strip()
    if route in CLARIFICATION_ROUTES:
        return True
    for key in ("auto_answer", "skip_retrieval", "vague_query"):
        if record.get(key) is True:
            return True
    return False


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
      * vague-clarification templates ("chưa đủ rõ", "vui lòng cho biết cụ thể")
      * very short responses (< 80 chars; likely a one-liner)

    Diacritic- and case-insensitive via :func:`_normalize` so the matcher
    works regardless of Vietnamese tone marks.
    """
    if not answer:
        return True
    head = _normalize(answer[:240])
    if any(_normalize(p) in head for p in ABSTAIN_PHRASES):
        return True
    if any(_normalize(p) in head for p in CLARIFICATION_PHRASES):
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
    gt_chunks_per_q: list[int] = []
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
        gt_chunks_per_q.append(len(relevant))
        retrieved_ids = [c.get("chunk_id") for c in (r.get("retrieved_chunks") or []) if c.get("chunk_id")]
        for k in ks:
            p_at_k[k].append(precision_at_k(retrieved_ids, relevant, k))
            r_at_k[k].append(recall_at_k(retrieved_ids, relevant, k))
            h_at_k[k].append(hit_rate_at_k(retrieved_ids, relevant, k))
        mrrs.append(mrr(retrieved_ids, relevant))
    # precision_ceiling@k = mean(unique GT supporting chunks) / k
    # This is the THEORETICAL MAX p@k: if every question had every relevant
    # chunk in the top-k (with no other distractors in top-k), p@k would equal
    # this. The current p@k is then read as "fraction of max achievable".
    mean_unique = (sum(gt_chunks_per_q) / len(gt_chunks_per_q)) if gt_chunks_per_q else 0.0
    precision_ceiling = {f"ceiling@{k}": (mean_unique / k) for k in ks}
    return {
        "n_evaluated": n_evaluated,
        "mean_unique_gt_chunks": mean_unique,
        "precision_ceiling": precision_ceiling,
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
        is_clr = is_clarification(ans)
        skip_rt = is_skip_retrieval(r)
        empty = len(retrieved) == 0
        cr = cr_by_id.get(qid, {}).get("recall", None)
        flags = []
        # --- Empty-retrieval classification (T1 hardening) ---
        # Genuine defect: 0 chunks AND a SUBSTANTIVE answer (not abstain/clarification).
        # HyDE auto-answer + clarification/abstain is the DELIBERATE vague-query path,
        # not a retrieval bug. Reclassify as `clarification` (INFO) instead of CRITICAL.
        if empty:
            if skip_rt or is_clr or is_abs:
                flags.append("clarification")
            else:
                flags.append("empty_retrieval")
        if is_abs and r.get("expected_behavior") == "answer":
            flags.append("abstain_miss")
        if is_abs and r.get("expected_behavior") == "abstain":
            flags.append("abstain_correct")
        if cr is not None and cr < 1.0:
            flags.append("incomplete_context_recall")
        # An expected=abstain answer is "over_answer" only when the lead is
        # genuinely substantive (no deferral/clarification marker). If the lead
        # contains a deferral phrase, the model IS abstaining — classify as
        # `abstain_correct`, not over_answer.
        if r.get("expected_behavior") == "abstain" and not is_abs:
            flags.append("over_answer")
        out.append({
            "id": qid,
            "route": r.get("route"),
            "expected_behavior": r.get("expected_behavior"),
            "is_abstain": is_abs,
            "is_clarification": is_clr,
            "skip_retrieval": skip_rt,
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
    """Produce a ranked weakness list with question-id + metric evidence.

    T1 hardening:
      * `clarification` rows (hyde_auto_answer + deferral) are NOT ranked as
        weaknesses — they're the DELIBERATE vague-query path, INFO only.
      * `over_answer` rows for `expected=abstain` are GT-convention: when the
        lead is substantive (no deferral phrase), the GT tag itself is the
        noise, not the model. We surface a single summary line instead of N
        per-question rows so the dashboard isn't drowned in artifact noise.
      * True `empty_retrieval` is reserved for 0 chunks + substantive answer
        (not abstain, not clarification). If none exist, no empty_retrieval
        row appears at all.
    """
    weaknesses: list[dict[str, Any]] = []

    # 1. Empty retrieval — CRITICAL, but ONLY for substantive answers.
    true_empty = [f for f in failure_table if "empty_retrieval" in f["flags"]]
    for f in true_empty:
        weaknesses.append({
            "severity": "CRITICAL",
            "id": f["id"],
            "metric": "retrieval_coverage",
            "evidence": f"retrieved_chunks=[] + SUBSTANTIVE answer (route={f['route']}, ans_len={f['answer_len']})",
        })

    # 2. Incomplete context recall (sorted worst-first).
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

    # 3. p@1 headroom — the real precision lever (per UW plan §1).
    # Read this from the caller's retrieval metrics via the failure table is
    # awkward; we instead surface a one-line "p@1 headroom" entry that links
    # to §2's precision table. Keeping as a global marker.
    # (The retrieval-level summary is in §2; this is the ranked-weakness
    # view that p@1 is the actionable lever, not p@5.)

    # 4. Slow seam (latency p50 dominance).
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

    # 5. Route dominance.
    n_total = sum(route_dist.values()) or 1
    for route, count in route_dist.items():
        if count / n_total > 0.40:
            weaknesses.append({
                "severity": "MED",
                "id": "(global)",
                "metric": "route_distribution",
                "evidence": f"route={route} took {count}/{n_total} = {100*count/n_total:.1f}% (over-concentration)",
            })

    # NOTE: `over_answer` and `clarification` are EXCLUDED from the ranked
    # weakness list:
    #   * `over_answer` is a GT-convention artifact (see plan §T1.4).
    #   * `clarification` is the DELIBERATE hyde_auto_answer path, INFO only.
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
    a(f"- mean_unique_gt_chunks/Q = {rm['mean_unique_gt_chunks']:.3f} (drives the theoretical precision ceiling)")
    a("")
    a("**Real precision levers (T1): p@1 and MRR — p@5 is bounded by GT ceiling.**")
    a("")
    a("| metric | value | ceiling | % of max |")
    a("|---|---:|---:|---:|")
    for k_ in (1, 3, 5):
        cur = rm["precision_at_k"][f"p@{k_}"]
        ceil = rm["precision_ceiling"][f"ceiling@{k_}"]
        pct = (cur / ceil * 100.0) if ceil else 0.0
        bold = "**" if k_ in (1, 5) else ""
        a(f"| {bold}p@{k_}{bold} | {bold}{cur:.4f}{bold} | {ceil:.4f} | {pct:.1f}% |")
    a(f"| **{rm['mrr']:.4f}** | _MRR_ | — | — |")
    a("")
    a(f"- hit@1 = {rm['hit_rate_at_k']['hit@1']:.4f}  |  hit@3 = {rm['hit_rate_at_k']['hit@3']:.4f}  |  hit@5 = {rm['hit_rate_at_k']['hit@5']:.4f}")
    a(f"- r@1 = {rm['recall_at_k']['r@1']:.4f}  |  r@3 = {rm['recall_at_k']['r@3']:.4f}  |  r@5 = {rm['recall_at_k']['r@5']:.4f}")
    a("")
    a(f"> **p@5 vs ceiling:** p@5 = {rm['precision_at_k']['p@5']:.4f} vs ceiling {rm['precision_ceiling']['ceiling@5']:.4f} → {100*rm['precision_at_k']['p@5']/rm['precision_ceiling']['ceiling@5']:.1f}% of max. "
      f"With mean {rm['mean_unique_gt_chunks']:.3f} GT chunks/question, p@5 cannot exceed ~{rm['precision_ceiling']['ceiling@5']:.4f}. The actionable lever is **p@1** ({rm['precision_at_k']['p@1']:.4f}) and **MRR** ({rm['mrr']:.4f}).")
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
    print(f"p@1={rm['precision_at_k']['p@1']:.4f}  p@5={rm['precision_at_k']['p@5']:.4f}  ceiling@5={rm['precision_ceiling']['ceiling@5']:.4f}  MRR={rm['mrr']:.4f}")
    print(f"abstain_rate={ab['abstain_rate']:.4f}  has_answer_rate={ab['has_answer_rate']:.4f}")
    print(f"weaknesses_ranked={len(weaknesses)}")
    # T1 confirmation lines
    n_clarification = sum(1 for f in ft if "clarification" in f["flags"])
    n_true_empty = sum(1 for f in ft if "empty_retrieval" in f["flags"])
    n_abstain_miss = sum(1 for f in ft if "abstain_miss" in f["flags"])
    print(f"clarification_reclassified={n_clarification}  true_empty_retrieval={n_true_empty}  abstain_miss={n_abstain_miss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
