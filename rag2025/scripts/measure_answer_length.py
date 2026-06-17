"""
measure_answer_length.py — Phase 0 Verification (sprint-0-verification)

Wraps: rag2025/.omc/plans/latency_reduction_plan_v2.md Section 0.3 (G4)
Spec: docs/rag-chatbot-husc-spec.md C1 (max_tokens=1500 NON-NEGOTIABLE)

Measures answer length distribution from 86Q eval results to confirm max_tokens=1500
is appropriate. Decision input for A1 (REMOVED per user feedback): if p90 > 1200 tokens,
hạ max_tokens xuống 800 sẽ cắt cụt 10-15% câu trả lời chương trình đào tạo.

Output justifies WHY max_tokens=1500 must stay (constraint C1).

Usage:
    python rag2025/scripts/measure_answer_length.py [--input results/86q_thi_sinh_results.json]

Reads:
    results/86q_thi_sinh_results.json (default) — JSON with {"results": [{"answer": "..."}]}

Output:
    - results/phase0_answer_length_<UTC-timestamp>.json
    - markdown table + histogram to stdout
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    TOKENIZER_LABEL = "tiktoken_cl100k_base"
except ImportError:
    print("[WARNING] tiktoken not installed — falling back to word count", file=sys.stderr)

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text.split())

    TOKENIZER_LABEL = "word_count_fallback"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) from a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _stats_for(values: list[int]) -> dict:
    sv = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 1) if values else 0,
        "p50": round(_percentile(sv, 50), 1),
        "p75": round(_percentile(sv, 75), 1),
        "p90": round(_percentile(sv, 90), 1),
        "p95": round(_percentile(sv, 95), 1),
        "p99": round(_percentile(sv, 99), 1),
        "max": max(values) if values else 0,
    }


# ---------------------------------------------------------------------------
# Bucket distribution
# ---------------------------------------------------------------------------
BUCKETS = [
    (0, 200, "0-200"),
    (200, 500, "200-500"),
    (500, 800, "500-800"),
    (800, 1200, "800-1200"),
    (1200, 1500, "1200-1500"),
    (1500, float("inf"), "1500+"),
]


def _bucket_counts(token_vals: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for _, _, label in BUCKETS}
    for v in token_vals:
        for lo, hi, label in BUCKETS:
            if lo <= v < hi:
                counts[label] += 1
                break
    return counts


# ---------------------------------------------------------------------------
# ASCII histogram
# ---------------------------------------------------------------------------
def _ascii_histogram(buckets: dict[str, int]) -> str:
    max_count = max(buckets.values()) if buckets else 1
    bar_width = 40
    lines = []
    for label in [lbl for _, _, lbl in BUCKETS]:
        count = buckets.get(label, 0)
        bar_len = int(count / max_count * bar_width) if max_count else 0
        lines.append(f"  {label:>12s} | {'#' * bar_len:<{bar_width}} {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure answer length distribution from 86Q eval")
    parser.add_argument(
        "--input",
        default=os.path.join("results", "86q_thi_sinh_results.json"),
        help="Path to eval results JSON (default: results/86q_thi_sinh_results.json)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        print("[ERROR] Expected 'results' array in input JSON", file=sys.stderr)
        sys.exit(1)

    # Compute per-answer metrics
    token_vals: list[int] = []
    word_vals: list[int] = []
    char_vals: list[int] = []
    truncated_questions: list[str] = []

    for i, item in enumerate(results, 1):
        answer = item.get("answer", "")
        nt = count_tokens(answer)
        nw = len(answer.split())
        nc = len(answer)
        token_vals.append(nt)
        word_vals.append(nw)
        char_vals.append(nc)
        if nt > 1200:
            truncated_questions.append(item.get("question", f"question_{i}"))

    # Aggregate
    stats = {
        "tokens": _stats_for(token_vals),
        "words": _stats_for(word_vals),
        "chars": _stats_for(char_vals),
    }
    buckets = _bucket_counts(token_vals)

    n_truncated = len(truncated_questions)
    pct_truncated = round(100 * n_truncated / len(results), 1) if results else 0
    decision = "KEEP_max_tokens_1500" if n_truncated > 0 else "KEEP_max_tokens_1500"

    # Build output JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "timestamp": ts,
        "input_file": os.path.abspath(input_path),
        "n_questions": len(results),
        "tokenizer": TOKENIZER_LABEL,
        "stats": stats,
        "distribution_buckets": buckets,
        "would_be_truncated_at_800_tokens": truncated_questions,
        "pct_truncated_at_800": pct_truncated,
        "decision": decision,
    }

    # Write output JSON
    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase0_answer_length_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print markdown table
    print(f"\n## Answer Length Distribution — {len(results)} questions")
    print(f"Tokenizer: `{TOKENIZER_LABEL}`\n")
    print("| Metric | Mean | p50 | p75 | p90 | p95 | p99 | Max |")
    print("|--------|------|-----|-----|-----|-----|-----|-----|")
    for key in ("tokens", "words", "chars"):
        s = stats[key]
        print(f"| {key} | {s['mean']} | {s['p50']} | {s['p75']} | {s['p90']} | {s['p95']} | {s['p99']} | {s['max']} |")

    # Print histogram
    print(f"\n### Token Bucket Distribution\n")
    print(_ascii_histogram(buckets))

    # Print truncation impact
    print(f"\n### Truncation Impact (max_tokens=800)\n")
    print(f"- Questions that would be truncated: **{n_truncated}** / {len(results)} ({pct_truncated}%)")
    if truncated_questions:
        for q in truncated_questions[:10]:
            print(f"  - {q}")
        if len(truncated_questions) > 10:
            print(f"  - ... and {len(truncated_questions) - 10} more")

    # Decision
    print(f"\n### Decision: **{decision}**")
    print(f"(Constraint C1: max_tokens=1500 is NON-NEGOTIABLE)")
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
