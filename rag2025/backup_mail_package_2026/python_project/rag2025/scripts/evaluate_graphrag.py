"""
GraphRAG Evaluation Script

Prints a formatted comparison table: PaddedRAG (Baseline) vs GraphRAG.
Reads from results/final_metrics.json.

Usage:
    cd rag2025
    python scripts/evaluate_graphrag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS_PATH = Path(__file__).parent.parent / "results" / "final_metrics.json"


def _pct(a: float, b: float) -> str:
    """Format percentage improvement from b to a."""
    delta = a - b
    pct = (delta / b * 100) if b else 0.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f} ({sign}{pct:.1f}%)"


def print_table(metrics: dict) -> None:
    baseline = metrics["naive_rag"]
    graphrag = metrics["graphrag"]
    meta = metrics["metadata"]

    header = f"{'Metric':<30} {'PaddedRAG':>12} {'GraphRAG':>12} {'Δ':>18}"
    sep = "─" * len(header)

    print()
    print("=" * len(header))
    print("  GraphRAG Evaluation Report – HUSC Admission Chatbot")
    print("=" * len(header))
    print(f"  Embedding : {meta['embedding_model']}")
    print(f"  Graph     : {meta['graph_stats']['nodes']} nodes / {meta['graph_stats']['edges']} edges")
    print(f"  PPR α     : {meta['ppr_params']['alpha']}  |  Fusion: {meta['ppr_params']['fusion_alpha']}·vec + {meta['ppr_params']['fusion_beta']}·ppr")
    print(f"  N         : {meta['n_questions']} questions ({meta['category_breakdown']})")
    print()

    categories = [("overall", "OVERALL"), ("simple", "Simple (1-hop)"), ("multihop", "Multi-hop"), ("comparative", "Comparative")]

    for cat_key, cat_label in categories:
        if cat_key == "overall":
            b_cat = baseline["overall"]
            g_cat = graphrag["overall"]
        else:
            b_cat = baseline["by_category"][cat_key]
            g_cat = graphrag["by_category"][cat_key]

        print(sep)
        print(f"  ▶ {cat_label}")
        print(sep)
        print(header)
        print(sep)

        metrics_to_show = [
            ("MRR", "mrr"),
            ("NDCG@5", "ndcg_at_5"),
            ("Faithfulness", "faithfulness"),
            ("Latency P50 (ms)", "latency_median_ms"),
            ("Latency P95 (ms)", "latency_p95_ms"),
        ]

        for label, key in metrics_to_show:
            b_val = b_cat.get(key, "N/A")
            g_val = g_cat.get(key, "N/A")
            if isinstance(b_val, float) and isinstance(g_val, float):
                delta = _pct(g_val, b_val)
                print(f"  {label:<28} {b_val:>12.3f} {g_val:>12.3f} {delta:>18}")
            else:
                print(f"  {label:<28} {str(b_val):>12} {str(g_val):>12} {'N/A':>18}")

        print()

    print(sep)
    print("  KEY FINDING:")
    imp = metrics["improvement_over_baseline"]
    print(f"  Multi-hop MRR   : {imp['multihop_highlight']['mrr_delta']}")
    print(f"  Multi-hop NDCG  : {imp['multihop_highlight']['ndcg_delta']}")
    print(f"  Overall latency : {imp['overall']['latency_overhead_ms']}")
    print(f"  Note            : {imp['multihop_highlight']['note']}")
    print("=" * len(header))
    print()


def main() -> None:
    if not METRICS_PATH.exists():
        print(f"ERROR: Metrics file not found: {METRICS_PATH}", file=sys.stderr)
        sys.exit(1)

    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)

    print_table(metrics)


if __name__ == "__main__":
    main()
