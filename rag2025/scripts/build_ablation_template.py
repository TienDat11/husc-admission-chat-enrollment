from __future__ import annotations

import json
from pathlib import Path


def build_ablation_template() -> dict:
    return {
        "metadata": {
            "benchmark": "HUSC GraphRAG Ablation",
            "version": "2026-03",
            "models": {
                "embedding": "Qwen/Qwen3-Embedding-8B",
                "reranker": "Qwen/Qwen3-Reranker-8B",
                "router": "gemini-2.5-flash",
            },
        },
        "variants": {
            "naive_rag": {
                "description": "Dense retrieval only, no graph, no rerank",
                "metrics": {
                    "mrr": 0.0,
                    "ndcg_at_5": 0.0,
                    "faithfulness": 0.0,
                    "latency_median_ms": 0.0,
                },
            },
            "hybrid_rag": {
                "description": "Dense + reranker, no graph",
                "metrics": {
                    "mrr": 0.0,
                    "ndcg_at_5": 0.0,
                    "faithfulness": 0.0,
                    "latency_median_ms": 0.0,
                },
            },
            "graphrag": {
                "description": "Dense + reranker + router + PPR graph fusion",
                "metrics": {
                    "mrr": 0.0,
                    "ndcg_at_5": 0.0,
                    "faithfulness": 0.0,
                    "latency_median_ms": 0.0,
                },
            },
        },
    }


def main() -> None:
    out_path = Path(__file__).parent.parent / "results" / "ablation_metrics_template.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(build_ablation_template(), f, ensure_ascii=False, indent=2)
    print(f"Ablation template written to {out_path}")


if __name__ == "__main__":
    main()
