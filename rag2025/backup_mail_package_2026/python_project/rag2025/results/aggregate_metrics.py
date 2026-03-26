"""
Aggregate evaluation metrics and compute final results.

Since the experiment failed due to infrastructure issues, this file generates
EXPERIMENT_FAILED markers for all metrics as required by prompt.md rules.
"""
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


def aggregate_metrics() -> Dict[str, Any]:
    """
    Aggregate metrics from evaluation results.
    Returns EXPERIMENT_FAILED markers for all metrics due to infrastructure issues.
    """
    # Load test questions to get metadata
    with open('results/test_questions.json', 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    # Count questions by category
    category_counts = {
        "simple": sum(1 for q in test_questions if q["category"] == "simple"),
        "multihop": sum(1 for q in test_questions if q["category"] == "multihop"),
        "comparative": sum(1 for q in test_questions if q["category"] == "comparative")
    }
    
    # Create EXPERIMENT_FAILED structure
    failed_metrics = {
        "mrr": "EXPERIMENT_FAILED",
        "ndcg": "EXPERIMENT_FAILED",
        "faithfulness": "EXPERIMENT_FAILED",
        "latency_median_ms": "EXPERIMENT_FAILED",
        "latency_p95_ms": "EXPERIMENT_FAILED"
    }
    
    # Build final metrics structure
    final_metrics = {
        "naive_rag": {
            "overall": failed_metrics.copy(),
            "by_category": {
                "simple": failed_metrics.copy(),
                "multihop": failed_metrics.copy(),
                "comparative": failed_metrics.copy()
            }
        },
        "graphrag": {
            "overall": failed_metrics.copy(),
            "by_category": {
                "simple": failed_metrics.copy(),
                "multihop": failed_metrics.copy(),
                "comparative": failed_metrics.copy()
            }
        },
        "metadata": {
            "n_questions": len(test_questions),
            "date_run": datetime.now().isoformat(),
            "naive_rag_model": "EXPERIMENT_FAILED",
            "graphrag_model": "EXPERIMENT_FAILED",
            "neo4j_version": "N/A (system uses Qdrant)",
            "experiment_status": "FAILED - Qdrant connectivity issues",
            "preflight_checks": {
                "neo4j": "N/A (system doesn't use Neo4j)",
                "qdrant": "FAILED (connection failed to cloud Qdrant instance)",
                "paddedrag": "PARTIAL (settings load but Qdrant connectivity fails)"
            },
            "category_breakdown": category_counts
        }
    }
    
    return final_metrics


def save_final_metrics():
    """
    Compute and save final metrics to results/final_metrics.json
    """
    final_metrics = aggregate_metrics()
    
    with open('results/final_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    
    print("Final metrics saved to results/final_metrics.json")
    print("Status: EXPERIMENT_FAILED - All metrics marked as EXPERIMENT_FAILED")
    
    return final_metrics


if __name__ == "__main__":
    metrics = save_final_metrics()
    
    # Print summary
    print(f"\nExperiment Summary:")
    print(f"Total questions: {metrics['metadata']['n_questions']}")
    print(f"Simple questions: {metrics['metadata']['category_breakdown']['simple']}")
    print(f"Multi-hop questions: {metrics['metadata']['category_breakdown']['multihop']}")
    print(f"Comparative questions: {metrics['metadata']['category_breakdown']['comparative']}")
    print(f"\nAll metrics: EXPERIMENT_FAILED")
    print(f"Reason: {metrics['metadata']['experiment_status']}")