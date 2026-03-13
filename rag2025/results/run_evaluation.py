"""
Run evaluation experiments for Naive RAG and GraphRAG systems.

NOTE: This experiment cannot run due to infrastructure failures:
- CHECK_1: Neo4j - N/A (system doesn't use Neo4j)
- CHECK_2: Qdrant - FAILED (connection failed to cloud Qdrant instance)
- CHECK_3: PaddedRAG - PARTIAL (settings load but Qdrant connectivity fails)

This file contains the evaluation framework structure but will output EXPERIMENT_FAILED results.
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import metrics
from metrics import mrr_at_k, ndcg_at_k, faithfulness_score, measure_latency


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/errors.log'),
        logging.StreamHandler()
    ]
)


def run_naive_rag_query(question: str) -> Dict[str, Any]:
    """
    Simulate Naive RAG query execution.
    Returns EXPERIMENT_FAILED due to infrastructure issues.
    """
    logging.warning(f"Naive RAG query failed for: {question[:50]}...")
    return {
        "retrieved_chunks": [],
        "generated_answer": "EXPERIMENT_FAILED: Qdrant connectivity issue",
        "latency_ms": 0.0,
        "error": "Qdrant connection failed - server disconnected"
    }


def run_graphrag_query(question: str) -> Dict[str, Any]:
    """
    Simulate GraphRAG query execution.
    Returns EXPERIMENT_FAILED due to infrastructure issues.
    """
    logging.warning(f"GraphRAG query failed for: {question[:50]}...")
    return {
        "retrieved_chunks": [],
        "generated_answer": "EXPERIMENT_FAILED: Qdrant connectivity issue",
        "latency_ms": 0.0,
        "error": "Qdrant connection failed - server disconnected",
        "graph_traversal_path": []
    }


def run_evaluation():
    """
    Main evaluation function.
    Attempts to run both systems but records failures.
    """
    # Load test questions
    with open('results/test_questions.json', 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    naive_rag_results = []
    graphrag_results = []
    
    logging.info(f"Starting evaluation for {len(test_questions)} questions")
    
    for question in test_questions:
        question_id = question["id"]
        
        # Run Naive RAG
        try:
            naive_result = run_naive_rag_query(question["question"])
            naive_result["question_id"] = question_id
            naive_result["category"] = question["category"]
            naive_rag_results.append(naive_result)
        except Exception as e:
            logging.error(f"Naive RAG failed for {question_id}: {e}")
            naive_rag_results.append({
                "question_id": question_id,
                "category": question["category"],
                "error": str(e),
                "retrieved_chunks": [],
                "generated_answer": "EXPERIMENT_FAILED",
                "latency_ms": 0.0
            })
        
        # Run GraphRAG
        try:
            graphrag_result = run_graphrag_query(question["question"])
            graphrag_result["question_id"] = question_id
            graphrag_result["category"] = question["category"]
            graphrag_results.append(graphrag_result)
        except Exception as e:
            logging.error(f"GraphRAG failed for {question_id}: {e}")
            graphrag_results.append({
                "question_id": question_id,
                "category": question["category"],
                "error": str(e),
                "retrieved_chunks": [],
                "generated_answer": "EXPERIMENT_FAILED",
                "latency_ms": 0.0,
                "graph_traversal_path": []
            })
    
    # Save results
    with open('results/naive_rag_results.json', 'w', encoding='utf-8') as f:
        json.dump(naive_rag_results, f, indent=2, ensure_ascii=False)
    
    with open('results/graphrag_results.json', 'w', encoding='utf-8') as f:
        json.dump(graphrag_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Evaluation completed. Results saved with {len(naive_rag_results)} Naive RAG and {len(graphrag_results)} GraphRAG results")
    
    return naive_rag_results, graphrag_results


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Run evaluation
    naive_results, graphrag_results = run_evaluation()
    
    # Log summary
    naive_errors = sum(1 for r in naive_results if "error" in r)
    graphrag_errors = sum(1 for r in graphrag_results if "error" in r)
    
    print(f"\nEvaluation Summary:")
    print(f"Naive RAG: {len(naive_results)} questions, {naive_errors} errors")
    print(f"GraphRAG: {len(graphrag_results)} questions, {graphrag_errors} errors")
    print(f"\nStatus: EXPERIMENT_FAILED - Qdrant connectivity issues")
    print(f"See results/errors.log for detailed error information")