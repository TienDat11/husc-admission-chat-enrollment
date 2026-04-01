"""
Retrieval Quality Comparison: Harrier-OSS-v1 vs Qwen3-Embedding

Chạy: python rag2025/scripts/compare_retrieval.py

Script này so sánh retrieval quality bằng cách:
1. Load test queries với ground truth relevant chunks
2. Encode queries với embedding model
3. Retrieve from LanceDB
4. Tính precision@K, NDCG@K, MRR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embedding import EmbeddingService
from src.services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument
from config.settings import RAGSettings


EMBEDDING_CANDIDATES = [
    {
        "key": "qwen3_4b",
        "model": "Qwen/Qwen3-Embedding-4B",
        "dim": 2560,
        "provider": "qwen",
    },
    {
        "key": "harrier_0_6b",
        "model": "microsoft/harrier-oss-v1-0.6b",
        "dim": 1024,
        "provider": "harrier",
    },
    {
        "key": "bge_m3",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "provider": "bge",
    },
]


# Default ground truth test queries
DEFAULT_TEST_QUERIES = [
    {
        "id": "q001",
        "query": "Điều kiện tuyển sinh ngành Công nghệ thông tin?",
        "relevant_sources": ["tuyen_sinh", "cntt", "thong_tin"],
        "expected_keywords": ["tốt nghiệp", "THPT", "điểm thi", "18"]
    },
    {
        "id": "q002",
        "query": "Học phí năm 2025 là bao nhiêu?",
        "relevant_sources": ["hoc_phi", "nam_2025"],
        "expected_keywords": ["triệu", "đồng", "năm"]
    },
    {
        "id": "q003",
        "query": "Điểm chuẩn ngành Kinh tế?",
        "relevant_sources": ["diem_chuan", "kinh_te"],
        "expected_keywords": ["điểm", "chuẩn", "tối thiểu"]
    },
    {
        "id": "q004",
        "query": "Thời hạn nộp hồ sơ tuyển sinh?",
        "relevant_sources": ["tuyen_sinh", "han_nop"],
        "expected_keywords": ["hạn", "nộp", "hồ sơ", "tháng"]
    },
    {
        "id": "q005",
        "query": "Khoa Công nghệ thông tin ở đâu?",
        "relevant_sources": ["khoa_cntt", "dia_chi"],
        "expected_keywords": ["khoa", "công nghệ", "thông tin", "địa chỉ"]
    },
    {
        "id": "q006",
        "query": "Chương trình đào tạo ngành Marketing?",
        "relevant_sources": ["chuong_trinh", "marketing"],
        "expected_keywords": ["marketing", "chương trình", "đào tạo", "tín chỉ"]
    },
    {
        "id": "q007",
        "query": "Học bổng tuyển sinh có những loại nào?",
        "relevant_sources": ["hoc_bong", "tuyen_sinh"],
        "expected_keywords": ["học bổng", "loại", "tài năng", "khuyến khích"]
    },
    {
        "id": "q008",
        "query": "Ngành nào có điểm chuẩn cao nhất?",
        "relevant_sources": ["diem_chuan", "cao_nhat"],
        "expected_keywords": ["điểm", "chuẩn", "cao", "nhất"]
    },
    {
        "id": "q009",
        "query": "Địa chỉ trường Đại học Khoa học?",
        "relevant_sources": ["dia_chi", "truong"],
        "expected_keywords": ["địa chỉ", "khoa học", "đường", "quận"]
    },
    {
        "id": "q010",
        "query": "Số điện thoại liên hệ tư vấn tuyển sinh?",
        "relevant_sources": ["dien_thoai", "tu_van"],
        "expected_keywords": ["điện thoại", "0", "tư vấn", "hotline"]
    }
]


def load_ground_truth(path: Path) -> List[Dict]:
    """Load ground truth test queries."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_TEST_QUERIES


def evaluate_single_query(
    query: str,
    retriever: LanceDBRetriever,
    emb_service: EmbeddingService,
    relevant_sources: List[str],
    k: int = 5
) -> Dict:
    """Evaluate retrieval for a single query."""
    # Encode query
    query_vec = emb_service.encode_query(query)

    # Retrieve
    result = retriever.retrieve(query_vec.tolist(), top_k=k)

    # Calculate metrics
    retrieved_docs = result.documents
    num_retrieved = len(retrieved_docs)

    # Check keyword overlap
    retrieved_texts = " ".join([doc.text.lower() for doc in retrieved_docs])
    keyword_matches = sum(
        1 for kw in relevant_sources
        if any(kw.lower() in doc.text.lower() for doc in retrieved_docs)
    )

    # Precision based on keyword match
    precision = keyword_matches / max(len(relevant_sources), 1)

    # Score-based metrics
    top_score = retrieved_docs[0].score if retrieved_docs else 0.0
    avg_score = np.mean([doc.score for doc in retrieved_docs]) if retrieved_docs else 0.0

    return {
        "query": query,
        "num_retrieved": num_retrieved,
        "keyword_matches": keyword_matches,
        "precision": precision,
        "top_score": top_score,
        "avg_score": avg_score,
        "retrieved_texts": [doc.text[:100] + "..." for doc in retrieved_docs[:3]]
    }


def calculate_ndcg(relevance_scores: List[float], k: int) -> float:
    """Calculate NDCG@k."""
    if not relevance_scores:
        return 0.0

    # DCG
    dcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k]))

    # IDCG (ideal DCG)
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(relevant_docs: Set[str], retrieved_docs: List[RetrievedDocument]) -> float:
    """Calculate Mean Reciprocal Rank."""
    for idx, doc in enumerate(retrieved_docs, 1):
        if doc.chunk_id and any(rel in doc.text for rel in relevant_docs):
            return 1.0 / idx
    return 0.0


def run_retrieval_evaluation(
    model_name: str,
    dim: int,
    provider: str,
    test_queries: List[Dict],
    k: int = 5
) -> Dict:
    """Run full retrieval evaluation."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL EVALUATION: {model_name}")
    print(f"{'='*70}")

    # Create settings
    settings = RAGSettings()
    settings.EMBEDDING_PROVIDER = provider
    settings.EMBEDDING_MODEL = model_name
    settings.EMBEDDING_DIM = dim

    # Load embedding service
    print(f"Loading model: {model_name}...")
    emb_service = EmbeddingService(settings)

    # Load retriever
    print(f"Connecting to LanceDB: {settings.LANCEDB_URI}...")
    retriever = LanceDBRetriever.from_env()

    # Check collection
    collection_info = retriever.check_collection()
    print(f"LanceDB collection: {collection_info}")

    if not collection_info.get("exists"):
        print("ERROR: LanceDB table not found! Run setup_data.bat first.")
        return {"error": "LanceDB not initialized"}

    # Evaluate each query
    results = []
    all_precisions = []
    all_ndcgs = []
    all_mrrs = []

    for item in test_queries:
        query_id = item.get("id", f"q{len(results)+1}")
        query = item["query"]
        relevant_sources = item.get("relevant_sources", [])

        print(f"\n[{query_id}] {query[:50]}...")

        # Evaluate
        eval_result = evaluate_single_query(
            query=query,
            retriever=retriever,
            emb_service=emb_service,
            relevant_sources=relevant_sources,
            k=k
        )

        results.append(eval_result)
        all_precisions.append(eval_result["precision"])

        # Calculate NDCG
        relevance_scores = [1.0 if eval_result["keyword_matches"] > 0 else 0.0]
        ndcg = calculate_ndcg(relevance_scores, k)
        all_ndcgs.append(ndcg)

        # Calculate MRR
        relevant_set = set(relevant_sources)
        mrr = calculate_mrr(relevant_set, eval_result.get("retrieved_docs", []))
        all_mrrs.append(mrr)

        print(f"  Precision@{k}: {eval_result['precision']:.2f}")
        print(f"  Top score: {eval_result['top_score']:.4f}")
        print(f"  Retrieved: {eval_result['num_retrieved']} docs")

    # Summary
    summary = {
        "model": model_name,
        "dimension": dim,
        "provider": provider,
        "num_queries": len(test_queries),
        "k": k,
        "precision_at_k": {
            "mean": np.mean(all_precisions),
            "std": np.std(all_precisions),
            "min": np.min(all_precisions),
            "max": np.max(all_precisions),
        },
        "ndcg": {
            "mean": np.mean(all_ndcgs),
        },
        "mrr": {
            "mean": np.mean(all_mrrs),
        },
        "detailed_results": results
    }

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Precision@{k}: {summary['precision_at_k']['mean']:.3f} ± {summary['precision_at_k']['std']:.3f}")
    print(f"NDCG@{k}: {summary['ndcg']['mean']:.3f}")
    print(f"MRR: {summary['mrr']['mean']:.3f}")

    return summary


def print_comparison_matrix(results_by_key: Dict[str, Dict]) -> None:
    """Print side-by-side comparison for all embedding candidates."""
    print(f"\n{'='*80}")
    print("RETRIEVAL QUALITY COMPARISON (QWEN vs HARRIER vs BGE)")
    print(f"{'='*80}")

    ordered_keys = ["qwen3_4b", "harrier_0_6b", "bge_m3"]
    available = [k for k in ordered_keys if k in results_by_key]

    print(f"\n{'Model Key':<18} {'Precision@K':<14} {'NDCG@K':<12} {'MRR':<10}")
    print("-" * 60)
    for key in available:
        r = results_by_key[key]
        print(
            f"{key:<18} "
            f"{r['precision_at_k']['mean']:<14.3f} "
            f"{r['ndcg']['mean']:<12.3f} "
            f"{r['mrr']['mean']:<10.3f}"
        )

    winner_precision = max(available, key=lambda k: results_by_key[k]["precision_at_k"]["mean"])
    winner_ndcg = max(available, key=lambda k: results_by_key[k]["ndcg"]["mean"])
    winner_mrr = max(available, key=lambda k: results_by_key[k]["mrr"]["mean"])

    print("\nMetric winners:")
    print(f"- Precision@K winner: {winner_precision}")
    print(f"- NDCG@K winner: {winner_ndcg}")
    print(f"- MRR winner: {winner_mrr}")


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval quality between embedding models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--dim", type=int, help="Embedding dimension")
    parser.add_argument("--provider", type=str, default="qwen", choices=["qwen", "harrier", "bge"])
    parser.add_argument("--k", type=int, default=5, help="K for Precision@K")
    parser.add_argument("--compare", action="store_true", help="Compare Qwen3 vs Harrier vs BGE")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--queries", type=str, help="Path to test queries JSON file")
    args = parser.parse_args()

    # Load test queries
    queries_path = Path(args.queries) if args.queries else None
    test_queries = load_ground_truth(queries_path) if queries_path else DEFAULT_TEST_QUERIES

    print(f"Loaded {len(test_queries)} test queries")

    if args.compare:
        # Compare three models
        print("\n" + "="*70)
        print("RETRIEVAL COMPARISON: Qwen3-Embedding-4B vs Harrier-OSS-v1-0.6B vs BGE-M3")
        print("="*70)

        results_by_key: Dict[str, Dict] = {}
        for candidate in EMBEDDING_CANDIDATES:
            results_by_key[candidate["key"]] = run_retrieval_evaluation(
                model_name=candidate["model"],
                dim=candidate["dim"],
                provider=candidate["provider"],
                test_queries=test_queries,
                k=args.k,
            )

        # Print comparison
        print_comparison_matrix(results_by_key)

        # Save results
        if args.output:
            output = {
                "results": results_by_key,
                "test_queries": test_queries,
                "winners": {
                    "precision_at_k": max(results_by_key.items(), key=lambda kv: kv[1]["precision_at_k"]["mean"])[0],
                    "ndcg": max(results_by_key.items(), key=lambda kv: kv[1]["ndcg"]["mean"])[0],
                    "mrr": max(results_by_key.items(), key=lambda kv: kv[1]["mrr"]["mean"])[0],
                },
            }
            Path(args.output).write_text(
                json.dumps(output, indent=2, ensure_ascii=False)
            )
            print(f"\nResults saved to: {args.output}")

    elif args.model and args.dim:
        # Single model evaluation
        results = run_retrieval_evaluation(
            model_name=args.model,
            dim=args.dim,
            provider=args.provider,
            test_queries=test_queries,
            k=args.k
        )

        if args.output:
            Path(args.output).write_text(
                json.dumps(results, indent=2, ensure_ascii=False)
            )
            print(f"\nResults saved to: {args.output}")
    else:
        # Default to compare mode without recursion
        print("\nNo specific model provided. Running 3-model retrieval comparison...")
        results_by_key: Dict[str, Dict] = {}
        for candidate in EMBEDDING_CANDIDATES:
            results_by_key[candidate["key"]] = run_retrieval_evaluation(
                model_name=candidate["model"],
                dim=candidate["dim"],
                provider=candidate["provider"],
                test_queries=test_queries,
                k=args.k,
            )

        print_comparison_matrix(results_by_key)

        if args.output:
            output = {
                "results": results_by_key,
                "test_queries": test_queries,
                "winners": {
                    "precision_at_k": max(results_by_key.items(), key=lambda kv: kv[1]["precision_at_k"]["mean"])[0],
                    "ndcg": max(results_by_key.items(), key=lambda kv: kv[1]["ndcg"]["mean"])[0],
                    "mrr": max(results_by_key.items(), key=lambda kv: kv[1]["mrr"]["mean"])[0],
                },
            }
            Path(args.output).write_text(
                json.dumps(output, indent=2, ensure_ascii=False)
            )
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
