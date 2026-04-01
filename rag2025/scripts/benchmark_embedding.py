"""
Benchmark Script: So sánh Harrier-OSS-v1 vs Qwen3-Embedding

Chạy: python rag2025/scripts/benchmark_embedding.py

Script này đo:
1. Query latency (ms)
2. Indexing throughput (docs/sec)
3. Memory usage (GB)
4. Retrieval quality (precision@K nếu có ground truth)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService


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


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def measure_memory() -> float:
    """Get current process memory usage in GB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        # Fallback: estimate from available memory
        import os
        if sys.platform == 'win32':
            return 0.0  # Can't easily measure on Windows without psutil
        else:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / (1024 ** 2)  # KB to GB
    return 0.0


def load_test_chunks(data_dir: Path, max_chunks: int = 100) -> List[str]:
    """Load test texts from rag2025/data with safe fallbacks."""
    candidate_files = []

    chunked_dir = data_dir / "chunked"
    if chunked_dir.exists():
        candidate_files.extend(sorted(chunked_dir.glob("chunked_*.jsonl")))

    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        candidate_files.extend(sorted(raw_dir.glob("*.jsonl")))

    candidate_files.extend(
        sorted(
            f for f in data_dir.rglob("*.jsonl")
            if f not in candidate_files
        )
    )

    chunks = []
    for f in candidate_files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(obj, dict):
                    text = obj.get("text") or obj.get("content") or obj.get("question") or ""
                else:
                    text = str(obj)

                if text:
                    chunks.append(text)
                    if len(chunks) >= max_chunks:
                        return chunks[:max_chunks]

    return chunks[:max_chunks]


def load_test_queries(queries_file: Optional[Path] = None) -> List[str]:
    """Load test queries for benchmarking."""
    # Try to load from file if exists
    if queries_file and queries_file.exists():
        try:
            data = json.loads(queries_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [q.get("query", q) if isinstance(q, dict) else str(q) for q in data]
        except Exception:
            pass

    # Default test queries (Vietnamese admission FAQs)
    return [
        "Điều kiện tuyển sinh ngành Công nghệ thông tin là gì?",
        "Học phí năm 2025 bao nhiêu?",
        "Ngành nào có điểm chuẩn cao nhất?",
        "Thời hạn nộp hồ sơ tuyển sinh?",
        "Khoa Công nghệ thông tin ở đâu?",
        "Cho tôi biết về chương trình đào tạo ngành Marketing",
        "Điểm chuẩn các ngành năm 2024",
        "Học bổng tuyển sinh có những loại nào?",
        "Địa chỉ trường đại học Khoa học?",
        "Số điện thoại liên hệ tuyển sinh?",
        "Ngành Kinh tế học ở khoa nào?",
        "Học phí ngành Kinh tế là bao nhiêu?",
    ]


def benchmark_query_latency(
    emb_service: EmbeddingService,
    queries: List[str],
    num_runs: int = 3
) -> Dict:
    """Benchmark query encoding latency."""
    results = {
        "latencies_ms": [],
        "avg_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "p95_ms": 0.0,
    }

    all_latencies = []

    for run in range(num_runs):
        run_latencies = []
        for q in queries:
            start = time.perf_counter()
            _ = emb_service.encode_query(q)
            elapsed_ms = (time.perf_counter() - start) * 1000
            run_latencies.append(elapsed_ms)

        all_latencies.extend(run_latencies)

        # Warm up
        if run == 0:
            _ = emb_service.encode_query(queries[0])

    results["latencies_ms"] = all_latencies
    results["avg_ms"] = np.mean(all_latencies)
    results["min_ms"] = np.min(all_latencies)
    results["max_ms"] = np.max(all_latencies)
    results["p95_ms"] = np.percentile(all_latencies, 95)

    return results


def benchmark_batch_encoding(
    emb_service: EmbeddingService,
    documents: List[str],
    batch_sizes: List[int] = [1, 8, 32, 64]
) -> Dict:
    """Benchmark batch document encoding."""
    results = {}

    for batch_size in batch_sizes:
        if batch_size > len(documents):
            continue

        # Create batches
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]

        start = time.perf_counter()
        for batch in batches:
            _ = emb_service.encode_documents(batch)
        elapsed = time.perf_counter() - start

        total_docs = len(documents)
        throughput = total_docs / elapsed
        results[f"batch_{batch_size}"] = {
            "total_time_sec": elapsed,
            "throughput_docs_per_sec": throughput,
            "docs_processed": total_docs,
        }

    return results


def benchmark_indexing(
    emb_service: EmbeddingService,
    documents: List[str]
) -> Dict:
    """Benchmark full indexing process."""
    start = time.perf_counter()
    vectors = emb_service.encode_documents(documents)
    indexing_time = time.perf_counter() - start

    return {
        "total_chunks": len(documents),
        "total_time_sec": indexing_time,
        "throughput_chunks_per_sec": len(documents) / indexing_time,
        "vector_shape": vectors.shape,
        "vector_dtype": str(vectors.dtype),
    }


def run_benchmark(model_name: str, dim: int, provider: str = "qwen") -> Dict:
    """Run complete benchmark for a model."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*70}")

    # Memory before
    mem_before = measure_memory()
    print(f"Memory before: {mem_before:.2f} GB")

    # Create settings and embedding service
    settings = RAGSettings()

    # Override for this benchmark
    original_model = settings.EMBEDDING_MODEL
    original_dim = settings.EMBEDDING_DIM

    settings.EMBEDDING_PROVIDER = provider
    settings.EMBEDDING_MODEL = model_name
    settings.EMBEDDING_DIM = dim

    # Load embedding service
    print(f"Loading model: {model_name}...")
    load_start = time.perf_counter()
    emb_service = EmbeddingService(settings)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    mem_after_load = measure_memory()
    print(f"Memory after load: {mem_after_load:.2f} GB (delta: {mem_after_load - mem_before:.2f} GB)")

    # Load test data
    data_dir = Path(__file__).parent.parent / "data"
    test_documents = load_test_chunks(data_dir, max_chunks=100)
    test_queries = load_test_queries()

    print(f"\nTest data: {len(test_documents)} documents, {len(test_queries)} queries")

    # Run benchmarks
    results = {
        "model": model_name,
        "dimension": dim,
        "provider": provider,
        "load_time_sec": load_time,
        "memory_delta_gb": mem_after_load - mem_before,
    }

    # Query latency
    print("\n[1/3] Benchmarking query latency...")
    query_results = benchmark_query_latency(emb_service, test_queries)
    results["query_latency"] = query_results
    print(f"  Avg: {query_results['avg_ms']:.1f}ms, P95: {query_results['p95_ms']:.1f}ms")

    # Batch encoding
    print("\n[2/3] Benchmarking batch encoding...")
    batch_results = benchmark_batch_encoding(emb_service, test_documents)
    results["batch_encoding"] = batch_results
    for bs, data in batch_results.items():
        print(f"  {bs}: {data['throughput_docs_per_sec']:.1f} docs/sec")

    # Full indexing
    print("\n[3/3] Benchmarking indexing throughput...")
    index_results = benchmark_indexing(emb_service, test_documents)
    results["indexing"] = index_results
    print(f"  {index_results['throughput_chunks_per_sec']:.1f} chunks/sec")

    # Cleanup
    del emb_service
    gc.collect()

    mem_final = measure_memory()
    print(f"\nMemory after cleanup: {mem_final:.2f} GB")

    return results


def print_comparison_matrix(results_by_key: Dict[str, Dict]) -> None:
    """Print side-by-side comparison for all embedding candidates."""
    print(f"\n{'='*90}")
    print("COMPARISON SUMMARY (QWEN vs HARRIER vs BGE)")
    print(f"{'='*90}")

    ordered_keys = ["qwen3_4b", "harrier_0_6b", "bge_m3"]
    available = [k for k in ordered_keys if k in results_by_key]

    if len(available) < 2:
        print("Need at least 2 models to compare.")
        return

    print("\nModel overview:")
    for key in available:
        r = results_by_key[key]
        print(f"- {key}: {r['model']} | dim={r['dimension']} | provider={r['provider']}")

    def _winner(metric_getter, better: str = "min") -> tuple[str, float]:
        pairs = [(k, metric_getter(results_by_key[k])) for k in available]
        if better == "min":
            return min(pairs, key=lambda x: x[1])
        return max(pairs, key=lambda x: x[1])

    w_query, query_val = _winner(lambda r: r["query_latency"]["avg_ms"], better="min")
    w_index, index_val = _winner(lambda r: r["indexing"]["throughput_chunks_per_sec"], better="max")
    w_mem, mem_val = _winner(lambda r: r["memory_delta_gb"], better="min")
    w_load, load_val = _winner(lambda r: r["load_time_sec"], better="min")

    print("\nMetric winners:")
    print(f"- Query latency winner: {w_query} ({query_val:.1f} ms)")
    print(f"- Indexing throughput winner: {w_index} ({index_val:.1f} chunks/s)")
    print(f"- Memory delta winner: {w_mem} ({mem_val:.2f} GB)")
    print(f"- Load time winner: {w_load} ({load_val:.1f} s)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--model", type=str, help="Specific model to benchmark")
    parser.add_argument("--dim", type=int, help="Embedding dimension")
    parser.add_argument("--provider", type=str, default="qwen", choices=["qwen", "harrier", "bge"],
                        help="Provider type")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison between Qwen3-4B, Harrier-0.6B, and BGE-M3")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    if args.compare:
        # Compare three embeddings: Qwen3-4B vs Harrier-0.6B vs BGE-M3
        print("\n" + "="*70)
        print("EMBEDDING MODEL COMPARISON: Qwen3-Embedding-4B vs Harrier-OSS-v1-0.6B vs BGE-M3")
        print("="*70)

        results_by_key: Dict[str, Dict] = {}
        for candidate in EMBEDDING_CANDIDATES:
            gc.collect()
            results_by_key[candidate["key"]] = run_benchmark(
                model_name=candidate["model"],
                dim=candidate["dim"],
                provider=candidate["provider"],
            )

        # Print comparison
        print_comparison_matrix(results_by_key)

        # Save results
        output = {
            "results": results_by_key,
            "winners": {
                "query_latency": min(results_by_key.items(), key=lambda kv: kv[1]["query_latency"]["avg_ms"])[0],
                "indexing_throughput": max(results_by_key.items(), key=lambda kv: kv[1]["indexing"]["throughput_chunks_per_sec"])[0],
                "memory_delta": min(results_by_key.items(), key=lambda kv: kv[1]["memory_delta_gb"])[0],
                "load_time": min(results_by_key.items(), key=lambda kv: kv[1]["load_time_sec"])[0],
            }
        }

        if args.output:
            output_file = Path(args.output)
            output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"\nResults saved to: {output_file}")

    elif args.model and args.dim:
        # Benchmark single model
        results = run_benchmark(
            model_name=args.model,
            dim=args.dim,
            provider=args.provider
        )

        if args.output:
            output_file = Path(args.output)
            output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"\nResults saved to: {output_file}")
    else:
        # Default to compare mode without recursion
        print("\nNo specific model provided. Running 3-model comparison...")
        results_by_key: Dict[str, Dict] = {}
        for candidate in EMBEDDING_CANDIDATES:
            gc.collect()
            results_by_key[candidate["key"]] = run_benchmark(
                model_name=candidate["model"],
                dim=candidate["dim"],
                provider=candidate["provider"],
            )

        print_comparison_matrix(results_by_key)

        if args.output:
            output = {
                "results": results_by_key,
                "winners": {
                    "query_latency": min(results_by_key.items(), key=lambda kv: kv[1]["query_latency"]["avg_ms"])[0],
                    "indexing_throughput": max(results_by_key.items(), key=lambda kv: kv[1]["indexing"]["throughput_chunks_per_sec"])[0],
                    "memory_delta": min(results_by_key.items(), key=lambda kv: kv[1]["memory_delta_gb"])[0],
                    "load_time": min(results_by_key.items(), key=lambda kv: kv[1]["load_time_sec"])[0],
                }
            }
            output_file = Path(args.output)
            output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
