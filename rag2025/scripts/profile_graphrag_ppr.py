"""
Throwaway diagnostic: profile GraphRAG PPR path on the real knowledge graph.

Measures (median of N runs, time.perf_counter):
  - _pagerank_view() alone (MultiDiGraph -> DiGraph collapse)
  - personalized_pagerank([seed]) first call (miss + build view + nx.pagerank)
  - personalized_pagerank([seed]) second call (cache hit, with view rebuild)
  - personalized_pagerank([seed]) forced-miss call (cache cleared)
  - ppr_scores_by_chunk([seed]) end-to-end
  - raw nx.pagerank on a pre-built view (isolates networkx cost)
  - aggregation step (node_scores -> chunk_scores) on its own

Usage:
    cd rag2025
    D:/miniconda3/python.exe scripts/profile_graphrag_ppr.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List

# Ensure repo-root-relative imports work when invoked from `rag2025/`.
THIS_DIR = Path(__file__).resolve().parent
RAG_ROOT = THIS_DIR.parent
sys.path.insert(0, str(RAG_ROOT))

from src.domain.graph import KnowledgeGraph  # noqa: E402

GRAPH_PATH = RAG_ROOT / "data" / "graph" / "knowledge_graph.graphml"

# Seeds chosen to match what production sees: a NGANH node + a couple of
# realistic candidates. The first NGANH:cong_nghe_thong_tin is the production
# "CNTT" / "IT" hub.
DEFAULT_SEEDS = [
    "NGANH:cong_nghe_thong_tin",
    "TO_HOP:a00",
    "DIEM_CHUAN:cong_nghe_thong_tin_2024",
]

RUNS = 5
WARMUPS = 1


def median_ms(samples: List[float]) -> float:
    return statistics.median(samples) * 1000.0


def time_call(fn: Callable[[], None], runs: int, warmups: int = 0) -> List[float]:
    for _ in range(warmups):
        fn()
    out: List[float] = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def fmt_samples(samples: List[float]) -> str:
    samples_ms = [s * 1000.0 for s in samples]
    return (
        f"min={min(samples_ms):.1f} med={median_ms(samples):.1f} "
        f"max={max(samples_ms):.1f}  (runs={len(samples)})"
    )


def main() -> int:
    if not GRAPH_PATH.exists():
        print(f"BLOCKED: graph file not found at {GRAPH_PATH}")
        return 2

    print(f"Loading graph: {GRAPH_PATH}")
    t0 = time.perf_counter()
    kg = KnowledgeGraph.load(GRAPH_PATH)
    load_ms = (time.perf_counter() - t0) * 1000.0
    nodes = kg.node_count
    edges = kg.edge_count
    print(f"  load: {load_ms:.0f} ms")
    print(f"  nodes={nodes}  edges={edges}")

    stats = kg.stats()
    print(
        f"  avg_degree={stats['avg_degree']:.2f}  "
        f"max_degree={stats['max_degree']}  "
        f"schema={stats['schema_version']}"
    )

    # Pick seeds that exist in the graph.
    seeds = [s for s in DEFAULT_SEEDS if kg.has_node(s)]
    if not seeds:
        # Fallback: first NGANH node we can find
        for nid in kg._graph.nodes:
            if nid.startswith("NGANH:"):
                seeds = [nid]
                break
    if not seeds:
        print("BLOCKED: no NGANH seed in graph")
        return 3
    print(f"  seeds ({len(seeds)}): {seeds}")

    # 1) _pagerank_view() alone
    samples = time_call(lambda: kg._pagerank_view(), runs=RUNS, warmups=WARMUPS)
    print(f"\n[1] _pagerank_view() alone        : {fmt_samples(samples)} ms")

    # 2) personalized_pagerank([seed]) — FIRST call (cache miss, builds view)
    kg.clear_caches()
    samples = time_call(
        lambda: kg.personalized_pagerank(seeds, alpha=0.85, max_iter=100),
        runs=1,
        warmups=0,
    )
    print(
        f"[2] personalized_pagerank FIRST  : {fmt_samples(samples)} ms  "
        f"(cache miss + view build + nx.pagerank)"
    )

    # 3) personalized_pagerank([seed]) — SECOND call (cache hit, but view
    #    still rebuilt every call per current code path).
    samples = time_call(
        lambda: kg.personalized_pagerank(seeds, alpha=0.85, max_iter=100),
        runs=RUNS,
        warmups=WARMUPS,
    )
    print(
        f"[3] personalized_pagerank CACHE  : {fmt_samples(samples)} ms  "
        f"(cache hit, but _pagerank_view() still rebuilt)"
    )

    # 4) personalized_pagerank forced-miss (cache cleared each call). This
    #    isolates the "real" cost per query in production where seeds vary
    #    and the cache key (tuple(sorted_seeds)) rarely collides.
    def _forced_miss():
        kg.clear_caches()
        return kg.personalized_pagerank(seeds, alpha=0.85, max_iter=100)

    samples = time_call(_forced_miss, runs=RUNS, warmups=WARMUPS)
    print(
        f"[4] personalized_pagerank MISSxN : {fmt_samples(samples)} ms  "
        f"(cache cleared per call — production-like)"
    )

    # 5) ppr_scores_by_chunk([seed]) end-to-end
    samples = time_call(
        lambda: kg.ppr_scores_by_chunk(seeds, alpha=0.85),
        runs=RUNS,
        warmups=WARMUPS,
    )
    print(f"[5] ppr_scores_by_chunk e2e      : {fmt_samples(samples)} ms")

    # Reproduce the ~1300 chunk figure
    chunk_scores = kg.ppr_scores_by_chunk(seeds, alpha=0.85)
    print(f"    ppr_scores_by_chunk returned: {len(chunk_scores)} chunks")

    # 6) raw nx.pagerank on a pre-built view (isolates networkx cost)
    view = kg._pagerank_view()
    personalization = {node: 1.0 / len(seeds) for node in seeds}
    samples = time_call(
        lambda: __import__("networkx").pagerank(
            view,
            alpha=0.85,
            personalization=personalization,
            max_iter=100,
            weight="weight",
        ),
        runs=RUNS,
        warmups=WARMUPS,
    )
    print(
        f"[6] nx.pagerank on PREBUILT view : {fmt_samples(samples)} ms  "
        f"(view already collapsed; isolates networkx only)"
    )

    # 7) aggregation cost (node_scores -> chunk_scores)
    node_scores = kg.personalized_pagerank(seeds, alpha=0.85, max_iter=100)
    samples = time_call(
        lambda: _aggregate_only(node_scores, kg),
        runs=RUNS,
        warmups=WARMUPS,
    )
    print(
        f"[7] aggregate nodes->chunks      : {fmt_samples(samples)} ms  "
        f"({len(node_scores)} nodes -> chunks)"
    )

    # 8) lower max_iter cost (max_iter=20)
    kg.clear_caches()
    samples = time_call(
        lambda: kg.personalized_pagerank(seeds, alpha=0.85, max_iter=20),
        runs=RUNS,
        warmups=WARMUPS,
    )
    print(
        f"[8] personalized_pagerank iter=20: {fmt_samples(samples)} ms  "
        f"(max_iter lowered from 100)"
    )

    # Top-10 chunk_ids by PPR score
    if chunk_scores:
        top10 = sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop-10 chunks by PPR score:")
        for cid, sc in top10:
            print(f"  {sc:.6f}  {cid}")

    return 0


def _aggregate_only(node_scores, kg: KnowledgeGraph):
    chunk_scores = {}
    for node_id, score in node_scores.items():
        for cid in kg.get_chunk_ids_for_entity(node_id):
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + score
    if not chunk_scores:
        return {}
    max_s = max(chunk_scores.values())
    if max_s > 0:
        return {k: v / max_s for k, v in chunk_scores.items()}
    return chunk_scores


if __name__ == "__main__":
    sys.exit(main())
