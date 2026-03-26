# Hybrid Retrieval Integration Design Spec (v3)

**Date:** 2026-03-25
**Author:** Claude (Workflow Agent)
**Status:** Draft v3 — HybridSearchService Architecture
**Epic:** Hybrid Retrieval Enhancement

---

## Executive Summary

Enable BM25 hybrid retrieval (dense + sparse) in the RAG pipeline to improve recall by 20-40%. The active retrieval stack uses `LanceDBRetriever` which returns `RetrievedDocument` objects. Rather than adapter-wrapping the existing `HybridRetriever` (which returns incompatible `SearchResult` types), this spec defines a new **`HybridSearchService`** class that works natively with `RetrievedDocument` and integrates directly with the `LanceDBRetriever` pipeline in `main.py`.

---

## Problem Statement

### Current State
- `main.py` uses **dense-only retrieval** via `LanceDBRetriever` (lines 486-531)
- `LanceDBRetriever.retrieve()` returns `RetrievalResult` containing `List[RetrievedDocument]`
- `sparse_terms` are already extracted during chunking and stored in LanceDB
- Research shows hybrid search (dense + BM25) improves recall by 20-40% over dense-only

### Why the Existing `HybridRetriever` Cannot Be Reused
`HybridRetriever` in `retriever.py` depends on `VectorStore` (abstract base class — not instantiable) and returns `SearchResult` objects (fields: `doc_id`, `chunk_id`, `score`, `text`, `metadata`). The pipeline expects `RetrievedDocument` (fields: `text`, `source`, `chunk_id`, `metadata`, `score`, `point_id`). Bridging this requires:
- A concrete `VectorStore` subclass wrapping `LanceDBRetriever` (~80 LOC)
- A `SearchResult` → `RetrievedDocument` field-mapping adapter (`doc_id` vs `source`)
- An `EmbeddingService` interface wrapper

Total adapter complexity: ~150 LOC with high fragility risk. Instead, this spec defines a focused 130-LOC class that does exactly what the pipeline needs.

### Gap
- No BM25 index is built from corpus at startup
- Current retrieval flow bypasses sparse retrieval entirely
- Keyword-heavy queries (e.g., "học phí ngành Công nghệ thông tin") miss lexical matches

---

## Goals

1. **Create `HybridSearchService`** — new class working natively with `RetrievedDocument`
2. **Build BM25 index at startup** from LanceDB using column-selective `to_pandas()`
3. **Implement RRF fusion** (Reciprocal Rank Fusion, k=60) over dense + sparse results
4. **Preserve backward compatibility** — keep existing dense-only path as fallback
5. **Enable toggle** via `USE_HYBRID_RETRIEVAL` config flag (default=False)
6. **Graceful degradation** — 3 levels: full hybrid → BM25 fails (dense-only) → flag=False

---

## Non-Goals

- Modifying existing `HybridRetriever`, `LanceDBRetriever`, or `VectorStore` classes
- Changing chunking pipeline (sparse_terms already extracted)
- Re-ingesting data (sparse_terms already in LanceDB)
- GraphRAG integration (separate concern)

---

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           main.py                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Startup:                                                    │ │
│  │   1. LanceDB init (existing)                               │ │
│  │   2. IF USE_HYBRID_RETRIEVAL:                              │ │
│  │        hybrid_search_service = HybridSearchService(        │ │
│  │            lancedb_retriever_service, settings)            │ │
│  │        success = hybrid_search_service.build_bm25_index()  │ │
│  │        if not success: hybrid_search_service = None        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ /query endpoint:                                           │ │
│  │   1. HYDE enhancement (existing)                           │ │
│  │   2. Encode query vector (existing)                        │ │
│  │   3. IF hybrid_search_service:                             │ │
│  │        result = await hybrid_search_service.retrieve(      │ │
│  │            query, query_vector, top_k, metadata_filter)    │ │
│  │      ELSE:                                                 │ │
│  │        result = lancedb_retriever_service.retrieve(...)    │ │
│  │   4. Reranking (existing, unchanged)                       │ │
│  │   5. LLM generation (existing, unchanged)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### New File: `rag2025/src/services/hybrid_search.py`

```python
"""
HybridSearchService — dense + BM25 sparse retrieval with RRF fusion.

Works natively with RetrievedDocument (no adapter needed).
Integrates directly with LanceDBRetriever.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument, RetrievalResult
from config.settings import RAGSettings

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid dense + BM25 retrieval with RRF fusion.

    Lifecycle:
      1. __init__(lancedb_retriever, settings)
      2. build_bm25_index() → bool   (call once at startup)
      3. await retrieve(query, query_vector, top_k) → RetrievalResult
    """

    def __init__(self, lancedb_retriever: LanceDBRetriever, settings: RAGSettings) -> None:
        self._retriever = lancedb_retriever
        self._settings = settings
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_docs: List[RetrievedDocument] = []  # parallel to BM25 index

    # ──────────────────────────────────────────────────────────────────────
    # Index Build
    # ──────────────────────────────────────────────────────────────────────

    def build_bm25_index(self) -> bool:
        """
        Scan LanceDB, build in-memory BM25 index.

        Uses column-selective to_pandas() to avoid loading 4096-dim vectors.
        Returns True on success, False on failure (caller should set service = None).
        """
        try:
            adapter = self._retriever._adapter
            table = adapter._table

            # Column-selective load — skip large vector columns (~160MB saved)
            cols = ["text", "sparse_terms", "source", "chunk_id"]
            available_cols = table.schema.names
            load_cols = [c for c in cols if c in available_cols]

            df = table.to_pandas(columns=load_cols)

            if df.empty:
                logger.warning("HybridSearchService: LanceDB table is empty, BM25 not built")
                return False

            tokenized_corpus: List[List[str]] = []
            self._corpus_docs = []

            for _, row in df.iterrows():
                # Prefer sparse_terms (pre-extracted), fallback to text tokenization
                sparse_terms = row.get("sparse_terms")
                if sparse_terms and isinstance(sparse_terms, list) and len(sparse_terms) > 0:
                    tokens = [str(t) for t in sparse_terms]
                else:
                    tokens = str(row.get("text", "")).lower().split()

                tokenized_corpus.append(tokens)

                # Build parallel RetrievedDocument list for result mapping
                self._corpus_docs.append(RetrievedDocument(
                    text=str(row.get("text", "")),
                    source=str(row.get("source", "")),
                    chunk_id=str(row.get("chunk_id", "")),
                    metadata={},
                    score=0.0,
                    point_id=None,
                ))

            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(
                f"HybridSearchService: BM25 index built — {len(self._corpus_docs)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"HybridSearchService: BM25 build failed: {e}", exc_info=True)
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: dense search + BM25 sparse → RRF fusion.

        Falls back to dense-only if BM25 index unavailable.
        """
        # Dense retrieval (existing LanceDB path).
        # NOTE: `retrieve()` is async but calls sync LanceDB — no run_in_threadpool needed
        # because LanceDB is embedded (in-process); there is no network or kernel I/O
        # that would block the event loop.  Wrapping in a threadpool would add overhead
        # for zero benefit.  If LanceDB is ever switched to a remote mode, revisit.
        dense_result: RetrievalResult = self._retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k * 2,  # over-fetch for fusion
            metadata_filter=metadata_filter,
        )
        dense_docs: List[RetrievedDocument] = dense_result.documents

        if self._bm25 is None:
            logger.warning("HybridSearchService: BM25 not ready, using dense-only")
            return dense_result

        # BM25 sparse retrieval
        sparse_docs = self._bm25_search(query, top_k=top_k * 2)

        # RRF fusion → RetrievedDocument list
        fused = self._rrf_fusion(dense_docs, sparse_docs, top_k=top_k)

        # Confidence = mean of top-k scores
        confidence = (
            sum(d.score for d in fused[:top_k]) / len(fused[:top_k])
            if fused else 0.0
        )

        return RetrievalResult(documents=fused[:top_k], confidence=confidence)

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Return top-k docs from BM25 index as RetrievedDocument list."""
        tokens = query.lower().split()
        if not tokens:
            logger.warning("_bm25_search called with empty token list — returning []")
            return []
        scores = self._bm25.get_scores(tokens)

        # Pair scores with docs, sort descending
        scored = sorted(
            zip(scores, self._corpus_docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in scored[:top_k]:
            if score <= 0:
                break
            results.append(RetrievedDocument(
                text=doc.text,
                source=doc.source,
                chunk_id=doc.chunk_id,
                metadata=doc.metadata,
                score=float(score),
                point_id=doc.point_id,
            ))
        return results

    def _rrf_fusion(
        self,
        dense_docs: List[RetrievedDocument],
        sparse_docs: List[RetrievedDocument],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """
        Reciprocal Rank Fusion (RRF) over dense + sparse result lists.

        RRF score = Σ weight_i / (k + rank_i)
        k=60 is the standard parameter from Cormack et al. 2009.

        Dense/sparse weights come from settings:
          HYBRID_FUSION_DENSE_WEIGHT  (default 0.6)
          HYBRID_FUSION_SPARSE_WEIGHT (default 0.4)
        """
        dense_weight = self._settings.HYBRID_FUSION_DENSE_WEIGHT
        sparse_weight = self._settings.HYBRID_FUSION_SPARSE_WEIGHT

        # Map chunk_id → cumulative RRF score + doc reference
        scores: Dict[str, float] = {}
        docs_map: Dict[str, RetrievedDocument] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank)
            if cid not in docs_map:
                docs_map[cid] = doc

        for rank, doc in enumerate(sparse_docs, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank)
            if cid not in docs_map:
                docs_map[cid] = doc

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused = []
        for cid in sorted_ids[:top_k]:
            doc = docs_map[cid]
            fused.append(RetrievedDocument(
                text=doc.text,
                source=doc.source,
                chunk_id=doc.chunk_id,
                metadata=doc.metadata,
                score=scores[cid],   # RRF score (not raw cosine/BM25)
                point_id=doc.point_id,
            ))

        return fused
```

### Configuration Changes (`config/settings.py`)

Add 4 new settings in the Retrieval section:

```python
# Hybrid Retrieval Configuration
USE_HYBRID_RETRIEVAL: bool = Field(
    default=False,
    description="Enable hybrid retrieval (dense + BM25 sparse with RRF fusion)"
)

HYBRID_FUSION_DENSE_WEIGHT: float = Field(
    default=0.6,
    ge=0.0,
    le=1.0,
    description="Weight for dense retrieval in RRF fusion (0.0-1.0)"
)

HYBRID_FUSION_SPARSE_WEIGHT: float = Field(
    default=0.4,
    ge=0.0,
    le=1.0,
    description="Weight for sparse (BM25) retrieval in RRF fusion (0.0-1.0)"
)

BM25_INDEX_PATH: Optional[str] = Field(
    default=None,
    description="Reserved for v2 BM25 persistence. None = in-memory only (v1)."
)
```

### Startup Integration (`main.py`)

```python
# Global service (add after lancedb_retriever_service declaration)
hybrid_search_service: HybridSearchService | None = None

@app.on_event("startup")
async def startup_event():
    global lancedb_retriever_service, hybrid_search_service

    # ... existing LanceDB init ...

    if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:
        logger.info("Initializing HybridSearchService (dense + BM25)...")
        try:
            from services.hybrid_search import HybridSearchService
            svc = HybridSearchService(lancedb_retriever_service, settings)
            success = svc.build_bm25_index()
            if success:
                hybrid_search_service = svc
                logger.info("HybridSearchService ready")
            else:
                logger.warning("BM25 build returned False — using dense-only fallback")
        except Exception as e:
            logger.error(f"HybridSearchService init failed: {e} — using dense-only fallback")
            hybrid_search_service = None
```

### Query Endpoint Routing (`main.py`)

```python
# In /query endpoint, replace single retrieval call with:
if hybrid_search_service:
    retrieval_result = await hybrid_search_service.retrieve(
        query=query_text,
        query_vector=query_vector,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
else:
    retrieval_result = lancedb_retriever_service.retrieve(
        query_vector=query_vector,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
```

---

## Data Flow

### Startup (Index Build)
```
App starts
├── LanceDB init (existing)
├── USE_HYBRID_RETRIEVAL = True?
│   ├── YES
│   │   ├── HybridSearchService.__init__(lancedb_retriever, settings)
│   │   ├── build_bm25_index()
│   │   │   ├── table.to_pandas(columns=["text","sparse_terms","source","chunk_id"])
│   │   │   ├── For each row: use sparse_terms OR fallback to text.split()
│   │   │   ├── BM25Okapi(tokenized_corpus)  ← builds in-memory index
│   │   │   └── Returns True (success) or False (empty table / exception)
│   │   ├── SUCCESS → hybrid_search_service = svc
│   │   └── FAILURE → hybrid_search_service = None (dense-only fallback)
│   └── NO → skip, dense-only
└── API accepts requests
```

### Query Time (Hybrid Path)
```
User query
├── HYDE enhancement (existing)
├── Encode query → query_vector (existing)
├── hybrid_search_service.retrieve(query, query_vector, top_k)
│   ├── lancedb_retriever.retrieve(query_vector, top_k*2) → dense_docs
│   ├── bm25.get_scores(query.lower().split()) → sparse_docs (top_k*2)
│   ├── _rrf_fusion(dense_docs, sparse_docs, top_k, k=60)
│   │   ├── For each dense doc (rank 1..N): score += 0.6 / (60 + rank)
│   │   ├── For each sparse doc (rank 1..N): score += 0.4 / (60 + rank)
│   │   └── Sort by RRF score → top_k docs as RetrievedDocument list
│   └── Returns RetrievalResult(documents, confidence)
├── Reranking (existing, unchanged — takes List[RetrievedDocument])
└── LLM generation (existing, unchanged)
```

---

## Graceful Degradation Model (3 Levels)

```
Level 1: USE_HYBRID_RETRIEVAL=True + BM25 build SUCCESS
    → Full hybrid pipeline (dense + BM25 + RRF + reranker)
    → Best recall (+20-40% expected)

Level 2: USE_HYBRID_RETRIEVAL=True + BM25 build FAILS
    → hybrid_search_service = None (set during startup exception)
    → /query detects None → auto-falls back to dense-only
    → WARNING logged, no service interruption

Level 3: USE_HYBRID_RETRIEVAL=False (default)
    → Dense-only path (current production behavior)
    → Zero risk, backward-compatible baseline
```

**Specific failure scenarios:**

| Failure | Effect | Recovery |
|---------|--------|----------|
| LanceDB table empty | BM25 not built (returns False) → dense fallback | Auto-recover when data ingested + restart |
| `rank_bm25` missing | ImportError → except block → dense fallback | `pip install rank_bm25` |
| `sparse_terms` absent | Per-row fallback to `text.split()` | No action needed |
| Column not in schema | `load_cols` filter skips it gracefully | No action needed |
| Partial BM25 exception | Error caught → dense fallback | Check logs |

---

## BM25 Index Lifecycle

### Build Strategy: Startup, In-Memory

BM25 index built **once at startup**. Rationale:
- BM25Okapi for ~1000 chunks: < 3 seconds build, ~20MB RAM
- Corpus is read-heavy; admission data updates < 2x per semester
- No serialization complexity in v1

### Corpus Sync

When new chunks are added:
- **Dense retrieval:** LanceDB updates automatically (existing behavior)
- **BM25 index:** becomes **stale** — does NOT auto-update in v1

**v1 approach:** Restart service to rebuild. Acceptable because dense covers new chunks immediately.

**v2 consideration:** `POST /admin/hybrid/rebuild` endpoint guarded by `x-admin-token`.

### Persistence

`BM25_INDEX_PATH` field reserved in settings but `None` (unused) in v1. If corpus grows beyond 50k chunks and rebuild exceeds 30s, serialize via `pickle` to path, stamp with row count for stale detection.

---

## Memory & Performance Profile

| Corpus Size | BM25 Build Time | BM25 RAM | Query Latency Added |
|------------|----------------|----------|---------------------|
| 1,000 chunks | < 3s | ~20 MB | +20-50ms |
| 10,000 chunks | < 15s | ~150 MB | +50-100ms |
| 50,000 chunks | < 60s | ~600 MB | +100-200ms |

**Column-selective `to_pandas()` benefit:** Skips loading the 4096-dim embedding vectors (~160 MB saved for 10k chunks). Only loads `text`, `sparse_terms`, `source`, `chunk_id`.

---

## Testing Strategy

### Unit Tests (`tests/test_hybrid_search.py`)

1. `test_build_bm25_index_success` — mock LanceDB table with 5 rows → assert BM25 built
2. `test_build_bm25_index_empty_table` → assert returns False
3. `test_build_bm25_index_exception` → assert returns False (no crash)
4. `test_bm25_search_returns_relevant` — query "học phí" → ranked result
5. `test_rrf_fusion_deduplication` — same chunk in dense + sparse → appears once
6. `test_rrf_fusion_weights` — dense_weight=1.0, sparse_weight=0.0 → dense ranking preserved
7. `test_retrieve_fallback_to_dense_when_bm25_none` — bm25=None → dense result returned
8. `test_retrieve_hybrid_path` — BM25 built → fused result has correct type

### Integration Tests

- `USE_HYBRID_RETRIEVAL=False` → dense-only path (baseline)
- `USE_HYBRID_RETRIEVAL=True` → hybrid path
- Compare top-5 results for 5 keyword-heavy test queries (listed below)

### A/B Test Queries

```
1. "học phí ngành Công nghệ thông tin"    (keyword-heavy)
2. "điều kiện xét tuyển thẳng"            (lexical match)
3. "thông tư 08/2022"                      (citation query)
4. "ngưỡng đầu vào là gì"                 (definition query)
5. "các ngành tuyển sinh năm 2025"         (overview query)
```

---

## Success Metrics

| Metric | Baseline (Dense-only) | Target (Hybrid) |
|--------|----------------------|--------------------|
| Recall@5 (keyword queries) | ~65% | ≥80% (+15%) |
| Query Latency p95 | 250ms | ≤350ms (+100ms OK) |
| Startup Time | 8s | ≤16s (+8s OK) |
| Answer Quality (human eval) | 3.5/5 | ≥4.0/5 |

---

## Files Changed

| File | Change | LOC Delta |
|------|--------|-----------|
| `rag2025/src/services/hybrid_search.py` | **NEW** | +130 |
| `rag2025/config/settings.py` | Add 4 settings | +20 |
| `rag2025/src/main.py` | Import + init + routing | +25 |
| `rag2025/tests/test_hybrid_search.py` | **NEW** | +120 |

**Total: ~295 LOC added, 0 existing LOC modified**

---

## Decisions

| # | Decision | Alternatives considered | Reason chosen |
|---|----------|-------------------------|---------------|
| 1 | Create new `HybridSearchService` class instead of adapting `HybridRetriever` | Adapter wrapping `VectorStore` ABC + `SearchResult`→`RetrievedDocument` converter | `VectorStore` is ABC (not instantiable). Adapter would be ~150 LOC + field mapping fragility. New class is 130 LOC with zero type mismatch risk. |
| 2 | Work natively with `RetrievedDocument` (not `SearchResult`) | Use `SearchResult` + convert at call site | `RetrievedDocument` is what `main.py` and the reranker expect. No conversion layer needed. |
| 3 | Column-selective `to_pandas(columns=[...])` | `table.to_pandas()` (full load) | Avoids loading 4096-dim embedding vectors. ~160MB saved for 10k chunks. No embedding data needed for BM25 indexing. |
| 4 | RRF fusion with k=60 (Cormack et al. 2009 standard) | Linear combination of scores, Borda count | RRF is rank-based (not score-based) — immune to scale differences between cosine similarity and BM25 scores. k=60 is empirically optimal. |
| 5 | Prefer `sparse_terms` (pre-extracted), fallback to `text.split()` | Always use `text`, always use `sparse_terms` | `sparse_terms` are linguistically cleaned and domain-specific. Per-row fallback handles missing column gracefully. |
| 6 | Over-fetch top_k*2 for dense and sparse before fusion | Fetch exactly top_k | Fusion may promote different docs. Over-fetching ensures best candidates are available for RRF without inflating final output. |
| 7 | Default `USE_HYBRID_RETRIEVAL=False` (opt-in) | Default to True | Preserve backward compatibility. Safe rollout: dev → staging → production. |
| 8 | In-memory BM25 only (v1), `BM25_INDEX_PATH=None` | Persist to disk immediately | Build is fast (< 3s for 1k chunks). Avoids serialization complexity. Path setting reserved for v2 if needed. |

---

## Dependencies

### Existing (already in codebase)
- `rank_bm25` — `BM25Okapi` class (`requirements.txt`)
- `LanceDBRetriever`, `RetrievedDocument`, `RetrievalResult` — `services/lancedb_retrieval.py`
- `RAGSettings` — `config/settings.py`
- `sparse_terms` field in LanceDB table (extracted during chunking)

### New
- `rag2025/src/services/hybrid_search.py` — the new class (this spec)
- 4 new settings in `config/settings.py`

---

## References

- Active retrieval path: `rag2025/src/main.py` lines 486-531
- LanceDB types: `rag2025/src/services/lancedb_retrieval.py`
- `VectorStore` ABC: `rag2025/src/services/vector_store.py`
- `HybridRetriever` (not used): `rag2025/src/services/retriever.py`
- RRF: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009)

---

**End of Spec v3**
