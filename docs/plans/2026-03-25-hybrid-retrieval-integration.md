# Implementation Plan: Hybrid Retrieval Integration

**Date:** 2026-03-25
**Task:** task_01 — hybrid-retrieval-integration
**Branch:** workflow/hybrid-retrieval-integration
**Spec:** `docs/specs/2026-03-25-hybrid-retrieval-integration-design.md` (v3, APPROVED)

---

## Overview

This plan implements `HybridSearchService` — a new class that combines dense (LanceDB vector) and sparse (BM25) retrieval using Reciprocal Rank Fusion (RRF). It integrates directly into the existing `main.py` pipeline without modifying any existing service classes.

**Expected outcome:** 20-40% recall improvement on keyword-heavy queries (e.g., "học phí ngành Công nghệ thông tin") while preserving full backward compatibility via an opt-in config flag (`USE_HYBRID_RETRIEVAL=False` by default).

---

## Files to Create / Modify

| File | Action | Estimated LOC Delta |
|------|--------|---------------------|
| `rag2025/src/services/hybrid_search.py` | CREATE | +130 |
| `rag2025/config/settings.py` | MODIFY — add 4 fields to `RAGSettings` | +20 |
| `rag2025/src/main.py` | MODIFY — global var + startup init + query routing | +25 |
| `rag2025/tests/test_hybrid_search.py` | CREATE | +120 |

**Total: ~295 LOC added, 0 existing LOC modified.**

---

## Implementation Order

### Step 1 — Add settings fields to `RAGSettings`

**File:** `rag2025/config/settings.py`

Locate the `# ========== Retrieval Parameters ==========` section (currently ending at `TOP_K_SPARSE` on line 77). Insert a new `# ========== Hybrid Retrieval Configuration ==========` block immediately after:

```python
# ========== Hybrid Retrieval Configuration ==========
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

No validator needed — `ge`/`le` constraints on `Field` are handled by Pydantic automatically.

**Verify:** `from config.settings import RAGSettings; s = RAGSettings(); assert s.USE_HYBRID_RETRIEVAL is False`

---

### Step 2 — Create `rag2025/src/services/hybrid_search.py`

Create the file at the exact path `rag2025/src/services/hybrid_search.py`. Full implementation below.

#### Class: `HybridSearchService`

**Imports required:**
```python
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument, RetrievalResult
from config.settings import RAGSettings
```

Note: `rank_bm25` is already listed in `rag2025/requirements.txt`. Do not add it again.

#### Module-level logger

`logger = logging.getLogger(__name__)` is declared at **module level** (top of `hybrid_search.py`, immediately after the imports), not as an instance attribute. All log calls throughout the class (`logger.warning(...)`, `logger.info(...)`, `logger.error(...)`) reference this module-level variable. Do NOT add `self._logger` to `__init__`.

#### `__init__` method

```python
def __init__(self, lancedb_retriever: LanceDBRetriever, settings: RAGSettings) -> None:
    self._retriever = lancedb_retriever
    self._settings = settings
    self._bm25: Optional[BM25Okapi] = None
    self._corpus_docs: List[RetrievedDocument] = []  # parallel index to BM25
```

#### `build_bm25_index` method

Signature: `def build_bm25_index(self) -> bool`

Responsibilities:
1. Access `self._retriever._adapter._table` to get the LanceDB table handle
2. Load only columns `["text", "sparse_terms", "source", "chunk_id"]` (column-selective `to_pandas` — skips 4096-dim embedding vectors, saves ~160MB for 10k chunks)
3. Filter to columns that actually exist in the table schema via `table.schema.names`
4. Guard: if `df.empty`, log a warning and return `False`
5. For each row:
   - If `sparse_terms` column is present, is a non-empty list: use it as tokens
   - Otherwise: tokenize `text` field with `str(row.get("text", "")).lower().split()`
6. Build `BM25Okapi(tokenized_corpus)` and store in `self._bm25`
7. Build a parallel `self._corpus_docs` list of `RetrievedDocument` objects (one per row) for result mapping
8. On any exception: log error with `exc_info=True`, return `False`
9. On success: log info with chunk count, return `True`

**Critical guard:** The `tokenized_corpus` list must never contain an empty token list at the document level (BM25Okapi handles internal document-level empty lists, but for clarity, rows with zero-length text should still produce `[""]` rather than `[]` to avoid IDF calculation anomalies). The per-query empty-token guard is in `_bm25_search`.

#### `retrieve` method (async)

Signature:
```python
async def retrieve(
    self,
    query: str,
    query_vector: List[float],
    top_k: int = 10,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> RetrievalResult:
```

Responsibilities:
1. Call `self._retriever.retrieve(query_vector=query_vector, top_k=top_k * 2, metadata_filter=metadata_filter)` — note this is a **synchronous** call; no `run_in_threadpool` wrapper needed because LanceDB is embedded (in-process, no network I/O that would block the event loop)
2. Store result as `dense_result`; extract `dense_result.documents` as `dense_docs`
3. If `self._bm25 is None`: log warning, return `dense_result` directly (graceful degradation Level 2)
4. Call `self._bm25_search(query, top_k=top_k * 2)` to get `sparse_docs`
5. Call `self._rrf_fusion(dense_docs, sparse_docs, top_k=top_k)` to get `fused`
6. Compute `confidence = sum(d.score for d in fused) / len(fused) if fused else 0.0`
7. Return `RetrievalResult(documents=fused, confidence=confidence)`

#### `_bm25_search` method (private)

Signature: `def _bm25_search(self, query: str, top_k: int) -> List[RetrievedDocument]`

Responsibilities:
1. `tokens = query.lower().split()`
2. **Empty-token guard:** if `not tokens`: log warning "\_bm25\_search called with empty token list — returning []", return `[]`
3. `scores = self._bm25.get_scores(tokens)` — returns a numpy array parallel to `self._corpus_docs`
4. Pair `zip(scores, self._corpus_docs)`, sort descending by score
5. For each `(score, doc)` in top-k: if `score <= 0`, stop (all remaining are zero)
6. Build and return a new `RetrievedDocument` per hit (copy fields, set `score=float(score)`)

#### `_rrf_fusion` method (private)

Signature:
```python
def _rrf_fusion(
    self,
    dense_docs: List[RetrievedDocument],
    sparse_docs: List[RetrievedDocument],
    top_k: int,
    k: int = 60,
) -> List[RetrievedDocument]:
```

Algorithm (Cormack et al. SIGIR 2009):
- `RRF_score(doc) = Σ weight_i / (k + rank_i)`
- `dense_weight = self._settings.HYBRID_FUSION_DENSE_WEIGHT` (default 0.6)
- `sparse_weight = self._settings.HYBRID_FUSION_SPARSE_WEIGHT` (default 0.4)

Responsibilities:
1. Maintain `scores: Dict[str, float]` keyed by `chunk_id`
2. Maintain `docs_map: Dict[str, RetrievedDocument]` (first-seen doc wins for metadata)
3. Iterate `dense_docs` with `rank` starting at 1: `scores[cid] += dense_weight / (k + rank)`
4. Iterate `sparse_docs` with `rank` starting at 1: `scores[cid] += sparse_weight / (k + rank)`
5. Sort `chunk_id` keys by descending RRF score
6. Build output list: for each top-k `chunk_id`, create a new `RetrievedDocument` with `score=scores[cid]` (the RRF score replaces the original cosine/BM25 score)
7. Return the fused list

**Deduplication guarantee:** `chunk_id`-keyed dicts ensure a document appearing in both dense and sparse lists is counted once with both contributions added.

---

### Step 3 — Integrate into `main.py`

**File:** `rag2025/src/main.py`

#### 3a. Add global variable (after line 217, after existing globals)

In the `# ========== Global Services ==========` block, add:

```python
hybrid_search_service: "HybridSearchService | None" = None
```

Use a string annotation to avoid a top-level import at module load time (the import is deferred to startup).

#### 3b. Add to `startup_event` (after GraphRAG pipeline init block)

In `startup_event()`, replace the existing `global` declaration line (line 233) with the following exact line:

```python
global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_encoder, unified_pipeline, reranker_service, query_cache, guardrail_service, hybrid_search_service
```

Then, after the GraphRAG block (around line 296), add:

```python
# Hybrid retrieval (dense + BM25 + RRF)
if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:
    logger.info("Initializing HybridSearchService (dense + BM25 RRF)...")
    try:
        from services.hybrid_search import HybridSearchService
        _svc = HybridSearchService(lancedb_retriever_service, settings)
        _success = _svc.build_bm25_index()
        if _success:
            hybrid_search_service = _svc
            logger.info("HybridSearchService ready")
        else:
            logger.warning(
                "HybridSearchService: BM25 build returned False — dense-only fallback active"
            )
    except Exception as e:
        logger.error(
            f"HybridSearchService init failed: {e} — dense-only fallback active"
        )
        hybrid_search_service = None
```

Placement rationale: after LanceDB init (dependency), after GraphRAG init (parallel concern), before "API Ready" log line.

#### 3c. Modify the `/query` retrieval loop

Locate the `for i, variant in enumerate(variants):` loop in the `/query` endpoint (around lines 493-531). Currently each iteration calls `lancedb_retriever_service.retrieve(...)` directly.

**Change:** wrap the per-variant retrieval call with a hybrid branch. Replace:

```python
retrieval_result = lancedb_retriever_service.retrieve(
    query_vector=query_vector,
    top_k=retrieval_top_k,
    metadata_filter=metadata_filter,
)
```

with:

```python
if hybrid_search_service:
    retrieval_result = await hybrid_search_service.retrieve(
        query=variant,
        query_vector=query_vector,
        top_k=retrieval_top_k,
        metadata_filter=metadata_filter,
    )
else:
    retrieval_result = lancedb_retriever_service.retrieve(
        query_vector=query_vector,
        top_k=retrieval_top_k,
        metadata_filter=metadata_filter,
    )
```

Note: `hybrid_search_service.retrieve()` is `async`, so `await` is required.

The retrieval loop also contains a **no-filter fallback call** (around line 519) that fires when the filtered call returns 0 docs. This fallback must also be wrapped with the same hybrid branch. Replace:

```python
retrieval_result = lancedb_retriever_service.retrieve(
    query_vector=query_vector,
    top_k=retrieval_top_k,
    metadata_filter=None,
)
```

with:

```python
if hybrid_search_service:
    retrieval_result = await hybrid_search_service.retrieve(
        query=variant,
        query_vector=query_vector,
        top_k=retrieval_top_k,
        metadata_filter=None,
    )
else:
    retrieval_result = lancedb_retriever_service.retrieve(
        query_vector=query_vector,
        top_k=retrieval_top_k,
        metadata_filter=None,
    )
```

Both calls — the primary filtered call and the no-filter fallback — must use the hybrid path when `hybrid_search_service` is active. Leaving the fallback as dense-only would create an inconsistency where a filtered hybrid query that returns 0 results silently falls back to dense-only, bypassing the configured hybrid path.

---

### Step 4 — Create `rag2025/tests/test_hybrid_search.py`

Create unit tests covering 8 scenarios:

1. **`test_build_bm25_index_success`**
   - Mock `LanceDBRetriever` with a table returning a 5-row DataFrame (columns: `text`, `sparse_terms`, `source`, `chunk_id`)
   - Assert `build_bm25_index()` returns `True`
   - Assert `len(service._corpus_docs) == 5`
   - Assert `service._bm25 is not None`

2. **`test_build_bm25_index_empty_table`**
   - Mock table returning an empty DataFrame
   - Assert `build_bm25_index()` returns `False`
   - Assert `service._bm25 is None`

3. **`test_build_bm25_index_exception`**
   - Mock table `.to_pandas()` raising a `RuntimeError`
   - Assert `build_bm25_index()` returns `False` (no exception propagated)

4. **`test_bm25_search_returns_relevant`**
   - Build index with 3 docs, one containing "học phí" tokens
   - Call `_bm25_search("học phí", top_k=3)`
   - Assert first result matches expected doc

5. **`test_bm25_search_empty_query_guard`**
   - Build valid index
   - Call `_bm25_search("", top_k=5)` (empty string → empty tokens after split)
   - Assert result is `[]` (guard triggered, no BM25 exception)

6. **`test_rrf_fusion_deduplication`**
   - Create dense_docs and sparse_docs sharing one `chunk_id`
   - Call `_rrf_fusion(dense_docs, sparse_docs, top_k=10)`
   - Assert the shared doc appears exactly once in output

7. **`test_rrf_fusion_weights`**
   - Set `HYBRID_FUSION_DENSE_WEIGHT=1.0`, `HYBRID_FUSION_SPARSE_WEIGHT=0.0`
   - Call `_rrf_fusion(dense_docs, sparse_docs, top_k=5)`
   - Assert output ranking matches dense-only order

8. **`test_retrieve_fallback_to_dense_when_bm25_none`**
   - Leave `service._bm25 = None` (do not call `build_bm25_index`)
   - Call `await service.retrieve(query="test", query_vector=[0.1]*4096, top_k=5)`
   - Assert result is the `RetrievalResult` returned by `lancedb_retriever_service.retrieve`

9. **`test_retrieve_hybrid_path`**
   - Build BM25 index (mock table)
   - Mock `lancedb_retriever_service.retrieve` to return a `RetrievalResult`
   - Call `await service.retrieve(...)`
   - Assert result is a `RetrievalResult` with `documents` being a list of `RetrievedDocument`
   - Assert no document chunk_id appears twice

---

## New File: Full Implementation Spec

### `rag2025/src/services/hybrid_search.py`

```python
"""
HybridSearchService — dense + BM25 sparse retrieval with RRF fusion.

Works natively with RetrievedDocument (no adapter needed).
Integrates directly with LanceDBRetriever.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument, RetrievalResult
from config.settings import RAGSettings

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid dense + BM25 retrieval with RRF fusion.

    Lifecycle:
      1. __init__(lancedb_retriever, settings)
      2. build_bm25_index() -> bool   (call once at startup)
      3. await retrieve(query, query_vector, top_k) -> RetrievalResult
    """

    def __init__(self, lancedb_retriever: LanceDBRetriever, settings: RAGSettings) -> None:
        self._retriever = lancedb_retriever
        self._settings = settings
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_docs: List[RetrievedDocument] = []

    # ------------------------------------------------------------------
    # Index Build
    # ------------------------------------------------------------------

    def build_bm25_index(self) -> bool:
        """
        Scan LanceDB, build in-memory BM25 index.

        Uses column-selective to_pandas() to avoid loading 4096-dim vectors.
        Returns True on success, False on failure (caller should set service = None).
        """
        try:
            adapter = self._retriever._adapter
            table = adapter._table

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
                sparse_terms = row.get("sparse_terms")
                if sparse_terms and isinstance(sparse_terms, list) and len(sparse_terms) > 0:
                    tokens = [str(t) for t in sparse_terms]
                else:
                    raw = str(row.get("text", "")).lower().split()
                    tokens = raw if raw else [""]

                tokenized_corpus.append(tokens)
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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: dense search + BM25 sparse -> RRF fusion.

        Falls back to dense-only if BM25 index unavailable.
        NOTE: retrieve() is async but calls sync LanceDB — no run_in_threadpool needed
        because LanceDB is embedded (in-process); there is no network or kernel I/O
        that would block the event loop. If LanceDB is ever switched to remote mode, revisit.
        """
        dense_result: RetrievalResult = self._retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k * 2,
            metadata_filter=metadata_filter,
        )
        dense_docs: List[RetrievedDocument] = dense_result.documents

        if self._bm25 is None:
            logger.warning("HybridSearchService: BM25 not ready, using dense-only")
            return dense_result

        sparse_docs = self._bm25_search(query, top_k=top_k * 2)
        fused = self._rrf_fusion(dense_docs, sparse_docs, top_k=top_k)

        confidence = (
            sum(d.score for d in fused) / len(fused)
            if fused else 0.0
        )

        return RetrievalResult(documents=fused, confidence=confidence)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Return top-k docs from BM25 index as RetrievedDocument list."""
        tokens = query.lower().split()
        if not tokens:
            logger.warning("_bm25_search called with empty token list — returning []")
            return []

        scores = self._bm25.get_scores(tokens)
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

        RRF score = sum(weight_i / (k + rank_i))
        k=60 is the standard parameter from Cormack et al. SIGIR 2009.
        """
        dense_weight = self._settings.HYBRID_FUSION_DENSE_WEIGHT
        sparse_weight = self._settings.HYBRID_FUSION_SPARSE_WEIGHT

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

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused = []
        for cid in sorted_ids[:top_k]:
            doc = docs_map[cid]
            fused.append(RetrievedDocument(
                text=doc.text,
                source=doc.source,
                chunk_id=doc.chunk_id,
                metadata=doc.metadata,
                score=scores[cid],
                point_id=doc.point_id,
            ))

        return fused
```

---

## Risk Assessment

| Risk | Probability | Severity | Mitigation |
|------|-------------|----------|------------|
| `_adapter._table` attribute path changes | Low | High | Integration test checks attribute access at startup; BM25 build failure → graceful dense fallback |
| `sparse_terms` column absent from LanceDB schema | Medium | Low | Per-row fallback to `text.split()` handles missing column silently |
| `rank_bm25` not installed | Low | Medium | `requirements.txt` already lists it; ImportError caught by startup try/except |
| LanceDB table empty (no data ingested) | Low | Low | `build_bm25_index` returns `False`, dense-only fallback, WARNING logged |
| Multi-variant loop: `hybrid_search_service.retrieve` called per-variant | Medium | Medium | Each call is in-process BM25 scan (cheap); dense call also per-variant already |
| `chunk_id` is `None` for some docs | Low | Medium | RRF map uses `str(chunk_id)` — None becomes "None" string; no crash but dedup may merge unrelated docs. Mitigated by ensuring chunker always sets chunk_id |
| Memory: BM25 index RAM growth | Low | Low | ~20MB for 1k chunks, ~150MB for 10k chunks — within typical server limits |

---

## Decisions

| # | Decision | Alternatives | Reason |
|---|----------|-------------|--------|
| 1 | Create new `HybridSearchService` instead of adapting `HybridRetriever` | Adapter wrapping `VectorStore` ABC | `VectorStore` is abstract (not instantiable); adapter would be ~150 LOC with type-mapping fragility. New class is 130 LOC, zero mismatch risk. |
| 2 | Work natively with `RetrievedDocument` | Use `SearchResult` + convert at call site | `RetrievedDocument` is what `main.py` and the reranker already consume. No conversion layer. |
| 3 | Column-selective `to_pandas(columns=[...])` | `table.to_pandas()` full load | Avoids loading 4096-dim embedding vectors; ~160MB saved for 10k chunks. |
| 4 | RRF with k=60 (Cormack et al. 2009) | Linear score combination, Borda count | RRF is rank-based — immune to scale differences between cosine similarity and BM25 scores. |
| 5 | Prefer `sparse_terms` per row, fallback to `text.split()` | Always use text only | `sparse_terms` are pre-extracted, linguistically cleaned. Per-row fallback handles missing column gracefully. |
| 6 | Over-fetch `top_k * 2` for both dense and sparse | Fetch exactly `top_k` | Fusion may promote different docs; over-fetching ensures best candidates available without inflating final output. |
| 7 | `USE_HYBRID_RETRIEVAL=False` by default | Default True | Backward-compatible opt-in. Safe rollout: dev → staging → production. |
| 8 | In-memory BM25 only (v1), `BM25_INDEX_PATH=None` | Persist to disk | Build is fast (<3s for 1k chunks). Avoids serialization complexity. Path setting reserved for v2. |
| 9 | Deferred import in `startup_event` | Top-level import in main.py | Avoids module-load failure if `rank_bm25` is missing when feature is disabled. |

---

## Test Strategy

### Unit Tests (`rag2025/tests/test_hybrid_search.py`)

Use `unittest.mock.MagicMock` to mock `LanceDBRetriever` and `LanceDBAdapter`. Use `pytest-asyncio` for `async` test cases.

Test scenarios (see Step 4 above for full descriptions):

1. `test_build_bm25_index_success`
2. `test_build_bm25_index_empty_table`
3. `test_build_bm25_index_exception`
4. `test_bm25_search_returns_relevant`
5. `test_bm25_search_empty_query_guard`
6. `test_rrf_fusion_deduplication`
7. `test_rrf_fusion_weights`
8. `test_retrieve_fallback_to_dense_when_bm25_none`
9. `test_retrieve_hybrid_path`

### Integration Test Queries (A/B)

Run with `USE_HYBRID_RETRIEVAL=False` (baseline) and `USE_HYBRID_RETRIEVAL=True` (hybrid). Compare top-5 results:

1. "học phí ngành Công nghệ thông tin" (keyword-heavy)
2. "điều kiện xét tuyển thẳng" (lexical match)
3. "thông tư 08/2022" (citation query)
4. "ngưỡng đầu vào là gì" (definition query)
5. "các ngành tuyển sinh năm 2025" (overview query)

### Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Recall@5 (keyword queries) | ~65% | ≥80% |
| Query latency p95 | 250ms | ≤350ms |
| Startup time | 8s | ≤16s |

---

## Graceful Degradation Model

```
Level 1: USE_HYBRID_RETRIEVAL=True + BM25 build SUCCESS
    -> Full hybrid (dense + BM25 + RRF + reranker)

Level 2: USE_HYBRID_RETRIEVAL=True + BM25 build FAILS
    -> hybrid_search_service = None
    -> /query uses dense-only path automatically
    -> WARNING logged, no service interruption

Level 3: USE_HYBRID_RETRIEVAL=False (default)
    -> Dense-only path (current production behavior)
    -> Zero risk, full backward compatibility
```

---

**End of Plan**
