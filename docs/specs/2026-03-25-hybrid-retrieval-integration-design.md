# Hybrid Retrieval Integration Design Spec

**Date:** 2026-03-25
**Author:** Claude (Workflow Agent)
**Status:** Draft
**Epic:** Hybrid Retrieval Enhancement

---

## Executive Summary

Enable BM25 hybrid retrieval (dense + sparse) in the RAG pipeline to improve recall by 20-40% based on research findings. The codebase already has a complete `HybridRetriever` class with BM25 support, but it's not wired into the active retrieval path in `main.py`. This spec outlines minimal changes to integrate hybrid retrieval without breaking existing functionality.

---

## Problem Statement

### Current State
- `main.py` uses **dense-only retrieval** via `LanceDBRetriever` (vector similarity search)
- `HybridRetriever` class exists in `rag2025/src/services/retriever.py` with full BM25 + RRF + reranking pipeline
- `sparse_terms` are already extracted during chunking and stored in JSONL files
- Research shows hybrid search (dense + BM25) improves recall by 20-40% over dense-only

### Gap
- `HybridRetriever` is **not instantiated or used** in `main.py`
- No BM25 index is built from corpus at startup
- Current retrieval flow bypasses hybrid pipeline entirely

### Impact
- Lower recall on keyword-heavy queries (e.g., "học phí ngành Công nghệ thông tin")
- Missed opportunities for lexical matching that dense embeddings may not capture
- Suboptimal answer quality for admission chatbot compared to competitors

---

## Goals

1. **Wire `HybridRetriever` into `main.py`** as an optional retrieval path
2. **Build BM25 index at startup** from existing `sparse_terms` in LanceDB
3. **Preserve backward compatibility** — keep existing dense-only path as fallback
4. **Minimal code changes** — reuse existing `HybridRetriever` class without modification
5. **Enable A/B testing** — allow toggling between dense-only and hybrid via config flag

---

## Non-Goals

- Modifying `HybridRetriever` class logic (already well-designed)
- Changing chunking pipeline (sparse_terms already extracted)
- Re-ingesting data (sparse_terms already in LanceDB)
- GraphRAG integration (separate concern, handled by `UnifiedRAGPipeline`)

---

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Startup: Load corpus from LanceDB                     │  │
│  │   → Extract (text, sparse_terms, chunk_id)           │  │
│  │   → Build BM25 index via HybridRetriever              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Query Flow:                                           │  │
│  │   1. HYDE enhancement (existing)                      │  │
│  │   2. Encode query variants (existing)                 │  │
│  │   3. IF USE_HYBRID_RETRIEVAL:                         │  │
│  │        → HybridRetriever.retrieve()                   │  │
│  │          - Dense search (vector)                      │  │
│  │          - Sparse search (BM25)                       │  │
│  │          - RRF fusion                                 │  │
│  │          - Cross-encoder reranking                    │  │
│  │      ELSE:                                            │  │
│  │        → LanceDBRetriever.retrieve() (existing)       │  │
│  │   4. LLM generation (existing)                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Configuration (`config/settings.py`)
Add new settings:
```python
# Hybrid Retrieval Configuration
USE_HYBRID_RETRIEVAL: bool = Field(
    default=False,
    description="Enable hybrid retrieval (dense + BM25 sparse)"
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
    description="Path to persist BM25 index (None = in-memory only)"
)
```

#### 2. Startup Initialization (`main.py`)
```python
# Global services
hybrid_retriever_service: HybridRetriever | None = None

@app.on_event("startup")
async def startup_event():
    global hybrid_retriever_service

    if settings.USE_HYBRID_RETRIEVAL:
        logger.info("Initializing Hybrid Retriever (dense + BM25)...")

        try:
            # Load corpus from LanceDB
            corpus_texts, corpus_ids = await load_corpus_for_bm25()

            # Initialize HybridRetriever
            from services.retriever import HybridRetriever
            from services.embedding import EmbeddingService
            from services.vector_store import VectorStore

            embedding_service = EmbeddingService(settings)
            vector_store = VectorStore(settings)  # Wraps LanceDB

            hybrid_retriever_service = HybridRetriever(
                embedding_service=embedding_service,
                vector_store=vector_store,
                settings=settings
            )

            # Build BM25 index (in-memory)
            hybrid_retriever_service.build_bm25_index(corpus_texts, corpus_ids)
            logger.info(f"Hybrid Retriever ready: {len(corpus_texts)} chunks indexed")

        except Exception as e:
            logger.error(f"Failed to initialize Hybrid Retriever: {e}")
            logger.warning("Falling back to dense-only retrieval")
            hybrid_retriever_service = None
            # Continue startup - dense-only path will be used
```

#### 3. Corpus Loading Helper
```python
async def load_corpus_for_bm25() -> tuple[list[str], list[str]]:
    """Load all chunks from LanceDB for BM25 indexing."""
    if not lancedb_retriever_service:
        raise RuntimeError("LanceDB not initialized")

    # Query all chunks (no vector search, just scan)
    adapter = lancedb_retriever_service._adapter
    table = adapter._table

    rows = table.to_pandas()  # Load all rows

    corpus_texts = []
    corpus_ids = []

    for _, row in rows.iterrows():
        chunk_id = row.get("chunk_id") or row.get("id")

        # Prefer sparse_terms if available, else use text
        sparse_terms = row.get("sparse_terms")
        if sparse_terms and isinstance(sparse_terms, list):
            text = " ".join(sparse_terms)
        else:
            text = row.get("text", "")

        corpus_texts.append(text)
        corpus_ids.append(str(chunk_id))

    logger.info(f"Loaded {len(corpus_texts)} chunks for BM25 indexing")
    return corpus_texts, corpus_ids
```

#### 4. Query Endpoint Modification (`/query`)
```python
@app.post("/query", response_model=QueryResponse)
async def query(request: SimpleQueryRequest, raw_request: Request):
    # ... existing HYDE enhancement ...

    # Step 2: Retrieval
    if settings.USE_HYBRID_RETRIEVAL and hybrid_retriever_service:
        # Hybrid retrieval path
        for variant in variants:
            results, confidence = hybrid_retriever_service.retrieve(
                query=variant,
                top_k=top_k
            )
            # Merge results (existing deduplication logic)
            all_chunks.extend([r.to_dict() for r in results])
            all_scores.extend([r.score for r in results])
    else:
        # Dense-only path (existing)
        for variant in variants:
            query_vector = embedding_encoder.encode(variant).tolist()
            retrieval_result = lancedb_retriever_service.retrieve(
                query_vector=query_vector,
                top_k=top_k
            )
            # ... existing logic ...

    # ... rest of pipeline unchanged ...
```

### Data Flow

1. **Startup:**
   - Load all chunks from LanceDB table
   - Extract `sparse_terms` (or fallback to `text`)
   - Build BM25 index in memory via `rank_bm25.BM25Okapi`
   - **If build fails:** log error, set `hybrid_retriever_service = None`, continue with dense-only

2. **Query Time:**
   - User query → HYDE variants (existing)
   - For each variant:
     - **Dense search:** Encode variant → LanceDB vector search → top-K results
     - **Sparse search:** Tokenize variant → BM25 scoring → top-K results
     - **RRF fusion:** Merge dense + sparse with configurable weights (`HYBRID_FUSION_DENSE_WEIGHT` / `HYBRID_FUSION_SPARSE_WEIGHT`)
   - **Reranking:** Cross-encoder on top-50 fused results
   - **Deduplication:** Merge across variants (existing logic)
   - LLM generation (existing)

---

## BM25 Index Lifecycle

### Build Strategy: Startup (In-Memory)
The BM25 index is built **once at startup** from the full LanceDB corpus:

```
App starts
    ├── LanceDB table loaded (existing)
    ├── USE_HYBRID_RETRIEVAL = True?
    │       ├── YES → scan all rows → build BM25Okapi in RAM → ready
    │       │          (on any error → warn → hybrid_retriever_service = None)
    │       └── NO  → skip (dense-only)
    └── API accepts requests
```

**Rationale:** BM25 index build for ~1000 chunks takes <3 seconds. In-memory BM25 lookup is O(n) per query (~1ms at 1k chunks). The corpus is read-heavy and rarely mutated, so startup build is optimal.

### Persistence Strategy: In-Memory Only (v1)
The BM25 index is **not persisted to disk** in v1:
- `BM25_INDEX_PATH = None` by default
- On every restart, the index rebuilds from LanceDB in <5s
- This avoids serialization complexity and stale-index risks

**v2 consideration:** If corpus grows beyond 50k chunks and startup rebuild exceeds 30s, persist via `pickle` to `BM25_INDEX_PATH`. Stamp the file with LanceDB's row count; invalidate on mismatch.

### Corpus Sync Strategy
When new chunks are added (via `/v2/graph/update` or ingestion scripts):
- **Dense retrieval:** LanceDB updates automatically (existing behavior)
- **BM25 index:** becomes **stale** — it does NOT update automatically in v1

**Sync approach for v1:** Manual rebuild triggered by calling the rebuild endpoint or restarting the service. Acceptable because:
- Admission data is updated infrequently (1-2 times per semester)
- Dense retrieval still covers new chunks immediately
- BM25 only affects keyword-heavy queries

**v2 consideration:** Add a `POST /admin/hybrid/rebuild` endpoint guarded by `x-admin-token` to trigger BM25 rebuild on demand without restart.

---

## Implementation Plan

### Phase 1: Configuration & Adapter (1 hour)
- [ ] Add `USE_HYBRID_RETRIEVAL`, `HYBRID_FUSION_DENSE_WEIGHT`, `HYBRID_FUSION_SPARSE_WEIGHT`, `BM25_INDEX_PATH` to `config/settings.py`
- [ ] Create `VectorStore` wrapper class for `HybridRetriever` compatibility
- [ ] Create `EmbeddingService` wrapper for `HybridRetriever` compatibility
- [ ] Update `HybridRetriever._reciprocal_rank_fusion()` call to pass weights from `settings`

### Phase 2: Startup Integration (2 hours)
- [ ] Implement `load_corpus_for_bm25()` helper
- [ ] Add `HybridRetriever` initialization in `startup_event()` with try-catch
- [ ] Add BM25 index build step
- [ ] Add graceful fallback: if BM25 build fails, log warning and set `hybrid_retriever_service = None`
- [ ] Verify dense-only path still works when hybrid init fails

### Phase 3: Query Path Integration (2 hours)
- [ ] Modify `/query` endpoint to check `USE_HYBRID_RETRIEVAL` flag
- [ ] Wire hybrid retrieval into variant loop
- [ ] Preserve existing deduplication logic
- [ ] Add logging for hybrid vs dense path

### Phase 4: Testing & Validation (2 hours)
- [ ] Test with `USE_HYBRID_RETRIEVAL=False` (baseline)
- [ ] Test with `USE_HYBRID_RETRIEVAL=True` (hybrid)
- [ ] Compare recall on 20 test queries
- [ ] Verify no regressions in existing functionality

**Total Estimated Time:** 7 hours

---

## Testing Strategy

### Unit Tests
- `test_hybrid_retriever.py` (already exists, verify it passes)
- `test_corpus_loading.py` (new: verify BM25 index build)

### Integration Tests
- Query with `USE_HYBRID_RETRIEVAL=False` → verify dense-only path
- Query with `USE_HYBRID_RETRIEVAL=True` → verify hybrid path
- Compare top-5 results for 20 test queries

### Performance Tests
- Measure startup time with BM25 index build (expect +2-5 seconds for 1000 chunks)
- Measure query latency (expect +50-100ms for hybrid vs dense-only)

### A/B Test Queries
```
1. "học phí ngành Công nghệ thông tin"  (keyword-heavy)
2. "điều kiện xét tuyển thẳng"          (lexical match)
3. "các ngành tuyển sinh năm 2025"      (overview query)
4. "ngưỡng đầu vào là gì"               (definition query)
5. "thông tư 08/2022"                   (citation query)
```

---

## Fallback Strategy

The system follows a **graceful degradation** model with three levels:

```
Level 1: USE_HYBRID_RETRIEVAL=True + BM25 build SUCCESS
    → Full hybrid pipeline (dense + BM25 + RRF + reranker)
    → Best recall

Level 2: USE_HYBRID_RETRIEVAL=True + BM25 build FAILS
    → hybrid_retriever_service = None (set during startup catch)
    → /query endpoint detects None → auto-falls back to dense-only
    → WARNING logged: "Hybrid init failed, using dense-only fallback"
    → No service interruption, graceful degradation

Level 3: USE_HYBRID_RETRIEVAL=False (default)
    → Dense-only path (current production behavior)
    → Zero risk, baseline behavior
```

**On BM25 build failure:**
- Exception caught in `startup_event()` try-catch block
- `hybrid_retriever_service` remains `None`
- App continues startup and serves requests
- All queries use dense-only path silently
- Operators are alerted via `logger.error()` message with full traceback
- No retry on startup (avoid hanging); operator must fix and restart

**Failure scenarios covered:**
| Failure | Effect | Recovery |
|---------|--------|----------|
| LanceDB table empty | Warn + skip BM25 (empty corpus is valid) | Auto-recover when data ingested + restart |
| `rank_bm25` missing | Error + fallback to dense | Install dependency |
| LanceDB scan timeout | Error + fallback to dense | Restart service |
| `sparse_terms` column absent | Fallback to `text` field per-row | No action needed |

---

## Fusion Weights Configuration

Currently, RRF fusion weights are **hardcoded** in `retriever.py` line 328:
```python
# retriever.py line 327-328 (current)
fused_results = self._reciprocal_rank_fusion(
    dense_results, sparse_ids, weight_dense=0.6, weight_sparse=0.4
)
```

**Fix:** Wire `HYBRID_FUSION_DENSE_WEIGHT` and `HYBRID_FUSION_SPARSE_WEIGHT` from `settings` through the `retrieve()` call chain, so they can be tuned without code changes:

```python
# In HybridRetriever.retrieve() - use settings fields
fused_results = self._reciprocal_rank_fusion(
    dense_results,
    sparse_ids,
    weight_dense=self.settings.HYBRID_FUSION_DENSE_WEIGHT,   # 0.6 default
    weight_sparse=self.settings.HYBRID_FUSION_SPARSE_WEIGHT,  # 0.4 default
)
```

**Default values rationale:**
- `dense=0.6, sparse=0.4` is the empirically established baseline from academic literature (Ma et al. 2022)
- Vietnamese text has richer morphology → slightly prefer dense embeddings (Qwen3 trained multilingually)
- Operators can tune via `.env` without code changes: `HYBRID_FUSION_DENSE_WEIGHT=0.5` for more balanced fusion

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| BM25 index build fails at startup | HIGH | Try-catch in startup → log error → set `hybrid_retriever_service = None` → auto-fallback to dense-only. No service interruption. |
| Memory overhead for BM25 index | MEDIUM | Monitor memory usage. BM25Okapi for 10k chunks ~50MB. Acceptable for modern servers (8GB+ RAM). |
| Increased query latency | MEDIUM | Make hybrid retrieval opt-in via config flag. Measure p95 latency in staging before production rollout. |
| Breaking existing dense-only path | HIGH | Preserve existing code path with `if hybrid_retriever_service else` branch. Feature flag defaults to `False`. |
| `sparse_terms` missing in some chunks | LOW | Fallback to `text` field for BM25 indexing (per-row check in `load_corpus_for_bm25()`). |
| BM25 index becomes stale after new chunks added | MEDIUM | Document manual rebuild requirement. Dense retrieval covers new chunks immediately. Consider v2 auto-rebuild endpoint. |
| Fusion weights not tunable | LOW | Wire `HYBRID_FUSION_DENSE_WEIGHT` / `HYBRID_FUSION_SPARSE_WEIGHT` from settings. Operators can tune via `.env`. |

---

## Rollout Plan

### Stage 1: Development (Week 1)
- Implement changes in feature branch
- Run unit + integration tests
- Verify no regressions with `USE_HYBRID_RETRIEVAL=False`

### Stage 2: Staging (Week 2)
- Deploy to staging with `USE_HYBRID_RETRIEVAL=True`
- Run A/B test on 100 queries
- Measure recall improvement (target: +15-20%)

### Stage 3: Production (Week 3)
- Deploy to production with `USE_HYBRID_RETRIEVAL=False` (safe default)
- Enable for 10% of traffic via feature flag
- Monitor latency, error rate, user satisfaction
- Gradually increase to 100% if metrics improve

---

## Success Metrics

| Metric | Baseline (Dense-only) | Target (Hybrid) |
|--------|----------------------|-----------------|
| Recall@5 | 65% | 80% (+15%) |
| Query Latency (p95) | 250ms | 350ms (+100ms acceptable) |
| Answer Quality (human eval) | 3.5/5 | 4.2/5 (+0.7) |
| Startup Time | 8s | 13s (+5s acceptable) |

---

## Dependencies

### Existing (Already in codebase)
- `HybridRetriever` class (`rag2025/src/services/retriever.py`)
- `rank_bm25` library (already in `requirements.txt`)
- `sentence_transformers.CrossEncoder` (already loaded)
- `sparse_terms` field in chunks (already extracted)

### New (Need to create)
- `VectorStore` wrapper class (thin adapter for `LanceDBRetriever`)
- `EmbeddingService` wrapper class (thin adapter for `SentenceTransformer`)
- `load_corpus_for_bm25()` helper function

---

## Alternatives Considered

### Alternative 1: Modify LanceDBRetriever to support BM25
**Rejected:** Would require significant refactoring of existing retriever. `HybridRetriever` already exists and is well-designed.

### Alternative 2: Use Qdrant's built-in sparse vectors
**Rejected:** Codebase uses LanceDB, not Qdrant. Migration would be high-risk and out of scope.

### Alternative 3: Build separate BM25 service
**Rejected:** `HybridRetriever` already integrates BM25 + dense + reranking. No need to duplicate.

### Alternative 4: Use LanceDB FTS (full-text search)
**Rejected:** LanceDB FTS is experimental and less mature than BM25Okapi. Hybrid approach is proven.

---

## Open Questions

1. **Q:** Should we persist BM25 index to disk to avoid rebuild on restart?
   **A:** Not in v1. In-memory build is fast (<5s for 1000 chunks). `BM25_INDEX_PATH` added to settings for v2 extensibility, but defaults to `None`. Consider persistence in v2 if corpus grows >50k chunks and rebuild exceeds 30s.

2. **Q:** Should hybrid retrieval be the default (`USE_HYBRID_RETRIEVAL=True`)?
   **A:** No. Start with `False` for safety, enable after A/B testing validates improvement. Gradual rollout: dev → staging → 10% prod → 100% prod.

3. **Q:** What if `sparse_terms` field is missing in some chunks?
   **A:** Fallback to `text` field. BM25 will tokenize on-the-fly (slightly slower but functional). `load_corpus_for_bm25()` handles this per-row.

4. **Q:** Should we expose hybrid vs dense choice to end users?
   **A:** No. This is an internal optimization. Users should not need to know retrieval strategy.

5. **Q:** How to keep BM25 index in sync when new chunks are added?
   **A:** v1 accepts staleness (admission data updates infrequently). Dense retrieval covers new chunks immediately. Operators can restart service to rebuild BM25. v2 will add `POST /admin/hybrid/rebuild` endpoint for on-demand rebuild without restart.

6. **Q:** What happens if BM25 build fails mid-startup?
   **A:** Exception caught → `hybrid_retriever_service = None` → app continues → all queries use dense-only fallback → operator alerted via logs. No retry to avoid hanging startup.

---

## Decisions

| # | Decision | Alternatives considered | Reason chosen |
|---|----------|-------------------------|---------------|
| 1 | Reuse existing `HybridRetriever` class without modification | Build new hybrid retriever, modify LanceDBRetriever | `HybridRetriever` is already complete, tested, and well-designed. Minimal risk. |
| 2 | Build BM25 index at startup from LanceDB corpus (in-memory only) | Build on-demand per query, persist to disk | Startup build is fast (<5s) and simplifies query path. Persistence adds complexity for v1. Corpus updates are infrequent. |
| 3 | Use `sparse_terms` field for BM25, fallback to `text` | Always use `text`, re-extract sparse terms | `sparse_terms` already extracted during chunking. Reuse existing data. |
| 4 | Make hybrid retrieval opt-in via `USE_HYBRID_RETRIEVAL` flag (default=False) | Make it default, remove dense-only path | Preserve backward compatibility. Enable safe rollout with A/B testing. |
| 5 | Preserve existing dense-only path as fallback | Remove dense-only path entirely | Minimize risk. If hybrid fails, system degrades gracefully to dense-only. |
| 6 | Create thin adapter wrappers (`VectorStore`, `EmbeddingService`) | Refactor `HybridRetriever` to accept `LanceDBRetriever` directly | Adapters isolate changes. `HybridRetriever` remains reusable across different backends. |
| 7 | Make RRF fusion weights configurable via settings | Keep hardcoded 0.6/0.4 weights | Enable tuning without code changes. Operators can experiment with different weights via `.env`. |
| 8 | Load entire corpus into memory for BM25 | Use disk-based BM25 index | In-memory is faster for query time. Corpus size (<10k chunks) fits in memory (~50MB). |
| 9 | Accept BM25 index staleness when new chunks added | Auto-rebuild on every chunk addition | Admission data updates infrequently (1-2x per semester). Dense retrieval covers new chunks immediately. Manual rebuild acceptable for v1. |
| 10 | Graceful fallback on BM25 build failure (no retry) | Retry build, fail fast and block startup | Avoid hanging startup. Dense-only fallback ensures service availability. Operators alerted via logs. |
| 11 | Add `BM25_INDEX_PATH` setting but default to `None` (no persistence) | Omit setting entirely, always in-memory | Future-proof for v2 persistence. Setting exists but unused in v1. Clear migration path. |

---

## References

- `HybridRetriever` implementation: `rag2025/src/services/retriever.py`
- Current retrieval flow: `rag2025/src/main.py` lines 486-531
- Sparse terms extraction: `rag2025/src/chunker.py` lines 377-404
- Research: "Hybrid search improves recall by 20-40%" (task context)

---

**End of Spec**
