# Plan Review — hybrid-retrieval-integration
**Phase:** code_review
**Loop:** 2
**Commit:** d464f13
**Verdict:** approved

## Summary

Both required fixes from code-review loop 1 are correctly implemented. The startup block in `main.py` now matches the spec's guard condition and isolation pattern exactly. The test file now has 10 tests covering all 9 plan-specified scenarios plus one additional smoke test. All test assertions are logically correct and will exercise the right code paths. No new issues introduced.

---

## Issue 1 Resolution — `main.py` startup block (lines 301-318)

**Status: RESOLVED — CORRECT.**

```python
# ── Hybrid Search (optional) ──────────────────────────────────────
if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:
    try:
        from services.hybrid_search import HybridSearchService as _HybridSearchService
        hybrid_search_service = _HybridSearchService(
            lancedb_retriever=lancedb_retriever_service,
            settings=settings,
        )
        if not hybrid_search_service.build_bm25_index():
            logger.warning(
                "HybridSearchService: BM25 index build failed — disabling hybrid search"
            )
            hybrid_search_service = None
        else:
            logger.info("HybridSearchService: BM25 index built successfully")
    except Exception as e:
        logger.error(f"HybridSearchService: startup init failed: {e}", exc_info=True)
        hybrid_search_service = None
```

All three required elements are present:

| Check | Result |
|-------|--------|
| Guard: `if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:` | PASS (line 302) |
| Deferred import inside try block | PASS (line 304) |
| Inner `except Exception` sets `hybrid_search_service = None` with `exc_info=True` | PASS (lines 316-318) |
| "API Ready" log at line 320 is outside the try/except — will always execute | PASS |
| Graceful degradation: build failure → `hybrid_search_service = None` (line 313) | PASS |

The inner try/except correctly isolates `ImportError` (missing `rank_bm25`), constructor errors, and any unexpected exceptions from propagating into the outer `startup_event` try/except — ensuring "API Ready" is always logged.

---

## Issue 2 Resolution — `test_hybrid_search.py` (10 tests)

**Status: RESOLVED — CORRECT.**

**Test count: 10** (up from 8). All plan items now covered.

| Plan Item | Required Test Name | Implementation | Assertions | Status |
|-----------|-------------------|----------------|------------|--------|
| 1 | `test_build_bm25_index_success` | Present | `assertTrue`, `assertIsNotNone`, `assertEqual(len, 2)` | PASS |
| 2 | `test_build_bm25_index_empty_table` | Present | `assertFalse`, `assertIsNone` | PASS |
| 3 | `test_build_bm25_index_exception` | `test_build_bm25_index_failure_graceful` | `assertFalse`, `assertIsNone` | PASS |
| 4 | `test_bm25_search_returns_relevant` | **ADDED** | `assertGreater(len, 0)`, `assertEqual(result[0].chunk_id, "c2")` | PASS |
| 5 | `test_bm25_search_empty_query_guard` | Present | `assertEqual(result, [])` | PASS |
| 6 | `test_rrf_fusion_deduplication` | `test_rrf_fusion_correctness` | `assertIn`, `assertGreater`, **`assertEqual(count("c2"), 1)`** | PASS |
| 7 | `test_rrf_fusion_weights` | **ADDED** | `assertLess(index("c1"), index("c2"))` | PASS |
| 8 | `test_retrieve_fallback_to_dense` | `test_hybrid_retrieve_bm25_not_ready_fallback` | `assertEqual(result.documents, docs)` | PASS |
| 9 | `test_retrieve_hybrid_path` | `test_hybrid_retrieve_full_path` | `assertIsInstance`, `assertGreater(len, 0)` | PASS |

### Assertion correctness verification

**`test_bm25_search_returns_relevant` (lines 102-114):**
Corpus: c1=`"tuyển sinh đại học"`, c2=`"học phí ngành công nghệ"`, c3=`"thời gian nộp hồ sơ"`. Query: `"học phí"` → tokens `["học", "phí"]`. c2 is the only document containing both query tokens; BM25Okapi will score it highest. Assertion `result[0].chunk_id == "c2"` is mathematically guaranteed to hold. ✓

**`test_rrf_fusion_weights` (lines 141-152):**
`dense_w=1.0, sparse_w=0.0, k=60`. Computed scores: c1 = 1.0/(60+1) ≈ 0.01639; c2 = 1.0/(60+2) + 0.0/(60+1) ≈ 0.01613; c3 = 0.0/(60+2) = 0.0. Sorted: `[c1, c2, c3]`. Assertion `index("c1") < index("c2")` → `0 < 1` → TRUE. ✓

**`test_rrf_fusion_correctness` dedup assertion (line 139):**
`chunk_ids.count("c2") == 1` — since `_rrf_fusion` uses a `chunk_id`-keyed dict, c2 can appear at most once in `sorted_ids`. This assertion correctly verifies the structural deduplication guarantee. ✓

---

## Full Implementation Re-Verification (no regressions)

Spot-checked unchanged files to confirm the loop-1 approved elements remain intact:

| Component | Status |
|-----------|--------|
| `hybrid_search.py`: module-level `logger`, no `self._logger` | PASS (unchanged) |
| `hybrid_search.py`: `_rrf_fusion` k=60, weight settings access | PASS (unchanged) |
| `hybrid_search.py`: `_bm25_search` empty-token guard | PASS (unchanged) |
| `settings.py`: 4 hybrid fields with correct defaults | PASS (unchanged) |
| `main.py` global declaration includes `hybrid_search_service` | PASS (line 236, unchanged) |
| `main.py` primary retrieval call: `if hybrid_search_service:` branch | PASS (lines 530-542, unchanged) |
| `main.py` no-filter fallback: `if hybrid_search_service:` branch | PASS (lines 546-558, unchanged) |

---

## Decision

approved
