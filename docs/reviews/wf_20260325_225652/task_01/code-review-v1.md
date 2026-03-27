# Plan Review — hybrid-retrieval-integration
**Phase:** code_review
**Loop:** 1 (first code review loop)
**Verdict:** issues

## Summary

The core implementation of `HybridSearchService` is correct, clean, and functionally sound. All three issues from plan-review-v1 were correctly resolved in the plan and carried through into implementation. The `hybrid_search.py` class, `settings.py` changes, and both retrieval call wraps in `main.py` are all correct.

Two issues require fixes before approval:

1. **[REQUIRED]** The startup block in `main.py` omits the `lancedb_retriever_service` null guard specified in the spec, and does not wrap the hybrid init in an inner try/except — meaning an unexpected exception from the deferred import or constructor would propagate into the outer `startup_event` try/except and potentially abort other startup steps that follow, such as the "API Ready" log (they would not, since the hybrid block IS inside the outer try, but the spec's pattern was safer).

2. **[REQUIRED]** The test file has 8 tests (not 9 as specified in the plan) and is missing two specific test cases the plan explicitly required: `test_bm25_search_returns_relevant` (plan item 4) and `test_rrf_fusion_weights` with `dense_w=1.0, sparse_w=0.0` (plan item 7). The existing `test_rrf_fusion_correctness` does not explicitly assert deduplication (no `count == 1` or `len == len(set)` check) nor weight isolation.

---

## File-by-File Verification

### `rag2025/src/services/hybrid_search.py` — PASS

All checklist items verified:

| Check | Result |
|-------|--------|
| Module-level `logger = logging.getLogger(__name__)` at line 17 | PASS |
| No `self._logger` in `__init__` | PASS |
| `build_bm25_index()`: column-selective `to_pandas(columns=load_cols)` | PASS |
| `build_bm25_index()`: empty-table returns `False` | PASS |
| `build_bm25_index()`: exception caught, returns `False` | PASS |
| Empty-token guard in `build_bm25_index`: `tokens = raw if raw else [""]` | PASS (line 70) |
| `retrieve()`: over-fetch `top_k * 2` for dense and sparse | PASS (lines 115, 124) |
| `retrieve()`: BM25 None → dense-only fallback (returns `dense_result`) | PASS (lines 120-122) |
| `retrieve()`: calls `_rrf_fusion` then `RetrievalResult(documents=fused, confidence=...)` | PASS |
| `_rrf_fusion()`: `k=60` default | PASS (line 164) |
| `_rrf_fusion()`: uses `HYBRID_FUSION_DENSE_WEIGHT` and `HYBRID_FUSION_SPARSE_WEIGHT` | PASS (lines 172-173) |
| `_rrf_fusion()`: deduplication by `chunk_id` (dict keys) | PASS (structural guarantee) |
| `_bm25_search()`: empty-query guard returning `[]` | PASS (lines 136-138) |

The full implementation code is correct and matches the plan's canonical code block exactly.

---

### `rag2025/config/settings.py` — PASS

All 4 fields verified at lines 79-102:

| Field | Default | Constraints | Result |
|-------|---------|-------------|--------|
| `USE_HYBRID_RETRIEVAL: bool` | `False` | — | PASS |
| `HYBRID_FUSION_DENSE_WEIGHT: float` | `0.6` | `ge=0.0, le=1.0` | PASS |
| `HYBRID_FUSION_SPARSE_WEIGHT: float` | `0.4` | `ge=0.0, le=1.0` | PASS |
| `BM25_INDEX_PATH: Optional[str]` | `None` | — | PASS |

Inserted immediately after `TOP_K_SPARSE` (line 77), in the correct location. `Optional` was already imported. No existing fields were modified.

---

### `rag2025/src/main.py` — MOSTLY PASS, one issue

| Check | Result |
|-------|--------|
| `hybrid_search_service: Optional["HybridSearchService"] = None` in globals block | PASS (line 223) |
| `global` declaration includes `hybrid_search_service` appended | PASS (line 236, exact line matches plan requirement) |
| Deferred import `from services.hybrid_search import HybridSearchService as _HybridSearchService` | PASS (line 303) |
| Hybrid init block: `if settings.USE_HYBRID_RETRIEVAL:` guard | PASS (line 302) |
| Build call + graceful disable on failure (`hybrid_search_service = None`) | PASS (lines 308-313) |
| Hybrid init placed after GraphRAG block, before "API Ready" log | PASS (lines 301-315 vs 317-319) |
| Primary retrieval call wrapped with `if hybrid_search_service:` | PASS (lines 530-542) |
| No-filter fallback call wrapped with `if hybrid_search_service:` | PASS (lines 546-558) |
| `await hybrid_search_service.retrieve(...)` used correctly | PASS |
| `query=variant` passed to `retrieve()` in both branches | PASS |

**Issue found:** The startup init block at line 302 uses:

```python
if settings.USE_HYBRID_RETRIEVAL:
    from services.hybrid_search import HybridSearchService as _HybridSearchService
    hybrid_search_service = _HybridSearchService(
        lancedb_retriever=lancedb_retriever_service,
        settings=settings,
    )
```

The spec (line 366) specifies: `if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:`. The implementation omits the `lancedb_retriever_service` null guard. Additionally, the plan (Step 3b) showed an inner try/except wrapping the entire hybrid init block — the implementation has no inner try/except; only the outer `startup_event` try/except exists.

**Functional impact analysis:** When `lancedb_retriever_service is None` (e.g., LanceDB timeout scenario), `HybridSearchService(None, settings)` is constructed without error (no guard in `__init__`). Then `build_bm25_index()` calls `self._retriever._adapter` on `None`, which raises `AttributeError`, which is caught by `build_bm25_index()`'s own `except Exception`, returning `False`. The outer handler then sets `hybrid_search_service = None`. So the outcome is **functionally correct** — graceful degradation works.

However: if a future `ImportError` for `rank_bm25` or a constructor error occurs, it propagates to the outer `startup_event` try/except (line 321), which logs and does NOT re-raise — so "API Ready" log is also suppressed. The plan's inner try/except pattern would have isolated the failure and let startup continue cleanly. This is the safer pattern and matches what the plan specified.

**Severity: REQUIRED** — the guard `and lancedb_retriever_service` is a one-word addition that the spec explicitly mandated. The missing inner try/except means a deferred-import failure (`rank_bm25` missing when `USE_HYBRID_RETRIEVAL=True`) would silently swallow the "API Ready" log. Both should be fixed for correctness.

---

### `rag2025/tests/test_hybrid_search.py` — FAIL

**Test count:** 8 tests implemented; plan requires 9.

**Mapping of plan test items to implementation:**

| Plan Item | Plan Test Name | Implementation | Status |
|-----------|---------------|----------------|--------|
| 1 | `test_build_bm25_index_success` | `test_build_bm25_index_success` | PASS |
| 2 | `test_build_bm25_index_empty_table` | `test_build_bm25_index_empty_table` | PASS |
| 3 | `test_build_bm25_index_exception` | `test_build_bm25_index_failure_graceful` | PASS (equivalent) |
| 4 | `test_bm25_search_returns_relevant` | **MISSING** | FAIL |
| 5 | `test_bm25_search_empty_query_guard` | `test_bm25_search_empty_query_guard` | PASS |
| 6 | `test_rrf_fusion_deduplication` | `test_rrf_fusion_correctness` (partial) | PARTIAL FAIL |
| 7 | `test_rrf_fusion_weights` | **MISSING** | FAIL |
| 8 | `test_retrieve_fallback_to_dense_when_bm25_none` | `test_hybrid_retrieve_bm25_not_ready_fallback` | PASS |
| 9 | `test_retrieve_hybrid_path` | `test_hybrid_retrieve_full_path` | PASS |

**Issue 2a — Missing `test_bm25_search_returns_relevant` (plan item 4):**
The plan requires: "Build index with 3 docs, one containing 'học phí' tokens; call `_bm25_search('học phí', top_k=3)`; assert first result matches expected doc." This test is entirely absent. Without it, there is no test verifying that BM25 actually retrieves a relevant document by keyword matching — which is the core value proposition of this feature.

**Issue 2b — Missing `test_rrf_fusion_weights` (plan item 7):**
The plan requires: "Set `HYBRID_FUSION_DENSE_WEIGHT=1.0`, `HYBRID_FUSION_SPARSE_WEIGHT=0.0`; call `_rrf_fusion(dense_docs, sparse_docs, top_k=5)`; assert output ranking matches dense-only order." This test is completely absent (the `1.0` that appears in the file is only the `score=1.0` default in `_make_doc`). This test verifies the weight mechanism is wired correctly, which is critical for the RRF fusion logic.

**Issue 2c — `test_rrf_fusion_correctness` does not assert deduplication:**
The plan (item 6) requires: "Assert the shared doc appears exactly once in output." The existing test only asserts `assertIn("c2", chunk_ids)` and `assertGreater(score, 0)`. There is no assertion that `c2` does NOT appear twice (e.g., `self.assertEqual(chunk_ids.count("c2"), 1)` or `self.assertEqual(len(chunk_ids), len(set(chunk_ids)))`). While the `_rrf_fusion` implementation is structurally correct (dict keys guarantee dedup), the test does not verify this invariant.

**Extra test added (not in plan):** `test_dense_only_when_flag_false` is a low-value test that only asserts `settings.USE_HYBRID_RETRIEVAL is False` — it does not test any code in `hybrid_search.py` and adds no meaningful coverage. Not a blocking issue, but it substitutes for one of the missing plan tests.

---

## Issues Summary

### Issue 1 — REQUIRED: Missing `lancedb_retriever_service` guard and inner try/except in startup init

**File:** `rag2025/src/main.py`, lines 302-315

The spec requires `if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:`. The implementation uses only `if settings.USE_HYBRID_RETRIEVAL:`. Additionally, the plan's startup block wrapped the entire hybrid init in an inner try/except; the implementation relies on the outer `startup_event` try/except, which means an `ImportError` from `rank_bm25` (when `USE_HYBRID_RETRIEVAL=True`) would cause the outer handler to log and exit, preventing the "API Ready" log from appearing.

**Fix:** Add `and lancedb_retriever_service` to the guard condition. Add an inner try/except block wrapping the import, instantiation, and build call, matching the plan's Step 3b pattern.

### Issue 2 — REQUIRED: Three test gaps in `test_hybrid_search.py`

**File:** `rag2025/tests/test_hybrid_search.py`

a. Add `test_bm25_search_returns_relevant`: build a 3-doc index, call `_bm25_search("học phí", top_k=3)`, assert the doc containing "học phí" tokens is ranked first.

b. Add `test_rrf_fusion_weights`: configure `dense_w=1.0, sparse_w=0.0`, call `_rrf_fusion(dense_docs, sparse_docs, top_k=5)`, assert ranking matches dense-only input order (c1 before c2).

c. Strengthen `test_rrf_fusion_correctness` to add an explicit deduplication assertion: `self.assertEqual(chunk_ids.count("c2"), 1)` or `self.assertEqual(len(chunk_ids), len(set(chunk_ids)))`.

---

## Positive Observations

- `hybrid_search.py` is clean, well-documented, and implementation-complete. The spec's 130-LOC estimate was met accurately.
- The `Optional["HybridSearchService"] = None` string annotation at module level correctly avoids a circular/deferred import problem.
- Both retrieval calls in the `/query` loop (primary and no-filter fallback) are correctly wrapped — this was the hardest issue from plan-review-v1 and was correctly implemented.
- The `global` declaration at line 236 correctly includes `hybrid_search_service` as the last item.
- The `build_bm25_index()` empty-token guard (`tokens = raw if raw else [""]`) is correctly implemented, matching the plan's critical guard note.
- The RRF `k=60` and weight settings access are correct.
- No existing symbols were modified — blast radius is zero.

---

## Decision

issues:

1. **[REQUIRED]** `main.py` startup block: add `and lancedb_retriever_service` to the `if settings.USE_HYBRID_RETRIEVAL:` guard, and wrap the hybrid init (import + instantiate + build) in an inner try/except with explicit error logging and `hybrid_search_service = None` fallback.

2. **[REQUIRED]** `test_hybrid_search.py`: (a) add `test_bm25_search_returns_relevant`, (b) add `test_rrf_fusion_weights` with `dense_w=1.0`, (c) add dedup assertion to `test_rrf_fusion_correctness`.
