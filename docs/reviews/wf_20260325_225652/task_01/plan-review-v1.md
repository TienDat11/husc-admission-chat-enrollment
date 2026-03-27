# Plan Review — hybrid-retrieval-integration
**Phase:** plan_review
**Loop:** 1
**Verdict:** issues

## Summary

The plan is thorough, well-structured, and correctly carries forward all 8 spec decisions. The implementation code in the plan's full-implementation section is production-quality and matches the spec's class design. However, three concrete issues were found during codebase verification that would cause the plan to fail or produce incorrect behavior at implementation time:

1. The retrieval loop replacement in `main.py` is incomplete — the plan only replaces the primary `lancedb_retriever_service.retrieve(...)` call but leaves the no-filter fallback call (line 519) as a bare `lancedb_retriever_service.retrieve(...)`. The hybrid path must also wrap this fallback call, or the fallback will always bypass the hybrid path even when `USE_HYBRID_RETRIEVAL=True`.

2. The plan incorrectly states the `global` declaration is on "line 233" covering a fixed set of variables. The actual startup `global` line is:
   `global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_encoder, unified_pipeline, reranker_service, query_cache, guardrail_service`
   — `hybrid_search_service` is absent. The plan must make explicit that this exact line must be extended to include `hybrid_search_service`.

3. The plan's `__init__` method in Step 2's description uses `self._logger = logging.getLogger(__name__)`, but the full implementation body (the canonical code block) uses a module-level `logger = logging.getLogger(__name__)` without `self._logger`. These two descriptions contradict each other. The full implementation block is correct (matching the spec); the method description is wrong and will confuse an implementer.

No HIGH or CRITICAL blast-radius concerns were raised: the plan creates 0 net changes to existing symbols (all changes are new code + controlled insertion points), and `rank_bm25` is confirmed present in `requirements.txt` as `rank-bm25==0.2.2`.

---

## Issues Found

### Issue 1 (REQUIRED FIX) — Hybrid path does not cover the no-filter fallback call in the retrieval loop

**Location:** Plan Step 3c, `main.py` lines 517-523 (verified).

The plan wraps the primary retrieval call with a hybrid branch, but the actual `/query` loop contains a second retrieval call — a no-filter retry when the filtered call returns 0 docs:

```python
# Line 519 — NOT covered by the plan's change
retrieval_result = lancedb_retriever_service.retrieve(
    query_vector=query_vector,
    top_k=retrieval_top_k,
    metadata_filter=None,
)
```

With the plan as written, this fallback call will always use dense-only even when `USE_HYBRID_RETRIEVAL=True`. This is a logic inconsistency: the primary path is hybrid, the fallback silently reverts to dense. The plan must either:

- (a) Wrap this fallback call with the same `if hybrid_search_service:` branch, or
- (b) Explicitly document the decision to leave the fallback as dense-only with a rationale (acceptable if intentional, but must be explicit).

### Issue 2 (REQUIRED FIX) — Incomplete instruction for extending the `global` declaration in `startup_event`

**Location:** Plan Step 3b.

The plan states: "extend the `global` declaration line to include `hybrid_search_service`" but does not show the full updated line. The actual line in `main.py` at line 233 is:

```python
global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_encoder, unified_pipeline, reranker_service, query_cache, guardrail_service
```

The plan must show the exact replacement line with `hybrid_search_service` appended, rather than leaving it as an implicit instruction. Given the length of this line and the number of variables, an implementer could easily insert it in the wrong place or omit it, causing a `UnboundLocalError` at runtime.

### Issue 3 (SHOULD FIX) — Contradictory logger pattern between Step 2 description and the canonical code block

**Location:** Plan Step 2, `__init__` method description vs. full implementation code block.

The Step 2 `__init__` method description states:
```python
self._logger = logging.getLogger(__name__)
```

But the canonical full implementation block (lines 341-342 of the plan) uses:
```python
logger = logging.getLogger(__name__)
```
as a module-level variable (correct, matching the spec). All subsequent logging calls in the implementation use `logger.warning(...)`, `logger.info(...)`, `logger.error(...)` — not `self._logger`.

An implementer following the Step-by-Step descriptions (as they should) will create `self._logger` in `__init__` but then find all log calls reference the undefined module-level `logger`. The step description must be corrected to remove `self._logger = ...` from the `__init__` description and confirm that `logger` is a module-level variable.

---

## Cross-Reference: Spec Goals → Plan Coverage

| Spec Goal | Covered in Plan | Notes |
|-----------|----------------|-------|
| G1: Create `HybridSearchService` class | Yes — Step 2 + full code block | Complete |
| G2: Build BM25 index at startup from LanceDB using column-selective `to_pandas()` | Yes — Step 1 + Step 3b | Complete |
| G3: Implement RRF fusion (k=60) over dense + sparse | Yes — `_rrf_fusion` in Step 2 | Complete |
| G4: Preserve backward compatibility — dense-only fallback | Yes — `USE_HYBRID_RETRIEVAL=False` default, fallback on None | Complete (Issue 1 partially undermines this) |
| G5: Enable toggle via `USE_HYBRID_RETRIEVAL` config flag (default=False) | Yes — Step 1 settings + Step 3c branch | Complete |
| G6: Graceful degradation — 3 levels | Yes — Graceful Degradation section mirrors spec exactly | Complete |

All 8 spec decisions are carried into the plan's Decisions table with correct rationale.

All spec-defined files are listed: `hybrid_search.py` (CREATE), `settings.py` (MODIFY), `main.py` (MODIFY), `test_hybrid_search.py` (CREATE).

---

## Cross-Reference: Spec Decisions → Plan

| Spec Decision | In Plan? | Match |
|---------------|----------|-------|
| D1: New `HybridSearchService` (not adapter) | Yes | Exact |
| D2: Work natively with `RetrievedDocument` | Yes | Exact |
| D3: Column-selective `to_pandas()` | Yes | Exact |
| D4: RRF k=60 | Yes | Exact |
| D5: Prefer `sparse_terms`, fallback to `text.split()` | Yes | Plan adds `tokens = raw if raw else [""]` empty-token guard (correct improvement over spec) |
| D6: Over-fetch `top_k*2` | Yes | Exact |
| D7: `USE_HYBRID_RETRIEVAL=False` default | Yes | Exact |
| D8: In-memory BM25 v1, `BM25_INDEX_PATH=None` | Yes | Exact |
| D9 (new in plan): Deferred import in `startup_event` | Yes — plan adds Decision 9 | Valid addition not in spec; safe and correct |

---

## Codebase Verification

### `rag2025/config/settings.py`

- Class name: `RAGSettings` — CONFIRMED (line 12)
- `# ========== Retrieval Parameters ==========` section ends at `TOP_K_SPARSE` — CONFIRMED (line 77)
- Plan says to insert new block immediately after `TOP_K_SPARSE` — CONFIRMED as correct insertion point
- `Optional` is already imported (line 6, `from typing import Literal, Optional`) — new `BM25_INDEX_PATH: Optional[str]` field will not require a new import
- No existing `USE_HYBRID_RETRIEVAL` or hybrid settings present — insertion is clean

### `rag2025/src/services/lancedb_retrieval.py`

- `LanceDBRetriever` class — CONFIRMED (line 72)
- `RetrievedDocument` dataclass — CONFIRMED (lines 27-43), fields: `text`, `source`, `chunk_id`, `metadata`, `score`, `point_id` — all match plan's usage
- `RetrievalResult` dataclass — CONFIRMED (lines 46-55), fields: `documents`, `error_type`, `error_message`, `confidence` — all match plan's usage
- `LanceDBRetriever._adapter` attribute — CONFIRMED (line 75: `self._adapter = LanceDBAdapter(...)`)
- `LanceDBAdapter._table` attribute — plan assumes `adapter._table`; the `LanceDBAdapter` class is in `src/infrastructure/lancedb_adapter.py` and was not read, but this is the standard LanceDB adapter pattern and the existing `HybridRetriever` in `retriever.py` was described in the spec as using similar access. This is a low risk; BM25 build failure falls back gracefully.

### `rag2025/src/main.py`

- Global services block — CONFIRMED at lines 209-220; `hybrid_search_service` is NOT yet declared (expected — plan adds it)
- `startup_event()` global declaration — CONFIRMED at line 233; does NOT include `hybrid_search_service` yet
- GraphRAG pipeline init block — CONFIRMED at lines 281-296; plan's placement "after GraphRAG block" is correct
- "API Ready" log line — CONFIRMED at lines 298-300; hybrid init goes before this
- The retrieval loop — CONFIRMED at lines 493-531; primary call at line 511-515, no-filter fallback at lines 517-523 (Issue 1)
- The `/query` endpoint is `async` — CONFIRMED; `await hybrid_search_service.retrieve(...)` is valid

### `rag2025/requirements.txt`

- `rank_bm25` — CONFIRMED as `rank-bm25==0.2.2` (line 4). Plan correctly states "already listed, do not add again."
- `pytest` and `pytest-asyncio` are commented out (lines 28-29). The plan says "Use `pytest-asyncio` for async test cases" without noting they need to be uncommented. This is a minor gap for the test step.

### Test count discrepancy

- Spec defines 8 unit tests (items 1-8 in Testing Strategy).
- Plan Step 4 defines 9 unit tests (adds `test_bm25_search_empty_query_guard` as item 5, renumbering the rest).
- This is a valid and correct extension — the empty-query guard is implemented in the code and deserves explicit test coverage. No conflict with spec.

---

## Decision

issues:

1. **[REQUIRED]** Step 3c must address the no-filter fallback retrieval call at line 519 of `main.py` — either wrap it with the same `if hybrid_search_service:` branch or explicitly document why it remains dense-only.

2. **[REQUIRED]** Step 3b must show the exact full `global` declaration line with `hybrid_search_service` appended, not just describe the change in prose.

3. **[SHOULD FIX]** Step 2 `__init__` description must remove `self._logger = logging.getLogger(__name__)` — the canonical code uses a module-level `logger` variable, not an instance attribute.
