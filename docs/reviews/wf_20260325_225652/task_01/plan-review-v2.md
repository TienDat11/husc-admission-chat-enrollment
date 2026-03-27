# Plan Review — hybrid-retrieval-integration
**Phase:** plan_review
**Loop:** 2
**Verdict:** approved

## Summary

All three issues raised in loop 1 have been correctly resolved. The revised plan is internally consistent with the approved spec, covers all requirements, and is implementable as written. No new issues were introduced by the revision.

---

## Issue Resolution Verification

### Issue 1 (REQUIRED) — No-filter fallback call coverage

**Status: RESOLVED.**

Step 3c now explicitly addresses both retrieval calls in the `/query` loop:

1. The primary filtered call (lines 256-268 of plan) is wrapped with `if hybrid_search_service:`.
2. The no-filter fallback call (lines 286-298 of plan) is identically wrapped with its own `if hybrid_search_service:` branch.

The rationale is also documented: "Both calls — the primary filtered call and the no-filter fallback — must use the hybrid path when `hybrid_search_service` is active." This closes the logic inconsistency completely.

### Issue 2 (REQUIRED) — Exact `global` declaration line

**Status: RESOLVED.**

Step 3b now provides the exact replacement line verbatim:

```python
global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_encoder, unified_pipeline, reranker_service, query_cache, guardrail_service, hybrid_search_service
```

`hybrid_search_service` is appended at the end. An implementer can directly copy-paste this line over the existing one at `main.py:233` with no ambiguity.

### Issue 3 (SHOULD FIX) — `self._logger` contradiction removed

**Status: RESOLVED.**

A new "Module-level logger" subsection has been added to Step 2 (lines 93-95 of plan):

> `logger = logging.getLogger(__name__)` is declared at **module level** (top of `hybrid_search.py`, immediately after the imports), not as an instance attribute. All log calls throughout the class (`logger.warning(...)`, `logger.info(...)`, `logger.error(...)`) reference this module-level variable. Do NOT add `self._logger` to `__init__`.

The `__init__` method description code block contains no `self._logger` reference. The canonical full implementation code block (lines 380-572) is consistent: `logger` is declared at module level, all log calls use `logger.*`. The contradiction is eliminated.

---

## New Issues Check

No new issues found. The revision is clean:

- The plan's full implementation code block is unchanged and remains correct.
- Step 3c's addition of the fallback branch is syntactically correct — the fallback uses `metadata_filter=None` explicitly (not the variable), correctly mirroring the original dense-only fallback behavior.
- The `await hybrid_search_service.retrieve(...)` call in both branches is correct since the `/query` endpoint is `async`.
- The `variant` variable (the text query string passed as `query=variant`) is the correct variable in scope at that point in the loop, confirmed against `main.py` line 493.
- No LOC delta estimates were changed — the plan still targets ~295 LOC added, 0 existing LOC modified, which is consistent (the fallback branch adds ~8 lines but was already implied in the original "+25" estimate for `main.py`).

---

## Spec Coverage Confirmation (unchanged from loop 1, all pass)

| Spec Goal | Covered | Status |
|-----------|---------|--------|
| G1: Create `HybridSearchService` | Yes | Pass |
| G2: Build BM25 index at startup via column-selective `to_pandas()` | Yes | Pass |
| G3: RRF fusion (k=60) | Yes | Pass |
| G4: Backward compatibility / dense fallback | Yes | Pass (now fully consistent) |
| G5: `USE_HYBRID_RETRIEVAL` toggle (default=False) | Yes | Pass |
| G6: Graceful degradation — 3 levels | Yes | Pass |

All 9 decisions (8 from spec + D9 deferred import, plan addition) are present and correctly reasoned.

All 4 required files are listed with correct paths and action types.

9 unit tests cover all specified scenarios (8 from spec + 1 additional empty-query guard test, which is a correct addition).

---

## Decision

approved
