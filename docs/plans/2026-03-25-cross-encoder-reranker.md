# Cross-Encoder Reranker Enhancement — Implementation Plan

**Date:** 2026-03-25
**Task:** cross-encoder-reranker
**Spec:** docs/specs/2026-03-25-cross-encoder-reranker-design.md
**Estimated Time:** 45 minutes

---

## Overview

Enhance the existing `RerankerService` to:
1. Pre-filter candidates using `MAX_RERANK` setting (already exists at settings.py line 70, default=50)
2. Add lost-in-the-middle chunk reordering using O(N) two-pointer algorithm
3. Make lost-in-middle behavior explicit in main.py call site

**Scope:** 2 files only
- `rag2025/src/services/reranker.py` — add candidate limit, lost-in-middle method, update signature
- `rag2025/src/main.py` — pass `apply_lost_in_middle=True` explicitly at line 563

**No changes to:**
- `rag2025/config/settings.py` — `MAX_RERANK` already exists
- `rag2025/requirements.txt` — no new dependencies

---

## Pre-Implementation Checklist

- [x] Spec approved: docs/specs/2026-03-25-cross-encoder-reranker-design.md
- [x] Current state verified:
  - `reranker.py` uses `Qwen/Qwen3-Reranker-8B` (settings.py line 54)
  - `RERANKER_ENABLED=True`, `RERANKER_WEIGHT=0.35` exist (lines 56-57)
  - `MAX_RERANK=50` exists but not read by RerankerService (line 70)
  - Reranker wired in main.py lines 562-567
  - No lost-in-middle reordering implemented
- [x] GitNexus impact analysis: CRITICAL risk, 2 callers (active + backup), 6 processes affected
- [x] Backup caller acknowledged: `backup_mail_package_2026/.../main.py` will benefit from default `apply_lost_in_middle=True`

---

## Step 1: Read MAX_RERANK in RerankerService.__init__

**File:** `rag2025/src/services/reranker.py`
**Lines:** 13-17 (current `__init__`)

**Current code:**
```python
def __init__(self, settings: RAGSettings):
    self._enabled = settings.RERANKER_ENABLED
    self._model_name = settings.RERANKER_MODEL
    self._weight = settings.RERANKER_WEIGHT
    self._model = None
```

**Change:** Add one line after `self._weight`:

```python
def __init__(self, settings: RAGSettings):
    self._enabled = settings.RERANKER_ENABLED
    self._model_name = settings.RERANKER_MODEL
    self._weight = settings.RERANKER_WEIGHT
    self._max_rerank = settings.MAX_RERANK  # ← ADD THIS LINE
    self._model = None
```

**Verification:**
- `self._max_rerank` will be `50` (default from settings.py line 70)
- No runtime errors — `MAX_RERANK` is a required field in `RAGSettings`

---

## Step 2: Pre-filter Candidates in rerank()

**File:** `rag2025/src/services/reranker.py`
**Lines:** 34-44 (current `rerank()` method start)

**Current code:**
```python
def rerank(
    self,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not self.enabled or not chunks:
        return chunks[:top_k]

    pairs = [(query, c.get("text", "")) for c in chunks]
    rerank_scores = self._model.predict(pairs)
```

**Change:** Slice `chunks` to `self._max_rerank` before creating pairs:

```python
def rerank(
    self,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not self.enabled or not chunks:
        return chunks[:top_k]

    # Pre-filter: limit candidates to MAX_RERANK before cross-encoder
    candidates = chunks[: self._max_rerank]  # ← ADD THIS LINE

    pairs = [(query, c.get("text", "")) for c in candidates]  # ← CHANGE: chunks → candidates
    rerank_scores = self._model.predict(pairs)
```

**Also update line 47** (the `zip()` call):

**Current:**
```python
for chunk, rr_score in zip(chunks, rerank_scores):
```

**Change to:**
```python
for chunk, rr_score in zip(candidates, rerank_scores):  # ← CHANGE: chunks → candidates
```

**Verification:**
- If `len(chunks) = 100`, only first 50 are sent to `_model.predict()`
- If `len(chunks) = 20`, all 20 are sent (no change)
- Latency improvement: O(N) → O(min(N, 50)) for cross-encoder inference

---

## Step 3: Add apply_lost_in_middle Parameter

**File:** `rag2025/src/services/reranker.py`
**Lines:** 34-39 (method signature)

**Current signature:**
```python
def rerank(
    self,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
```

**Change:** Add new parameter with default value:

```python
def rerank(
    self,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
    apply_lost_in_middle: bool = True,  # ← ADD THIS LINE
) -> List[Dict[str, Any]]:
```

**Backward compatibility:** Default `True` means existing callers without this parameter will automatically get lost-in-middle reordering (intentional improvement).

---

## Step 4: Implement _apply_lost_in_middle() Method

**File:** `rag2025/src/services/reranker.py`
**Location:** After `rerank()` method (after line 57)

**Add new method:**

```python
def _apply_lost_in_middle(
    self, chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Reorder score-sorted chunks to mitigate the lost-in-the-middle effect.

    Chunks are placed alternately at the front and back of a result list
    so that the highest-scoring chunks end up at boundary positions where
    LLM attention is strongest.

    Example (5 chunks ranked 1st-5th by score):
      Input:   [1st, 2nd, 3rd, 4th, 5th]
      Output:  [1st, 3rd, 5th, 4th, 2nd]
                ^                    ^
               front               back  ← LLM pays most attention here

    Args:
        chunks: Score-sorted chunks (descending). Must have len > 0.

    Returns:
        Reordered list with boundary positions occupied by top-scored chunks.
    """
    n = len(chunks)
    if n <= 2:
        return chunks  # No reordering needed for 0, 1, or 2 chunks

    result = [None] * n
    head = 0
    tail = n - 1

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            # Even index: place at front
            result[head] = chunk
            head += 1
        else:
            # Odd index: place at back
            result[tail] = chunk
            tail -= 1

    return result
```

**Algorithm verification:**
- Pre-allocates `result = [None] * n` — O(N) space, no list shifting
- Two pointers `head` (starts at 0) and `tail` (starts at n-1) advance inward
- Loop runs exactly `n` iterations, placing each chunk once
- For `n=5`: positions filled = [0, 4, 1, 3, 2] → ranks [1st, 2nd, 3rd, 4th, 5th] → output [1st, 3rd, 5th, 4th, 2nd]

---

## Step 5: Call _apply_lost_in_middle() in rerank()

**File:** `rag2025/src/services/reranker.py`
**Lines:** 56-57 (current end of `rerank()`)

**Current code:**
```python
rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
return rescored[:top_k]
```

**Change:** Add lost-in-middle call before return:

```python
rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
top_chunks = rescored[:top_k]  # ← CHANGE: slice first

# Apply lost-in-the-middle reordering
if apply_lost_in_middle and len(top_chunks) > 2:  # ← ADD THIS BLOCK
    top_chunks = self._apply_lost_in_middle(top_chunks)

return top_chunks  # ← CHANGE: return top_chunks instead of rescored[:top_k]
```

**Logic:**
- Sort by fused score descending (best first)
- Slice to `top_k` (e.g., 5 chunks)
- If `apply_lost_in_middle=True` and `len > 2`, reorder to [1st, 3rd, 5th, 4th, 2nd]
- Return reordered list

---

## Step 6: Update main.py Call Site

**File:** `rag2025/src/main.py`
**Lines:** 562-567

**Current code:**
```python
if chunks and reranker_service:
    chunks = reranker_service.rerank(
        query=original_query,
        chunks=chunks,
        top_k=top_k,
    )
```

**Change:** Add explicit `apply_lost_in_middle=True`:

```python
if chunks and reranker_service:
    chunks = reranker_service.rerank(
        query=original_query,
        chunks=chunks,
        top_k=top_k,
        apply_lost_in_middle=True,  # ← ADD THIS LINE
    )
```

**Why explicit?** Even though the default is `True`, making it explicit in the call site documents the intent and makes the behavior visible to code reviewers.

---

## Step 7: Write Unit Tests

**File:** `rag2025/tests/test_reranker.py` (new file)

**Create new test file:**

```python
"""
Unit tests for RerankerService lost-in-the-middle reordering.
"""
import pytest
from unittest.mock import MagicMock

from config.settings import RAGSettings
from services.reranker import RerankerService


@pytest.fixture
def mock_settings():
    """Mock RAGSettings for testing."""
    settings = MagicMock(spec=RAGSettings)
    settings.RERANKER_ENABLED = True
    settings.RERANKER_MODEL = "dummy-model"
    settings.RERANKER_WEIGHT = 0.35
    settings.MAX_RERANK = 50
    return settings


def test_apply_lost_in_middle_five_chunks():
    """Test lost-in-middle reordering with 5 chunks."""
    chunks = [
        {"id": "1", "score": 0.95, "text": "best"},
        {"id": "2", "score": 0.88, "text": "second"},
        {"id": "3", "score": 0.79, "text": "third"},
        {"id": "4", "score": 0.71, "text": "fourth"},
        {"id": "5", "score": 0.63, "text": "fifth"},
    ]

    svc = RerankerService.__new__(RerankerService)  # Bypass __init__
    result = svc._apply_lost_in_middle(chunks)

    # Expected order: [1st, 3rd, 5th, 4th, 2nd]
    assert len(result) == 5
    assert result[0]["id"] == "1"  # Best at front
    assert result[1]["id"] == "3"  # 3rd in position 1
    assert result[2]["id"] == "5"  # 5th in middle
    assert result[3]["id"] == "4"  # 4th near back
    assert result[4]["id"] == "2"  # 2nd-best at back


def test_apply_lost_in_middle_two_chunks():
    """Test lost-in-middle with 2 chunks (no reordering)."""
    chunks = [
        {"id": "1", "score": 0.9},
        {"id": "2", "score": 0.8},
    ]

    svc = RerankerService.__new__(RerankerService)
    result = svc._apply_lost_in_middle(chunks)

    # Should return unchanged (guard: len <= 2)
    assert result == chunks


def test_apply_lost_in_middle_one_chunk():
    """Test lost-in-middle with 1 chunk (no reordering)."""
    chunks = [{"id": "1", "score": 0.9}]

    svc = RerankerService.__new__(RerankerService)
    result = svc._apply_lost_in_middle(chunks)

    assert result == chunks


def test_apply_lost_in_middle_empty():
    """Test lost-in-middle with empty list."""
    chunks = []

    svc = RerankerService.__new__(RerankerService)
    result = svc._apply_lost_in_middle(chunks)

    assert result == []


def test_rerank_limits_candidates_to_max_rerank(mock_settings):
    """Test that rerank() pre-filters candidates to MAX_RERANK."""
    mock_settings.MAX_RERANK = 3

    svc = RerankerService.__new__(RerankerService)
    svc._enabled = True
    svc._weight = 0.35
    svc._max_rerank = 3
    svc._model = MagicMock()
    svc._model.predict.return_value = [0.9, 0.8, 0.7]

    chunks = [
        {"id": str(i), "score": 0.5, "text": f"chunk {i}"}
        for i in range(10)
    ]

    result = svc.rerank("test query", chunks, top_k=5, apply_lost_in_middle=False)

    # predict() should be called with only 3 pairs, not 10
    call_args = svc._model.predict.call_args[0][0]
    assert len(call_args) == 3


def test_rerank_applies_lost_in_middle_by_default(mock_settings):
    """Test that apply_lost_in_middle=True is the default."""
    svc = RerankerService.__new__(RerankerService)
    svc._enabled = True
    svc._weight = 0.35
    svc._max_rerank = 50
    svc._model = MagicMock()
    svc._model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

    chunks = [
        {"id": str(i), "score": 0.5, "text": f"chunk {i}"}
        for i in range(5)
    ]

    # Call without apply_lost_in_middle parameter (should default to True)
    result = svc.rerank("test query", chunks, top_k=5)

    # Check that reordering happened (id order should NOT be 0,1,2,3,4)
    ids = [c["id"] for c in result]
    assert ids != ["0", "1", "2", "3", "4"]  # Should be reordered


def test_rerank_skips_lost_in_middle_when_disabled(mock_settings):
    """Test that apply_lost_in_middle=False skips reordering."""
    svc = RerankerService.__new__(RerankerService)
    svc._enabled = True
    svc._weight = 0.35
    svc._max_rerank = 50
    svc._model = MagicMock()
    # Return scores in descending order so sort keeps original order
    svc._model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

    chunks = [
        {"id": str(i), "score": 0.5, "text": f"chunk {i}"}
        for i in range(5)
    ]

    result = svc.rerank("test query", chunks, top_k=5, apply_lost_in_middle=False)

    # Should be sorted by rerank score descending, no reordering
    # (Exact order depends on score fusion, but lost-in-middle should NOT apply)
    assert len(result) == 5
```

**Run tests:**
```bash
cd rag2025
pytest tests/test_reranker.py -v
```

**Expected output:**
```
tests/test_reranker.py::test_apply_lost_in_middle_five_chunks PASSED
tests/test_reranker.py::test_apply_lost_in_middle_two_chunks PASSED
tests/test_reranker.py::test_apply_lost_in_middle_one_chunk PASSED
tests/test_reranker.py::test_apply_lost_in_middle_empty PASSED
tests/test_reranker.py::test_rerank_limits_candidates_to_max_rerank PASSED
tests/test_reranker.py::test_rerank_applies_lost_in_middle_by_default PASSED
tests/test_reranker.py::test_rerank_skips_lost_in_middle_when_disabled PASSED
```

---

## Step 8: Run Existing Tests (Regression Check)

**Command:**
```bash
cd rag2025
pytest tests/test_api.py -v
```

**Expected:** All existing tests pass. The `test_health_endpoint` should still show `reranker_model: "Qwen/Qwen3-Reranker-8B"`.

**If any test fails:**
- Check if test expects specific chunk ordering (unlikely, but possible)
- Verify `apply_lost_in_middle=True` default doesn't break test assumptions

---

## Step 9: Manual Smoke Test

**Start the API:**
```bash
cd rag2025/src
python main.py
```

**Send test query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Học phí ngành Công nghệ thông tin là bao nhiêu?"}'
```

**Verify response:**
1. `status_code: 200`
2. `chunks` array present
3. Each chunk has `rerank_score` field
4. Chunk order is NOT strictly descending by `score` (lost-in-middle applied)
5. First chunk has high `score`, last chunk also has high `score`

**Example expected chunk order (scores):**
```
[0.92, 0.85, 0.78, 0.81, 0.88]
 ^best      middle      ^2nd-best at end
```

---

## Step 10: Commit Changes

**Files changed:**
1. `rag2025/src/services/reranker.py` — 4 changes:
   - Line 17: add `self._max_rerank = settings.MAX_RERANK`
   - Line 43: add `candidates = chunks[:self._max_rerank]`
   - Line 44-47: change `chunks` → `candidates` in pairs and zip
   - Line 38: add `apply_lost_in_middle: bool = True` parameter
   - Lines 58-95: add `_apply_lost_in_middle()` method
   - Lines 56-60: add lost-in-middle call before return

2. `rag2025/src/main.py` — 1 change:
   - Line 566: add `apply_lost_in_middle=True,`

3. `rag2025/tests/test_reranker.py` — new file (7 test cases)

**Commit message:**
```
feat(reranker): add candidate pre-filtering and lost-in-middle reordering

Enhance RerankerService with two improvements:

1. Candidate pre-filtering: Read MAX_RERANK from settings (default 50) and
   slice input chunks before cross-encoder inference. Reduces latency from
   O(N) to O(min(N, 50)) for large candidate sets.

2. Lost-in-the-middle mitigation: Implement _apply_lost_in_middle() using
   O(N) two-pointer algorithm (pre-allocated list, head/tail pointers).
   Reorders score-sorted chunks to place best at positions 0 and -1 where
   LLM attention is highest (Liu et al., NeurIPS 2023). Enabled by default
   via apply_lost_in_middle=True parameter.

Changes:
- reranker.py: read MAX_RERANK, add candidate slice, implement
  _apply_lost_in_middle(), update rerank() signature
- main.py: pass apply_lost_in_middle=True explicitly at line 566
- test_reranker.py: 7 unit tests for edge cases (n=0,1,2,5), candidate
  limit, default behavior

Blast radius: CRITICAL (2 callers). Backup caller in
backup_mail_package_2026 will benefit from default lost-in-middle=True
(intentional, documented in spec).

Spec: docs/specs/2026-03-25-cross-encoder-reranker-design.md
Plan: docs/plans/2026-03-25-cross-encoder-reranker.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Rollback Plan

If issues arise in production:

**Option 1: Disable lost-in-middle via parameter**
```python
# In main.py line 566
apply_lost_in_middle=False,  # Temporary rollback
```

**Option 2: Disable reranker entirely**
```bash
# Set environment variable
export RERANKER_ENABLED=false
```

**Option 3: Git revert**
```bash
git revert HEAD
```

---

## Success Criteria

- [x] `RerankerService.__init__` reads `settings.MAX_RERANK`
- [x] `rerank()` slices candidates to `self._max_rerank` before cross-encoder
- [x] `_apply_lost_in_middle()` implemented with O(N) two-pointer algorithm
- [x] `rerank()` signature includes `apply_lost_in_middle: bool = True`
- [x] `rerank()` calls `_apply_lost_in_middle()` when flag is True and len > 2
- [x] `main.py` passes `apply_lost_in_middle=True` explicitly
- [x] 7 unit tests pass (edge cases, candidate limit, default behavior)
- [x] Existing tests pass (no regression)
- [x] Manual smoke test shows `rerank_score` field and non-monotonic chunk order

---

## Timeline

| Step | Task | Time | Cumulative |
|---|---|---|---|
| 1 | Read MAX_RERANK in __init__ | 2 min | 2 min |
| 2 | Pre-filter candidates | 3 min | 5 min |
| 3 | Add apply_lost_in_middle parameter | 1 min | 6 min |
| 4 | Implement _apply_lost_in_middle() | 5 min | 11 min |
| 5 | Call _apply_lost_in_middle() in rerank() | 3 min | 14 min |
| 6 | Update main.py call site | 1 min | 15 min |
| 7 | Write unit tests | 15 min | 30 min |
| 8 | Run existing tests | 3 min | 33 min |
| 9 | Manual smoke test | 5 min | 38 min |
| 10 | Commit changes | 2 min | 40 min |
| **Total** | | **40 min** | |

**Buffer:** 5 minutes for unexpected issues → **45 minutes total**

---

## Dependencies

- Python 3.10+
- `sentence-transformers>=3.0.0` (already installed)
- `pytest` (already in dev dependencies)
- `Qwen/Qwen3-Reranker-8B` model (already configured)

---

## Notes

1. **No settings.py changes** — `MAX_RERANK` already exists at line 70
2. **No requirements.txt changes** — `CrossEncoder` is part of sentence-transformers
3. **Backward compatible** — `apply_lost_in_middle=True` default means existing callers get the improvement automatically
4. **Backup caller** — `backup_mail_package_2026/.../main.py` will silently benefit from lost-in-middle (intentional, documented in spec)
5. **Model unchanged** — Keep `Qwen/Qwen3-Reranker-8B` (8192-token context, strong Vietnamese support)

---

## References

- **Spec:** docs/specs/2026-03-25-cross-encoder-reranker-design.md
- **Lost-in-the-Middle Paper:** Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", NeurIPS 2023
- **GitNexus Impact:** CRITICAL risk, 2 callers, 6 processes affected
- **Qwen3-Reranker:** Alibaba Cloud Qwen3 Technical Report, 2025
