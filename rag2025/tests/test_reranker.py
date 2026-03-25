"""
Unit tests for RerankerService._apply_lost_in_middle() and candidate pre-filtering.

Run with:
    cd rag2025
    python -m pytest tests/test_reranker.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src/ to path so that `services.reranker` is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.reranker import RerankerService


def _make_service(max_rerank: int = 50, weight: float = 0.35) -> RerankerService:
    """Build a RerankerService with a mocked CrossEncoder (bypasses model download)."""
    svc = RerankerService.__new__(RerankerService)
    svc._enabled = True
    svc._model_name = "dummy"
    svc._weight = weight
    svc._max_rerank = max_rerank
    svc._model = MagicMock()
    return svc


def _chunks(n: int) -> list:
    """Create n descending-scored dummy chunks: score 1.0, 0.9, 0.8, …"""
    return [
        {"id": str(i), "score": round(1.0 - i * 0.1, 2), "text": f"chunk {i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _apply_lost_in_middle — edge cases
# ---------------------------------------------------------------------------


def test_apply_lost_in_middle_empty():
    """n=0: should return empty list unchanged."""
    svc = RerankerService.__new__(RerankerService)
    result = svc._apply_lost_in_middle([])
    assert result == []


def test_apply_lost_in_middle_one():
    """n=1: should return single-item list unchanged."""
    svc = RerankerService.__new__(RerankerService)
    chunks = [{"id": "0", "score": 0.9}]
    result = svc._apply_lost_in_middle(chunks)
    assert result == chunks


def test_apply_lost_in_middle_two():
    """n=2: guard kicks in (len <= 2), returns list unchanged."""
    svc = RerankerService.__new__(RerankerService)
    chunks = [{"id": "0", "score": 0.9}, {"id": "1", "score": 0.8}]
    result = svc._apply_lost_in_middle(chunks)
    assert result == chunks
    assert result[0]["id"] == "0"
    assert result[1]["id"] == "1"


def test_apply_lost_in_middle_five():
    """n=5: verify exact boundary placement.

    Input ranked [1st, 2nd, 3rd, 4th, 5th]:
    Expected output: [1st, 3rd, 5th, 4th, 2nd]
      - index 0 = 1st (best)
      - index 4 = 2nd (second-best)
      - index 1 = 3rd
      - index 3 = 4th
      - index 2 = 5th (worst in middle)
    """
    svc = RerankerService.__new__(RerankerService)
    chunks = _chunks(5)  # ids "0".."4", scores 1.0..0.6

    result = svc._apply_lost_in_middle(chunks)

    assert len(result) == 5
    assert result[0]["id"] == "0"  # 1st → front
    assert result[4]["id"] == "1"  # 2nd → back
    assert result[1]["id"] == "2"  # 3rd → 2nd from front
    assert result[3]["id"] == "3"  # 4th → 2nd from back
    assert result[2]["id"] == "4"  # 5th → middle


# ---------------------------------------------------------------------------
# rerank() — candidate pre-filtering
# ---------------------------------------------------------------------------


def test_max_rerank_limits_candidates():
    """predict() must be called with at most MAX_RERANK pairs."""
    svc = _make_service(max_rerank=3)
    svc._model.predict.return_value = [0.9, 0.8, 0.7]

    chunks = _chunks(10)  # 10 chunks; only 3 should reach predict()
    svc.rerank("query", chunks, top_k=5, apply_lost_in_middle=False)

    pairs_passed = svc._model.predict.call_args[0][0]
    assert len(pairs_passed) == 3


# ---------------------------------------------------------------------------
# rerank() — apply_lost_in_middle behaviour
# ---------------------------------------------------------------------------


def test_lost_in_middle_default_true():
    """Calling rerank() without apply_lost_in_middle kwarg must reorder output.

    With 5 input chunks and scores [0.9, 0.8, 0.7, 0.6, 0.5] from predict(),
    a plain score-sort would yield ids in order 0,1,2,3,4 (since fused scores
    keep the same relative order). After lost-in-middle the order must differ.
    """
    svc = _make_service()
    # predict returns descending scores → sort preserves input order
    svc._model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

    chunks = _chunks(5)
    result = svc.rerank("query", chunks, top_k=5)  # apply_lost_in_middle=True by default

    ids = [c["id"] for c in result]
    # Strict sorted order would be ["0","1","2","3","4"] — must NOT be that
    assert ids != ["0", "1", "2", "3", "4"]
    # Best chunk must still be at the front
    assert result[0]["id"] == "0"
    # 2nd-best must be at the back
    assert result[-1]["id"] == "1"


def test_lost_in_middle_disabled():
    """apply_lost_in_middle=False must return score-sorted order, no reordering."""
    svc = _make_service()
    svc._model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

    chunks = _chunks(5)
    result = svc.rerank("query", chunks, top_k=5, apply_lost_in_middle=False)

    # Should be strictly descending by fused score (which mirrors predict order)
    scores = [c["score"] for c in result]
    assert scores == sorted(scores, reverse=True)
