"""Tests for S15.5 — Precision cutoff (ADR-E + MUST-FIX#4 + CF-1/CF-2 + CF-5).

Contract pinned for the pure helper `apply_rerank_cutoff(docs, ratio, min_keep)`:
  - Drop docs whose `.score < ratio * top_score`.
  - Always keep at least `min_keep` docs (fall back to highest-scored).
  - NEVER drop a doc whose `metadata.get("booster_injected")` is truthy.
  - NEVER drop a doc whose `metadata.get("graph_injected")` is truthy.
  - `ratio <= 0` → cutoff disabled, returns all docs.
  - `top_score == 0` → return all docs (no division-by-zero, no aggressive
    drop when nothing has meaningful score).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.graphrag_retriever import apply_rerank_cutoff  # noqa: E402
from src.services.lancedb_retrieval import RetrievedDocument  # noqa: E402


def _doc(cid: str, score: float, metadata: Optional[dict] = None) -> RetrievedDocument:
    return RetrievedDocument(
        text=f"t-{cid}",
        source=cid,
        chunk_id=cid,
        metadata=metadata or {},
        score=score,
        point_id=cid,
    )


def _ids(docs: List[RetrievedDocument]) -> List[str]:
    return [d.chunk_id for d in docs]


# ─────────────────────────────────────────────────────────────────────────────
# Core behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_low_score_docs_are_dropped():
    """Docs whose score < ratio * top_score get filtered out."""
    docs = [
        _doc("c1", 1.0),
        _doc("c2", 0.8),
        _doc("c3", 0.5),  # exactly 0.5 — keep (>= threshold)
        _doc("c4", 0.4),  # below 0.5 → drop
        _doc("c5", 0.1),
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.5, min_keep=2)
    ids = _ids(kept)
    assert "c1" in ids
    assert "c2" in ids
    assert "c3" in ids, f"score==threshold must be kept, got {ids}"
    assert "c4" not in ids
    assert "c5" not in ids


def test_min_keep_honored_when_many_below_threshold():
    """If cutoff would leave fewer than min_keep, fill from top of input."""
    docs = [
        _doc("c1", 1.0),
        _doc("c2", 0.1),   # below 0.5 threshold
        _doc("c3", 0.05),
        _doc("c4", 0.02),
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.5, min_keep=3)
    ids = _ids(kept)
    assert len(kept) >= 3, f"min_keep=3 must be honored, got {ids}"
    # The top-3 by score in original order should be kept.
    assert "c1" in ids
    assert "c2" in ids
    assert "c3" in ids


def test_ratio_zero_disables_cutoff():
    """ratio=0 → return all docs untouched."""
    docs = [_doc("c1", 1.0), _doc("c2", 0.01), _doc("c3", 0.0001)]
    kept = apply_rerank_cutoff(docs, ratio=0.0, min_keep=2)
    assert _ids(kept) == ["c1", "c2", "c3"]


def test_top_score_zero_keeps_all():
    """When the highest doc has score==0, cutoff would be 0 → keep all."""
    docs = [_doc("c1", 0.0), _doc("c2", 0.0), _doc("c3", 0.0)]
    kept = apply_rerank_cutoff(docs, ratio=0.5, min_keep=2)
    assert _ids(kept) == ["c1", "c2", "c3"]


def test_empty_docs_returns_empty_list():
    """Edge case: empty input → empty output."""
    kept = apply_rerank_cutoff([], ratio=0.5, min_keep=2)
    assert kept == []


# ─────────────────────────────────────────────────────────────────────────────
# Exemption — CF-1 / CF-2 (booster + graph_injected survivors)
# ─────────────────────────────────────────────────────────────────────────────


def test_booster_injected_chunk_survives_below_cutoff():
    """CF-2: a booster-tagged chunk MUST survive even below the cutoff."""
    docs = [
        _doc("c_top", 1.0),
        _doc("c_booster", 0.01, metadata={"booster_injected": True}),
        _doc("c_drop", 0.01),  # same low score — but no exemption → drop
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.5, min_keep=1)
    ids = _ids(kept)
    assert "c_top" in ids
    assert "c_booster" in ids, (
        f"booster_injected chunk must NEVER be dropped, got {ids}"
    )
    assert "c_drop" not in ids


def test_graph_injected_chunk_survives_below_cutoff():
    """CF-1: a graph_injected chunk MUST survive even below the cutoff.

    The whole S15.1 expander work is wasted if S15.5 then drops what we just
    fetched.
    """
    docs = [
        _doc("c_top", 1.0),
        _doc("c_graph", 0.01, metadata={"graph_injected": True}),
        _doc("c_drop", 0.01),
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.5, min_keep=1)
    ids = _ids(kept)
    assert "c_top" in ids
    assert "c_graph" in ids, (
        f"graph_injected chunk must NEVER be dropped, got {ids}"
    )
    assert "c_drop" not in ids


def test_both_markers_survive_together():
    """A doc with BOTH markers (defensive — shouldn't happen) still survives."""
    docs = [
        _doc("c_top", 1.0),
        _doc(
            "c_both",
            0.001,
            metadata={"booster_injected": True, "graph_injected": True},
        ),
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.99, min_keep=1)
    assert "c_both" in _ids(kept)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / robustness
# ─────────────────────────────────────────────────────────────────────────────


def test_negative_ratio_treated_as_disabled():
    """Defensive: negative ratio behaves like ratio=0 (no cutoff)."""
    docs = [_doc("c1", 1.0), _doc("c2", 0.001)]
    kept = apply_rerank_cutoff(docs, ratio=-1.0, min_keep=1)
    assert _ids(kept) == ["c1", "c2"]


def test_single_doc_always_kept():
    """A single-doc input survives any cutoff (min_keep=1 default behavior)."""
    docs = [_doc("c1", 0.5)]
    kept = apply_rerank_cutoff(docs, ratio=0.99, min_keep=1)
    assert _ids(kept) == ["c1"]
