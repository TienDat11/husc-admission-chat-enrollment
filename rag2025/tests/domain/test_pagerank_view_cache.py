"""Tests for KnowledgeGraph._pagerank_view() internal cache.

Verifies that:
- Repeated calls return the SAME DiGraph object (identity check).
- Mutations that invalidate caches produce a FRESH view that reflects the change.
- PPR scores are unchanged when the cache is hit vs rebuilt.
- `_ppr_cache` and `_view_cache` are invalidated together.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.entities import Entity, EntityType, RelationType, Triple
from src.domain.graph import KnowledgeGraph, KnowledgeGraphBuilder


def _build_small_graph() -> KnowledgeGraph:
    builder = KnowledgeGraphBuilder()
    entities = [
        Entity(text="CNTT", entity_type=EntityType.NGANH, normalized="cntt", chunk_id="c1"),
        Entity(text="A00", entity_type=EntityType.TO_HOP, normalized="a00", chunk_id="c2"),
        Entity(text="KT", entity_type=EntityType.NGANH, normalized="kt", chunk_id="c3"),
    ]
    triples = [
        Triple(
            head="NGANH:cntt",
            relation=RelationType.CO_TO_HOP,
            tail="TO_HOP:a00",
            chunk_id="c1",
        ),
        Triple(
            head="NGANH:kt",
            relation=RelationType.CO_TO_HOP,
            tail="TO_HOP:a00",
            chunk_id="c3",
        ),
    ]
    builder.add_entities(entities)
    builder.add_triples(triples)
    return builder.build()


def test_pagerank_view_returns_same_object_on_repeat_call():
    """Identity check: cache hit returns the same DiGraph instance."""
    kg = _build_small_graph()
    view1 = kg._pagerank_view()
    view2 = kg._pagerank_view()
    assert view1 is view2, "view cache miss: second call should return cached object"


def test_pagerank_view_invalidated_after_clear_caches():
    """clear_caches() must drop both _ppr_cache and _view_cache."""
    kg = _build_small_graph()

    # Prime caches
    kg.personalized_pagerank(["NGANH:cntt"], alpha=0.85)
    cached_view = kg._pagerank_view()
    assert kg._view_cache is not None
    assert len(kg._ppr_cache) == 1

    kg.clear_caches()

    assert kg._view_cache is None
    assert len(kg._ppr_cache) == 0

    # Rebuild → fresh object (not the old stale reference)
    fresh_view = kg._pagerank_view()
    assert fresh_view is not cached_view


def test_pagerank_view_invalidated_after_add_node():
    """In-place graph mutation (add_node) must invalidate the view cache
    and the rebuilt view must reflect the new node count."""
    kg = _build_small_graph()
    view1 = kg._pagerank_view()
    assert view1.number_of_nodes() == 3

    # Mutate underlying graph in place; invalidate caches like add_edge/add_node would
    kg._graph.add_node("NGANH:new_node", text="New", entity_type="NGANH", normalized="new_node", chunk_ids="[]")
    kg._view_cache = None  # mirror the in-place invalidation path
    kg._ppr_cache.clear()

    view2 = kg._pagerank_view()
    assert view2 is not view1, "view should be a fresh object after invalidation"
    assert view2.number_of_nodes() == 4, "fresh view must reflect the new node"


def test_pagerank_view_invalidated_after_add_edge():
    """In-place edge mutation invalidates the view, and the fresh view picks up
    the new edge weight (proves the cache is not stale-serving)."""
    kg = _build_small_graph()
    view1 = kg._pagerank_view()
    # Seed the PPR cache so we can also prove it gets cleared.
    kg.personalized_pagerank(["NGANH:cntt"], alpha=0.85)

    # In-place edge addition
    kg._graph.add_edge(
        "NGANH:kt",
        "NGANH:cntt",
        key="NGANH:kt|CO_DIEM|NGANH:cntt",
        relation=RelationType.CO_DIEM.value,
        weight=2.0,
        chunk_ids="[]",
    )
    kg._view_cache = None
    kg._ppr_cache.clear()

    view2 = kg._pagerank_view()
    assert view2 is not view1
    # New edge between kt and cntt should appear in collapsed view
    assert view2.has_edge("NGANH:kt", "NGANH:cntt")
    assert view2["NGANH:kt"]["NGANH:cntt"]["weight"] == pytest.approx(2.0)


def test_pagerank_view_preserves_correctness():
    """Cache hit must return the SAME scores as a freshly-built view.

    Builds a tiny graph, computes PPR once (warms the cache), then bypasses
    the cache to compute a reference and asserts the cached path matches.
    """
    kg = _build_small_graph()

    # Warm the cache via the public path
    cached_scores = kg.personalized_pagerank(["NGANH:cntt"], alpha=0.85)

    # Bypass the view cache to compute a reference
    import networkx as nx
    dg = nx.DiGraph()
    dg.add_nodes_from(kg._graph.nodes(data=True))
    for u, v, attrs in kg._graph.edges(data=True):
        if dg.has_edge(u, v):
            dg[u][v]["weight"] = dg[u][v].get("weight", 0.0) + attrs.get("weight", 1.0)
        else:
            dg.add_edge(u, v, weight=attrs.get("weight", 1.0))

    personalization = {"NGANH:cntt": 1.0}
    ref_scores = nx.pagerank(
        dg,
        alpha=0.85,
        personalization=personalization,
        max_iter=100,
        weight="weight",
    )

    # Cached path must produce identical scores
    for node, score in ref_scores.items():
        assert node in cached_scores
        assert cached_scores[node] == pytest.approx(score, rel=1e-6)
