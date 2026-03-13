"""
Domain Graph Logic – Scalable Knowledge Graph with PPR

Scalability design (10-20 year horizon):
- Graph stored as GraphML (XML-based, human-readable, version-agnostic)
- Entity index stored as JSON (schema-versioned, append-friendly)
- Incremental update: add new chunks WITHOUT rebuilding the whole graph
- Schema version in GraphML metadata: allows future migration scripts
- Nodes are content-addressed (entity_type:normalized) – deduplication is automatic
- Edges are keyed by (head, relation, tail) – same relation type never duplicates,
  different relations between the same pair co-exist as separate edges (MultiDiGraph)
- Edges accumulate weight across chunks – no duplicates, only weight++

PPR implementation:
- NetworkX `pagerank()` with personalization dict on a collapsed DiGraph view
- Convergence: max_iter=100, tol=1e-6 (NetworkX default)
- Score aggregation: sum of PPR scores for all nodes linked to a chunk_id

Infrastructure: NetworkX injected. No Qdrant/LLM here (pure domain logic).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    logger.warning("networkx not installed – graph operations disabled. pip install networkx")

from src.domain.entities import Entity, Triple

# Schema version embedded in graph metadata – increment on breaking schema changes
GRAPH_SCHEMA_VERSION = "2.0"


def _edge_key(head: str, relation: str, tail: str) -> str:
    """Deterministic edge key for MultiDiGraph – prevents duplicate edges."""
    return f"{head}|{relation}|{tail}"


class KnowledgeGraph:
    """Directed knowledge graph over admission domain entities.

    Uses MultiDiGraph so that two nodes can be connected by multiple edges
    as long as they carry different relation types (e.g., CO_TO_HOP vs CO_DIEM).
    Edge key = "{head}|{relation}|{tail}" – globally unique, collision-free.

    Node ID format: "{EntityType}:{normalized_name}"
    e.g., "NGANH:cong_nghe_thong_tin", "TO_HOP:a00"

    Node attributes:
        text         (str)  : human-readable surface form
        entity_type  (str)  : EntityType value
        normalized   (str)  : normalized key
        chunk_ids    (str)  : JSON-serialized list of source chunk IDs

    Edge attributes:
        relation     (str)  : RelationType value
        weight       (float): cumulative occurrence count
        chunk_ids    (str)  : JSON-serialized list of source chunk IDs

    Note: GraphML does not support list attributes → chunk_ids stored as JSON string.
    """

    def __init__(self, graph: "nx.MultiDiGraph") -> None:
        if not _NX_AVAILABLE:
            raise RuntimeError("networkx is required: pip install networkx")
        self._graph = graph
        self._ppr_cache: Dict[Tuple[Tuple[str, ...], float, int], Dict[str, float]] = {}

    @classmethod
    def empty(cls) -> "KnowledgeGraph":
        if not _NX_AVAILABLE:
            raise RuntimeError("networkx is required")
        g = nx.MultiDiGraph()
        g.graph["schema_version"] = GRAPH_SCHEMA_VERSION
        return cls(g)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def schema_version(self) -> str:
        return self._graph.graph.get("schema_version", "unknown")

    # ── Graph queries ────────────────────────────────────────────────────────

    def get_chunk_ids_for_entity(self, node_id: str) -> Set[str]:
        """Return chunk_ids that contributed to an entity node."""
        node_data = self._graph.nodes.get(node_id, {})
        raw = node_data.get("chunk_ids", "[]")
        try:
            return set(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return set()

    def neighbors(self, node_id: str) -> List[str]:
        """Direct successors of a node."""
        return list(self._graph.successors(node_id))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def _pagerank_view(self) -> "nx.DiGraph":
        """Collapse MultiDiGraph to DiGraph for PageRank (sum parallel edge weights)."""
        dg = nx.DiGraph()
        dg.add_nodes_from(self._graph.nodes(data=True))
        for u, v, attrs in self._graph.edges(data=True):
            if dg.has_edge(u, v):
                dg[u][v]["weight"] = dg[u][v].get("weight", 0.0) + attrs.get("weight", 1.0)
            else:
                dg.add_edge(u, v, weight=attrs.get("weight", 1.0))
        return dg

    # ── PPR ─────────────────────────────────────────────────────────────────

    def personalized_pagerank(
        self,
        seed_entities: List[str],
        alpha: float = 0.85,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """Personalized PageRank starting from seed entity nodes.

        PPR formula:
            π_S(v) = α · Σ_{u→v} π_S(u)/d_out(u)  +  (1-α) · [v∈S] / |S|

        Uses a collapsed DiGraph view (parallel edges summed) for PageRank.

        Args:
            seed_entities: Entity node_ids as personalization seeds.
            alpha: Damping factor (standard: 0.85).
            max_iter: Maximum power iteration steps.

        Returns:
            Dict[node_id → PPR score] (probability distribution, sums to 1).
        """
        if self._graph.number_of_nodes() == 0:
            return {}

        valid_seeds = sorted([e for e in seed_entities if e in self._graph])
        if not valid_seeds:
            logger.debug("No valid seed entities in graph – returning empty PPR")
            return {}

        cache_key = (tuple(valid_seeds), round(alpha, 4), int(max_iter))
        cached = self._ppr_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        personalization = {node: 1.0 / len(valid_seeds) for node in valid_seeds}
        view = self._pagerank_view()

        try:
            scores = nx.pagerank(
                view,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                weight="weight",
            )
            self._ppr_cache[cache_key] = scores
            return dict(scores)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR did not converge – falling back to empty scores")
            return {}

    def ppr_scores_by_chunk(
        self,
        seed_entities: List[str],
        alpha: float = 0.85,
    ) -> Dict[str, float]:
        """Aggregate PPR node scores to chunk level.

        For each chunk_id: score = sum of PPR scores of all nodes linked to it.
        Result is normalized to [0, 1].

        Args:
            seed_entities: Personalization seeds (entity node_ids).
            alpha: PPR damping factor.

        Returns:
            Dict[chunk_id → normalized aggregated PPR score].
        """
        node_scores = self.personalized_pagerank(seed_entities, alpha=alpha)
        if not node_scores:
            return {}

        chunk_scores: Dict[str, float] = {}
        for node_id, score in node_scores.items():
            for cid in self.get_chunk_ids_for_entity(node_id):
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + score

        max_score = max(chunk_scores.values(), default=1.0)
        if max_score > 0:
            return {k: v / max_score for k, v in chunk_scores.items()}
        return chunk_scores

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize graph to GraphML (schema-versioned)."""
        self._graph.graph["schema_version"] = GRAPH_SCHEMA_VERSION
        nx.write_graphml(self._graph, str(path))
        logger.info(
            f"Graph saved → {path}  "
            f"(nodes={self.node_count}, edges={self.edge_count}, "
            f"schema_version={GRAPH_SCHEMA_VERSION})"
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        """Load graph from GraphML file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not _NX_AVAILABLE:
            raise RuntimeError("networkx is required")
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        graph = nx.read_graphml(str(path))

        # Convert to MultiDiGraph if needed
        if not isinstance(graph, nx.MultiDiGraph):
            graph = nx.MultiDiGraph(graph)

        version = graph.graph.get("schema_version", "unknown")
        logger.info(
            f"Graph loaded ← {path}  "
            f"(nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, "
            f"schema_version={version})"
        )
        return cls(graph)

    def merge(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Merge another graph into this one (incremental update).

        Nodes with same ID: chunk_ids lists are merged.
        Edges with same (head, relation, tail) key: weights accumulate, chunk_ids merge.

        Returns:
            New merged KnowledgeGraph (self is not mutated).
        """
        merged_g = nx.MultiDiGraph(self._graph)

        for node_id, attrs in other._graph.nodes(data=True):
            if merged_g.has_node(node_id):
                existing = set(json.loads(merged_g.nodes[node_id].get("chunk_ids", "[]")))
                incoming = set(json.loads(attrs.get("chunk_ids", "[]")))
                merged_g.nodes[node_id]["chunk_ids"] = json.dumps(sorted(existing | incoming))
            else:
                merged_g.add_node(node_id, **attrs)

        for head, tail, key, attrs in other._graph.edges(data=True, keys=True):
            rel = attrs.get("relation", "LIEN_QUAN")
            edge_key = _edge_key(head, rel, tail)
            if merged_g.has_edge(head, tail, key=edge_key):
                ed = merged_g[head][tail][edge_key]
                ed["weight"] = ed.get("weight", 1.0) + attrs.get("weight", 1.0)
                existing = set(json.loads(ed.get("chunk_ids", "[]")))
                incoming = set(json.loads(attrs.get("chunk_ids", "[]")))
                ed["chunk_ids"] = json.dumps(sorted(existing | incoming))
            else:
                merged_g.add_edge(head, tail, key=edge_key, **attrs)

        merged_g.graph["schema_version"] = GRAPH_SCHEMA_VERSION
        return KnowledgeGraph(merged_g)

    def clear_caches(self) -> None:
        self._ppr_cache.clear()

    def stats(self) -> Dict:
        """Return graph statistics for monitoring."""
        degree_seq = sorted([d for _, d in self._graph.degree()], reverse=True)
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "schema_version": self.schema_version,
            "avg_degree": sum(degree_seq) / len(degree_seq) if degree_seq else 0.0,
            "max_degree": degree_seq[0] if degree_seq else 0,
        }


class KnowledgeGraphBuilder:
    """Stateful builder: accumulate entities/triples → build KnowledgeGraph.

    Supports incremental builds: start from an existing graph via `from_graph()`.

    Deduplication guarantees (Neo4j-inspired):
    - Node key = "{EntityType}:{normalized}" → same entity always maps to same node
    - Edge key = "{head}|{relation}|{tail}" → same triple never duplicates
    - When a duplicate is detected: chunk_ids merge, weight accumulates

    Usage (fresh build):
        builder = KnowledgeGraphBuilder()
        builder.add_entities(entities)
        builder.add_triples(triples)
        graph = builder.build()

    Usage (incremental update on existing graph):
        builder = KnowledgeGraphBuilder.from_graph(existing_graph)
        builder.add_entities(new_entities)
        builder.add_triples(new_triples)
        updated_graph = builder.build()
    """

    def __init__(self, base_graph: Optional["nx.MultiDiGraph"] = None) -> None:
        if not _NX_AVAILABLE:
            raise RuntimeError("networkx is required: pip install networkx")
        if base_graph is not None:
            self._graph: "nx.MultiDiGraph" = nx.MultiDiGraph(base_graph)
        else:
            self._graph = nx.MultiDiGraph()
            self._graph.graph["schema_version"] = GRAPH_SCHEMA_VERSION

    @classmethod
    def from_graph(cls, kg: KnowledgeGraph) -> "KnowledgeGraphBuilder":
        """Create builder pre-seeded with existing graph (incremental update)."""
        return cls(base_graph=kg._graph)

    def add_entities(self, entities: List[Entity]) -> None:
        """Add entity nodes. Same-ID entities merge their chunk_ids."""
        for entity in entities:
            node_id = entity.node_id
            if self._graph.has_node(node_id):
                existing = set(json.loads(self._graph.nodes[node_id].get("chunk_ids", "[]")))
                existing.add(entity.chunk_id)
                self._graph.nodes[node_id]["chunk_ids"] = json.dumps(sorted(existing))
            else:
                self._graph.add_node(
                    node_id,
                    text=entity.text,
                    entity_type=entity.entity_type.value,
                    normalized=entity.normalized,
                    chunk_ids=json.dumps([entity.chunk_id]),
                )

    def add_triples(self, triples: List[Triple]) -> None:
        """Add directed edges keyed by (head, relation, tail).

        Same (head, relation, tail) → weight accumulates, chunk_ids merge.
        Different relations between the same pair → separate edges.
        """
        for triple in triples:
            if not triple.head or not triple.tail:
                continue
            edge_key = _edge_key(triple.head, triple.relation.value, triple.tail)
            if self._graph.has_edge(triple.head, triple.tail, key=edge_key):
                ed = self._graph[triple.head][triple.tail][edge_key]
                ed["weight"] = ed.get("weight", 1.0) + triple.weight
                existing = set(json.loads(ed.get("chunk_ids", "[]")))
                existing.add(triple.chunk_id)
                ed["chunk_ids"] = json.dumps(sorted(existing))
            else:
                self._graph.add_edge(
                    triple.head,
                    triple.tail,
                    key=edge_key,
                    relation=triple.relation.value,
                    weight=triple.weight,
                    chunk_ids=json.dumps([triple.chunk_id]),
                )

    def build(self) -> KnowledgeGraph:
        """Finalize and return the KnowledgeGraph."""
        self._graph.graph["schema_version"] = GRAPH_SCHEMA_VERSION
        kg = KnowledgeGraph(self._graph)
        logger.info(
            f"KnowledgeGraph built: {kg.node_count} nodes, {kg.edge_count} edges"
        )
        return kg
