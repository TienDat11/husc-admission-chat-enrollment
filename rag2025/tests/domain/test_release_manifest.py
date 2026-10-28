"""TDD: rag2025/src/domain/release.py — release manifest + transition domain models.
M1-T1: manifest roundtrip is field-preserving; corpus_sha256 is order-stable.
M1-T7: VersionTransition refuses to be constructed without evidence_span.
(M1-T2, M1-T6 land later in this same file — see Slice 3.)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.release import (  # noqa: E402
    ReleaseManifest,
    ScopeLevel,
    TransitionKind,
    VersionTransition,
    compute_corpus_sha256,
)


def test_manifest_roundtrip():
    corpus_hash = compute_corpus_sha256(["aaa", "bbb", "ccc"])
    m = ReleaseManifest(
        release_id="R-2026-08-01",
        document_count=10,
        chunk_count=100,
        corpus_sha256=corpus_hash,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
    )
    d = m.to_dict()
    m2 = ReleaseManifest.from_dict(d)
    assert m2 == m

    h1 = compute_corpus_sha256(["aaa", "bbb", "ccc"])
    h2 = compute_corpus_sha256(["ccc", "aaa", "bbb"])
    assert h1 == h2 == corpus_hash


def test_transition_requires_evidence():
    with pytest.raises(ValueError):
        VersionTransition(
            transition_id="VT-0001",
            kind=TransitionKind.REPLACES,
            subject_doc_id="doc-new",
            object_doc_id="doc-old",
            effective_from=date(2026, 8, 1),
            evidence_doc_id="doc-new",
            evidence_span="",
            evidence_offset=(0, 0),
            detector="manual",
            detector_confidence=1.0,
            scope=ScopeLevel.husc,
        )


def test_atomic_pointer_swap(tmp_path):
    import json

    from src.infrastructure.canonical_store import publish_pointer

    pointer = tmp_path / "active_release.json"
    publish_pointer(pointer, "R-2026-07-01")
    original_bytes = pointer.read_bytes()

    tmp = pointer.with_suffix(pointer.suffix + ".tmp")
    tmp.write_text("{not valid json", encoding="utf-8")

    assert pointer.read_bytes() == original_bytes
    data = json.loads(pointer.read_text(encoding="utf-8"))
    assert data["release_id"] == "R-2026-07-01"

    publish_pointer(pointer, "R-2026-08-01")
    assert json.loads(pointer.read_text(encoding="utf-8"))["release_id"] == "R-2026-08-01"


def test_canonical_immutable(tmp_path):
    from src.domain.release import ReleaseManifest
    from src.infrastructure.canonical_store import mark_release_immutable, write_manifest

    release_dir = tmp_path / "R-2026-07-01"
    m = ReleaseManifest(release_id="R-2026-07-01")
    out = write_manifest(release_dir, m)
    mark_release_immutable(release_dir)
    with pytest.raises(PermissionError):
        out.write_text("tampered", encoding="utf-8")

# ---------------------------------------------------------------------------
# FIX-02 — the graph is part of the release unit, and the pin is a whole thing
# ---------------------------------------------------------------------------

def test_a_partial_graph_pin_is_refused():
    """A pin that names a path but no digest pins something unverifiable.

    The manifest refuses to exist in that state rather than storing a safety
    field that cannot be checked — the failure mode the dead ``graph_sha256``
    field was.
    """
    from src.domain.release import ReleaseManifest

    with pytest.raises(ValueError, match="complete or absent"):
        ReleaseManifest(
            release_id="R-1",
            graph_path="graph/knowledge_graph.graphml",
            graph_sha256="",
            graph_schema_version="2.0",
        )

    # Absent is legal: a release published before the graph was part of the unit.
    assert ReleaseManifest(release_id="R-1").pins_graph is False
    # Complete is legal.
    pinned = ReleaseManifest(
        release_id="R-1",
        graph_path="graph/knowledge_graph.graphml",
        graph_sha256="ab" * 32,
        graph_schema_version="2.0",
    )
    assert pinned.pins_graph is True

def test_a_graph_pin_that_escapes_its_release_is_refused(tmp_path):
    """The pin is resolved under its own release directory, or refused.

    A pin that resolved elsewhere would let a rollback serve another release's
    graph — the exact mixture the pin exists to prevent — so it raises rather
    than being clamped into the release.
    """
    from src.domain.release import ReleaseManifest

    for bad in ("../other/graph.graphml", "/abs/graph.graphml", "graph/../../x.graphml"):
        manifest = ReleaseManifest(
            release_id="R-1",
            graph_path=bad,
            graph_sha256="ab" * 32,
            graph_schema_version="2.0",
        )
        with pytest.raises(ValueError):
            manifest.pinned_graph_path(tmp_path / "R-1")

    good = ReleaseManifest(
        release_id="R-1",
        graph_path="graph/knowledge_graph.graphml",
        graph_sha256="ab" * 32,
        graph_schema_version="2.0",
    )
    resolved = good.pinned_graph_path(tmp_path / "R-1")
    assert resolved == (tmp_path / "R-1" / "graph" / "knowledge_graph.graphml").resolve()

def test_a_pin_round_trips_through_the_manifest_file(tmp_path):
    """Publish writes the pin; the manifest reads it back byte-identically."""
    import networkx as nx

    from src.domain.graph import GRAPH_SCHEMA_VERSION, KnowledgeGraph
    from src.domain.release import ReleaseManifest
    from src.infrastructure.canonical_store import read_manifest, write_manifest
    from src.infrastructure.graph_artifact import publish_release_graph

    release_dir = tmp_path / "R-2026-08-01"
    write_manifest(release_dir, ReleaseManifest(release_id="R-2026-08-01"))

    g = nx.MultiDiGraph()
    g.graph["schema_version"] = GRAPH_SCHEMA_VERSION
    g.add_node("NGANH:cntt", text="CNTT", entity_type="NGANH",
               normalized="cntt", chunk_ids='["c-1"]')
    build = tmp_path / "built.graphml"
    KnowledgeGraph(g).save(build)

    updated = publish_release_graph(release_dir, build)

    assert updated.pins_graph is True
    assert updated.graph_schema_version == GRAPH_SCHEMA_VERSION
    assert (release_dir / updated.graph_path).is_file()
    # The artifact in the release is the build, byte for byte.
    assert (release_dir / updated.graph_path).read_bytes() == build.read_bytes()

    # And the pin survives the manifest's own serialization.
    on_disk = read_manifest(release_dir)
    assert on_disk.graph_sha256 == updated.graph_sha256
    assert on_disk.graph_path == updated.graph_path
    assert on_disk.graph_schema_version == updated.graph_schema_version
