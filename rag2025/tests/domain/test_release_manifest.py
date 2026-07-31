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
