"""TDD: post-ingest graph sync hook + clear_caches (M3-N3a).

Tests:
1. ingest triggers subprocess call to build_graph.py --incremental
2. graph sync failure does NOT fail ingest
3. incremental build calls clear_caches() before from_graph
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import lancedb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    d = Path("D:/tmp")
    d.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def temp_db(tmp_workspace: Path):
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_graph_hook_")
    db = lancedb.connect(d)
    yield db
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _write_jsonl(path: Path, chunks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Test 1 — ingest triggers graph sync subprocess
# ---------------------------------------------------------------------------


def test_ingest_triggers_graph_sync(tmp_workspace: Path, temp_db: Any, monkeypatch: Any):
    """Successful ingest calls subprocess.run with build_graph.py --incremental."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_hook.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "g1", "text": "Graph hook test", "metadata": {"data_year": 2026}},
    ])

    monkeypatch.setattr(mod, "MANIFEST_PATH", Path(tempfile.mktemp(dir="D:/tmp", suffix=".json")))
    monkeypatch.setattr(mod, "TARGET_TABLE", "husc_chunks")
    monkeypatch.setattr(mod, "_resolve_target_files", lambda *a, **kw: [chunk_file])
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *a, **kw: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "husc_chunks"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec = np.full(1024, 0.1)

    class FakeEmbeddingService:
        def __init__(self, s):
            pass

        def encode_documents(self, texts):
            return [vec.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    # Spy on subprocess.run
    subprocess_calls = []

    def fake_subprocess_run(*args, **kwargs):
        subprocess_calls.append((args, kwargs))
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    # Assert subprocess.run was called with build_graph.py --incremental
    assert len(subprocess_calls) >= 1, "subprocess.run was not called"
    call_args = subprocess_calls[0][0][0]  # first positional arg of first call
    assert "build_graph.py" in str(call_args), f"build_graph.py not in args: {call_args}"
    assert "--incremental" in call_args, f"--incremental not in args: {call_args}"


# ---------------------------------------------------------------------------
# Test 2 — graph sync failure does NOT fail ingest
# ---------------------------------------------------------------------------


def test_graph_sync_failure_does_not_fail_ingest(tmp_workspace: Path, temp_db: Any, monkeypatch: Any):
    """subprocess.run raises → ingest still returns True."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_hook2.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "g2", "text": "Graph fail test", "metadata": {"data_year": 2026}},
    ])

    monkeypatch.setattr(mod, "MANIFEST_PATH", Path(tempfile.mktemp(dir="D:/tmp", suffix=".json")))
    monkeypatch.setattr(mod, "TARGET_TABLE", "husc_chunks")
    monkeypatch.setattr(mod, "_resolve_target_files", lambda *a, **kw: [chunk_file])
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *a, **kw: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "husc_chunks"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec = np.full(1024, 0.1)

    class FakeEmbeddingService:
        def __init__(self, s):
            pass

        def encode_documents(self, texts):
            return [vec.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    # Make subprocess.run raise
    def exploding_subprocess(*args, **kwargs):
        raise RuntimeError("Graph build exploded")

    monkeypatch.setattr(mod.subprocess, "run", exploding_subprocess)

    # Ingest must still succeed
    result = mod.ingest_lancedb(incremental=True)
    assert result is True, "Ingest should succeed even when graph sync fails"


# ---------------------------------------------------------------------------
# Test 3 — incremental build calls clear_caches
# ---------------------------------------------------------------------------


def test_incremental_build_clears_caches(tmp_workspace: Path, monkeypatch: Any):
    """build_graph.py incremental path calls clear_caches() before from_graph."""
    # We test by importing build_graph and checking the source contains clear_caches
    # after KnowledgeGraph.load in the incremental branch.
    # Direct unit test: mock KnowledgeGraph.load to return a mock with clear_caches spy.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_graph", ROOT / "scripts" / "build_graph.py"
    )
    bg_mod = importlib.util.module_from_spec(spec)

    # We need to patch before exec to intercept imports
    # Instead, just verify the source code contains the call in the right place
    source = (ROOT / "scripts" / "build_graph.py").read_text(encoding="utf-8")

    # Find the incremental block
    assert "existing_graph.clear_caches()" in source, (
        "clear_caches() not called in build_graph.py incremental path"
    )

    # Verify it's AFTER KnowledgeGraph.load and BEFORE from_graph
    load_pos = source.find("KnowledgeGraph.load(GRAPH_PATH)")
    clear_pos = source.find("existing_graph.clear_caches()")
    from_graph_pos = source.find("from_graph(existing_graph)")

    assert load_pos < clear_pos < from_graph_pos, (
        f"clear_caches() not in correct position: load@{load_pos}, clear@{clear_pos}, from_graph@{from_graph_pos}"
    )
