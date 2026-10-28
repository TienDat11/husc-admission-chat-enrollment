"""T11: /health must report the LOADED embedder's dimension and whether the
stored LanceDB table's vector column actually matches it.

Contract pinned:
  1. No .env override → RAGSettings defaults: EMBEDDING_MODEL =
     Qwen/Qwen3-Embedding-0.6B, EMBEDDING_DIM = 1024 (khớp .env production),
     RERANKER_ENABLED = False (8B không vừa VM 8GB).
  2. GET /health with a real LanceDB table (tmp_path) + fake embedder:
     embedding_dim comes from the service's `expected_dim`, and
     vector_dim_match compares it against the table schema's vector dim.
  3. Embedder dim 2560 against a 1024-dim table → vector_dim_match=False.

Seam: HTTP through the real FastAPI app, fake embedder service, real LanceDB
table under tmp_path. No LLM provider, no model download.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. Defaults
# ─────────────────────────────────────────────────────────────────────────────


def test_defaults_match_production_env() -> None:
    """Class-level defaults: 0.6B/1024 (khớp .env production) + reranker off.

    The CLASS field default is the no-.env startup value — a .env on disk or
    in os.environ overrides it, so this must not read the file.
    """
    import config.settings as cfg

    fields = cfg.RAGSettings.model_fields
    assert fields["EMBEDDING_MODEL"].default == "Qwen/Qwen3-Embedding-0.6B"
    assert fields["EMBEDDING_DIM"].default == 1024
    assert fields["RERANKER_ENABLED"].default is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. /health embedding_dim + vector_dim_match
# ─────────────────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Duck-typed stand-in: exposes `expected_dim` like EmbeddingService."""

    def __init__(self, expected_dim: int) -> None:
        self.expected_dim = expected_dim
        self.model_name = "fake"


def _load_main(monkeypatch, tmp_path: Path, dim: int, vector_dim: int, table_rows: int = 3) -> Any:
    import lancedb
    import pyarrow as pa

    db_uri = str(tmp_path / "lancedb")
    table = "husc_chunks"
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )
    rows = [
        {"id": f"c{i}", "text": f"chunk {i}", "vector": [0.1] * vector_dim}
        for i in range(table_rows)
    ]
    db = lancedb.connect(db_uri)
    db.create_table(table, data=rows, schema=schema, mode="overwrite")

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.setenv("LANCEDB_URI", db_uri)
    monkeypatch.setenv("LANCEDB_TABLE", table)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    spec = importlib.util.spec_from_file_location("t11_test_main", SRC / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Real retriever against the tmp table (real adapter + real schema read).
    from src.services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig

    module.lancedb_retriever_service = LanceDBRetriever(
        LanceDBRetrieverConfig(uri=db_uri, table_name=table, embedding_dim=dim)
    )
    module.embedding_service = _FakeEmbedder(expected_dim=dim)
    module.reranker_service = None
    module.unified_pipeline = None
    module.hybrid_search_service = None
    module.app.state.warmup_stage = "warming"
    return module


def test_health_reports_dim_match(tmp_path: Path, monkeypatch) -> None:
    main = _load_main(monkeypatch, tmp_path, dim=1024, vector_dim=1024)
    body = TestClient(main.app).get("/health").json()

    assert body["embedding_dim"] == 1024
    assert body["vector_dim_match"] is True
    assert body["lancedb_connected"] is True


def test_health_reports_dim_mismatch(tmp_path: Path, monkeypatch) -> None:
    """Embedder dim 2560 against a 1024-dim table → NOT a match."""
    main = _load_main(monkeypatch, tmp_path, dim=2560, vector_dim=1024)
    body = TestClient(main.app).get("/health").json()

    assert body["embedding_dim"] == 2560
    assert body["vector_dim_match"] is False
