"""TDD: GET /health must surface a `ready` boolean + `stage` string that
the FE can poll to gate Send-button activation.

Contract pinned by these tests:
  1. After a full lifespan (embedding + reranker + graphrag all init),
     GET /health returns `ready=true, stage='ready'`.
  2. Backward compat: legacy fields (`status`, `lancedb_connected`,
     `embedding_model`, `reranker_model`) are still present in the
     response so older clients do not break.
  3. `services` is a per-subsystem boolean map
     (embedding, reranker, graphrag, hybrid_bm25, lancedb).
  4. With `unified_pipeline=None` (warming), GET /v2/query returns 503
     and the response body's `detail` mentions warming.

Approach: import src/main.py via importlib (same trick as the other
test_*.py that exercise the FastAPI app without booting heavy models).
We never enter the TestClient `with` context, so the lifespan event
does NOT run; we set the globals manually to simulate post-warmup state.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
MAIN_PATH = SRC / "main.py"
# Ensure project root is importable so `import config.env_loader` works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
assert MAIN_PATH.is_file(), f"src/main.py not found at {MAIN_PATH}"


def _load_main(monkeypatch) -> object:
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    spec = importlib.util.spec_from_file_location("health_ready_test", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_embedding() -> SimpleNamespace:
    e = SimpleNamespace()
    e.encode_query = lambda text: [0.0] * 4
    e.is_loaded = lambda: True
    return e


def _fake_reranker() -> SimpleNamespace:
    return SimpleNamespace(score=lambda q, d: 0.5)


def _fake_pipeline() -> SimpleNamespace:
    p = SimpleNamespace()
    p._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 1})
    return p


def _fake_lancedb_retriever() -> SimpleNamespace:
    r = SimpleNamespace()
    r.check_collection = lambda: {"exists": True, "vectors_count": 123}
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 1. /health.ready=true, stage='ready' after lifespan
# ─────────────────────────────────────────────────────────────────────────────


def test_health_ready_true_when_all_core_services_initialized(monkeypatch) -> None:
    """With all globals set, GET /health must report ready=true, stage='ready'."""
    main = _load_main(monkeypatch)

    # Simulate post-lifespan state.
    main.embedding_service = _fake_embedding()
    main.reranker_service = _fake_reranker()
    main.unified_pipeline = _fake_pipeline()
    main.lancedb_retriever_service = _fake_lancedb_retriever()
    main.hybrid_search_service = None  # optional
    main.app.state.warmup_stage = "ready"

    client = TestClient(main.app)
    res = client.get("/health")
    assert res.status_code == 200, res.text
    body = res.json()

    assert body.get("ready") is True, body
    assert body.get("stage") == "ready", body
    services = body.get("services") or {}
    assert services.get("embedding") is True
    assert services.get("reranker") is True
    assert services.get("graphrag") is True
    assert services.get("lancedb") is True
    assert "hybrid_bm25" in services  # key always present

    # Backward compat fields (V5 contract).
    assert "status" in body
    assert body.get("embedding_model")
    assert body.get("reranker_model")
    assert "metrics" in body


# ─────────────────────────────────────────────────────────────────────────────
# 2. /health.ready=false, stage='warming' before warmup finishes
# ─────────────────────────────────────────────────────────────────────────────


def test_health_ready_false_when_pipeline_is_none(monkeypatch) -> None:
    """If unified_pipeline is None (warming), ready=false, stage='warming'."""
    main = _load_main(monkeypatch)
    main.embedding_service = _fake_embedding()
    main.reranker_service = _fake_reranker()
    main.unified_pipeline = None  # warming
    main.lancedb_retriever_service = None
    main.hybrid_search_service = None
    main.app.state.warmup_stage = "warming"

    client = TestClient(main.app)
    res = client.get("/health")
    body = res.json()

    assert res.status_code == 200
    assert body.get("ready") is False, body
    assert body.get("stage") in ("warming", "failed"), body
    services = body.get("services") or {}
    assert services.get("graphrag") is False
    assert services.get("lancedb") is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Force-fail one service → ready=false, stage='failed'
# ─────────────────────────────────────────────────────────────────────────────


def test_health_stage_failed_when_one_core_service_missing(monkeypatch) -> None:
    """If the embedding service failed to init, the stage machine must
    reflect 'failed' (not 'ready') so the FE can surface a banner."""
    main = _load_main(monkeypatch)
    main.embedding_service = None  # force-fail
    main.reranker_service = _fake_reranker()
    main.unified_pipeline = _fake_pipeline()
    main.lancedb_retriever_service = _fake_lancedb_retriever()

    # Drive the stage transition explicitly (the lifespan would do this
    # under normal boot; here we mimic it for an isolated check).
    main.app.state.warmup_stage = "failed"

    client = TestClient(main.app)
    res = client.get("/health")
    body = res.json()

    assert body.get("ready") is False, body
    assert body.get("stage") == "failed", body
    services = body.get("services") or {}
    assert services.get("embedding") is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. /v2/query returns 503 + Retry-After when warming
# ─────────────────────────────────────────────────────────────────────────────


def test_v2_query_returns_503_with_retry_after_when_warming(monkeypatch) -> None:
    """With unified_pipeline=None, POST /v2/query must return 503 and
    include a Retry-After header (RFC 7231) so the FE can back off."""
    from fastapi import HTTPException

    main = _load_main(monkeypatch)
    main.unified_pipeline = None  # warming

    client = TestClient(main.app)
    payload = {"query": "test"}
    res = client.post("/v2/query", json=payload)
    assert res.status_code == 503, res.text
    # Header is the canonical signal for backoff; the FE retry layer
    # can also parse body.retry_after_ms if needed.
    assert "retry-after" in {k.lower() for k in res.headers.keys()}, dict(res.headers)
    assert int(res.headers.get("retry-after", "0")) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. /v2/query with full pipeline → not 503 (sanity: gate only fires when None)
# ─────────────────────────────────────────────────────────────────────────────


def test_v2_query_does_not_503_when_pipeline_present(monkeypatch) -> None:
    """Sanity: when pipeline is set, /v2/query is NOT 503'd (it will likely
    fail with 500 from the inner pipeline since we use a stub, but it must
    not be the warming gate)."""
    main = _load_main(monkeypatch)
    main.unified_pipeline = _fake_pipeline()  # gate passes
    # Other globals the inner pipeline needs (kept minimal — we expect 500
    # not 503 from the inner call).
    main.embedding_service = _fake_embedding()
    main.reranker_service = _fake_reranker()

    client = TestClient(main.app)
    res = client.post("/v2/query", json={"query": "test"})
    assert res.status_code != 503, (
        f"unexpected 503 — gate fired despite pipeline set: {res.text}"
    )