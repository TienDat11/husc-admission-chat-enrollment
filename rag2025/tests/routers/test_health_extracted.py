"""Tests: extracted HealthResponse model + GET /health from routers/health.

Verifies:
  1. HealthResponse model can be imported from routers.health.
  2. The router is mounted on the FastAPI app and responds at GET /health.
  3. The HealthResponse re-export from main works for backward compat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# 1. HealthResponse model import + shape contract
# ---------------------------------------------------------------------------


def test_health_response_model_has_all_fields() -> None:
    """Extracted HealthResponse must carry every field the FE depends on."""
    from routers.health import HealthResponse

    inst = HealthResponse(status="ok")
    assert inst.status == "ok"
    # Legacy fields
    assert inst.lancedb_connected is False
    assert inst.vectors_count == 0
    assert inst.collection == ""
    assert inst.embedding_model == ""
    assert inst.reranker_model == ""
    assert inst.metrics == {}
    # Warmup readiness fields
    assert inst.ready is False
    assert inst.stage == "warming"
    assert inst.services == {}


# ---------------------------------------------------------------------------
# 2. Router is mounted on the app
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Load main with env overrides and return TestClient."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    import importlib.util

    main_path = SRC / "main.py"
    assert main_path.is_file(), f"main.py not found at {main_path}"

    spec = importlib.util.spec_from_file_location("health_extracted_test", main_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return TestClient(module.app)


def test_health_endpoint_returns_200_with_model_fields(client: TestClient) -> None:
    """GET /health must respond with all expected fields."""
    res = client.get("/health")
    assert res.status_code == 200, res.text
    body = res.json()

    assert "status" in body
    assert "lancedb_connected" in body
    assert "vectors_count" in body
    assert "collection" in body
    assert "embedding_model" in body
    assert "reranker_model" in body
    assert "metrics" in body
    assert "ready" in body
    assert "stage" in body
    assert "services" in body


# ---------------------------------------------------------------------------
# 3. Backward-compat re-export from main
# ---------------------------------------------------------------------------


def test_health_response_re_exported_from_main() -> None:
    """from main import HealthResponse must work (P1 backward compat)."""
    import importlib.util

    main_path = SRC / "main.py"
    spec = importlib.util.spec_from_file_location("health_reexport_test", main_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    HR = getattr(module, "HealthResponse", None)
    assert HR is not None, "HealthResponse not re-exported from main"
    inst = HR(status="test")
    assert inst.status == "test"
