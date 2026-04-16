"""
API tests aligned with current FastAPI contract.
"""
import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Fixture for FastAPI test client."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "test-admin-token")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("RAMCLOUDS_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("ZAI_API_KEY", "")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import main
    importlib.reload(main)

    return TestClient(main.app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RAG API 2025"
    assert "features" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "lancedb_connected" in data
    assert "embedding_model" in data
    assert "reranker_model" in data


def test_query_endpoint_schema_when_available(client):
    payload = {"query": "What is the admission deadline?", "force_rag_only": True}
    response = client.post("/query", json=payload)

    if response.status_code == 200:
        data = response.json()
        assert "original_query" in data
        assert "enhanced_query" in data
        assert "query_type" in data
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "provider" in data
    else:
        assert response.status_code == 503


def test_query_endpoint_validation(client):
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422

    response = client.post("/query", json={"query": "a" * 1001})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "metrics" in data
    assert "query_total" in data["metrics"]
    assert "query_low_groundedness" in data["metrics"]


def test_query_pii_guardrail_block(client):
    payload = {"query": "Số CCCD của tôi là 012345678901, cho tôi hỏi học phí CNTT", "force_rag_only": True}
    response = client.post("/query", json=payload)

    if response.status_code == 200:
        data = response.json()
        assert data["status_code"] == "SENSITIVE_PII_DETECTED"
        assert data["pii_detected"] is True
        assert data["trace_id"]
        assert "groundedness_score" in data
    else:
        assert response.status_code == 503


def test_guardrail_service_detects_pii(client):
    import main
    from services.guardrail import GuardrailService

    guardrail = GuardrailService(main.settings)
    decision = asyncio.run(guardrail.precheck("Số CCCD của tôi là 012345678901"))
    assert decision.internal_code == "SENSITIVE_PII_DETECTED"
    assert decision.pii_detected is True


def test_query_response_contains_trace_and_groundedness(client):
    payload = {"query": "What is the admission deadline?", "force_rag_only": True}
    response = client.post("/query", json=payload)

    if response.status_code == 200:
        data = response.json()
        assert "trace_id" in data
        assert "groundedness_score" in data
    else:
        assert response.status_code == 503


def test_unified_query_trace_id_on_503(client):
    payload = {"query": "test", "top_k": 5}
    response = client.post("/v2/query", json=payload)
    assert response.status_code == 503


def test_docs_endpoint(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    assert "/query" in schema["paths"]
    assert "/v2/query" in schema["paths"]


def test_startup_degraded_mode_still_serves_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/query" in schema["paths"]
