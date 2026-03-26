"""
API tests aligned with current FastAPI contract.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Fixture for FastAPI test client."""
    os.environ.setdefault("ADMIN_API_TOKEN", "test-admin-token")
    os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from main import app

    return TestClient(app)


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


def test_unified_query_validation(client):
    response = client.post("/v2/query", json={"query": "test", "top_k": 0})
    assert response.status_code == 422

    response = client.post("/v2/query", json={"query": "", "top_k": 5})
    assert response.status_code == 422


def test_graph_update_requires_admin_token(client):
    response = client.post("/v2/graph/update", json={"chunks": []})
    assert response.status_code == 403


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
