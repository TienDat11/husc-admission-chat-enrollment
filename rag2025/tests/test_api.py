"""
Unit tests for FastAPI endpoints (Epic 3.3)
"""
import pytest
from fastapi.testclient import TestClient

# Note: This test requires the app to be running with initialized services
# For full integration testing, run after building index


@pytest.fixture
def client():
    """Fixture for FastAPI test client."""
    # Import here to avoid circular imports
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from main import app

    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "2025 RAG API"
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")

    # May return 503 if services not initialized
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_store_count" in data
        assert "embedding_model" in data
        assert "reranker_model" in data
    else:
        assert response.status_code == 503


def test_query_endpoint_schema(client):
    """Test query endpoint request/response schema."""
    payload = {
        "query": "What is the admission deadline?",
        "top_k": 3,
        "force_rag_only": True,
    }

    response = client.post("/query", json=payload)

    # May return 503 if services not initialized, 200 if initialized
    if response.status_code == 200:
        data = response.json()

        # Verify response schema
        assert "query" in data
        assert "results" in data
        assert "confidence" in data
        assert "routing_decision" in data
        assert "threshold" in data
        assert "total_results" in data

        # Verify query matches
        assert data["query"] == payload["query"]

        # Verify results structure
        if data["total_results"] > 0:
            result = data["results"][0]
            assert "doc_id" in result
            assert "chunk_id" in result
            assert "text" in result
            assert "score" in result
            assert "metadata" in result

    else:
        assert response.status_code in [503, 500]


def test_query_endpoint_validation(client):
    """Test query endpoint input validation."""
    # Empty query
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422  # Validation error

    # Invalid top_k
    response = client.post("/query", json={"query": "test", "top_k": 0})
    assert response.status_code == 422

    # top_k too large
    response = client.post("/query", json={"query": "test", "top_k": 100})
    assert response.status_code == 422


def test_query_endpoint_force_rag_only(client):
    """Test force_rag_only flag."""
    payload = {
        "query": "Test query",
        "top_k": 5,
        "force_rag_only": True,
    }

    response = client.post("/query", json=payload)

    if response.status_code == 200:
        data = response.json()
        # Should return RAG results without LLM fallback
        assert data["routing_decision"] in [
            "rag_direct",
            "rag_low_confidence",
            "fallback_disabled",
        ]


def test_docs_endpoint(client):
    """Test that OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test OpenAPI schema generation."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema

    # Verify /query endpoint in schema
    assert "/query" in schema["paths"]
    assert "post" in schema["paths"]["/query"]
