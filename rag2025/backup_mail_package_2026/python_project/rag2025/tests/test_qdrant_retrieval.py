"""
Unit tests for Phase 7: Qdrant Retrieval Layer

Tests cover:
- query_points() API compliance
- Error handling (fail-safe behavior)
- Payload validation
- Collection schema
- Multi-vector extensibility
"""
from unittest.mock import MagicMock, Mock, patch
from dataclasses import asdict

import numpy as np
import pytest

from services.qdrant_retrieval import (
    QdrantRetriever,
    QdrantRetrieverConfig,
    RetrievedDocument,
    RetrievalResult,
    RetrievalError,
    retrieve_with_qdrant,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def qdrant_config():
    """Fixture for Qdrant retriever config."""
    return QdrantRetrieverConfig(
        url="http://localhost:6333",
        api_key=None,
        collection_name="rag2025",
        embedding_dim=1024,
        default_top_k=5,
        timeout=30,
    )


@pytest.fixture
def sample_query_vector():
    """Fixture for sample query vector (1024-dim for BGE-M3)."""
    # Create normalized random vector
    vec = np.random.randn(1024).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


@pytest.fixture
def mock_query_response():
    """Fixture for mock Qdrant QueryResponse."""
    from qdrant_client.http.models import QueryResponse, ScoredPoint

    # Create mock points with proper payload structure
    points = []
    for i in range(3):
        point = Mock(spec=ScoredPoint)
        point.id = f"point_{i}"
        point.score = 0.9 - (i * 0.1)  # 0.9, 0.8, 0.7
        point.payload = {
            "text": f"This is chunk {i} with relevant content",
            "source": f"doc_{i}",
            "chunk_id": f"chunk_{i}",
            "metadata": {
                "title": f"Document {i}",
                "language": "en",
                "created_at": "2025-01-01",
            },
        }
        points.append(point)

    response = Mock(spec=QueryResponse)
    response.points = points
    return response


@pytest.fixture
def mock_qdrant_client():
    """Fixture for mock QdrantClient."""
    client = MagicMock()

    # Mock query_points response
    from qdrant_client.http.models import QueryResponse, ScoredPoint

    points = []
    for i in range(3):
        point = Mock(spec=ScoredPoint)
        point.id = f"point_{i}"
        point.score = 0.9 - (i * 0.1)
        point.payload = {
            "text": f"This is chunk {i} with relevant content",
            "source": f"doc_{i}",
            "chunk_id": f"chunk_{i}",
            "metadata": {"title": f"Document {i}"},
        }
        points.append(point)

    response = Mock(spec=QueryResponse)
    response.points = points

    client.query_points.return_value = response
    return client


# =============================================================================
# QdrantRetrieverConfig Tests
# =============================================================================


def test_config_initialization(qdrant_config):
    """Test config initialization with defaults."""
    assert qdrant_config.url == "http://localhost:6333"
    assert qdrant_config.api_key is None
    assert qdrant_config.collection_name == "rag2025"
    assert qdrant_config.embedding_dim == 1024
    assert qdrant_config.default_top_k == 5
    assert qdrant_config.timeout == 30


# =============================================================================
# RetrievedDocument Tests
# =============================================================================


def test_retrieved_document_creation():
    """Test RetrievedDocument dataclass."""
    doc = RetrievedDocument(
        text="Sample text",
        source="doc_1",
        chunk_id="chunk_1",
        metadata={"title": "Test"},
        score=0.85,
        point_id="point_1",
    )

    assert doc.text == "Sample text"
    assert doc.source == "doc_1"
    assert doc.chunk_id == "chunk_1"
    assert doc.metadata == {"title": "Test"}
    assert doc.score == 0.85
    assert doc.point_id == "point_1"


def test_retrieved_document_to_dict():
    """Test RetrievedDocument.to_dict() method."""
    doc = RetrievedDocument(
        text="Sample text",
        source="doc_1",
        chunk_id="chunk_1",
        metadata={"title": "Test"},
        score=0.85,
    )

    result = doc.to_dict()

    assert result == {
        "text": "Sample text",
        "source": "doc_1",
        "chunk_id": "chunk_1",
        "metadata": {"title": "Test"},
        "score": 0.85,
    }


def test_retrieved_document_defaults():
    """Test RetrievedDocument with default values."""
    doc = RetrievedDocument(text="Sample text")

    assert doc.text == "Sample text"
    assert doc.source is None
    assert doc.chunk_id is None
    assert doc.metadata == {}
    assert doc.score == 0.0
    assert doc.point_id is None


# =============================================================================
# RetrievalResult Tests
# =============================================================================


def test_retrieval_result_success():
    """Test successful RetrievalResult."""
    docs = [
        RetrievedDocument(text="Doc 1", score=0.9),
        RetrievedDocument(text="Doc 2", score=0.8),
    ]
    result = RetrievalResult(documents=docs, confidence=0.85)

    assert result.is_success is True
    assert result.is_empty is False
    assert result.error_type is None
    assert result.error_message is None
    assert len(result.documents) == 2
    assert result.confidence == 0.85


def test_retrieval_result_error():
    """Test RetrievalResult with error."""
    result = RetrievalResult(
        documents=[],
        error_type=RetrievalError.NETWORK_ERROR,
        error_message="Connection failed",
    )

    assert result.is_success is False
    assert result.is_empty is False
    assert result.error_type == RetrievalError.NETWORK_ERROR
    assert result.error_message == "Connection failed"
    assert len(result.documents) == 0


def test_retrieval_result_empty():
    """Test RetrievalResult with empty results (not an error)."""
    result = RetrievalResult(
        documents=[],
        error_type=RetrievalError.EMPTY_RESULT,
        error_message="No matches found",
    )

    assert result.is_success is False
    assert result.is_empty is True
    assert result.error_type == RetrievalError.EMPTY_RESULT


# =============================================================================
# QdrantRetriever Tests
# =============================================================================


def test_retriever_initialization(qdrant_config, mock_qdrant_client):
    """Test retriever initialization with mock client."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)

    assert retriever._config == qdrant_config
    assert retriever._client == mock_qdrant_client


def test_retrieve_success(qdrant_config, mock_qdrant_client, sample_query_vector):
    """Test successful retrieval."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    result = retriever.retrieve(sample_query_vector, top_k=3)

    assert result.is_success is True
    assert len(result.documents) == 3
    assert result.documents[0].score == 0.9
    assert result.documents[0].text == "This is chunk 0 with relevant content"
    assert result.confidence > 0


def test_retrieve_uses_query_points_api(qdrant_config, mock_qdrant_client, sample_query_vector):
    """Test that retrieve() uses query_points() API (not search/search_batch)."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    retriever.retrieve(sample_query_vector)

    # Verify query_points was called
    mock_qdrant_client.query_points.assert_called_once()

    # Verify the call arguments
    call_args = mock_qdrant_client.query_points.call_args
    assert call_args[1]["collection_name"] == "rag2025"
    assert call_args[1]["limit"] == 5
    assert call_args[1]["with_payload"] is True
    assert call_args[1]["with_vector"] is False


def test_retrieve_empty_vector(qdrant_config, mock_qdrant_client):
    """Test retrieval with empty query vector."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    result = retriever.retrieve([])

    assert result.is_success is False
    assert result.error_type == RetrievalError.INVALID_PAYLOAD
    assert "empty" in result.error_message.lower()


def test_retrieve_wrong_dimension(qdrant_config, mock_qdrant_client):
    """Test retrieval with wrong vector dimension."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)

    # 512-dim vector (wrong for 1024-dim collection)
    wrong_vector = [0.1] * 512
    result = retriever.retrieve(wrong_vector)

    assert result.is_success is False
    assert result.error_type == RetrievalError.INVALID_PAYLOAD
    assert "dimension mismatch" in result.error_message.lower()


def test_retrieve_missing_text_field(qdrant_config, sample_query_vector):
    """Test retrieval handles missing 'text' field in payload."""
    from qdrant_client.http.models import QueryResponse, ScoredPoint

    # Create mock point with missing text field
    point = Mock(spec=ScoredPoint)
    point.id = "point_1"
    point.score = 0.9
    point.payload = {"source": "doc_1"}  # Missing 'text'

    response = Mock(spec=QueryResponse)
    response.points = [point]

    mock_client = MagicMock()
    mock_client.query_points.return_value = response

    retriever = QdrantRetriever(config=qdrant_config, client=mock_client)
    result = retriever.retrieve(sample_query_vector)

    # Should skip invalid point and return empty
    assert result.is_empty is True
    assert len(result.documents) == 0


def test_retrieve_invalid_metadata(qdrant_config, sample_query_vector):
    """Test retrieval handles invalid metadata in payload."""
    from qdrant_client.http.models import QueryResponse, ScoredPoint

    point = Mock(spec=ScoredPoint)
    point.id = "point_1"
    point.score = 0.9
    point.payload = {
        "text": "Valid text",
        "metadata": "not_a_dict",  # Invalid metadata type
    }

    response = Mock(spec=QueryResponse)
    response.points = [point]

    mock_client = MagicMock()
    mock_client.query_points.return_value = response

    retriever = QdrantRetriever(config=qdrant_config, client=mock_client)
    result = retriever.retrieve(sample_query_vector)

    # Should replace invalid metadata with empty dict
    assert len(result.documents) == 1
    assert result.documents[0].metadata == {}


def test_confidence_calculation_single_result():
    """Test confidence calculation with single result."""
    docs = [RetrievedDocument(text="Doc", score=0.85)]
    retriever = QdrantRetriever(config=QdrantRetrieverConfig())
    confidence = retriever._calculate_confidence(docs)
    assert confidence == 0.85


def test_confidence_calculation_two_results():
    """Test confidence calculation with two results."""
    docs = [
        RetrievedDocument(text="Doc 1", score=0.9),
        RetrievedDocument(text="Doc 2", score=0.7),
    ]
    retriever = QdrantRetriever(config=QdrantRetrieverConfig())
    confidence = retriever._calculate_confidence(docs)
    # 0.7 * 0.9 + 0.3 * 0.7 = 0.63 + 0.21 = 0.84
    assert abs(confidence - 0.84) < 0.001


def test_confidence_calculation_three_results():
    """Test confidence calculation with three results."""
    docs = [
        RetrievedDocument(text="Doc 1", score=0.9),
        RetrievedDocument(text="Doc 2", score=0.7),
        RetrievedDocument(text="Doc 3", score=0.5),
    ]
    retriever = QdrantRetriever(config=QdrantRetrieverConfig())
    confidence = retriever._calculate_confidence(docs)
    # 0.5 * 0.9 + 0.3 * 0.7 + 0.2 * 0.5 = 0.45 + 0.21 + 0.1 = 0.76
    assert abs(confidence - 0.76) < 0.001


def test_confidence_calculation_empty():
    """Test confidence calculation with empty results."""
    retriever = QdrantRetriever(config=QdrantRetrieverConfig())
    confidence = retriever._calculate_confidence([])
    assert confidence == 0.0


def test_batch_retrieve(qdrant_config, mock_qdrant_client):
    """Test batch retrieval for multiple queries."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)

    # Create 3 query vectors
    query_vectors = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    results = retriever.batch_retrieve(query_vectors, top_k=2)

    assert len(results) == 3
    assert all(isinstance(r, RetrievalResult) for r in results)


# =============================================================================
# Convenience Function Tests
# =============================================================================


def test_retrieve_with_qdrant(sample_query_vector):
    """Test the convenience function retrieve_with_qdrant."""
    with patch("services.qdrant_retrieval.QdrantClient") as mock_client_class:
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        from qdrant_client.http.models import QueryResponse, ScoredPoint

        point = Mock(spec=ScoredPoint)
        point.id = "point_1"
        point.score = 0.9
        point.payload = {
            "text": "Sample text",
            "source": "doc_1",
            "chunk_id": "chunk_1",
            "metadata": {},
        }

        response = Mock(spec=QueryResponse)
        response.points = [point]
        mock_client.query_points.return_value = response

        # Call function
        docs = retrieve_with_qdrant(sample_query_vector, top_k=3)

        assert len(docs) == 1
        assert docs[0]["text"] == "Sample text"
        assert docs[0]["score"] == 0.9


def test_retrieve_with_qdrant_error_handling(sample_query_vector):
    """Test that retrieve_with_qdrant returns empty list on error."""
    with patch("services.qdrant_retrieval.QdrantClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.query_points.side_effect = Exception("Network error")

        # Call function - should return empty list (fail-safe)
        docs = retrieve_with_qdrant(sample_query_vector)

        assert docs == []


# =============================================================================
# Collection Info Tests
# =============================================================================


def test_check_collection_exists(qdrant_config):
    """Test check_collection when collection exists."""
    mock_client = MagicMock()

    # Mock collections response
    from qdrant_client.http.models import CollectionsResponse, CollectionDescription

    mock_collections = Mock(spec=CollectionsResponse)
    mock_collection = Mock(spec=CollectionDescription)
    mock_collection.name = "rag2025"
    mock_collections.collections = [mock_collection]

    # Mock collection info
    from qdrant_client.http.models import CollectionInfo

    mock_info = Mock(spec=CollectionInfo)
    mock_info.config.params.vectors.size = 1024
    mock_info.points_count = 1000
    mock_info.segments_count = 5
    mock_info.status = "green"

    mock_client.get_collections.return_value = mock_collections
    mock_client.get_collection.return_value = mock_info

    retriever = QdrantRetriever(config=qdrant_config, client=mock_client)
    info = retriever.check_collection()

    assert info["exists"] is True
    assert info["vectors_count"] == 1000


def test_check_collection_not_found(qdrant_config):
    """Test check_collection when collection doesn't exist."""
    mock_client = MagicMock()

    from qdrant_client.http.models import CollectionsResponse

    mock_collections = Mock(spec=CollectionsResponse)
    mock_collections.collections = []  # Empty - no collections

    mock_client.get_collections.return_value = mock_collections

    retriever = QdrantRetriever(config=qdrant_config, client=mock_client)
    info = retriever.check_collection()

    assert info["exists"] is False
    assert "not found" in info["error"]


# =============================================================================
# Extensibility Tests (Multi-Vector Ready)
# =============================================================================


def test_uses_prefetch_for_extensibility(qdrant_config, mock_qdrant_client, sample_query_vector):
    """Test that retrieve uses prefetch (for future multi-vector support)."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    retriever.retrieve(sample_query_vector)

    # Verify prefetch is used in the call
    call_args = mock_qdrant_client.query_points.call_args
    assert "prefetch" in call_args[1]
    assert len(call_args[1]["prefetch"]) > 0


# =============================================================================
# API Compliance Tests
# =============================================================================


def test_forbidden_search_api_not_used(qdrant_config, mock_qdrant_client, sample_query_vector):
    """Test that the deprecated search() API is NOT used."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    retriever.retrieve(sample_query_vector)

    # Verify search was NOT called
    assert not hasattr(mock_qdrant_client, "search") or not mock_qdrant_client.search.called


def test_forbidden_search_batch_api_not_used(qdrant_config, mock_qdrant_client, sample_query_vector):
    """Test that the deprecated search_batch() API is NOT used."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    retriever.retrieve(sample_query_vector)

    # Verify search_batch was NOT called
    assert not hasattr(mock_qdrant_client, "search_batch") or not mock_qdrant_client.search_batch.called


def test_mandatory_query_points_api_used(qdrant_config, mock_qdrant_client, sample_query_vector):
    """Test that ONLY query_points() API is used (mandatory)."""
    retriever = QdrantRetriever(config=qdrant_config, client=mock_qdrant_client)
    retriever.retrieve(sample_query_vector)

    # Verify query_points was called
    assert mock_qdrant_client.query_points.called
