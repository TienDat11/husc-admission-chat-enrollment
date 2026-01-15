"""
Unit tests for embedding service (Epic 2.1)
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from config.settings import RAGSettings
from services.embedding import EmbeddingService


@pytest.fixture
def settings():
    """Fixture for RAG settings."""
    return RAGSettings()


@pytest.fixture
def embedding_service(settings):
    """Fixture for embedding service."""
    return EmbeddingService(settings)


def test_embedding_service_initialization(embedding_service, settings):
    """Test embedding service initialization."""
    assert embedding_service.model_name == settings.EMBEDDING_MODEL
    assert embedding_service.expected_dim == settings.EMBEDDING_DIM
    assert embedding_service.batch_size == settings.EMBEDDING_BATCH_SIZE
    assert embedding_service.model is not None


def test_model_dimension_validation(embedding_service):
    """Test model dimension validation."""
    # Should not raise error (already validated in __init__)
    assert embedding_service.expected_dim == 768


def test_encode_single(embedding_service):
    """Test single text encoding."""
    text = "This is a test sentence for embedding."
    embedding = embedding_service.encode_single(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32


def test_encode_batch(embedding_service):
    """Test batch encoding."""
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence",
    ]
    embeddings = embedding_service.encode_batch(texts, show_progress=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32


def test_encode_empty_batch(embedding_service):
    """Test encoding empty batch."""
    embeddings = embedding_service.encode_batch([], show_progress=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 768)


def test_encode_query(embedding_service):
    """Test query encoding with e5 prefix."""
    query = "What is the admission deadline?"
    embedding = embedding_service.encode_query(query)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)

    # Query embedding should be different from document embedding
    doc_embedding = embedding_service.encode_single(query)
    assert not np.allclose(embedding, doc_embedding)


def test_encode_documents(embedding_service):
    """Test document encoding with e5 prefix."""
    documents = [
        "The admission deadline is June 30, 2025.",
        "Registration opens on May 1, 2025.",
    ]
    embeddings = embedding_service.encode_documents(documents)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 768)


def test_save_and_load_embeddings(embedding_service):
    """Test saving and loading embeddings."""
    # Create test embeddings
    texts = ["Test 1", "Test 2", "Test 3"]
    embeddings = embedding_service.encode_batch(texts)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f:
        temp_path = Path(f.name)

    try:
        embedding_service.save_embeddings(embeddings, temp_path)

        # Load embeddings
        loaded_embeddings = embedding_service.load_embeddings(temp_path)

        # Verify
        assert loaded_embeddings.shape == embeddings.shape
        assert np.allclose(loaded_embeddings, embeddings)

    finally:
        temp_path.unlink()


def test_embedding_normalization(embedding_service):
    """Test L2 normalization of embeddings."""
    text = "Test normalization"
    embedding = embedding_service.encode_single(text)

    # Check L2 norm is close to 1
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01  # Should be normalized


def test_batch_consistency(embedding_service):
    """Test that batch and single encoding produce same results."""
    text = "Consistency test"

    # Encode as single
    single_embedding = embedding_service.encode_single(text)

    # Encode as batch
    batch_embeddings = embedding_service.encode_batch([text])

    # Should be very similar (small numerical differences acceptable)
    assert np.allclose(single_embedding, batch_embeddings[0], atol=1e-5)
