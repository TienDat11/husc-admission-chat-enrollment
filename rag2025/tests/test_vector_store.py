"""
Unit tests for vector store (Epic 2.2)
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from services.vector_store import NumpyVectorStore, SearchResult


@pytest.fixture
def sample_vectors():
    """Fixture for sample vectors."""
    # Create 10 random 768-dim vectors
    vectors = np.random.randn(10, 768).astype(np.float32)
    # Normalize
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    return vectors


@pytest.fixture
def sample_ids():
    """Fixture for sample IDs."""
    return [f"chunk_{i}" for i in range(10)]


@pytest.fixture
def sample_metadatas():
    """Fixture for sample metadatas."""
    return [
        {
            "id": f"chunk_{i}",
            "doc_id": f"doc_{i // 3}",
            "chunk_id": i % 3,
            "text": f"This is chunk {i}",
            "metadata": {"source": "test"},
        }
        for i in range(10)
    ]


@pytest.fixture
def vector_store(sample_vectors, sample_ids, sample_metadatas):
    """Fixture for vector store with sample data."""
    store = NumpyVectorStore(dim=768)
    store.add_vectors(sample_vectors, sample_ids, sample_metadatas)
    return store


def test_vector_store_initialization():
    """Test vector store initialization."""
    store = NumpyVectorStore(dim=768)

    assert store.dim == 768
    assert store.vectors is None
    assert len(store.ids) == 0
    assert len(store.metadatas) == 0


def test_add_vectors(sample_vectors, sample_ids, sample_metadatas):
    """Test adding vectors to store."""
    store = NumpyVectorStore(dim=768)
    store.add_vectors(sample_vectors, sample_ids, sample_metadatas)

    assert store.count() == 10
    assert len(store.ids) == 10
    assert len(store.metadatas) == 10
    assert store.vectors.shape == (10, 768)


def test_add_vectors_dimension_mismatch():
    """Test adding vectors with wrong dimension."""
    store = NumpyVectorStore(dim=768)
    wrong_vectors = np.random.randn(5, 512).astype(np.float32)  # Wrong dim
    ids = [f"chunk_{i}" for i in range(5)]
    metadatas = [{"id": f"chunk_{i}"} for i in range(5)]

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add_vectors(wrong_vectors, ids, metadatas)


def test_add_vectors_length_mismatch():
    """Test adding vectors with mismatched IDs length."""
    store = NumpyVectorStore(dim=768)
    vectors = np.random.randn(5, 768).astype(np.float32)
    ids = [f"chunk_{i}" for i in range(3)]  # Wrong length
    metadatas = [{"id": f"chunk_{i}"} for i in range(5)]

    with pytest.raises(ValueError, match="IDs length mismatch"):
        store.add_vectors(vectors, ids, metadatas)


def test_search(vector_store):
    """Test vector search."""
    # Create query vector (similar to first vector)
    query_vector = vector_store.vectors[0] + np.random.randn(768) * 0.01
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)

    # Search
    results = vector_store.search(query_vector, top_k=3)

    assert len(results) == 3
    assert isinstance(results[0], SearchResult)
    assert results[0].score > results[1].score  # Sorted by score


def test_search_empty_store():
    """Test search on empty store."""
    store = NumpyVectorStore(dim=768)
    query_vector = np.random.randn(768).astype(np.float32)

    results = store.search(query_vector, top_k=5)

    assert len(results) == 0


def test_search_auto_scale_top_k(vector_store):
    """Test auto-scaling of top_k."""
    query_vector = np.random.randn(768).astype(np.float32)

    # Request more results than available
    results = vector_store.search(query_vector, top_k=100)

    # Should return all available (10)
    assert len(results) == 10


def test_save_and_load(vector_store):
    """Test saving and loading vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f:
        temp_path = Path(f.name)

    try:
        # Save
        vector_store.save(temp_path)

        # Load into new store
        new_store = NumpyVectorStore(dim=768)
        new_store.load(temp_path)

        # Verify
        assert new_store.count() == vector_store.count()
        assert new_store.ids == vector_store.ids
        assert np.allclose(new_store.vectors, vector_store.vectors)

    finally:
        temp_path.unlink()


def test_get_by_id(vector_store):
    """Test retrieving chunks by document ID."""
    results = vector_store.get_by_id("doc_0")

    # Should return 3 chunks (doc_0 has chunks 0, 1, 2)
    assert len(results) == 3
    assert all(r.doc_id == "doc_0" for r in results)


def test_count(vector_store):
    """Test count method."""
    assert vector_store.count() == 10


def test_cosine_similarity_scores(vector_store):
    """Test that scores are in valid cosine similarity range."""
    query_vector = np.random.randn(768).astype(np.float32)
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)

    results = vector_store.search(query_vector, top_k=5)

    for result in results:
        assert -1.0 <= result.score <= 1.0


def test_incremental_addition(sample_vectors, sample_ids, sample_metadatas):
    """Test adding vectors incrementally."""
    store = NumpyVectorStore(dim=768)

    # Add first 5
    store.add_vectors(
        sample_vectors[:5], sample_ids[:5], sample_metadatas[:5]
    )
    assert store.count() == 5

    # Add next 5
    store.add_vectors(
        sample_vectors[5:], sample_ids[5:], sample_metadatas[5:]
    )
    assert store.count() == 10
