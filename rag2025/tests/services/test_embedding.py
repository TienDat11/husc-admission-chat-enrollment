"""Tests for lazy EmbeddingService.
@spec(S10.B2) — lazy loading + thread-safe init + encode methods.
"""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.embedding import EmbeddingService
from config.settings import RAGSettings


@pytest.fixture
def mock_settings():
    """Settings fixture that avoids real model download."""
    return RAGSettings(
        EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B",
        EMBEDDING_DIM=1024,
        EMBEDDING_BATCH_SIZE=8,
        EMBEDDING_NORMALIZE=True,
        EMBEDDING_PROVIDER="qwen",
    )


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer that returns fixed-dim vectors."""
    model = MagicMock()

    def _encode(texts, *args, **kwargs):
        import numpy as np

        if isinstance(texts, str):
            return np.random.rand(1024).astype(np.float32)
        n = len(texts)
        return np.random.rand(n, 1024).astype(np.float32)

    model.encode.side_effect = _encode
    return model


class TestLazyLoading:
    """@spec(S10.B2): model not loaded at init, loaded on first encode."""

    def test_init_does_not_load_model(self, mock_settings):
        svc = EmbeddingService(mock_settings)
        assert svc._model is None
        assert not svc.is_loaded

    def test_ensure_loaded_loads_model(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            svc._ensure_loaded()
        assert svc._model is not None
        assert svc.is_loaded

    def test_ensure_loaded_idempotent(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st:
            svc._ensure_loaded()
            svc._ensure_loaded()
            # SentenceTransformer constructor called only once
            assert mock_st.call_count == 1


class TestThreadSafety:
    """@spec(S10.B2): thread-safe double-check locking."""

    def test_concurrent_ensure_loaded(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st:
            barriers = threading.Barrier(4, timeout=5)

            def load_model():
                barriers.wait()  # synchronize all threads
                svc._ensure_loaded()

            threads = [threading.Thread(target=load_model) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            # Model loaded exactly once despite 4 concurrent callers
            assert mock_st.call_count == 1
            assert svc.is_loaded


class TestEncodeMethods:
    """@spec(S10.B2): public encode methods trigger lazy load."""

    def test_encode_query_triggers_load(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = svc.encode_query("test query")
        assert isinstance(result, np.ndarray)
        assert svc.is_loaded

    def test_encode_single_triggers_load(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = svc.encode_single("test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (mock_settings.EMBEDDING_DIM,)
        assert svc.is_loaded

    def test_encode_batch_triggers_load(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = svc.encode_batch(["text1", "text2"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, mock_settings.EMBEDDING_DIM)
        assert svc.is_loaded

    def test_encode_documents_triggers_load(self, mock_settings, mock_model):
        svc = EmbeddingService(mock_settings)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = svc.encode_documents(["doc1", "doc2"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, mock_settings.EMBEDDING_DIM)
        assert svc.is_loaded

    def test_encode_empty_batch_no_load(self, mock_settings):
        """Empty batch returns correctly shaped zeros. Model may be loaded."""
        svc = EmbeddingService(mock_settings)
        result = svc.encode_batch([])
        assert result.shape == (0, mock_settings.EMBEDDING_DIM)

    def test_model_name_unchanged(self, mock_settings):
        """C7: model name must not change."""
        svc = EmbeddingService(mock_settings)
        assert svc.model_name == "Qwen/Qwen3-Embedding-0.6B"
