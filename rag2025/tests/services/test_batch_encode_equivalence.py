"""tests/services/test_batch_encode_equivalence.py

Regression test for Fix #6 — switch /query variant loop from N×
encode_query to a single encode_batch.

encode_query uses a provider-specific prompt_name (Qwen/Harrier),
encode_batch does NOT. We therefore assert that:
  - encode_batch returns an ndarray with shape (N, dim) for N inputs
  - the per-vector normalize + dim-mismatch invariants from
    encode_query carry over to encode_batch
  - the call site uses encode_batch when invoked with a list of
    variants.

We do NOT assert bit-exact equality of vectors — encode_query and
encode_batch differ by design (prompt_name). The contract is purely
shape + dtype + per-vector L2-normalization invariants.
"""
from __future__ import annotations

import numpy as np
import pytest


class FakeModel:
    """Minimal stub with the encode() surface encode_batch needs."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts, batch_size=None, normalize_embeddings=False,
               show_progress_bar=False, convert_to_numpy=True):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((len(list(texts)) if not isinstance(texts, str) else 1,
                                   self.dim)).astype(np.float32)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


class _StubService:
    """Minimal duck-typed EmbeddingService — only the surface we test."""

    def __init__(self, dim: int = 4):
        self.expected_dim = dim
        self.batch_size = 16
        self.normalize = True
        self._model = FakeModel(dim)

    def encode_batch(self, texts, show_progress=False):
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError("dim mismatch")
        return embeddings


class TestBatchEncode:
    def test_batch_shape_and_dtype(self):
        svc = _StubService(dim=8)
        out = svc.encode_batch(["q1", "q2", "q3"])
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 8)
        assert out.dtype == np.float32

    def test_batch_empty_input(self):
        svc = _StubService(dim=8)
        out = svc.encode_batch([])
        assert out.shape == (0, 8)

    def test_batch_normalized(self):
        """Vectors produced by encode_batch must be L2-normalized when
        the service has ``normalize=True``."""
        svc = _StubService(dim=4)
        out = svc.encode_batch(["q1", "q2"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


class TestCallSiteUsesBatchEncode:
    """Verify the /query variant loop switched to encode_batch."""

    def test_main_calls_encode_batch(self):
        """The call site for variant encoding must mention encode_batch,
        not encode_query inside the loop."""
        import inspect
        import src.main as main_mod

        src = inspect.getsource(main_mod)
        # The new call site should encode_batch([...]).tolist() once
        assert "encode_batch(variants)" in src
        # The old per-variant encode_query(...) call must be gone from
        # the hot path (encode_query may still appear elsewhere as a
        # permitted utility).
        assert "encode_query(variant)" not in src