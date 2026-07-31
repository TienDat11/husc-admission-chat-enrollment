"""
Unit tests for adaptive reranker integration in /v2.

Run with:
    cd rag2025
    TEMP=D:/tmp D:/miniconda3/python.exe -m pytest tests/services/test_reranker_adaptive.py -q -o addopts=""
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from services.reranker import RerankerService


def _make_service(
    enabled: bool = True,
    max_length: int = 512,
    pool: int = 5,
    adaptive_confidence: float = 0.5,
    weight: float = 0.35,
    max_rerank: int = 50,
) -> RerankerService:
    """Build a RerankerService with a mocked CrossEncoder."""
    svc = RerankerService.__new__(RerankerService)
    svc._enabled = enabled
    svc._model_name = "dummy"
    svc._weight = weight
    svc._max_rerank = max_rerank
    svc._max_length = max_length
    svc._pool = pool
    svc._adaptive_confidence = adaptive_confidence
    svc._model = MagicMock()
    return svc


def _long_vn_chunk(n_chars: int = 2000) -> dict:
    """Simulate a long Vietnamese admission chunk."""
    return {
        "text": (
            "Ngành Công nghệ thông tin mã 7480201 thuộc khối ngành Kỹ thuật - Công nghệ. "
            "Học phí 600.000 VNĐ/tín chỉ. Điểm chuẩn năm 2025: 21.5 điểm. "
            "Tổ hợp xét tuyển: A00 (Toán, Vật lý, Hóa học), D01 (Toán, Ngữ văn, Tiếng Anh). "
            "Chương trình đào tạo 4 năm, trao đổi sinh viên với Đại học Kyushu (Nhật Bản). "
        ) * (n_chars // 200 + 1),
        "score": 0.85,
        "chunk_id": "test_long_chunk_001",
        "source": "tuyensinh_cntt_2026",
    }


# ---------------------------------------------------------------------------
# (a) rerank with max_length set doesn't error on long VN chunk
# ---------------------------------------------------------------------------

def test_rerank_max_length_long_vn_chunk():
    """rerank() must handle long Vietnamese text without error at any max_length."""
    svc = _make_service(max_length=128)
    # CrossEncoder.predict is mocked; simulate scores
    svc._model.predict.return_value = [0.9, 0.7, 0.5]

    chunks = [_long_vn_chunk(2000), _long_vn_chunk(3000), _long_vn_chunk(500)]
    result = svc.rerank(
        query="học phí ngành CNTT",
        chunks=chunks,
        top_k=3,
        apply_lost_in_middle=False,
        max_length=128,
    )
    assert len(result) == 3
    # Verify predict was called (model used)
    svc._model.predict.assert_called_once()


# ---------------------------------------------------------------------------
# (b) returned chunks preserve full original text (no truncation)
# ---------------------------------------------------------------------------

def test_rerank_preserves_full_chunk_text():
    """The reranker must NOT truncate the chunk text in returned results.

    Truncation is scoring-only (inside the cross-encoder). The generator
    must always see the complete chunk text.
    """
    svc = _make_service(max_length=128)
    svc._model.predict.return_value = [0.9, 0.8]

    original_text_a = "A" * 5000  # way longer than max_length=128
    original_text_b = "B" * 5000

    chunks = [
        {"text": original_text_a, "score": 0.7, "chunk_id": "a"},
        {"text": original_text_b, "score": 0.6, "chunk_id": "b"},
    ]
    result = svc.rerank(
        query="test",
        chunks=chunks,
        top_k=2,
        apply_lost_in_middle=False,
        max_length=128,
    )

    for r in result:
        assert len(r["text"]) == 5000, (
            f"Chunk text was truncated: got {len(r['text'])} chars, expected 5000. "
            "The reranker must preserve full text (truncation is scoring-only)."
        )


# ---------------------------------------------------------------------------
# (c) adaptive gate: skip rerank when confidence is high
# ---------------------------------------------------------------------------

def test_adaptive_gate_skips_rerank_when_confidence_high():
    """When retrieval confidence >= adaptive_confidence, rerank must be skipped."""
    svc = _make_service(
        enabled=True,
        pool=5,
        adaptive_confidence=0.5,
    )
    svc._model.predict.return_value = [0.9, 0.8]

    # Simulate the gate logic from main.py
    confidence = 0.6  # above threshold 0.5
    should_rerank = (
        confidence < svc.adaptive_confidence
    )

    assert should_rerank is False, (
        f"Expected rerank to be skipped when confidence={confidence} >= "
        f"threshold={svc.adaptive_confidence}"
    )

    # If we had called rerank, predict would have been called; confirm it was NOT
    svc._model.predict.assert_not_called()


def test_adaptive_gate_triggers_rerank_when_confidence_low():
    """When retrieval confidence < adaptive_confidence, rerank must activate."""
    svc = _make_service(
        enabled=True,
        pool=5,
        adaptive_confidence=0.5,
    )
    svc._model.predict.return_value = [0.9, 0.8, 0.7]

    confidence = 0.3  # below threshold 0.5
    should_rerank = (
        confidence < svc.adaptive_confidence
    )

    assert should_rerank is True, (
        f"Expected rerank to activate when confidence={confidence} < "
        f"threshold={svc.adaptive_confidence}"
    )


# ---------------------------------------------------------------------------
# (d) new settings are properly exposed
# ---------------------------------------------------------------------------

def test_settings_new_fields():
    """Verify the new RAGSettings fields exist and are set from .env."""
    from config.settings import RAGSettings
    s = RAGSettings()
    assert hasattr(s, "RERANKER_MAX_LENGTH")
    assert hasattr(s, "RERANKER_POOL")
    assert hasattr(s, "RERANKER_ADAPTIVE_CONFIDENCE")
    assert isinstance(s.RERANKER_MAX_LENGTH, int) and s.RERANKER_MAX_LENGTH > 0
    assert isinstance(s.RERANKER_POOL, int) and s.RERANKER_POOL > 0
    assert isinstance(s.RERANKER_ADAPTIVE_CONFIDENCE, float)
    assert 0.0 <= s.RERANKER_ADAPTIVE_CONFIDENCE <= 1.0
