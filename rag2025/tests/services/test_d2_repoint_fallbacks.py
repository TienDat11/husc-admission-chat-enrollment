"""D2 repoint: fallback tests cho husc_chunks schema (không có chunk_id/source/faq_type top-level).

husc_chunks schema: id + metadata_json (faq_type/source/chunk_id nằm trong metadata_json),
KHÔNG có chunk_id/source/faq_type top-level. Các hàm build RetrievedDocument / BM25 corpus /
_apply_filter phải fallback sang id / metadata_json.

TDD RED: các test này FAIL trước khi code fix:
- lancedb_retrieval.py:550-551: `source=row.get("source"), chunk_id=row.get("chunk_id")` — không fallback.
- hybrid_search.py:104: `chunk_id=str(row.get("chunk_id",""))` — không fallback id.
- settings.py:134: `default="rag2025"` (phải là "husc_chunks" sau E4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# conftest.py (rag2025/conftest.py) đã add src/ + repo root vào sys.path;
# thêm safety path cho trường hợp chạy file đơn lẻ.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument
from services.hybrid_search import HybridSearchService
from config.settings import RAGSettings


def _make_retriever() -> LanceDBRetriever:
    """Bypass __init__ (touch LanceDB adapter) — chỉ test method thuần trên instance."""
    return object.__new__(LanceDBRetriever)


class TestChunkIdFallback:
    """chunk_id vắng → fallback id (lancedb_retrieval.py:551)."""

    def test_chunk_id_fallback_to_id(self):
        """Row không có chunk_id → RetrievedDocument.chunk_id = row['id']."""
        row = {"id": "test_chunk_001", "text": "hello", "metadata_json": "{}"}
        docs = _make_retriever()._normalize_rows([row])
        assert docs[0].chunk_id == "test_chunk_001"

    def test_chunk_id_prefers_explicit(self):
        """Row CÓ chunk_id → giữ chunk_id, không override bằng id."""
        row = {"id": "id_1", "chunk_id": "cid_1", "text": "hello", "metadata_json": "{}"}
        docs = _make_retriever()._normalize_rows([row])
        assert docs[0].chunk_id == "cid_1"


class TestSourceFallback:
    """source vắng → fallback metadata_json.source (lancedb_retrieval.py:550)."""

    def test_source_fallback_to_metadata_json(self):
        row = {"id": "x", "text": "t", "metadata_json": json.dumps({"source": "thongbao.php?id=76"})}
        docs = _make_retriever()._normalize_rows([row])
        assert docs[0].source == "thongbao.php?id=76"

    def test_source_prefers_top_level(self):
        row = {"id": "x", "source": "top.csv", "text": "t", "metadata_json": json.dumps({"source": "meta.json"})}
        docs = _make_retriever()._normalize_rows([row])
        assert docs[0].source == "top.csv"

    def test_source_empty_when_neither(self):
        """Không source top-level, không trong metadata → None (không crash)."""
        row = {"id": "x", "text": "t", "metadata_json": "{}"}
        docs = _make_retriever()._normalize_rows([row])
        assert docs[0].source is None


class TestHybridBm25Fallback:
    """BM25 corpus build: chunk_id vắng → dùng id (hybrid_search.py:104)."""

    def test_bm25_uses_id_when_no_chunk_id(self):
        """Corpus doc chunk_id fallback row['id'] khi column chunk_id không tồn tại.

        Interface test: build_bm25_index() yêu cầu adapter + table thật; test qua
        fixture object __new__ + monkeypatch _adapter để mô phỏng schema husc_chunks.
        """
        retriever = _make_retriever()
        svc = HybridSearchService(retriever, RAGSettings(_env_file=None))

        class FakeTable:
            schema = type("S", (), {"names": ["id", "text", "metadata_json"]})()

            def to_pandas(self):
                import pandas as pd

                return pd.DataFrame(
                    [{"id": "bm25_1", "text": "hello world", "metadata_json": "{}"}]
                )

        svc._retriever._adapter = type("FakeAdapter", (), {"_table": FakeTable()})()

        assert svc.build_bm25_index() is True
        assert svc._corpus_docs[0].chunk_id == "bm25_1"


class TestFaqFilterMetadataJson:
    """_apply_filter faq_type qua metadata_json (đã có fallback — confirm test)."""

    def test_faq_match_via_metadata_json(self):
        """metadata_json faq_type="tuition", filter {"faq_type": "tuition"} → match."""
        row = {"id": "c1", "text": "t", "metadata_json": json.dumps({"faq_type": "tuition"})}
        filtered = _make_retriever()._apply_filter([row], {"faq_type": "tuition"})
        assert filtered == [row]

    def test_faq_no_match(self):
        """metadata_json faq_type="overview", filter tuition → no match."""
        row = {"id": "c2", "text": "t", "metadata_json": json.dumps({"faq_type": "overview"})}
        filtered = _make_retriever()._apply_filter([row], {"faq_type": "tuition"})
        assert filtered == []


class TestSettingsDefault:
    def test_default_table_husc_chunks(self):
        """Settings default = husc_chunks (sau E4). FAIL trước E4 (default còn rag2025)."""
        assert RAGSettings(_env_file=None).LANCEDB_TABLE == "husc_chunks"


class TestLancedbUriResolve:
    """LANCEDB_URI relative → absolute (settings.py validator)."""

    def test_relative_uri_resolves_to_rag2025(self):
        """Relative ./data/lancedb → absolute path chứa rag2025/data/lancedb."""
        s = RAGSettings(_env_file=None)  # dùng default "./data/lancedb"
        assert "rag2025" in s.LANCEDB_URI or "data" in s.LANCEDB_URI
        assert s.LANCEDB_URI.endswith("data/lancedb") or s.LANCEDB_URI.endswith(r"data\lancedb")
        assert Path(s.LANCEDB_URI).is_absolute()

    def test_absolute_uri_unchanged(self):
        """Absolute path giữ nguyên."""
        s = RAGSettings(_env_file=None, LANCEDB_URI="D:/custom/path")
        assert s.LANCEDB_URI == "D:/custom/path"

    def test_scheme_uri_unchanged(self):
        """s3:// hoặc gs:// giữ nguyên."""
        s = RAGSettings(_env_file=None, LANCEDB_URI="s3://bucket/path")
        assert s.LANCEDB_URI == "s3://bucket/path"
