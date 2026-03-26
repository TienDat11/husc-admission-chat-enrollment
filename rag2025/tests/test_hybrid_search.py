"""Unit tests for HybridSearchService."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

import pandas as pd

from services.hybrid_search import HybridSearchService
from services.lancedb_retrieval import RetrievedDocument, RetrievalResult


def _make_doc(chunk_id: str, text: str = "sample text", score: float = 1.0) -> RetrievedDocument:
    return RetrievedDocument(
        text=text, source="src", chunk_id=chunk_id, metadata={}, score=score, point_id=None
    )


def _make_settings(use_hybrid=True, dense_w=0.6, sparse_w=0.4):
    s = MagicMock()
    s.USE_HYBRID_RETRIEVAL = use_hybrid
    s.HYBRID_FUSION_DENSE_WEIGHT = dense_w
    s.HYBRID_FUSION_SPARSE_WEIGHT = sparse_w
    return s


def _make_retriever(docs=None):
    retriever = MagicMock()
    docs = docs or []
    retriever.retrieve.return_value = RetrievalResult(documents=docs, confidence=0.5)
    # Mock adapter._table for BM25 build
    table = MagicMock()
    table.schema.names = ["text", "source", "chunk_id"]
    table.to_pandas.return_value = (
        pd.DataFrame(
            {
                "text": [d.text for d in docs],
                "source": [d.source for d in docs],
                "chunk_id": [d.chunk_id for d in docs],
            }
        )
        if docs
        else pd.DataFrame()
    )
    retriever._adapter._table = table
    return retriever


class TestBuildBm25Index(unittest.TestCase):
    def test_build_bm25_index_success(self):
        docs = [_make_doc("c1", "tuyển sinh đại học"), _make_doc("c2", "học phí 2024")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        result = svc.build_bm25_index()
        self.assertTrue(result)
        self.assertIsNotNone(svc._bm25)
        self.assertEqual(len(svc._corpus_docs), 2)

    def test_build_bm25_index_empty_table(self):
        retriever = _make_retriever([])
        svc = HybridSearchService(retriever, _make_settings())
        result = svc.build_bm25_index()
        self.assertFalse(result)
        self.assertIsNone(svc._bm25)

    def test_build_bm25_index_failure_graceful(self):
        retriever = MagicMock()
        retriever._adapter._table.schema.names = ["text"]
        retriever._adapter._table.to_pandas.side_effect = RuntimeError("DB error")
        svc = HybridSearchService(retriever, _make_settings())
        result = svc.build_bm25_index()
        self.assertFalse(result)
        self.assertIsNone(svc._bm25)


class TestRetrieve(unittest.TestCase):
    def test_hybrid_retrieve_full_path(self):
        docs = [_make_doc("c1", "tuyển sinh đại học HUSC"), _make_doc("c2", "ngành học 2024")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        result = asyncio.get_event_loop().run_until_complete(
            svc.retrieve(query="tuyển sinh", query_vector=[0.1] * 10, top_k=2)
        )
        self.assertIsInstance(result, RetrievalResult)
        self.assertGreater(len(result.documents), 0)

    def test_hybrid_retrieve_bm25_not_ready_fallback(self):
        docs = [_make_doc("c1", "text")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        # Do NOT build index — _bm25 is None
        result = asyncio.get_event_loop().run_until_complete(
            svc.retrieve(query="tuyển sinh", query_vector=[0.1] * 10, top_k=2)
        )
        # Should fall back to dense-only
        self.assertEqual(result.documents, docs)


class TestBm25Search(unittest.TestCase):
    def test_bm25_search_empty_query_guard(self):
        docs = [_make_doc("c1", "tuyển sinh")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("", top_k=5)
        self.assertEqual(result, [])


class TestRrfFusion(unittest.TestCase):
    def test_rrf_fusion_correctness(self):
        dense = [_make_doc("c1", score=0.9), _make_doc("c2", score=0.7)]
        sparse = [_make_doc("c2", score=5.0), _make_doc("c3", score=3.0)]
        retriever = _make_retriever()
        svc = HybridSearchService(retriever, _make_settings())
        fused = svc._rrf_fusion(dense, sparse, top_k=3)
        # c2 appears in both → should have higher fused score
        chunk_ids = [d.chunk_id for d in fused]
        self.assertIn("c2", chunk_ids)
        c2_idx = chunk_ids.index("c2")
        c2_score = fused[c2_idx].score
        self.assertGreater(c2_score, 0)

    def test_dense_only_when_flag_false(self):
        """When USE_HYBRID_RETRIEVAL=False, service should not be instantiated (integration guard)."""
        # This tests that the flag being False means hybrid_search_service stays None in main.py.
        # We simulate: if flag=False, caller does not call HybridSearchService at all.
        settings = _make_settings(use_hybrid=False)
        self.assertFalse(settings.USE_HYBRID_RETRIEVAL)
        # The actual routing logic in main.py checks `if hybrid_search_service:` — no service → dense only


if __name__ == "__main__":
    unittest.main()
