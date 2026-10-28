"""T15 AC3: a tombstoned legacy row is absent from `sources` on ALL
THREE chunk-serving read paths.

Route → read path mapping (the three retrieval routes of the 3-route
HTTP surface):
  * padded_rag  → dense arm  = `LanceDBRetriever.retrieve()`
  * graph_rag   → GraphRAG expander = `fetch_by_id` / `fetch_by_ids`
  * hybrid (v3) → BM25 corpus = `HybridSearchService.build_bm25_index`

Each path must drop a row stamped `is_tombstone=true` (both the flat
column and the metadata-mirror convention), on a table that carries
the column AND on a legacy table that does not (schema-gated).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

RAG_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RAG_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
)

EMBEDDING_DIM = 4


def _row_t15(
    cid: str,
    is_tombstone: Optional[bool],
    *,
    mirror: bool = False,
) -> Dict[str, Any]:
    """Row shaped like one returned by `LanceDBAdapter.search`."""
    meta: Dict[str, Any] = {"data_year": 2026}
    if mirror:
        meta["is_tombstone"] = True
    return {
        "text": f"text-{cid}",
        "source": "synthetic",
        "chunk_id": cid,
        "id": cid,
        "metadata_json": json.dumps(meta),
        "is_tombstone": is_tombstone,
    }

def _make_retriever(
    rows: List[Dict[str, Any]],
    *,
    has_tombstone_column: bool = True,
) -> LanceDBRetriever:
    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://",
        table_name="t",
        embedding_dim=EMBEDDING_DIM,
        default_top_k=5,
    )
    fake = MagicMock()
    fake.ensure_table = MagicMock()
    fake.has_tombstone_column = MagicMock(return_value=has_tombstone_column)
    fake.search = MagicMock(
        side_effect=lambda query_vector, top_k, where_clause=None: list(rows)
    )
    fake.fetch_rows_by_ids = MagicMock(
        side_effect=lambda ids: [
            r for r in rows if r.get("id") in set(ids or [])
        ]
    )
    fake.fetch_rows_by_doc_ids = MagicMock(return_value=[])
    retriever._adapter = fake
    return retriever


# ---------------------------------------------------------------------------
# Route 1 — padded_rag: dense arm (retrieve)
# ---------------------------------------------------------------------------


class TestPaddedRagDenseArm:
    def test_tombstoned_row_absent_from_retrieve_documents(self):
        rows = [
            _row_t15("normal", False),
            _row_t15("tombstoned", True),
        ]
        retriever = _make_retriever(rows)
        result = retriever.retrieve(
            query_vector=[0.0] * EMBEDDING_DIM, top_k=5, query="học phí"
        )
        got = {doc.chunk_id for doc in result.documents}
        assert "normal" in got
        assert "tombstoned" not in got

    def test_metadata_mirror_tombstone_absent(self):
        rows = [
            _row_t15("normal", None),
            _row_t15("mirror", None, mirror=True),
        ]
        retriever = _make_retriever(rows)
        result = retriever.retrieve(
            query_vector=[0.0] * EMBEDDING_DIM, top_k=5, query="học phí"
        )
        got = {doc.chunk_id for doc in result.documents}
        assert "normal" in got
        assert "mirror" not in got

    def test_legacy_table_without_column_still_drops_mirror(self):
        """Schema gate: no `is_tombstone` column → where_clause=None,
        but the post-fetch belt still catches the metadata mirror."""
        rows = [
            _row_t15("normal", None),
            _row_t15("mirror", None, mirror=True),
        ]
        retriever = _make_retriever(rows, has_tombstone_column=False)
        result = retriever.retrieve(
            query_vector=[0.0] * EMBEDDING_DIM, top_k=5, query="học phí"
        )
        got = {doc.chunk_id for doc in result.documents}
        assert "normal" in got
        assert "mirror" not in got


# ---------------------------------------------------------------------------
# Route 2 — graph_rag: expander single + batch fetch
# ---------------------------------------------------------------------------


class TestGraphRagExpander:
    def test_fetch_by_id_returns_none_for_tombstoned(self):
        rows = [_row_t15("tombstoned", True)]
        retriever = _make_retriever(rows)
        assert retriever.fetch_by_id("tombstoned") is None

    def test_fetch_by_ids_omits_tombstoned(self):
        rows = [
            _row_t15("ok", False),
            _row_t15("bad", True),
        ]
        retriever = _make_retriever(rows)
        out = retriever.fetch_by_ids(["ok", "bad"])
        assert "ok" in out
        assert "bad" not in out


# ---------------------------------------------------------------------------
# Route 3 — hybrid (v3): BM25 corpus build
# ---------------------------------------------------------------------------


class TestHybridBm25Corpus:
    def _build_service(self, rows: List[Dict[str, Any]]):
        import pandas as pd

        from services.hybrid_search import HybridSearchService

        retriever = _make_retriever(rows)
        table = MagicMock()
        table.schema.names = [
            "text", "source", "chunk_id", "metadata_json",
            "id", "release_id", "is_tombstone",
        ]
        df = pd.DataFrame(
            {
                "text": [r["text"] for r in rows],
                "source": [r["source"] for r in rows],
                "chunk_id": [r["chunk_id"] for r in rows],
                "metadata_json": [r["metadata_json"] for r in rows],
                "id": [r["id"] for r in rows],
                "release_id": ["R1" for _ in rows],
                "is_tombstone": [r["is_tombstone"] for r in rows],
            }
        )
        select_result = MagicMock()
        select_result.to_pandas.return_value = df
        table.select.return_value = select_result
        table.to_pandas.return_value = df
        retriever._adapter._table = table

        settings = MagicMock()
        settings.HYBRID_FUSION_DENSE_WEIGHT = 0.6
        settings.HYBRID_FUSION_SPARSE_WEIGHT = 0.4
        svc = HybridSearchService(retriever, settings)
        assert svc.build_bm25_index() is True
        return svc

    def test_tombstoned_row_absent_from_bm25_corpus(self):
        rows = [
            _row_t15("normal", False),
            _row_t15("tombstoned", True),
        ]
        svc = self._build_service(rows)
        got = {d.chunk_id for d in svc._corpus_docs}
        assert "normal" in got
        assert "tombstoned" not in got

    def test_null_tombstone_row_still_in_bm25_corpus(self):
        rows = [_row_t15("unstamped", None)]
        svc = self._build_service(rows)
        got = {d.chunk_id for d in svc._corpus_docs}
        assert "unstamped" in got
