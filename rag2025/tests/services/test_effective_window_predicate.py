"""T15 effectiveness law (D1 R03): status + effective-window predicate.

Focused unit tests for ``_drop_by_effectiveness`` / ``_current_as_of_int``:
the clock-based law (risk R5) wired into the retrieval hot path.

Clock source is ``services.temporal_authority._now`` (testable clock).
Tests pin it via monkeypatch to a fixed date (2026-10-09 → as_of=20261009).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
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
    _current_as_of_int,
    _drop_by_effectiveness,
)

AS_OF = 20261009  # 2026-10-09

EMBEDDING_DIM = 4


def _pin_clock(monkeypatch) -> None:
    import services.temporal_authority as ta

    monkeypatch.setattr(ta, "_now", lambda: datetime(2026, 10, 9, 12, 0, 0))


def _row(
    cid: str,
    *,
    status: Any = "ABSENT",
    effective_from: Any = "ABSENT",
    effective_to: Any = "ABSENT",
    meta_status: Any = "ABSENT",
    is_tombstone: Optional[bool] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "text": f"text-{cid}",
        "source": "synthetic",
        "chunk_id": cid,
        "id": cid,
        "_distance": 0.5,
    }
    if status != "ABSENT":
        row["status"] = status
    if effective_from != "ABSENT":
        row["effective_from"] = effective_from
    if effective_to != "ABSENT":
        row["effective_to"] = effective_to
    meta: Dict[str, Any] = {"data_year": 2026}
    if meta_status != "ABSENT":
        meta["status"] = meta_status
    if is_tombstone is not None:
        meta["is_tombstone"] = is_tombstone
    row["metadata_json"] = json.dumps(meta)
    return row


def _meta_of(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("metadata_json") or "{}"
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def test_current_as_of_int_follows_authority_clock(monkeypatch) -> None:
    _pin_clock(monkeypatch)
    assert _current_as_of_int() == AS_OF


def test_expired_row_dropped() -> None:
    row = _row("expired", effective_to=20261008)
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is True


def test_not_yet_effective_row_dropped() -> None:
    row = _row("future", effective_from=20261010)
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is True


def test_in_window_row_kept() -> None:
    row = _row("ok", effective_from=20260101, effective_to=20261231)
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is False


def test_null_window_kept() -> None:
    row = _row("nullwin", effective_from=None, effective_to=None)
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is False


def test_legacy_row_without_columns_kept() -> None:
    row = _row("legacy")
    assert "status" not in row
    assert "effective_from" not in row
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is False


def test_nan_window_kept() -> None:
    row = _row("nan", effective_from=float("nan"), effective_to=float("nan"))
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is False


@pytest.mark.parametrize("bad", ["archived", "draft", "review", "approved"])
def test_non_current_flat_status_dropped(bad: str) -> None:
    row = _row("st", status=bad)
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is True


@pytest.mark.parametrize("bad", ["draft", "review", "approved", "archived"])
def test_non_current_meta_status_dropped(bad: str) -> None:
    row = _row("mst", meta_status=bad)
    assert _drop_by_effectiveness(row, _meta_of(row), AS_OF) is True


def test_current_and_absent_status_kept() -> None:
    assert _drop_by_effectiveness(_row("a", status="current"), {"data_year": 2026}, AS_OF) is False
    assert _drop_by_effectiveness(_row("b"), {"data_year": 2026}, AS_OF) is False
    # Legacy rows may carry an unrelated meta status key — explicit-deny keeps them.
    assert _drop_by_effectiveness(_row("c"), {"status": "whatever"}, AS_OF) is False


# ---------------------------------------------------------------------------
# Belt wiring: expired / non-current rows absent from retrieve + fetch_by_id(s)
# ---------------------------------------------------------------------------


def _make_retriever(rows: List[Dict[str, Any]]) -> LanceDBRetriever:
    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://", table_name="t",
        embedding_dim=EMBEDDING_DIM, default_top_k=5,
    )
    fake = MagicMock()
    fake.ensure_table = MagicMock()
    fake.has_tombstone_column = MagicMock(return_value=False)
    fake.search = MagicMock(side_effect=lambda query_vector, top_k, where_clause=None: list(rows))
    fake.fetch_rows_by_ids = MagicMock(
        side_effect=lambda ids: [r for r in rows if r.get("id") in set(ids or [])]
    )
    fake.fetch_rows_by_doc_ids = MagicMock(return_value=[])
    retriever._adapter = fake
    return retriever


def _drive_in_season(retriever: LanceDBRetriever) -> None:
    from services.season import SeasonPhase

    retriever.get_admission_context = MagicMock(  # type: ignore[assignment]
        return_value={"year": 2026, "has_data": True, "season": SeasonPhase.IN_SEASON}
    )


def test_retrieve_drops_expired_and_non_current(monkeypatch) -> None:
    import os

    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.delenv("APP_ENV", raising=False)
    _pin_clock(monkeypatch)
    rows = [
        _row("ok", effective_from=20260101, effective_to=20261231),
        _row("expired", effective_to=20240101),
        _row("future", effective_from=20270101),
        _row("archived", status="archived"),
        _row("legacy"),
    ]
    retriever = _make_retriever(rows)
    _drive_in_season(retriever)
    result = retriever.retrieve(
        query_vector=[0.0] * EMBEDDING_DIM, top_k=5, query="học phí"
    )
    got = {doc.chunk_id for doc in result.documents}
    assert "ok" in got
    assert "legacy" in got
    assert "expired" not in got
    assert "future" not in got
    assert "archived" not in got


def test_fetch_by_id_returns_none_for_expired(monkeypatch) -> None:
    _pin_clock(monkeypatch)
    rows = [_row("expired", effective_to=20240101), _row("ok")]
    retriever = _make_retriever(rows)
    assert retriever.fetch_by_id("expired") is None
    assert retriever.fetch_by_id("ok") is not None


def test_fetch_by_ids_omits_expired(monkeypatch) -> None:
    _pin_clock(monkeypatch)
    rows = [_row("expired", effective_to=20240101), _row("ok")]
    retriever = _make_retriever(rows)
    out = retriever.fetch_by_ids(["expired", "ok"])
    assert "ok" in out
    assert "expired" not in out


class TestStaleSparseIndexExpiry:
    """The advisory blocker: BM25 builds once (day D), the chunk expires,
    retrieve() WITHOUT a rebuild must drop it. The build-time belt alone
    would keep the stale corpus doc; the per-request sidecar filter fixes it."""

    def _make_hybrid(self, rows: List[Dict[str, Any]]):
        import pandas as pd

        from services.hybrid_search import HybridSearchService

        from config.settings import RAGSettings

        retriever = _make_retriever(rows)
        hybrid = HybridSearchService.__new__(HybridSearchService)
        hybrid._retriever = retriever
        hybrid._settings = MagicMock(spec=RAGSettings)
        hybrid._settings.HYBRID_FUSION_DENSE_WEIGHT = 0.6
        hybrid._settings.HYBRID_FUSION_SPARSE_WEIGHT = 0.4
        hybrid._bm25 = None
        hybrid._corpus_docs = []
        hybrid._corpus_eligibility = {}
        hybrid._code_postings = {}
        hybrid._corpus_version = {}

        df_rows: Dict[str, List[Any]] = {
            "source": [r["source"] for r in rows],
            "chunk_id": [r["chunk_id"] for r in rows],
            "metadata_json": [r["metadata_json"] for r in rows],
            "id": [r["id"] for r in rows],
            "is_tombstone": [None for _ in rows],
        }
        if any("effective_from" in r for r in rows):
            df_rows["effective_from"] = [
                r.get("effective_from") for r in rows
            ]
        if any("effective_to" in r for r in rows):
            df_rows["effective_to"] = [r.get("effective_to") for r in rows]
        df = pd.DataFrame(df_rows)
        table = MagicMock()
        table.schema.names = list(df.columns)
        select_result = MagicMock()
        select_result.to_pandas.return_value = df
        table.select.return_value = select_result
        table.to_pandas.return_value = df
        retriever._adapter._table = table
        return hybrid, retriever

    @pytest.mark.asyncio
    async def test_sparse_chunk_expiring_after_build_disappears(
        self, monkeypatch
    ) -> None:
        import services.temporal_authority as ta

        # Day D: chunk still in-window → build succeeds, doc in corpus.
        monkeypatch.setattr(
            ta, "_now", lambda: datetime(2026, 6, 1, 12, 0, 0)
        )
        rows = [
            _row("fresh", effective_from=20260101, effective_to=20261231),
            _row("dying", effective_from=20260101, effective_to=20260615),
        ]
        hybrid, retriever = self._make_hybrid(rows)
        assert hybrid.build_bm25_index() is True
        assert "dying" in hybrid._corpus_eligibility

        # Advance past effective_to WITHOUT rebuilding: per-request
        # filter must drop the stale sparse doc via the sidecar.
        monkeypatch.setattr(
            ta, "_now", lambda: datetime(2026, 7, 1, 12, 0, 0)
        )
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        _drive_in_season(retriever)
        result = await hybrid.retrieve(
            query="fresh dying",
            query_vector=[0.0] * EMBEDDING_DIM,
            top_k=5,
        )
        got = {d.chunk_id for d in result.documents}
        assert "fresh" in got
        assert "dying" not in got
