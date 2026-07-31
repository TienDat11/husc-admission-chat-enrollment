"""TDD: _apply_temporal_filter emits degraded_filter flag (M3-N4b).

REAL LanceDB table under D:/tmp — no mocks on table ops.
Tests _apply_temporal_filter + _normalize_rows + RetrievalResult.degraded_filter.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import lancedb
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig


@pytest.fixture()
def temp_db():
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_degraded_")
    db = lancedb.connect(d)
    yield db
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _make_retriever(temp_db) -> LanceDBRetriever:
    """Create a retriever pointing at temp_db with a test table."""
    # Create minimal table
    temp_db.create_table("test_degraded", data=[
        {
            "id": "x1",
            "text": "dummy",
            "vector": [0.1] * 1024,
            "metadata_json": "{}",
        },
    ])
    config = LanceDBRetrieverConfig(
        uri=temp_db.uri,
        table_name="test_degraded",
        embedding_dim=1024,
        default_top_k=5,
    )
    return LanceDBRetriever(config=config)


# ---------------------------------------------------------------------------
# Test 1 — prior-year fallback sets degraded_filter
# ---------------------------------------------------------------------------


def test_prior_year_fallback_sets_degraded_filter(temp_db: Any):
    """Rows with data_year=2025, current=2026, season=PRE_SEASON_GAP → _degraded_filter=True."""
    retriever = _make_retriever(temp_db)

    rows = [
        {
            "id": "r1",
            "text": "Old year content",
            "metadata_json": json.dumps({"data_year": 2025}),
            "_distance": 0.5,
        },
        {
            "id": "r2",
            "text": "Another old",
            "metadata_json": json.dumps({"data_year": 2025}),
            "_distance": 0.6,
        },
    ]

    # Monkeypatch get_current_admission_year → 2026
    with patch("services.temporal_authority.get_current_admission_year", return_value=2026):
        from src.services.season import SeasonPhase
        from src.services.temporal_intent import TemporalIntent

        admission_ctx = {
            "year": 2026,
            "has_data": True,
            "season": SeasonPhase.PRE_SEASON_GAP,
        }

        result = retriever._apply_temporal_filter(
            rows,
            metadata_filter=None,
            temporal_intent=TemporalIntent.current,
            admission_ctx=admission_ctx,
        )

    # At least 1 row must have _degraded_filter=True
    assert any(r.get("_degraded_filter") is True for r in result), (
        f"No row has _degraded_filter=True: {result}"
    )

    # After _normalize_rows, doc.metadata["degraded_filter"] must be True
    docs = retriever._normalize_rows(result)
    assert any(d.metadata.get("degraded_filter") is True for d in docs), (
        f"No doc has metadata.degraded_filter=True: {[d.metadata for d in docs]}"
    )


# ---------------------------------------------------------------------------
# Test 2 — current-year rows do NOT get degraded_filter
# ---------------------------------------------------------------------------


def test_current_year_no_degraded_flag(temp_db: Any):
    """Rows with data_year=2026=current → no degraded_filter."""
    retriever = _make_retriever(temp_db)

    rows = [
        {
            "id": "c1",
            "text": "Current year content",
            "metadata_json": json.dumps({"data_year": 2026}),
            "_distance": 0.3,
        },
    ]

    with patch("services.temporal_authority.get_current_admission_year", return_value=2026):
        from src.services.season import SeasonPhase
        from src.services.temporal_intent import TemporalIntent

        admission_ctx = {
            "year": 2026,
            "has_data": True,
            "season": SeasonPhase.PRE_SEASON_GAP,
        }

        result = retriever._apply_temporal_filter(
            rows,
            metadata_filter=None,
            temporal_intent=TemporalIntent.current,
            admission_ctx=admission_ctx,
        )

    # No row should have _degraded_filter=True
    assert all(r.get("_degraded_filter") is not True for r in result), (
        f"Current-year row incorrectly flagged: {result}"
    )

    docs = retriever._normalize_rows(result)
    assert all(d.metadata.get("degraded_filter") is not True for d in docs)


# ---------------------------------------------------------------------------
# Test 3 — is_reference flat column: demote not drop
# ---------------------------------------------------------------------------


def test_is_reference_flat_column_demotes_not_drops(temp_db: Any):
    """Row with is_reference=True (flat column), data_year=2025, PRE_SEASON_GAP → kept + flagged."""
    retriever = _make_retriever(temp_db)

    rows = [
        {
            "id": "ref1",
            "text": "Reference prior-year",
            "metadata_json": json.dumps({"data_year": 2025}),
            "is_reference": True,  # flat column
            "_distance": 0.4,
        },
    ]

    with patch("services.temporal_authority.get_current_admission_year", return_value=2026):
        from src.services.season import SeasonPhase
        from src.services.temporal_intent import TemporalIntent

        admission_ctx = {
            "year": 2026,
            "has_data": True,
            "season": SeasonPhase.PRE_SEASON_GAP,
        }

        result = retriever._apply_temporal_filter(
            rows,
            metadata_filter=None,
            temporal_intent=TemporalIntent.current,
            admission_ctx=admission_ctx,
        )

    # Row must be KEPT (not dropped)
    assert len(result) == 1, f"Reference row was dropped: {result}"
    # Must have degraded_filter
    assert result[0].get("_degraded_filter") is True


# ---------------------------------------------------------------------------
# Test 4 — RetrievalResult aggregates degraded_filter
# ---------------------------------------------------------------------------


def test_retrieval_result_aggregates_degraded_flag(temp_db: Any):
    """retrieve() with old-year rows → result.degraded_filter is True."""
    retriever = _make_retriever(temp_db)

    # Monkeypatch adapter.search to return old-year rows
    old_rows = [
        {
            "id": "old1",
            "text": "Old content",
            "metadata_json": json.dumps({"data_year": 2025}),
            "_distance": 0.5,
        },
    ]

    with patch.object(retriever._adapter, "search", return_value=old_rows):
        with patch("services.temporal_authority.get_current_admission_year", return_value=2026):
            from src.services.season import SeasonPhase

            # Patch get_admission_context to return PRE_SEASON_GAP
            with patch.object(retriever, "get_admission_context", return_value={
                "year": 2026,
                "has_data": True,
                "season": SeasonPhase.PRE_SEASON_GAP,
            }):
                result = retriever.retrieve(
                    query_vector=[0.1] * 1024,
                    query="điểm chuẩn 2025",
                )

    assert result.is_success
    assert result.degraded_filter is True, (
        f"RetrievalResult.degraded_filter should be True, got {result.degraded_filter}"
    )
