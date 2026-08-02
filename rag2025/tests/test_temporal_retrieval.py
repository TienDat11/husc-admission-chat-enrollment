"""Tests for temporal retrieval filter in LanceDBRetriever."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add rag2025/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.lancedb_retrieval import (
    CURRENT_ADMISSION_YEAR,
    LanceDBRetriever,
    LanceDBRetrieverConfig,
    YEAR_BOOST_FACTOR,
    OLD_YEAR_DEMOTE_FACTOR,
    _detect_current_admission_year,
)


# --- Temporal filter unit tests ---


@pytest.fixture(autouse=True)
def _pin_admission_year_2026():
    """Pin the admission-year authority to 2026 for calendar-deterministic
    temporal-filter tests. `_apply_temporal_filter` resolves the current year
    per-call from `temporal_authority.get_current_admission_year()` (env-driven),
    so setting the env var makes these tests independent of the wall-clock
    calendar rollover (month>=8 -> year+1)."""
    os.environ["CURRENT_ADMISSION_YEAR"] = "2026"
    yield
    os.environ.pop("CURRENT_ADMISSION_YEAR", None)


def test_current_admission_year_is_2026_or_higher():
    year = _detect_current_admission_year()
    assert year >= 2026


def test_boost_factor_values():
    assert YEAR_BOOST_FACTOR == 1.5
    assert OLD_YEAR_DEMOTE_FACTOR == 0.7


def test_temporal_filter_boosts_current_year():
    retriever = _make_retriever()
    rows = [
        {"_distance": 0.1, "metadata": json.dumps({"data_year": 2025, "is_superseded": False})},
        {"_distance": 0.1, "metadata": json.dumps({"data_year": 2026, "is_superseded": False})},
    ]
    filtered = retriever._apply_temporal_filter(rows, None)
    # 2026 row should come first (boosted = lower effective distance)
    meta_0 = json.loads(filtered[0].get("metadata", "{}"))
    assert meta_0.get("data_year") == 2026


def test_temporal_filter_demotes_old_year():
    retriever = _make_retriever()
    rows = [
        {"_distance": 0.05, "metadata": json.dumps({"data_year": 2025, "is_superseded": False})},
        {"_distance": 0.10, "metadata": json.dumps({"data_year": 2026, "is_superseded": False})},
    ]
    filtered = retriever._apply_temporal_filter(rows, None)
    # Even though 2025 has better raw distance, boosted 2026 should rank first
    meta_0 = json.loads(filtered[0].get("metadata", "{}"))
    assert meta_0.get("data_year") == 2026


def test_temporal_filter_excludes_superseded():
    retriever = _make_retriever()
    rows = [
        {"_distance": 0.01, "metadata": json.dumps({"data_year": 2025, "is_superseded": True})},
        {"_distance": 0.10, "metadata": json.dumps({"data_year": 2026, "is_superseded": False})},
    ]
    filtered = retriever._apply_temporal_filter(rows, None)
    # Superseded 2025 should be demoted far down
    meta_last = json.loads(filtered[-1].get("metadata", "{}"))
    assert meta_last.get("is_superseded") is True


def test_temporal_filter_respects_explicit_year_query():
    retriever = _make_retriever()
    rows = [
        {"_distance": 0.05, "metadata": json.dumps({"data_year": 2025, "is_superseded": False})},
        {"_distance": 0.10, "metadata": json.dumps({"data_year": 2026, "is_superseded": False})},
    ]
    # When user explicitly asks for 2025, don't demote it
    filtered = retriever._apply_temporal_filter(rows, {"data_year": 2025})
    # 2025 should not be demoted
    for row in filtered:
        meta = json.loads(row.get("metadata", "{}"))
        if meta.get("data_year") == 2025:
            assert not row.get("_temporal_demoted", False)


def test_temporal_filter_empty_rows():
    retriever = _make_retriever()
    assert retriever._apply_temporal_filter([], None) == []


def test_2026_chunks_have_correct_metadata():
    """Verify 2026 tuition chunks in chunked_10.jsonl have required temporal fields."""
    chunked_path = Path(__file__).parent.parent / "data" / "chunked" / "chunked_10.jsonl"
    if not chunked_path.exists():
        pytest.skip("chunked_10.jsonl not found")

    found_2026 = []
    with open(chunked_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            meta = data.get("metadata", {})
            if meta.get("data_year") == 2026 or meta.get("year") == "2026":
                found_2026.append(data)

    assert len(found_2026) >= 2, f"Expected at least 2 chunks with data_year=2026, got {len(found_2026)}"
    for chunk in found_2026:
        meta = chunk.get("metadata", {})
        assert meta.get("is_superseded") is False
        assert meta.get("freshness_score") == 1.0
        assert "supersedes" in meta
        # Verify price fields
        assert meta.get("current_price_khoi_VII") == 525000 or meta.get("base_price_khoi_VII") == 525000
        assert meta.get("current_price_khoi_IV") == 540000 or meta.get("base_price_khoi_IV") == 540000
        assert meta.get("current_price_khoi_V") == 600000 or meta.get("base_price_khoi_V") == 600000


def test_2025_chunks_marked_expired():
    """Verify 2025 tuition chunks are marked as expired and superseded."""
    chunked_path = Path(__file__).parent.parent / "data" / "chunked" / "chunked_10.jsonl"
    if not chunked_path.exists():
        pytest.skip("chunked_10.jsonl not found")

    for line in chunked_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        chunk_id = data.get("id", "")
        if "hoc_phi" in chunk_id and "2025" in chunk_id:
            meta = data.get("metadata", {})
            assert meta.get("expired") is True, f"{chunk_id} should be expired"
            assert meta.get("superseded_by") is not None, f"{chunk_id} should have superseded_by"


def test_86q_no_flat_545000_answers():
    """Verify 86Q test file has no flat '545.000 VNĐ' tuition answers for current-year questions."""
    eval_path = Path(__file__).parent.parent / "data" / "eval" / "husc_86q_test.json"
    if not eval_path.exists():
        pytest.skip("husc_86q_test.json not found")

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    flat_545_count = 0
    for q in data:
        question = q.get("question", "")
        answer = q.get("answer", "")
        must_include = q.get("must_include", [])
        # Skip historical queries that explicitly ask about a past year
        is_historical = any(yr in question for yr in ["2025", "2024", "2023"])
        if is_historical:
            continue
        # Check answer text
        if "545.000 VNĐ/tín chỉ" in answer and "khối" not in answer.lower():
            flat_545_count += 1
        # Check must_include still referencing 545000 without context
        for item in must_include:
            if item == "545000" and "khối" not in answer.lower():
                flat_545_count += 1

    assert flat_545_count == 0, f"Found {flat_545_count} references to flat 545.000 tuition"


# --- Helpers ---


def _make_retriever() -> LanceDBRetriever:
    """Create a LanceDBRetriever with mocked adapter."""
    mock_adapter = MagicMock()
    mock_adapter.table_exists.return_value = True
    mock_adapter.count.return_value = 100
    mock_adapter._table = MagicMock()

    config = MagicMock(spec=LanceDBRetrieverConfig)
    config.uri = "test"
    config.table_name = "test"
    config.embedding_dim = 4096
    config.default_top_k = 5

    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = config
    retriever._adapter = mock_adapter
    return retriever
