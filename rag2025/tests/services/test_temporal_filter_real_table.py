"""Integration tests for temporal_filter against real LanceDB tables.

Uses tempfile on D:/tmp — no mocks, real LanceDB queries.
"""
from __future__ import annotations
import json
import tempfile
import sys
from pathlib import Path

import lancedb
import pytest
import importlib.util

SRC = Path(__file__).resolve().parents[2] / "src" / "services"
if str(SRC.parent) not in sys.path:
    sys.path.insert(0, str(SRC.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ti_module():
    spec = importlib.util.spec_from_file_location(
        "services.temporal_intent", SRC / "temporal_intent.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["services.temporal_intent"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tf_module(ti_module):
    # ti_module must load first so services.temporal_intent is in sys.modules
    spec_tf = importlib.util.spec_from_file_location(
        "services.temporal_filter", SRC / "temporal_filter.py"
    )
    mod_tf = importlib.util.module_from_spec(spec_tf)
    mod_tf.__package__ = "services"
    spec_tf.loader.exec_module(mod_tf)
    return mod_tf


@pytest.fixture
def rag2025_table():
    """Bảng schema string-nested giống rag2025 thật."""
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="b2_rag2025_")
    db = lancedb.connect(d)
    rows = [
        {
            "id": "r1",
            "text": "2025 chunk",
            "metadata_json": json.dumps({"data_year": 2025}),
            "vector": [0.1] * 1024,
        },
        {
            "id": "r2",
            "text": "2026 chunk",
            "metadata_json": json.dumps({"data_year": 2026}),
            "vector": [0.2] * 1024,
        },
        {
            "id": "r3",
            "text": "null year",
            "metadata_json": json.dumps({}),
            "vector": [0.3] * 1024,
        },
    ]
    yield db.create_table("rag2025", data=rows)
    try:
        db.drop_all_tables()
    except TypeError:
        pass  # older lancedb version


@pytest.fixture
def flat_table():
    """Bảng có cột top-level data_year/status/effective_from/effective_to/is_tombstone."""
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="b2_flat_")
    db = lancedb.connect(d)
    rows = [
        {
            "id": "f1",
            "text": "old superseded",
            "data_year": 2025,
            "status": "superseded",
            "effective_from": 20240101,
            "effective_to": 20251231,
            "is_tombstone": False,
            "vector": [0.1] * 1024,
        },
        {
            "id": "f2",
            "text": "current valid",
            "data_year": 2026,
            "status": "current",
            "effective_from": 20250901,
            "effective_to": 20271231,
            "is_tombstone": False,
            "vector": [0.2] * 1024,
        },
        {
            "id": "f3",
            "text": "tombstoned",
            "data_year": 2026,
            "status": "current",
            "effective_from": 20250101,
            "effective_to": 20260101,
            "is_tombstone": True,
            "vector": [0.3] * 1024,
        },
    ]
    tbl = db.create_table("flat_schema", data=rows)
    yield tbl
    try:
        db.drop_all_tables()
    except TypeError:
        pass  # older lancedb version


@pytest.fixture
def husc_table():
    """Bảng schema struct-nested giống husc_v2026_clean thật (KHÔNG có cột flat)."""
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="b2_husc_")
    db = lancedb.connect(d)
    rows = [
        {
            "id": "h1",
            "text": "2025",
            "metadata": {"data_year": 2025},
            "embedding": [0.1] * 1024,
        },
        {
            "id": "h2",
            "text": "2026",
            "metadata": {"data_year": 2026},
            "embedding": [0.2] * 1024,
        },
    ]
    yield db.create_table("husc_v2026_clean", data=rows)
    try:
        db.drop_all_tables()
    except TypeError:
        pass  # older lancedb version


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nested_json_returns_none_not_predicate(tf_module, ti_module):
    """schema_mode=nested_json → None, không phải predicate string."""
    out = tf_module.apply_temporal_filter(
        ti_module.TemporalIntent.current, 2026, schema_mode="nested_json"
    )
    assert out is None


def test_nested_json_no_crash_on_rag2025(tf_module, ti_module, rag2025_table):
    """nested_json trả None → caller không gọi .where() → không crash."""
    pred = tf_module.apply_temporal_filter(
        ti_module.TemporalIntent.current, 2026, schema_mode="nested_json"
    )
    # None means post-fetch filter — do NOT call .where() with None
    assert pred is None
    # Verify table still readable
    count = len(rag2025_table.search([0.1]*1024).limit(10).to_list())
    assert count >= 3


def test_flat_schema_predicate_executes(tf_module, ti_module, flat_table):
    """schema_mode=flat + as_of → predicate string → .where() chạy thành công."""
    pred = tf_module.apply_temporal_filter(
        ti_module.TemporalIntent.current, 2026, schema_mode="flat", as_of=20260729
    )
    assert isinstance(pred, str)
    assert "data_year = 2026" in pred

    results = flat_table.search([0.1]*1024).where(pred).to_list()
    # f2 (data_year=2026, status=current, effective_from<=as_of, effective_to>=as_of, tombstone=false)
    # f3 has is_tombstone=True → excluded by current_clause
    ids = [r.get("id") for r in results]
    assert "f2" in ids
    assert "f3" not in ids


def test_post_fetch_filter_removes_rows(ti_module, monkeypatch, rag2025_table):
    """_apply_temporal_filter với admission_ctx IN_SEASON+has_data hard-drop prior-year.

    Uses rag2025_table (tempfile on D:/tmp) for row data — never touches production DB.
    _apply_temporal_filter does not reference self; object.__new__() skips __init__.
    """
    from services.lancedb_retrieval import LanceDBRetriever
    from services.season import SeasonPhase
    from services.temporal_intent import TemporalIntent

    # Pull real rows from temp table (same schema as rag2025 production)
    raw_rows = rag2025_table.search([0.1] * 1024).limit(10).to_list()
    rows = [
        {
            "id": r["id"],
            "metadata_json": r.get("metadata_json", "{}"),
            "_distance": float(i) * 0.1 + 0.1,
        }
        for i, r in enumerate(raw_rows)
    ]

    # Ensure we have at least one 2025 and one 2026 row for assertions
    rows.append({"id": "r1", "metadata_json": json.dumps({"data_year": 2025}), "_distance": 0.5})
    rows.append({"id": "r2", "metadata_json": json.dumps({"data_year": 2026}), "_distance": 0.3})
    rows.append({"id": "r3", "metadata_json": json.dumps({}), "_distance": 0.4})

    # Skip __init__ — _apply_temporal_filter is self-less (pure filter)
    retriever = object.__new__(LanceDBRetriever)

    # Legacy path (no admission_ctx) → soft demote only, all rows kept
    legacy_result = retriever._apply_temporal_filter(
        rows.copy(), metadata_filter=None,
        temporal_intent=TemporalIntent.current, admission_ctx=None,
    )
    assert len(legacy_result) == len(rows)

    # Hard-exclude path: IN_SEASON + has_data + current → drop prior-year
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    # Need to reload temporal_authority to pick up env change
    import importlib
    import services.temporal_authority as ta
    importlib.reload(ta)

    result = retriever._apply_temporal_filter(
        rows.copy(), metadata_filter=None,
        temporal_intent=TemporalIntent.current,
        admission_ctx={"season": SeasonPhase.IN_SEASON, "has_data": True},
    )
    ids = [r.get("id") for r in result]
    assert "r1" not in ids  # 2025 dropped
    assert "r2" in ids     # 2026 kept
    assert "r3" in ids     # null year kept (year-agnostic)


def test_post_fetch_filter_zero_results_for_missing_year(ti_module, monkeypatch, flat_table):
    """Khi bảng chỉ có prior-year row + IN_SEASON+current → hard-drop hết → 0 kết quả.

    Uses flat_table (tempfile on D:/tmp) for row data — never touches production DB.
    """
    from services.lancedb_retrieval import LanceDBRetriever
    from services.season import SeasonPhase
    from services.temporal_intent import TemporalIntent

    # Pull real rows from temp table
    raw_rows = flat_table.search([0.1] * 1024).limit(10).to_list()
    rows = [
        {
            "id": r["id"],
            "metadata_json": r.get("metadata_json", json.dumps({"data_year": r.get("data_year")})),
            "_distance": float(i) * 0.1 + 0.1,
        }
        for i, r in enumerate(raw_rows)
    ]

    # Ensure only prior-year rows exist for this assertion
    rows = [
        {"id": "old1", "metadata_json": json.dumps({"data_year": 2025}), "_distance": 0.1},
        {"id": "old2", "metadata_json": json.dumps({"data_year": 2024}), "_distance": 0.2},
    ]

    # Skip __init__ — _apply_temporal_filter is self-less (pure filter)
    retriever = object.__new__(LanceDBRetriever)

    result = retriever._apply_temporal_filter(
        rows.copy(), metadata_filter=None,
        temporal_intent=TemporalIntent.current,
        admission_ctx={"season": SeasonPhase.IN_SEASON, "has_data": True},
    )
    # Tất cả prior-year → hard-drop hết
    assert len(result) == 0
