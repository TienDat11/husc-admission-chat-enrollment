"""D1 regression: `_probe_distinct_years` must run against a real LanceDB table.

History: the body called `table.to_pandas(columns=["data_year"])`, which raises
`TypeError: unexpected keyword argument 'columns'` on lancedb 0.29.2. Every
existing test monkey-patched `_probe_distinct_years` away
(test_admission_context.py:183,190), so the broken body never executed, and the
bare `except` at lancedb_retrieval.py:442 turned the TypeError into a
plausible-looking `True`. Green tests, wrong answers.

This test exercises the REAL body. No monkey-patching of the method under test.
"""
from __future__ import annotations

import lancedb
import pyarrow as pa
import pytest

from src.services.lancedb_retrieval import LanceDBRetriever


class _StubAdapter:
    """Minimal stand-in: `_probe_distinct_years` only touches `_table`."""

    def __init__(self, table):
        self._table = table


class _StubRetriever:
    def __init__(self, table):
        self._adapter = _StubAdapter(table)

    # Bind the real implementation — this is the code under test.
    _probe_distinct_years = LanceDBRetriever._probe_distinct_years


def _make_table(tmp_path, years):
    db = lancedb.connect(str(tmp_path / "lancedb"))
    schema = pa.schema(
        [pa.field("id", pa.string()), pa.field("data_year", pa.int64())]
    )
    rows = [{"id": f"r{i}", "data_year": y} for i, y in enumerate(years)]
    return db.create_table("probe", data=rows, schema=schema)


def test_returns_distinct_years_from_real_table(tmp_path):
    table = _make_table(tmp_path, [2024, 2025, 2026, 2026, 2025])

    got = _StubRetriever(table)._probe_distinct_years()

    assert got == {2024, 2025, 2026}


def test_zero_sentinel_is_kept_not_crashed_on(tmp_path):
    """Production husc_chunks carries `data_year == 0` for unknown-year rows."""
    table = _make_table(tmp_path, [0, 2026])

    got = _StubRetriever(table)._probe_distinct_years()

    assert got == {0, 2026}


def test_null_years_are_skipped(tmp_path):
    db = lancedb.connect(str(tmp_path / "lancedb"))
    schema = pa.schema(
        [pa.field("id", pa.string()), pa.field("data_year", pa.int64())]
    )
    table = db.create_table(
        "probe",
        data=[{"id": "a", "data_year": 2026}, {"id": "b", "data_year": None}],
        schema=schema,
    )

    got = _StubRetriever(table)._probe_distinct_years()

    assert got == {2026}, "NULL/NaN years must be skipped, not counted as 0"


def test_detect_current_admission_year_reads_the_table(tmp_path, monkeypatch):
    """Covers the sibling site at lancedb_retrieval.py:37, which had NO test.

    That call runs at module import and is wrapped in a bare `except: pass`
    falling back to the clock, so a revert to `to_pandas(columns=)` looks fine:
    the clock year happens to match the data year today. The seeded year here is
    deliberately far from any plausible clock year, so a fallback cannot fake it.
    """
    uri = str(tmp_path / "lancedb")
    db = lancedb.connect(uri)
    schema = pa.schema(
        [pa.field("id", pa.string()), pa.field("data_year", pa.int64())]
    )
    db.create_table("probe", data=[{"id": "a", "data_year": 2031}], schema=schema)

    import src.services.lancedb_retrieval as mod

    class _Settings:
        LANCEDB_URI = uri
        LANCEDB_TABLE = "probe"

    monkeypatch.setattr(mod, "RAGSettings", _Settings)

    assert mod._detect_current_admission_year() == 2031, (
        "must read data_year from the table, not fall back to the system clock"
    )


def test_scans_every_row_not_a_truncated_head(tmp_path):
    """`search()` must not silently cap the scan — DISTINCT needs all rows.

    Guards the projection-pushdown rewrite: a default row limit (10 in some
    LanceDB versions) would make the probe miss years beyond the head, so
    `has_current_year_data` would report False for data that is present.
    The rare year sits at the tail, past any such cap.
    """
    years = [2024] * 40 + [2026]
    table = _make_table(tmp_path, years)

    got = _StubRetriever(table)._probe_distinct_years()

    assert got == {2024, 2026}, (
        f"expected the tail year to be seen across {len(years)} rows, got {got}"
    )


def test_does_not_pass_columns_kwarg_to_to_pandas(tmp_path):
    """Falsification guard: pins the 0.29.2 incompatibility itself.

    If a future lancedb adds a `columns=` kwarg this fails, telling us the
    workaround can be dropped.
    """
    table = _make_table(tmp_path, [2026])

    with pytest.raises(TypeError, match="columns"):
        table.to_pandas(columns=["data_year"])
