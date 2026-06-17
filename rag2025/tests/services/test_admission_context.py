# @spec(S14.2 + S14.2b) test_admission_context — kill the import-freeze
"""TDD: prove per-call resolution of admission context.

These tests assert the contract spelled out in the S14.2b plan section
("ARCHITECT MUST-FIX #1 — blocks S14.3/S14.4"): every value the retriever
consumes from `CURRENT_ADMISSION_YEAR`, `has_current_year_data`, and
`SeasonPhase` must be resolved *per `retrieve()` call*, not at import time.

Key invariants:
  1. Changing `CURRENT_ADMISSION_YEAR` env at runtime flips
     `get_admission_context()["year"]` without reimporting `lancedb_retrieval`.
  2. The data-presence probe is per-call: faking a new distinct-year set
     between two calls in the same process must flip `has_current_year_data(2027)`
     from False to True.
  3. The probe must NEVER raise — on adapter failure it returns the
     conservative default `True` (so prod does not accidentally trigger
     empty-answer gap behavior).
  4. With year=2026, `_apply_temporal_filter` produces the SAME row order
     as the old hardcoded-2026 baseline (no-regression safety net).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# Make sure `src.services.lancedb_retrieval` resolves regardless of where
# pytest is invoked from (mirrors the convention used in
# tests/services/test_metadata_helpers.py).
RAG_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RAG_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
    CURRENT_ADMISSION_YEAR,
    YEAR_BOOST_FACTOR,
    OLD_YEAR_DEMOTE_FACTOR,
)
from services.temporal_authority import get_current_admission_year  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_adapter(row_count: int = 10) -> MagicMock:
    """Build a MagicMock that quacks like LanceDBAdapter for has_current_year_data.

    We intentionally bypass `LanceDBRetriever.__init__` so the test does not
    need a real LanceDB connection. The fake exposes:
      - ensure_table()  → no-op
      - _table.count_rows()  → row_count
      - _table.schema.names  → ['data_year', ...] so the column probe succeeds
    The actual `to_pandas` is overridden per-test (default: not called).
    """
    fake = MagicMock()
    fake.ensure_table = MagicMock()
    fake_table = MagicMock()
    fake_table.schema.names = ["data_year", "text", "vector", "source",
                              "metadata_json", "chunk_id", "id"]
    fake_table.count_rows = MagicMock(return_value=row_count)
    # to_pandas should not be reached in tests that monkey-patch the probe;
    # if it IS reached, return an empty DataFrame-shaped object.
    fake_table.to_pandas = MagicMock(return_value=_empty_year_df())
    fake._table = fake_table
    return fake


def _empty_year_df():
    """Minimal stand-in for a pandas DataFrame; only the iteration is used."""
    class _EmptyDF:
        def __getitem__(self, key):
            class _Col:
                def tolist(self_inner):
                    return []
            return _Col()
    return _EmptyDF()


def _make_retriever(row_count: int = 10) -> LanceDBRetriever:
    """Build a LanceDBRetriever with a fake adapter, bypassing __init__."""
    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://", table_name="t", embedding_dim=4, default_top_k=5
    )
    retriever._adapter = _make_fake_adapter(row_count=row_count)
    return retriever


def _row(text: str, distance: float, data_year, is_superseded: bool = False) -> Dict[str, Any]:
    """Build a synthetic row shaped like one returned by LanceDBAdapter.search."""
    return {
        "text": text,
        "source": "synthetic",
        "chunk_id": text,
        "id": text,
        "metadata_json": json.dumps(
            {"data_year": data_year, "is_superseded": is_superseded}
        ),
        "_distance": distance,
    }


# json is used in _row() above; import lazily so test collection does not
# require a specific order.
import json  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: per-call year resolution (proves no import-freeze)
# ---------------------------------------------------------------------------

class TestPerCallYearResolution:
    """Changing CURRENT_ADMISSION_YEAR at runtime must take effect immediately,
    without reimporting `services.lancedb_retrieval`. This is the core
    "import-freeze" fix."""

    def test_env_year_flips_year_field_without_reimport(self, monkeypatch):
        # Start with a known year.
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2030")
        monkeypatch.delenv("APP_ENV", raising=False)

        retriever = _make_retriever()
        ctx_a = retriever.get_admission_context()
        assert ctx_a["year"] == 2030, (
            "Per-call resolution broken: env CURRENT_ADMISSION_YEAR=2030 "
            f"should produce year=2030, got {ctx_a['year']}"
        )

        # Flip env WITHOUT touching the module or reimporting anything.
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2027")
        ctx_b = retriever.get_admission_context()
        assert ctx_b["year"] == 2027, (
            "Import-freeze regression: env change to 2027 should take "
            f"effect on next call; got {ctx_b['year']}"
        )

        # Flip back, for good measure.
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2025")
        ctx_c = retriever.get_admission_context()
        assert ctx_c["year"] == 2025

    def test_frozen_module_constant_is_not_the_authority(self, monkeypatch):
        """`CURRENT_ADMISSION_YEAR` (the module symbol) may be a stale initial
        value. The LIVE context must come from the authority, not the symbol.
        """
        # The module-level symbol is whatever it was at import time.
        # We do not assume its value here — we just verify that the live
        # context does NOT equal it once we change the env.
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2099")
        monkeypatch.delenv("APP_ENV", raising=False)

        retriever = _make_retriever()
        ctx = retriever.get_admission_context()
        assert ctx["year"] == 2099
        # If the symbol still equals 2099 it's a coincidence; what matters
        # is that the context tracks the env, not the symbol.
        assert ctx["year"] != CURRENT_ADMISSION_YEAR or CURRENT_ADMISSION_YEAR == 2099


# ---------------------------------------------------------------------------
# Test 2: data-presence probe flips between calls in the same process
# ---------------------------------------------------------------------------

class TestDataPresenceProbeFlips:
    """A change to the underlying distinct-year set must be visible on the
    next call, without reimport. The cache is keyed by row-count, so
    refreshing the row-count (or any cache bust) is the documented way to
    invalidate; tests use `_distinct_years_count` to force a refresh."""

    def test_has_current_year_data_flips_false_to_true(self, monkeypatch):
        retriever = _make_retriever(row_count=10)

        # Index currently has only 2026 data.
        monkeypatch.setattr(retriever, "_probe_distinct_years", lambda: {2026})
        assert retriever.has_current_year_data(2026) is True
        assert retriever.has_current_year_data(2027) is False

        # Now 2027 is ingested in the SAME process. Simulate by:
        #  (a) bumping the row count (matches the documented cache key), AND
        #  (b) updating what the probe returns.
        monkeypatch.setattr(retriever, "_probe_distinct_years",
                            lambda: {2026, 2027})
        retriever._adapter._table.count_rows.return_value = 11  # row-count changed
        assert retriever.has_current_year_data(2027) is True, (
            "Import-freeze regression: data-presence probe must reflect "
            "post-ingest state on next call, without reimport."
        )

    def test_has_current_year_data_uses_cache_when_row_count_unchanged(
        self, monkeypatch
    ):
        """When row count is unchanged, the cached distinct set is reused.

        This documents the stale-set staleness window explicitly per the
        pre-mortem in the plan. We force this by setting the cache directly
        and verifying the probe is NOT re-called.
        """
        retriever = _make_retriever(row_count=10)

        # Populate cache as if 2027 was already there.
        retriever._distinct_years_cache = {2026, 2027}
        retriever._distinct_years_count = 10

        # If the probe were called, this would explode — instead, the cache
        # must answer.
        def _explode():
            raise AssertionError("probe should not be called when cache is fresh")

        monkeypatch.setattr(retriever, "_probe_distinct_years", _explode)
        assert retriever.has_current_year_data(2027) is True


# ---------------------------------------------------------------------------
# Test 3: probe never raises — conservative default
# ---------------------------------------------------------------------------

class TestProbeNeverRaises:
    """On any error, the probe returns True (assume data present).

    Why True: in prod, a False here would flip the season to PRE_SEASON_GAP
    and emit the gap disclaimer + labeled prior-year context. That is a
    graceful degradation, but it should not happen by accident from a
    transient adapter error. Conservative = assume data is there, let
    retrieval proceed normally; if it is in fact empty, the empty-result
    path already handles that."""

    def test_adapter_ensure_table_failure_returns_true(self, monkeypatch):
        retriever = _make_retriever()
        # Make adapter.ensure_table() raise — must not propagate.
        retriever._adapter.ensure_table.side_effect = RuntimeError("table missing")
        assert retriever.has_current_year_data(2027) is True

    def test_count_rows_failure_returns_true(self, monkeypatch):
        retriever = _make_retriever()
        retriever._adapter._table.count_rows.side_effect = OSError("disk gone")
        assert retriever.has_current_year_data(2027) is True

    def test_probe_failure_returns_true(self, monkeypatch):
        retriever = _make_retriever()
        # Force a cache miss (different row-count), then make the probe raise.
        retriever._distinct_years_count = 0
        monkeypatch.setattr(
            retriever, "_probe_distinct_years",
            lambda: (_ for _ in ()).throw(ValueError("corrupt df"))
        )
        assert retriever.has_current_year_data(2027) is True

    def test_schema_missing_data_year_returns_true(self, monkeypatch):
        """If the table has no data_year column at all, the question is
        unanswerable — fall back to the conservative True."""
        retriever = _make_retriever()
        retriever._adapter._table.schema.names = ["text", "vector"]  # no data_year
        assert retriever.has_current_year_data(2027) is True


# ---------------------------------------------------------------------------
# Test 4: no-regression — byte-identical ordering for the 2026 case
# ---------------------------------------------------------------------------

class TestNoRegressionOrdering2026:
    """When env year (or dev fallback) == data-max == 2026, the row order
    produced by `_apply_temporal_filter` MUST be byte-identical to the old
    hardcoded-2026 baseline. This is the safety net promised in the task
    brief: "no code change in ordering for the 2026 case"."""

    def test_temporal_filter_ordering_matches_hardcoded_2026(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        assert get_current_admission_year() == 2026  # sanity

        retriever = _make_retriever()
        # Build 4 rows that exercise boost / demote / no-year / superseded.
        rows: List[Dict[str, Any]] = [
            _row("boost-2026",  distance=0.50, data_year=2026),  # → 0.50/1.5 = 0.333
            _row("demote-2025", distance=0.40, data_year=2025),  # → 0.40/0.7 = 0.571
            _row("demote-2025b", distance=0.30, data_year=2025),  # → 0.30/0.7 = 0.428
            _row("year-agnostic", distance=0.20, data_year=None),  # → 0.20 (unchanged)
        ]

        out = retriever._apply_temporal_filter(rows, metadata_filter=None)

        # Extract the effective distance in the order returned.
        got = [(r["chunk_id"], round(r.get("_distance", 0.0), 4)) for r in out]
        # Manually computed expected order (ascending distance):
        expected = [
            ("year-agnostic", round(0.20, 4)),
            ("boost-2026",    round(0.50 / YEAR_BOOST_FACTOR, 4)),
            ("demote-2025b",  round(0.30 / OLD_YEAR_DEMOTE_FACTOR, 4)),
            ("demote-2025",   round(0.40 / OLD_YEAR_DEMOTE_FACTOR, 4)),
        ]
        assert got == expected, (
            f"No-regression regression: expected {expected}, got {got}"
        )

        # Side-effect sanity: the boosted row should carry the flag.
        boost = next(r for r in out if r["chunk_id"] == "boost-2026")
        assert boost.get("_temporal_boosted") is True
        # Demoted rows should carry the demote flag.
        for cid in ("demote-2025", "demote-2025b"):
            r = next(r for r in out if r["chunk_id"] == cid)
            assert r.get("_temporal_demoted") is True, (
                f"{cid} should be flagged _temporal_demoted"
            )

    def test_temporal_filter_uses_per_call_year_for_boost(
        self, monkeypatch
    ):
        """Same rows, different env year → different boost target.

        With year=2025, the row tagged data_year=2025 should be boosted
        (not the 2026 one). This proves the function reads the LIVE year,
        not a frozen constant.
        """
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2025")
        monkeypatch.delenv("APP_ENV", raising=False)

        retriever = _make_retriever()
        rows = [
            _row("y2026", distance=0.50, data_year=2026),
            _row("y2025", distance=0.50, data_year=2025),
        ]
        out = retriever._apply_temporal_filter(rows, metadata_filter=None)
        # y2025 should be boosted → effective distance 0.50/1.5 = 0.333
        # y2026 should be demoted  → effective distance 0.50/0.7 = 0.714
        first = out[0]
        assert first["chunk_id"] == "y2025", (
            f"y2025 should rank first when current year is 2025; got {first['chunk_id']}"
        )
        assert first.get("_temporal_boosted") is True


# ---------------------------------------------------------------------------
# Test 5: get_admission_context shape and composition
# ---------------------------------------------------------------------------

class TestGetAdmissionContext:
    """The single helper S14.3/S14.4 will consume. Must expose year, has_data,
    season as a dict with the documented types."""

    def test_context_returns_required_keys(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        retriever = _make_retriever()
        monkeypatch.setattr(retriever, "_probe_distinct_years", lambda: {2026})
        ctx = retriever.get_admission_context()
        assert set(ctx.keys()) == {"year", "has_data", "season"}, (
            f"Context keys must be exactly {{year, has_data, season}}; got {set(ctx.keys())}"
        )
        assert isinstance(ctx["year"], int)
        assert isinstance(ctx["has_data"], bool)

    def test_context_season_is_in_season_when_data_present(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        from services.season import SeasonPhase
        retriever = _make_retriever()
        monkeypatch.setattr(retriever, "_probe_distinct_years", lambda: {2026})
        ctx = retriever.get_admission_context()
        assert ctx["has_data"] is True
        assert ctx["season"] == SeasonPhase.IN_SEASON

    def test_context_year_never_raises(self, monkeypatch):
        """Even with no env and a probe that errors, get_admission_context
        must not raise (probe returns True conservatively → IN_SEASON)."""
        monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)

        retriever = _make_retriever()
        retriever._adapter.ensure_table.side_effect = RuntimeError("nope")
        ctx = retriever.get_admission_context()  # must not raise
        assert isinstance(ctx["year"], int)
        assert ctx["has_data"] is True  # conservative
