# @spec(S14.3) test_lancedb_temporal_wiring — wire temporal_intent + temporal_filter
"""TDD: prove the retriever hot path applies the prior-year hard-EXCLUDE
POST-FETCH inside `_apply_temporal_filter` (not via a SQL `where_clause`
passed to `LanceDBAdapter.search`).

The real LanceDB table has NO top-level `data_year` column — the year
lives inside the `metadata_json` JSON-string column. A SQL
`where_clause = "data_year = N OR data_year IS NULL"` would crash with
`LanceError(Schema): No field named data_year`. Therefore:
  * `adapter.search` is ALWAYS called with `where_clause=None`.
  * The hard-EXCLUDE is applied POST-FETCH, in `_apply_temporal_filter`,
    which already parses `metadata_json` per row.

Key invariants (S14.3 plan + architect root-cause fix):
  1. IN_SEASON (has_data=True) + intent=current/ambiguous
     → rows whose parsed `data_year` is present AND `< current_year` are
       HARD-DROPPED from the final docs. `data_year == current_year` AND
       `data_year is None` (year-agnostic stable chunks) survive.
  2. IN_SEASON + current + only-prior-year rows → returned docs EMPTY
       (intended "don't show old chunks" behavior).
  3. PRE_SEASON_GAP (has_data=False) → no hard drop; prior-year rows
       are SOFT-DEMOTED but survive in the final docs.
  4. historical intent ("điểm chuẩn năm 2024") → all prior years survive.
  5. cross_year intent ("so sánh điểm chuẩn qua các năm") → all years survive.
  6. Superseded row IN_SEASON → dropped.
  7. Backward-compat: `retrieve(qv, top_k)` with NO `query` → no hard drop
       (legacy soft path; no crash; no where_clause).
  8. CRITICAL: `adapter.search` is called with `where_clause=None` in
       EVERY case (no code path may emit a `data_year` SQL string).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# Make sure `src.services.lancedb_retrieval` resolves regardless of where
# pytest is invoked from (mirrors the convention in test_admission_context.py).
RAG_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RAG_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
    TOMBSTONE_PREDICATE,
)
from services.season import SeasonPhase  # noqa: E402
from services.temporal_intent import TemporalIntent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 4  # small; tests don't care about vector content


def _make_fake_search_adapter(
    rows: List[Dict[str, Any]],
    has_tombstone_column: bool = True,
) -> MagicMock:
    """Build a MagicMock that quacks like LanceDBAdapter for retrieve().

    The fake (a) records the `where_clause` it was called with so tests
    can assert on it, and (b) returns ALL rows unchanged (because the
    contract is that no SQL pre-filtering of `data_year` is done — the
    real table has no `data_year` column).

    `has_tombstone_column` models the schema gate: when True the retriever
    passes TOMBSTONE_PREDICATE as `where_clause`; when False (legacy table)
    it passes None. No real LanceDB is touched.
    """
    fake = MagicMock()
    fake.ensure_table = MagicMock()
    fake.has_tombstone_column = MagicMock(return_value=has_tombstone_column)

    # Recording side-effect: capture the where_clause at call time.
    captured: Dict[str, Any] = {"where_clause": "__unset__"}
    fake._captured = captured

    def _record_search(query_vector, top_k, where_clause=None):
        captured["where_clause"] = where_clause
        return list(rows)  # all rows pass through; the post-fetch filter does the work

    fake.search.side_effect = _record_search
    # _table is needed by has_current_year_data but we monkey-patch
    # get_admission_context per-test, so a simple MagicMock is fine.
    fake._table = MagicMock()
    return fake


def _row_data_year(row: Dict[str, Any]) -> Optional[int]:
    """Extract `data_year` from a synthetic row's metadata_json."""
    raw = row.get("metadata_json")
    if not raw:
        return None
    try:
        meta = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None
    val = meta.get("data_year")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _make_retriever(
    rows: Optional[List[Dict[str, Any]]] = None,
    embedding_dim: int = EMBEDDING_DIM,
    has_tombstone_column: bool = True,
) -> LanceDBRetriever:
    """Build a LanceDBRetriever bypassing __init__ to avoid real LanceDB."""
    if rows is None:
        rows = []
    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://",
        table_name="t",
        embedding_dim=embedding_dim,
        default_top_k=5,
    )
    retriever._adapter = _make_fake_search_adapter(rows, has_tombstone_column=has_tombstone_column)
    return retriever


def _row(
    text: str,
    distance: float = 0.5,
    data_year: Optional[int] = None,
    is_superseded: bool = False,
) -> Dict[str, Any]:
    """Build a synthetic row shaped like one returned by LanceDBAdapter.search.

    `data_year=None` simulates year-agnostic stable chunks (e.g., địa chỉ,
    hotline) that the IN_SEASON+current POST-FETCH hard-drop MUST preserve.
    """
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


def _drive_season(retriever: LanceDBRetriever, *, year: int, has_data: bool) -> None:
    """Monkey-patch get_admission_context to return a fixed season/year tuple.

    This is the test seam: rather than mock the underlying temporal_authority
    / season modules, we patch the *consumer* helper that retrieve() calls.
    Deterministic across tests; no real LanceDB probe runs.
    """
    season = SeasonPhase.IN_SEASON if has_data else SeasonPhase.PRE_SEASON_GAP
    ctx = {"year": year, "has_data": has_data, "season": season}
    monkey = MagicMock(return_value=ctx)
    retriever.get_admission_context = monkey  # type: ignore[assignment]


def _make_vector(dim: int = EMBEDDING_DIM) -> List[float]:
    return [0.0] * dim


# ---------------------------------------------------------------------------
# Critical invariant: `adapter.search` is NEVER called with a data_year
# SQL clause. The real table has no such column. This assertion holds for
# EVERY case in this file.
# ---------------------------------------------------------------------------

def _assert_no_data_year_sql_clause(retriever: LanceDBRetriever) -> None:
    """The whole point of the S14.3 root-cause fix: no code path may emit
    a `data_year` SQL string. The real schema has no such column."""
    captured = retriever._adapter._captured["where_clause"]
    # Must be None on every call. We also accept the literal string "None"
    # (defensive) but the canonical form is None.
    if captured is None or captured == "None":
        return
    assert "data_year" not in str(captured), (
        "CRITICAL: adapter.search received a SQL clause containing "
        f"'data_year': {captured!r}. The real LanceDB table has no "
        "top-level `data_year` column (year lives in `metadata_json`). "
        "Post-fetch hard-EXCLUDE in _apply_temporal_filter is the only "
        "supported path."
    )


# ---------------------------------------------------------------------------
# Test 1: IN_SEASON + current → prior-year rows HARD-DROPPED, null-year SURVIVES
# ---------------------------------------------------------------------------

class TestInSeasonCurrentHardDrop:
    """The 10-year-durability core: once current-year data exists, prior-year
    chunks are HARD-DROPPED at the post-fetch layer (not via SQL)."""

    def test_prior_year_dropped_current_and_null_year_kept(self, monkeypatch):
        """IN_SEASON+current + rows {2026, 2025, null-year}
        → returned docs CONTAIN 2026 + null-year, and DO NOT contain 2025."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
            _row("stable-agnostic", data_year=None),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="điểm chuẩn năm nay",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}

        assert "current-2026" in chunk_ids, (
            f"IN_SEASON+current must KEEP the current-year row; got {chunk_ids}"
        )
        assert "stable-agnostic" in chunk_ids, (
            "IN_SEASON+current must KEEP year-agnostic stable rows "
            f"(data_year=None); got {chunk_ids}"
        )
        assert "prior-2025" not in chunk_ids, (
            "IN_SEASON+current must HARD-DROP prior-year rows "
            f"(data_year=2025); got {chunk_ids}"
        )
        _assert_no_data_year_sql_clause(retriever)

    def test_only_prior_year_rows_returns_empty(self, monkeypatch):
        """IN_SEASON+current with only-prior-year rows → returned docs EMPTY
        (intended "don't show old chunks" behavior). The over-fetch
        headroom (top_k = max(limit*5, limit)) does not change this: the
        POST-FETCH hard-EXCLUDE truly drops them."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("prior-2025", data_year=2025),
            _row("prior-2024", data_year=2024),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="điểm chuẩn năm nay",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        # All prior-year rows must be hard-dropped → final docs is empty.
        assert chunk_ids == set(), (
            f"IN_SEASON+current with only prior-year rows must return EMPTY "
            f"docs; got {chunk_ids}"
        )
        _assert_no_data_year_sql_clause(retriever)

    def test_ambiguous_intent_in_season_also_drops_prior_year(self, monkeypatch):
        """`ambiguous` is treated like `current` per the new contract:
        IN_SEASON+ambiguous → drop prior-year. The classifier returns
        `ambiguous` for queries that match neither current/historical/
        cross_year keywords."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        # A query that has no temporal keywords → ambiguous.
        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="học phí ngành CNTT",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "prior-2025" not in chunk_ids, (
            f"IN_SEASON+ambiguous must HARD-DROP prior-year; got {chunk_ids}"
        )
        assert "current-2026" in chunk_ids
        _assert_no_data_year_sql_clause(retriever)


# ---------------------------------------------------------------------------
# Test 2: superseded row IN_SEASON → dropped
# ---------------------------------------------------------------------------

class TestSupersededDropped:
    def test_superseded_row_dropped_in_in_season(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026, is_superseded=False),
            _row("superseded-2025", data_year=2025, is_superseded=True),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="điểm chuẩn năm nay",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "superseded-2025" not in chunk_ids, (
            f"Superseded row must be dropped in IN_SEASON+current; got {chunk_ids}"
        )
        _assert_no_data_year_sql_clause(retriever)


# ---------------------------------------------------------------------------
# Test 3: PRE_SEASON_GAP → no hard drop; prior-year rows SURVIVE (demoted)
# ---------------------------------------------------------------------------

class TestPreSeasonGapNoHardDrop:
    """During the pre-season gap, prior-year chunks are KEPT (shown labeled).
    No post-fetch hard drop; the soft demote path stays in effect."""

    def test_pre_season_gap_where_clause_is_tombstone_gate(self, monkeypatch):
        """T15: the dense arm now always passes TOMBSTONE_PREDICATE as the
        server-side `where_clause` (never a data_year clause). The
        PRE_SEASON_GAP invariant was 'no hard-dropping prior-year rows' —
        that is enforced POST-FETCH in `_apply_temporal_filter`, not by
        omitting the tombstone gate from the SQL clause. The gate only
        excludes rows stamped `is_tombstone=true`; it does not affect
        year-based soft/demote behavior."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=False)  # ← gap

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="điểm chuẩn năm nay",
        )

        captured = retriever._adapter._captured["where_clause"]
        assert captured == TOMBSTONE_PREDICATE, (
            "T15: dense arm must pass TOMBSTONE_PREDICATE; "
            f"got {captured!r}"
        )
        assert "data_year" not in captured

    def test_pre_season_gap_prior_year_rows_survive(self, monkeypatch):
        """Prior-year rows SURVIVE in PRE_SEASON_GAP (shown, just demoted)."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=False)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="điểm chuẩn năm nay",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "prior-2025" in chunk_ids, (
            "In PRE_SEASON_GAP, prior-year rows must NOT be hard-dropped; "
            f"got {chunk_ids}"
        )
        assert "current-2026" in chunk_ids
        _assert_no_data_year_sql_clause(retriever)


# ---------------------------------------------------------------------------
# Test 4: historical intent → all prior years SURVIVE
# ---------------------------------------------------------------------------

class TestHistoricalIntent:
    """Historical intent (e.g. 'điểm chuẩn năm 2024') MUST NOT hard-drop to
    current-year only. Prior years are explicitly allowed."""

    def test_historical_intent_all_years_visible(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
            _row("prior-2024", data_year=2024),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="điểm chuẩn năm 2024",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        for cid in ("current-2026", "prior-2025", "prior-2024"):
            assert cid in chunk_ids, (
                f"Historical intent must NOT hard-drop prior years; "
                f"missing {cid} in {chunk_ids}"
            )
        _assert_no_data_year_sql_clause(retriever)


# ---------------------------------------------------------------------------
# Test 5: cross_year intent → all years survive
# ---------------------------------------------------------------------------

class TestCrossYearIntent:
    """Cross-year comparisons ('so sánh điểm chuẩn qua các năm') need ALL
    years visible. No post-fetch hard drop."""

    def test_cross_year_all_years_visible(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
            _row("prior-2024", data_year=2024),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query="so sánh điểm chuẩn qua các năm",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        for cid in ("current-2026", "prior-2025", "prior-2024"):
            assert cid in chunk_ids, (
                f"cross_year must keep all years; missing {cid} in {chunk_ids}"
            )
        _assert_no_data_year_sql_clause(retriever)


# ---------------------------------------------------------------------------
# Test 6: backward compat — no query, no metadata_filter still works
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Existing callers pass (query_vector, top_k[, metadata_filter]) and
    nothing else. retrieve() must continue to work without a query string.
    No hard drop (legacy soft path). No crash. No where_clause."""

    def test_no_query_no_metadata_filter_works(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [_row("a", data_year=2026), _row("b", data_year=2025)]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        # No `query=` kwarg, no `metadata_filter=` — the legacy path.
        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
        )

        # No crash; both rows must survive (no hard drop without intent).
        assert isinstance(result.documents, list)
        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "a" in chunk_ids and "b" in chunk_ids, (
            f"Backward-compat: no query → no hard drop → both rows survive; "
            f"got {chunk_ids}"
        )
        _assert_no_data_year_sql_clause(retriever)

    def test_metadata_filter_with_query_runs_classification(self, monkeypatch):
        """Existing callers using `metadata_filter={'query': '...'}` (the
        pre-S14.3 fallback path) must still work: the classifier runs,
        `adapter.search` is called with `where_clause=None`, and the
        post-fetch hard-EXCLUDE would fire (it just never sees the rows
        because the pre-existing `_apply_filter` strips them out first
        — that legacy behavior is unchanged)."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("a-2026", data_year=2026),
            _row("b-2025", data_year=2025),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        # The legacy _apply_filter will strip both rows (their metadata
        # doesn't have a literal "query" column), so the post-fetch
        # hard-EXCLUDE never sees them. This is unchanged from pre-S14.3.
        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            metadata_filter={"query": "điểm chuẩn năm nay"},
        )

        # The contract under test: where_clause is None (no SQL emitted),
        # the call returned cleanly (no exception, well-formed result).
        _assert_no_data_year_sql_clause(retriever)
        assert isinstance(result.documents, list)


# ---------------------------------------------------------------------------
# Test 7: query param is the preferred seam (explicit param wins)
# ---------------------------------------------------------------------------

class TestQueryParamSeam:
    """The explicit `query=` parameter is the preferred path. When both
    `query=` and `metadata_filter['query']` are supplied, `query=` wins."""

    def test_explicit_query_param_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2024", data_year=2024),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        # Explicit param: "năm nay" (current). Pass an empty
        # metadata_filter so the legacy _apply_filter is a no-op and
        # the post-fetch hard-EXCLUDE result is observable in the docs.
        # (The original test that used `metadata_filter={"query": ...}`
        # relied on a `query` key match in row metadata that doesn't
        # exist in production rows, so it ended up with an empty result
        # set — pre-existing behavior, unchanged.)
        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            metadata_filter={},
            query="điểm chuẩn năm nay",
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        # Explicit query= wins → intent=current → IN_SEASON+current
        # post-fetch hard drop removes 2024 → only 2026 survives.
        assert "current-2026" in chunk_ids
        assert "prior-2024" not in chunk_ids, (
            f"Explicit query= should drive IN_SEASON+current hard drop; "
            f"got {chunk_ids}"
        )
        _assert_no_data_year_sql_clause(retriever)


# ---------------------------------------------------------------------------
# Test 8: invalid payload still returns INVALID_PAYLOAD (no crash)
# ---------------------------------------------------------------------------

class TestInputValidationUnchanged:
    """S14.3 must not break the existing input-validation guard rails."""

    def test_empty_query_vector_returns_invalid_payload(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        from services.lancedb_retrieval import RetrievalError

        retriever = _make_retriever(rows=[])
        result = retriever.retrieve(query_vector=[], top_k=5, query="hello")
        assert result.error_type == RetrievalError.INVALID_PAYLOAD

    def test_wrong_dim_query_vector_returns_invalid_payload(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        from services.lancedb_retrieval import RetrievalError

        retriever = _make_retriever(rows=[])
        result = retriever.retrieve(
            query_vector=[0.0] * (EMBEDDING_DIM + 1),
            top_k=5,
            query="hello",
        )
        assert result.error_type == RetrievalError.INVALID_PAYLOAD


# ---------------------------------------------------------------------------
# T15 invariant — `adapter.search` never receives a data_year clause.
# The dense arm now passes TOMBSTONE_PREDICATE (the is_tombstone gate)
# as its `where_clause`; that gate does NOT reference data_year, so the
# "no data_year clause" invariant is preserved and re-asserted here.
# ---------------------------------------------------------------------------

class TestNoDataYearSQLClauseEver:
    """Sweep the contract end-to-end: in every intent path, `adapter.search`
    MUST be called with a clause that does not contain `data_year`. Since
    T15 the only clause emitted is `TOMBSTONE_PREDICATE` (is_tombstone
    gate); the real table has no `data_year` column, so any clause
    containing that identifier would raise `LanceError(Schema)`."""

    @pytest.mark.parametrize(
        "query, expected_intent",
        [
            ("điểm chuẩn năm nay", TemporalIntent.current),
            ("điểm chuẩn năm 2024", TemporalIntent.historical),
            ("so sánh điểm chuẩn qua các năm", TemporalIntent.cross_year),
            ("học phí ngành CNTT", TemporalIntent.ambiguous),
        ],
    )
    def test_where_clause_is_none_in_every_intent(self, monkeypatch, query, expected_intent):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            query=query,
        )

        _assert_no_data_year_sql_clause(retriever)

# ---------------------------------------------------------------------------
# Test 9: the `query` key is a HINT, never an equality constraint
# ---------------------------------------------------------------------------

class TestQueryHintIsNotAFilter:
    """HybridSearchService injects the raw query into `metadata_filter` so the
    temporal classifier can read the user's wording. No row carries a `query`
    field, so matching it as equality dropped EVERY row and the dense arm
    returned EMPTY_RESULT on every hybrid call — the fused pool silently
    degenerated to BM25-only. The hint must be ignored, like `or_conditions`.
    """

    def test_query_hint_alone_does_not_drop_rows(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("stable-agnostic", data_year=None),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            metadata_filter={"query": "học phí ngành CNTT"},
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert chunk_ids == {"current-2026", "stable-agnostic"}, (
            "the `query` hint must not filter rows out; "
            f"expected both rows, got {chunk_ids}"
        )

    def test_temporal_hard_drop_still_fires_with_hint_present(self, monkeypatch):
        """A real rule still applies when the hint rides along in the filter."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)

        rows = [
            _row("current-2026", data_year=2026),
            _row("prior-2025", data_year=2025),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)

        result = retriever.retrieve(
            query_vector=_make_vector(),
            top_k=5,
            metadata_filter={"query": "học phí ngành CNTT"},
        )

        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "current-2026" in chunk_ids, (
            f"current-year row must survive; got {chunk_ids}"
        )
        assert "prior-2025" not in chunk_ids, (
            "the IN_SEASON+ambiguous hard-drop must still fire when the "
            f"hint rides along in metadata_filter; got {chunk_ids}"
        )


# ---------------------------------------------------------------------------
# T15: tombstone rows are absent from dense `sources` AND from fetch_by_id
# ---------------------------------------------------------------------------


def _row_t15(
    text: str,
    distance: float = 0.5,
    data_year: Optional[int] = None,
    is_tombstone: Optional[bool] = None,
) -> Dict[str, Any]:
    """Row with an explicit `is_tombstone` top-level column."""
    return {
        "text": text,
        "source": "synthetic",
        "chunk_id": text,
        "id": text,
        "metadata_json": json.dumps({"data_year": data_year, "is_tombstone": is_tombstone}),
        "is_tombstone": is_tombstone,
        "_distance": distance,
    }


class TestTombstoneDenseDropped:
    """T15(a): a row stamped `is_tombstone=true` must NOT appear in the
    dense `sources` list, regardless of temporal intent."""

    def test_tombstone_row_dropped_from_sources(self, monkeypatch):
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        rows = [
            _row_t15("normal-2026", data_year=2026, is_tombstone=False),
            _row_t15("tombstoned-2026", data_year=2026, is_tombstone=True),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)
        result = retriever.retrieve(query_vector=_make_vector(), top_k=5, query="điểm chuẩn")
        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "normal-2026" in chunk_ids
        assert "tombstoned-2026" not in chunk_ids, f"tombstoned row must be absent; got {chunk_ids}"

    def test_tombstone_row_dropped_even_for_historical_query(self, monkeypatch):
        """Unlike superseded rows, a tombstoned row is NEVER valid."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        rows = [
            _row_t15("tombstoned-prior", data_year=2024, is_tombstone=True),
            _row_t15("normal-prior", data_year=2024, is_tombstone=False),
        ]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)
        result = retriever.retrieve(
            query_vector=_make_vector(), top_k=5, query="điểm chuẩn năm 2024"
        )
        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "normal-prior" in chunk_ids
        assert "tombstoned-prior" not in chunk_ids, (
            "tombstoned row must never survive, even for historical queries; "
            f"got {chunk_ids}"
        )

    def test_null_tombstone_row_is_still_returned(self, monkeypatch):
        """The nullable-column trap: a row where `is_tombstone` is NULL
        (not tombstoned) must NOT be dropped — it is a legitimate chunk."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        rows = [_row_t15("unstamped", data_year=2026, is_tombstone=None)]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)
        result = retriever.retrieve(query_vector=_make_vector(), top_k=5, query="học phí")
        chunk_ids = {doc.chunk_id for doc in result.documents}
        assert "unstamped" in chunk_ids, (
            "a NULL is_tombstone row is a legitimate chunk and must be "
            f"returned; got {chunk_ids}"
        )

    def test_server_side_predicate_is_tombstone_gate_not_data_year(self, monkeypatch):
        """T15 decision: the dense arm passes TOMBSTONE_PREDICATE as the
        server-side `where_clause`. No `data_year` clause may appear."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        rows = [_row_t15("c", data_year=2026, is_tombstone=False)]
        retriever = _make_retriever(rows=rows)
        _drive_season(retriever, year=2026, has_data=True)
        retriever.retrieve(query_vector=_make_vector(), top_k=5, query="học phí")
        captured = retriever._adapter._captured["where_clause"]
        assert captured == TOMBSTONE_PREDICATE, (
            f"dense arm must pass the tombstone gate; got {captured!r}"
        )
        assert "data_year" not in captured

    def test_legacy_table_no_tombstone_column_searches_with_no_clause(self, monkeypatch):
        """T15 regression: when the table predates the `is_tombstone`
        column the dense arm must NOT reference it (LanceError(Schema)).
        `where_clause` is None and rows still flow through the post-fetch
        belt, which catches the metadata-mirror stamp."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
        monkeypatch.delenv("APP_ENV", raising=False)
        rows = [_row_t15("c", data_year=2026, is_tombstone=False)]
        retriever = _make_retriever(rows=rows, has_tombstone_column=False)
        _drive_season(retriever, year=2026, has_data=True)
        result = retriever.retrieve(query_vector=_make_vector(), top_k=5, query="học phí")
        captured = retriever._adapter._captured["where_clause"]
        assert captured is None, (
            f"legacy table (no is_tombstone column) must search with "
            f"where_clause=None; got {captured!r}"
        )
        # Rows still come back; the schema gate suppressed the clause, not the filter.
        assert {doc.chunk_id for doc in result.documents} == {"c"}


class TestTombstoneFetchById:
    """T15(c): the post-fetch belt in `fetch_by_id`/`fetch_by_ids` catches
    tombstoned rows that bypassed the server-side predicate."""

    def _make_retriever_with_fake_fetch(
        self, rows_by_id: Dict[str, List[Dict[str, Any]]]
    ) -> LanceDBRetriever:
        retriever = LanceDBRetriever.__new__(LanceDBRetriever)
        retriever._config = LanceDBRetrieverConfig(
            uri="memory://",
            table_name="t",
            embedding_dim=EMBEDDING_DIM,
            default_top_k=5,
        )
        fake_adapter = MagicMock()
        fake_adapter.ensure_table = MagicMock()

        def _fetch_by_ids(ids: Optional[List[str]]) -> List[Dict[str, Any]]:
            if not ids:
                return []
            out: List[Dict[str, Any]] = []
            for cid in ids:
                out.extend(rows_by_id.get(cid, []))
            return out

        fake_adapter.fetch_rows_by_ids.side_effect = _fetch_by_ids
        fake_adapter.fetch_rows_by_doc_ids.return_value = []
        retriever._adapter = fake_adapter
        return retriever

    def test_fetch_by_id_returns_none_for_tombstoned_row(self):
        rows = [
            {
                "id": "t1",
                "text": "tombstoned",
                "source": "synthetic",
                "chunk_id": "t1",
                "metadata_json": "{}",
                "is_tombstone": True,
            }
        ]
        retriever = self._make_retriever_with_fake_fetch({"t1": rows})
        doc = retriever.fetch_by_id("t1")
        assert doc is None, f"tombstoned chunk must return None; got {doc}"

    def test_fetch_by_ids_omits_tombstoned_row(self):
        ok = {
            "id": "ok1",
            "text": "normal",
            "source": "synthetic",
            "chunk_id": "ok1",
            "metadata_json": "{}",
            "is_tombstone": False,
        }
        bad = {
            "id": "bad1",
            "text": "tombstoned",
            "source": "synthetic",
            "chunk_id": "bad1",
            "metadata_json": "{}",
            "is_tombstone": True,
        }
        retriever = self._make_retriever_with_fake_fetch({"ok1": [ok], "bad1": [bad]})
        result = retriever.fetch_by_ids(["ok1", "bad1"])
        assert "ok1" in result
        assert "bad1" not in result, f"tombstoned chunk must be omitted; got {list(result)}"

    def test_fetch_by_id_returns_doc_for_null_tombstone_row(self):
        rows = [
            {
                "id": "null1",
                "text": "un-stamped",
                "source": "synthetic",
                "chunk_id": "null1",
                "metadata_json": "{}",
                # is_tombstone column absent → NULL semantics
            }
        ]
        retriever = self._make_retriever_with_fake_fetch({"null1": rows})
        doc = retriever.fetch_by_id("null1")
        assert doc is not None, "a NULL is_tombstone row must not be dropped"
        assert doc.chunk_id == "null1"
