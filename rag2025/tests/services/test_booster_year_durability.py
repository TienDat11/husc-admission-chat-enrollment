# @spec(G1) test_booster_year_durability — TIME-BOMB 2027 regression
"""TDD: aggregation_booster must derive the year at runtime from
`temporal_authority.get_current_admission_year()` instead of hardcoding
`_2026` chunk-ID suffixes / literal `2026` regex patterns.

Three failure modes the test pins down:
  (a) `"học phí năm 2027"` must match a tuition boost rule at clock=2027.
  (b) The rule's resolved chunk-IDs MUST contain `_2027` (not `_2026`).
  (c) A None fetch must emit a WARN log AND increment a
      `booster_chunk_miss` counter so the silent failure is visible.

Additionally: at the legacy year=2026 build, the produced chunk-ID set
MUST be byte-identical to the golden fixture
`tests/services/fixtures/booster_chunkids_2026.golden.json` (no drift
in current prod boosts).

NOTE: G1-T1 RED phase. The first run is expected to fail because the
booster still hardcodes `_2026` — G1-T2 parameterizes it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest


# Mirror the import convention from test_season.py / test_booster_tag.py.
RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

GOLDEN_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "booster_chunkids_2026.golden.json"
)


def _load_golden_ids() -> set:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return set(data["chunk_ids"])


def _all_built_chunk_ids() -> set:
    """Collect the chunk-ids the booster currently builds, regardless of
    the year literal it uses internally. We probe via the public API and
    a synthetic query that surfaces every rule's candidate list."""
    from services.aggregation_booster import _AGGREGATION_RULES
    ids = set()
    for rule in _AGGREGATION_RULES:
        ids.update(rule["chunks"])
    return ids


def _year_aware_built_chunk_ids() -> set:
    """The 'byte-identical' comparison set: the booster's chunk-ids that
    carry a year suffix (e.g. ``_2026``) or a year-comparison id
    (``so_sanh_hocphi_2025_vs_2026``). The golden fixture enumerates
    exactly these — the qa_*/husc_info entries that are year-agnostic
    anchors are intentionally OUT OF SCOPE for the year-rollover guard.
    """
    import re as _re
    built = _all_built_chunk_ids()
    yearish = set()
    for cid in built:
        if _re.search(r"_20\d{2}\b", cid) or "_2025" in cid or "_2026" in cid:
            yearish.add(cid)
    return yearish


def _make_fake_retriever(chunk_id_to_text: dict):
    """Fake LanceDB retriever for boost_with_aggregation tests."""
    from services.lancedb_retrieval import RetrievedDocument

    def fetch_by_id(cid: str):
        if cid not in chunk_id_to_text:
            return None
        return RetrievedDocument(
            text=chunk_id_to_text[cid],
            source="fake",
            chunk_id=cid,
            metadata={"source_kind": "fake"},
            score=1.0,
            point_id=cid,
        )

    fake = MagicMock()
    fake.fetch_by_id.side_effect = fetch_by_id
    return fake


def _baseline_doc(chunk_id: str):
    from services.lancedb_retrieval import RetrievedDocument
    return RetrievedDocument(
        text="baseline",
        source="baseline",
        chunk_id=chunk_id,
        metadata={"source_kind": "vector"},
        score=0.42,
    )


# ---------------------------------------------------------------------------
# (1) year=2026 regression: built set == golden fixture exactly
# ---------------------------------------------------------------------------

def test_year_2026_built_chunk_ids_match_golden_fixture(monkeypatch):
    """At year=2026 the booster's candidate chunk-id set MUST equal the
    golden fixture (45 ids) byte-for-byte. Catches accidental drift in
    G1-T2 parameterization.

    The booster must source the year from `get_current_admission_year()`
    so we monkeypatch that to 2026.
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.delenv("APP_ENV", raising=False)

    from services import temporal_authority, aggregation_booster

    # If the booster reads the year lazily, calling detect now resolves it.
    built = _all_built_chunk_ids()
    # The booster is required to read the year via temporal_authority.
    # Sanity: monkeypatch forces 2026.
    assert temporal_authority.get_current_admission_year() == 2026, (
        "Test setup error: temporal_authority must return 2026 here"
    )

    golden = _load_golden_ids()
    missing = golden - built
    extra = built - golden
    assert not missing, f"Booster at year=2026 is MISSING chunk-ids from golden: {sorted(missing)}"
    assert not extra, f"Booster at year=2026 has EXTRA chunk-ids not in golden: {sorted(extra)}"


# ---------------------------------------------------------------------------
# (2) year=2027: query matches + chunk-ids parameterized to _2027
# ---------------------------------------------------------------------------

def test_tuition_query_year_2027_matches_and_uses_2027_chunk_ids(monkeypatch):
    """At clock=2027, 'học phí năm 2027' must (a) trigger a tuition boost
    rule and (b) the rule's resolved chunk-ids must carry the `_2027`
    suffix — NOT `_2026`."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2027")
    monkeypatch.delenv("APP_ENV", raising=False)

    # The booster snapshots the year at module import. To simulate
    # clock-2027 we MUST reload both the temporal authority AND the
    # booster after the env patch so the snapshot picks up 2027.
    import importlib
    from src.services import temporal_authority
    importlib.reload(temporal_authority)
    from src.services import aggregation_booster
    importlib.reload(aggregation_booster)

    assert temporal_authority.get_current_admission_year() == 2027
    assert aggregation_booster._YEAR == 2027, (
        f"Booster year snapshot must be 2027, got {aggregation_booster._YEAR}"
    )

    chunk_ids = aggregation_booster.detect_aggregation_chunks("học phí năm 2027")
    assert chunk_ids, (
        "Expected a tuition rule to match 'học phí năm 2027' at year=2027; got []"
    )

    # The current-year chunk-id set must be present and use the 2027 suffix.
    has_2027 = any("_2027" in cid for cid in chunk_ids)
    has_2026 = any("_2026" in cid for cid in chunk_ids)
    assert has_2027, (
        f"Expected at least one chunk-id with '_2027' suffix at year=2027; got {chunk_ids}"
    )
    assert not has_2026, (
        f"Did NOT expect any '_2026' chunk-ids at year=2027 (would mean hardcoded); got {chunk_ids}"
    )


# ---------------------------------------------------------------------------
# (3) WARN + booster_chunk_miss counter on fetch_by_id -> None
# ---------------------------------------------------------------------------

def test_none_fetch_emits_warn_and_increments_booster_chunk_miss(monkeypatch, caplog):
    """When the booster asks fetch_by_id for a chunk-id that does not
    exist, the booster MUST log a warning (not debug) and bump a
    `booster_chunk_miss` counter so the silent failure is observable."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2027")
    monkeypatch.delenv("APP_ENV", raising=False)

    import importlib
    from services import aggregation_booster
    importlib.reload(aggregation_booster)  # ensure year is read fresh

    # An empty fake retriever => every fetch returns None.
    fake_retriever = _make_fake_retriever({})
    baseline = [_baseline_doc("v1"), _baseline_doc("v2")]

    # Capture loguru output via caplog (caplog works with stdlib logging
    # propagation; loguru's logger sinks may bypass it. We use a sink
    # monkeypatch instead — see below.)
    try:
        from services.aggregation_booster import (
            boost_with_aggregation,
            BOOSTER_CHUNK_MISS,
        )
    except ImportError:
        pytest.fail(
            "aggregation_booster must export BOOSTER_CHUNK_MISS counter "
            "(see G1-T3 — WARN log + booster_chunk_miss metric)."
        )

    # Patch logger.warning to capture invocations. loguru exposes logger
    # via `from loguru import logger`; we monkeypatch `.warning` on the
    # module-level instance.
    captured_warnings: List[str] = []
    from loguru import logger as _loguru_logger
    original_warning = _loguru_logger.warning

    def _capture(*args, **kwargs):
        # loguru.warning is variadic; stringify the first arg for assertion.
        msg = args[0] if args else kwargs.get("message", "")
        captured_warnings.append(str(msg))
        return original_warning(*args, **kwargs)

    monkeypatch.setattr(_loguru_logger, "warning", _capture)

    # Before the call: capture the counter baseline.
    before_miss = BOOSTER_CHUNK_MISS.get() if hasattr(BOOSTER_CHUNK_MISS, "get") else BOOSTER_CHUNK_MISS

    # A query that triggers a tuition rule at year=2027 -> the rule's
    # candidate chunk-ids (e.g. hocphi_2027) will be looked up; the fake
    # retriever has no entry for any of them -> all return None.
    result = boost_with_aggregation(
        query="học phí năm 2027",
        baseline_docs=baseline,
        lancedb_retriever=fake_retriever,
        top_k=5,
    )

    after_miss = BOOSTER_CHUNK_MISS.get() if hasattr(BOOSTER_CHUNK_MISS, "get") else BOOSTER_CHUNK_MISS
    assert after_miss > before_miss, (
        f"Expected booster_chunk_miss to increment on None fetch; "
        f"before={before_miss} after={after_miss}"
    )

    assert captured_warnings, (
        "Expected at least one logger.warning(...) call when fetch_by_id returns None"
    )
    # And at least one warning mentions a chunk-id and the miss.
    miss_warnings = [w for w in captured_warnings if "miss" in w.lower() or "fetch" in w.lower() or "None" in w]
    assert miss_warnings, (
        f"Expected a miss-related warning; captured warnings: {captured_warnings}"
    )

    # The booster should still hand back the baseline list (graceful degrade).
    assert len(result) >= 1
    assert {d.chunk_id for d in result} >= {"v1", "v2"}


# ---------------------------------------------------------------------------
# (4) detect_aggregation_chunks is callable + empty-query is safe
# ---------------------------------------------------------------------------

def test_detect_returns_list_for_unrelated_query(monkeypatch):
    """Sanity: detect must never raise and must return an empty list for
    a query that matches no rule."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.delenv("APP_ENV", raising=False)
    from services.aggregation_booster import detect_aggregation_chunks
    out = detect_aggregation_chunks("hello world random text")
    assert out == []
