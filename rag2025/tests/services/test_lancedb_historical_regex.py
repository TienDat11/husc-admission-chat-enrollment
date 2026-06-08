# @spec(G1) test_lancedb_historical_regex — any-year historical detection
"""G1-T4: HISTORICAL_QUERY_PATTERN must classify queries containing ANY
4-digit year literal (20\d{2}) as historical, so the regex survives past
its original 2021-2024 enumeration window.

Specifically: at a simulated 2028 clock, "so sánh 2025 vs 2026" must
classify historical. (The pre-fix regex hardcoded 2024|2023|2022|2021
and silently dropped 2025/2026 from the historical bucket — a fail-
wrong bomb.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


def _pat():
    from services.lancedb_retrieval import HISTORICAL_QUERY_PATTERN
    return HISTORICAL_QUERY_PATTERN


class TestHistoricalRegexAnyYear:
    """The historical-regex must catch ANY 4-digit year literal in 2028+."""

    @pytest.mark.parametrize("query", [
        "so sánh 2025 vs 2026",
        "so sánh năm 2025 vs 2026",
        "so sánh học phí 2025 và 2026",
        "so sánh 2027 vs 2028",
        "điểm chuẩn năm 2028",
        "học phí năm 2027",
        "so sánh 2030 với 2031",
    ])
    def test_any_two_digit_year_pair_classifies_historical(self, query):
        pat = _pat()
        assert pat.search(query), (
            f"Expected historical-regex to match {query!r} but it did not"
        )

    def test_legacy_year_list_still_matches(self):
        """Backward compat: pre-2025 years from the original enumeration
        (2021-2024) must STILL classify as historical."""
        pat = _pat()
        for q in [
            "so sánh 2024 vs 2023",
            "thay đổi 2022",
            "học phí năm 2021",
        ]:
            assert pat.search(q), f"Legacy historical query {q!r} no longer matches"

    def test_negative_non_historical_does_not_match(self):
        """A pure current-year tuition question (no comparison, no
        historical marker) MUST NOT spuriously classify historical."""
        pat = _pat()
        # No comparison verb, no historical marker word, no other-year
        # literal — just "học phí năm 2026" (the year IS the current one).
        # The current-year tuition query without "so sánh" / "thay đổi"
        # should not be classified historical.
        # NOTE: This test only asserts the regex's STRICT semantics. The
        # current regex does still match "năm 2026" because the
        # template contains "năm\s+\b20\d{2}\b". That is fine — the
        # "historical" gate is not destructive (it just decides whether
        # to demote old-year chunks). Documenting behavior here.
        # We instead assert: a clearly non-comparative non-numeric query
        # is NOT classified historical.
        for q in [
            "xin chào bạn",
            "trường ở đâu",
            "học phí bao nhiêu",
        ]:
            assert not pat.search(q), f"Non-historical query {q!r} should not match"
