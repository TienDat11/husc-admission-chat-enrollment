# @spec(S13.5) TemporalIntent classifier — regex pre-classify with LLM fallback hook
"""Classify a user query into a temporal intent so the retriever can apply
year-namespace hard filters.

This module is INDEPENDENT from query_router.py to avoid touching the hot path
during Phase 4 rollout. Wire-up to query_router.py happens in a follow-up.

Resolution order in classify_temporal():
  1. Explicit 4-digit year (2024-2030 range) → current/historical based on current_year.
  2. "năm nay | hiện tại | năm hiện tại | year of {current_year}" → current.
  3. "năm trước | năm ngoái | năm cũ" → historical.
  4. "so sánh | đối chiếu | qua các năm | từng năm" → cross_year.
  5. else → ambiguous (caller may LLM-fallback or default to current).

All matching is case-insensitive and Vietnamese-diacritics-aware (we accept the
NFC form used by Vietnamese keyboards).
"""
from __future__ import annotations
import re
from enum import Enum
from typing import Awaitable, Callable, Optional


class TemporalIntent(str, Enum):
    """User intent w.r.t. admissions year namespace."""
    current = "current"
    historical = "historical"
    ambiguous = "ambiguous"
    cross_year = "cross_year"


# Match 4-digit years in [1990, 2039] — aligned with year_ner.YEAR_LO/HI.
_YEAR_RX = re.compile(r"\b(199\d|20[0-3]\d)\b")
_CURRENT_RX = re.compile(r"\b(năm\s+nay|hiện\s+tại|năm\s+hiện\s+tại|hiện\s+nay)\b", re.IGNORECASE)
_HISTORICAL_RX = re.compile(r"\b(năm\s+trước|năm\s+ngoái|năm\s+cũ|năm\s+rồi)\b", re.IGNORECASE)
_FUTURE_RX = re.compile(r"\b(năm\s+sau|năm\s+tới|sang\s+năm)\b", re.IGNORECASE)
_CROSS_YEAR_RX = re.compile(
    r"\b(so\s+sánh|đối\s+chiếu|qua\s+các\s+năm|từng\s+năm|theo\s+năm|giữa\s+các\s+năm)\b",
    re.IGNORECASE,
)


def extract_explicit_year(query: str) -> Optional[int]:
    """Return the FIRST explicit 4-digit year in [1990, 2039]; else None."""
    if not isinstance(query, str):
        return None
    m = _YEAR_RX.search(query)
    if not m:
        return None
    return int(m.group(1))


def classify_temporal(query: str, current_year: int) -> TemporalIntent:
    """Regex-based temporal classification.

    Args:
        query: User question (Vietnamese).
        current_year: Current admission year (e.g. 2026); drives current/historical split.

    Returns:
        TemporalIntent enum.
    """
    if not query:
        return TemporalIntent.ambiguous

    # Cross-year keywords win — comparing across years is the most specific intent.
    if _CROSS_YEAR_RX.search(query):
        return TemporalIntent.cross_year

    # HIGH-3 fix: relative phrases take priority over explicit year.
    # User intent ("năm trước") is more reliable than incidental year mentions.
    if _CURRENT_RX.search(query):
        return TemporalIntent.current
    if _HISTORICAL_RX.search(query):
        return TemporalIntent.historical

    # Explicit year only when no relative phrase found.
    explicit = extract_explicit_year(query)
    if explicit is not None:
        if explicit >= current_year:
            return TemporalIntent.current
        return TemporalIntent.historical

    if _FUTURE_RX.search(query):
        # No future data; treat as ambiguous so caller can warn or fall back.
        return TemporalIntent.ambiguous
    return TemporalIntent.ambiguous


# Async fallback hook — reserved for future LLM-based disambiguation.
LLMClassifier = Callable[[str], Awaitable[TemporalIntent]]


async def classify_with_llm_fallback(
    query: str,
    current_year: int,
    *,
    llm_runner: Optional[LLMClassifier] = None,
) -> TemporalIntent:
    """Regex first, LLM fallback only on ambiguous (rare path — keep hot path light)."""
    intent = classify_temporal(query, current_year)
    if intent is not TemporalIntent.ambiguous or llm_runner is None:
        return intent
    return await llm_runner(query)
