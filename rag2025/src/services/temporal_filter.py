# @spec(S13.5) temporal filter — convert TemporalIntent to LanceDB SQL clause
"""Build a LanceDB filter clause string from a TemporalIntent.

Pure function — no LanceDB dependency. The retriever applies the returned
clause via LanceDB's `where=` argument before vector search.

Safety: current_year is an int (validated by caller). We DO NOT interpolate
user input into the SQL — the only dynamic value is `int(current_year)`.
"""
from __future__ import annotations
from typing import Optional

from .temporal_intent import TemporalIntent


def apply_temporal_filter(intent: TemporalIntent, current_year: int) -> Optional[str]:
    """Return LanceDB filter clause for the given intent.

    Args:
        intent: classified TemporalIntent.
        current_year: int — clamped to plausible admissions year window.

    Returns:
        SQL clause string when a hard filter applies; None otherwise (caller
        may default ambiguous queries to current-year filter, or fall back to
        no filter for cross_year).

    Notes:
        - TemporalIntent.historical produces "data_year < N" (all past years).
        - For "last year only" semantics, the caller should use current_year=N-1
          and TemporalIntent.current to produce "data_year = N-1".
    """
    # Defensive: reject non-int current_year to prevent injection via str format.
    if not isinstance(current_year, int):
        raise TypeError(f"current_year must be int, got {type(current_year).__name__}")
    if current_year < 2020 or current_year > 2039:
        raise ValueError(f"current_year out of plausible range [2020,2039]: {current_year}")

    if intent is TemporalIntent.current:
        return f"data_year = {current_year}"
    # NOTE: historical means ALL past years (data_year < current_year), not just
    # last year. For "năm trước" specifically (last year), callers can compute
    # current_year - 1 and use TemporalIntent.current with that anchor.
    if intent is TemporalIntent.historical:
        return f"data_year < {current_year}"
    # cross_year + ambiguous → no hard filter (caller decides ambiguous default).
    return None
