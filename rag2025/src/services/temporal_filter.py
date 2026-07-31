# @spec(S13.5) temporal filter — convert TemporalIntent to LanceDB SQL clause
"""Build a LanceDB filter clause string from a TemporalIntent.

Pure function — no LanceDB dependency. The retriever applies the returned
clause via LanceDB's `where=` argument before vector search.

Safety: current_year is an int (validated by caller). We DO NOT interpolate
user input into the SQL — the only dynamic value is `int(current_year)`.
"""
from __future__ import annotations
from typing import Optional, Literal

from .temporal_intent import TemporalIntent


def current_clause(as_of: int) -> str:
    """Return the 4-condition active-status predicate for a given as-of date."""
    return (
        f"(status = 'current' AND effective_from <= {as_of} "
        f"AND effective_to >= {as_of} AND is_tombstone = false)"
    )


def apply_temporal_filter(
    intent: TemporalIntent,
    current_year: int,
    *,
    schema_mode: Literal["flat", "nested_json"] = "flat",
    as_of: int | None = None,
) -> Optional[str]:
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
    if not isinstance(current_year, int):
        raise TypeError(f"current_year must be int, got {type(current_year).__name__}")
    if current_year < 2020 or current_year > 2039:
        raise ValueError(f"current_year out of plausible range [2020,2039]: {current_year}")

    # nested_json → no SQL predicate (data_year lives inside metadata_json string)
    if schema_mode == "nested_json":
        return None

    base = f"data_year = {current_year}" if intent is TemporalIntent.current else None
    if base is None and intent is TemporalIntent.historical:
        base = f"data_year < {current_year}"
    # cross_year + ambiguous → None (caller decides ambiguous default).
    if base is None:
        return None

    # Append active-status predicate only when as_of is provided with current intent.
    if intent is TemporalIntent.current and as_of is not None:
        base = f"{base} AND {current_clause(as_of)}"
    return base
