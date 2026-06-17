# @spec(S13)
"""Year-specific admission facts loader and Vietnamese briefing renderer.

Reads per-year JSON fact files from the data/year_facts/ directory and produces
a compact citation-bearing briefing block suitable for injection at a
{{YEAR_FACTS}} placeholder in generation prompts.  All I/O is confined to
get_year_facts(); render_year_briefing() is pure and side-effect-free.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------

class YearFact(BaseModel):
    """A single sourced admission fact for a given year."""

    field: str
    value: str
    source: str
    source_date: str


class YearFacts(BaseModel):
    """Collection of admission facts for one year."""

    year: int
    facts: list[YearFact]
    available: bool


# ---------------------------------------------------------------------------
# Module constant — data directory (resolved at import, never created here)
# ---------------------------------------------------------------------------

_YEAR_FACTS_DIR = Path(__file__).resolve().parents[2] / "data" / "year_facts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_year_facts(year: int, facts_dir: Path | None = None) -> YearFacts:
    """Load year-specific admission facts from a JSON file.

    Reads ``{facts_dir}/{year}.json`` which must have the shape::

        {
            "year": <int>,
            "facts": [
                {"field": ..., "value": ..., "source": ..., "source_date": ...},
                ...
            ]
        }

    On any failure (missing file, malformed JSON, validation error) returns
    ``YearFacts(year=year, facts=[], available=False)`` — **never raises**.

    Args:
        year: The admission year to look up.
        facts_dir: Optional override for the data directory (useful in tests).

    Returns:
        YearFacts with ``available=True`` on success, ``available=False`` otherwise.
    """
    base = facts_dir or _YEAR_FACTS_DIR
    filepath = base / f"{year}.json"

    try:
        raw = filepath.read_text(encoding="utf-8")
        data = json.loads(raw)
        result = YearFacts(year=data["year"], facts=data["facts"], available=True)
        return result
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return YearFacts(year=year, facts=[], available=False)


def render_year_briefing(facts: YearFacts) -> str:
    """Render a compact Vietnamese briefing block with citations.

    If ``facts.available`` is False or ``facts.facts`` is empty, returns a
    single-line sentinel.  Otherwise produces a header line followed by one
    bullet per fact, each including field, value, source, and source_date.

    Pure function — no I/O or side effects.

    Args:
        facts: A YearFacts object (typically from get_year_facts).

    Returns:
        A string suitable for injection at a ``{{YEAR_FACTS}}`` placeholder.
    """
    if not facts.available or not facts.facts:
        return f"(Chưa có dữ liệu chính thức cho năm {facts.year}.)"

    lines: list[str] = [f"Thông tin tuyển sinh năm {facts.year}:"]
    for fact in facts.facts:
        lines.append(f"- {fact.field}: {fact.value} (nguồn: {fact.source}, {fact.source_date})")
    return "\n".join(lines)
