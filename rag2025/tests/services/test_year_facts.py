"""Tests for rag2025.src.services.year_facts — get_year_facts + render_year_briefing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.year_facts import YearFact, YearFacts, get_year_facts, render_year_briefing


# ---------------------------------------------------------------------------
# get_year_facts — valid file
# ---------------------------------------------------------------------------

def test_get_year_facts_valid_file(tmp_path: Path) -> None:
    """Returns available=True with correctly parsed facts when a valid JSON exists."""
    year = 2025
    payload = {
        "year": year,
        "facts": [
            {
                "field": "Tổng chỉ tiêu",
                "value": "3.500",
                "source": "QB2025-01",
                "source_date": "2025-03-15",
            },
            {
                "field": "Phương thức tuyển sinh",
                "value": "3 phương thức",
                "source": "QB2025-02",
                "source_date": "2025-04-01",
            },
        ],
    }
    (tmp_path / f"{year}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = get_year_facts(year, facts_dir=tmp_path)

    assert result.available is True
    assert result.year == year
    assert len(result.facts) == 2
    assert result.facts[0].field == "Tổng chỉ tiêu"
    assert result.facts[0].value == "3.500"
    assert result.facts[0].source == "QB2025-01"
    assert result.facts[0].source_date == "2025-03-15"
    assert result.facts[1].field == "Phương thức tuyển sinh"


# ---------------------------------------------------------------------------
# get_year_facts — missing file
# ---------------------------------------------------------------------------

def test_get_year_facts_missing_file(tmp_path: Path) -> None:
    """Returns available=False with empty facts when the file is missing (no raise)."""
    result = get_year_facts(2099, facts_dir=tmp_path)

    assert result.available is False
    assert result.year == 2099
    assert result.facts == []


# ---------------------------------------------------------------------------
# get_year_facts — malformed JSON
# ---------------------------------------------------------------------------

def test_get_year_facts_malformed_json(tmp_path: Path) -> None:
    """Returns available=False when the file contains malformed JSON (no raise)."""
    year = 2025
    (tmp_path / f"{year}.json").write_text("{not json", encoding="utf-8")

    result = get_year_facts(year, facts_dir=tmp_path)

    assert result.available is False
    assert result.year == year
    assert result.facts == []


# ---------------------------------------------------------------------------
# render_year_briefing — sentinel when unavailable
# ---------------------------------------------------------------------------

def test_render_year_briefing_sentinel() -> None:
    """Returns the sentinel string containing 'Chưa có dữ liệu' when available=False."""
    facts = YearFacts(year=2099, facts=[], available=False)

    briefing = render_year_briefing(facts)

    assert "Chưa có dữ liệu" in briefing
    assert "2099" in briefing


# ---------------------------------------------------------------------------
# render_year_briefing — includes field value and source citation
# ---------------------------------------------------------------------------

def test_render_year_briefing_with_citations() -> None:
    """Includes the field value AND the source (citation) when available=True."""
    facts = YearFacts(
        year=2025,
        facts=[
            YearFact(
                field="Tổng chỉ tiêu",
                value="3.500",
                source="QB2025-01",
                source_date="2025-03-15",
            ),
        ],
        available=True,
    )

    briefing = render_year_briefing(facts)

    # Must contain the field value
    assert "3.500" in briefing
    # Must contain the source citation
    assert "QB2025-01" in briefing
    # Must contain the header
    assert "Thông tin tuyển sinh năm 2025" in briefing
    # Must contain the source date
    assert "2025-03-15" in briefing
