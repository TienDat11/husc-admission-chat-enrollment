# @spec(S13) test_llm_generator_year_injection
"""Verify {{YEAR_FACTS}} placeholder is replaced at LLMGenerator __init__ time.

These tests are pure-string assertions on the loaded prompt — they do NOT
construct a full LLMGenerator (which would require LLM provider env vars).
"""
from __future__ import annotations
from pathlib import Path

import pytest


def test_prompt_file_contains_placeholder():
    """The prompt file must contain the {{YEAR_FACTS}} placeholder so that
    the runtime injection in LLMGenerator.__init__ has a target to replace."""
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "generation_system_prompt.txt"
    text = prompt_path.read_text(encoding="utf-8")
    assert "{{YEAR_FACTS}}" in text, (
        "Prompt file is missing the {{YEAR_FACTS}} placeholder — "
        "year_facts wiring will be a no-op."
    )


def test_render_briefing_unavailable_year_returns_sentinel():
    """When data/year_facts/{year}.json doesn't exist, the briefing must be
    a sentinel containing the year — never raise, never empty."""
    from src.services.year_facts import get_year_facts, render_year_briefing

    facts = get_year_facts(2099, facts_dir=Path("/nonexistent/path"))
    briefing = render_year_briefing(facts)
    assert briefing  # non-empty
    assert "2099" in briefing
    assert "Chưa có dữ liệu" in briefing


def test_replacement_logic_substitutes_placeholder():
    """Simulate the __init__ substitution: replace() must remove the
    placeholder and insert the briefing string."""
    from src.services.year_facts import YearFacts, render_year_briefing

    sample_prompt = "HEADER\n\n{{YEAR_FACTS}}\n\n=== CONTEXT ==="
    facts = YearFacts(year=2030, facts=[], available=False)
    briefing = render_year_briefing(facts)
    out = sample_prompt.replace("{{YEAR_FACTS}}", briefing)
    assert "{{YEAR_FACTS}}" not in out
    assert "2030" in out
    assert "Chưa có dữ liệu" in out
