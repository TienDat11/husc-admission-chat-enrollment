# @spec(S14.6) test_generation_prompt_dehardcoded
r"""Regression guard for the LIVE generation system prompt de-hardcoding.

The prompt at `prompts/generation_system_prompt.txt` is loaded by
`LLMGenerator.__init__` (`src/services/llm_generator.py:42`). Its `{{YEAR_FACTS}}`
placeholder is replaced at runtime from `data/year_facts/<year>.json` via
`render_year_briefing()`.

Per S14.6: the static prompt body MUST NOT bake in year-locked facts (specific
major codes, HUSC tuition numbers, the 28-ngành enumeration, "năm 2026/2025"
literal claims). Those facts must arrive via the `{{YEAR_FACTS}}` injection and
the retrieved context, not the static prompt.

This test enforces:
  1. The `{{YEAR_FACTS}}` placeholder is still present in the static file.
  2. The canonical major-code shape (`7\d{6}([A-Z]{2})?` — covers 7480107TD,
     7440102SC, 7510302IC, 7480201VJ, 7460108, 7480103, ...) appears ZERO
     times in the static body.
  3. No bare 4-digit year literal (`\b20\d{2}\b`) appears in the static body.
  4. The behavior keywords that must survive the strip are still present:
     `context` (context-only answering) and `KHÔNG bịa` (no-fabrication).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import re


PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "generation_system_prompt.txt"
)

# Regexes mirror the gates the plan S14.6 calls out verbatim.
_MAJOR_CODE_RX = re.compile(r"7\d{6}([A-Z]{2})?")
_YEAR_LITERAL_RX = re.compile(r"\b20\d{2}\b")


@pytest.fixture(scope="module")
def prompt_text() -> str:
    """Read the static prompt once per test module.

    We deliberately read the file directly (not the post-`{{YEAR_FACTS}}` injected
    version) because the contract is about the *static* body — runtime injection
    is covered by `test_llm_generator_year_injection.py`.
    """
    assert PROMPT_PATH.exists(), f"Prompt file missing: {PROMPT_PATH}"
    return PROMPT_PATH.read_text(encoding="utf-8")


def test_prompt_file_exists_and_is_nontrivial(prompt_text: str) -> None:
    """Sanity: the file is present and not accidentally emptied."""
    assert len(prompt_text) > 1000, (
        f"Prompt file looks suspiciously short ({len(prompt_text)} chars); "
        "the de-hardcode edit may have over-stripped behavior rules."
    )


def test_year_facts_placeholder_is_preserved(prompt_text: str) -> None:
    """The `{{YEAR_FACTS}}` runtime injection point MUST remain in the file.

    If this is removed/renamed, `llm_generator.py:52-54` silently does nothing
    and the model falls back to no year-specific facts (stale-on-rollover
    regression).
    """
    assert "{{YEAR_FACTS}}" in prompt_text, (
        "Prompt file is missing the {{YEAR_FACTS}} placeholder. "
        "Runtime year_facts injection in LLMGenerator.__init__ will be a no-op."
    )
    # Count exactly once — duplicate placeholders would corrupt briefing blocks.
    assert prompt_text.count("{{YEAR_FACTS}}") == 1, (
        "Expected exactly one {{YEAR_FACTS}} placeholder; found "
        f"{prompt_text.count('{{YEAR_FACTS}}')}."
    )


def test_no_static_major_codes_in_prompt_body(prompt_text: str) -> None:
    """Zero matches of the canonical 7-digit (+ optional 2-letter) major-code shape.

    Catches: 7480107TD, 7440102SC, 7510302IC, 7480201VJ, 7460108, 7480103,
    7480201, 7440102, 7510302, 7310205, 7220104, 7310301, 7320101, 7480107,
    and any other 7xxxxxx / 7xxxxxxAB form.

    Per S14.6: static major codes must come from retrieved context +
    `{{YEAR_FACTS}}`, not the prompt — otherwise the model leans on baked-in
    codes (a hallucination + stale-on-rollover source).
    """
    matches = _MAJOR_CODE_RX.findall(prompt_text)
    assert matches == [], (
        "Prompt body still contains hardcoded major-code literals "
        f"({len(matches)} matches). First few: {matches[:5]}. "
        "Strip these — they belong in {{YEAR_FACTS}} or retrieved context, "
        "not the static prompt."
    )


def test_no_bare_year_literal_in_prompt_body(prompt_text: str) -> None:
    """Zero 4-digit year literals (`\\b20\\d{2}\\b`) in the static body.

    Catches "2024", "2025", "2026" anywhere in the file. Per S14.6: year facts
    must come from {{YEAR_FACTS}} injection at runtime, not from a literal
    "năm 2026" baked into the prompt.
    """
    matches = _YEAR_LITERAL_RX.findall(prompt_text)
    assert matches == [], (
        "Prompt body still contains year-literal claims "
        f"({len(matches)} matches, unique: {sorted(set(matches))}). "
        "Replace with 'năm hiện tại (N do {{YEAR_FACTS}} cung cấp)' or rely on "
        "the runtime injection. The static prompt must be year-agnostic."
    )


def test_context_only_behavior_marker_preserved(prompt_text: str) -> None:
    """The "answer from context" rule must survive the de-hardcode edit.

    A no-context marker would let the model hallucinate freely. Use lowercase
    match to be tolerant of case variations introduced by edits.
    """
    assert "context" in prompt_text.lower(), (
        "Behavior marker 'context' is missing — the context-only-answer rule "
        "appears to have been stripped. Re-add the rule that tells the model "
        "to use retrieved context as the source of truth."
    )


def test_no_fabrication_behavior_marker_preserved(prompt_text: str) -> None:
    """The "KHÔNG bịa" no-fabrication rule must survive the de-hardcode edit.

    Stripping this would let the model invent majors/codes/phones/addresses —
    the exact failure mode S14.6 is closing.
    """
    assert "KHÔNG bịa" in prompt_text, (
        "Behavior marker 'KHÔNG bịa' is missing — the no-fabrication rule "
        "appears to have been stripped. Re-add the explicit prohibition on "
        "inventing facts not grounded in context or {{YEAR_FACTS}}."
    )


def test_vietnamese_only_behavior_marker_preserved(prompt_text: str) -> None:
    """The Vietnamese-only rule must survive the de-hardcode edit.

    The model must not switch to English or mix languages when answering.
    """
    # Look for the canonical "TIẾNG VIỆT" marker (case-insensitive, accent-tolerant).
    assert "tiếng việt" in prompt_text.lower(), (
        "Behavior marker 'tiếng Việt' is missing — the Vietnamese-only rule "
        "appears to have been stripped."
    )


def test_answer_first_behavior_marker_preserved(prompt_text: str) -> None:
    """The 'ANSWER-FIRST' priority rule must survive the de-hardcode edit.

    If a single context chunk is even loosely relevant, the model must answer
    rather than refuse. This is the year-agnostic core behavior.
    """
    assert "ANSWER-FIRST" in prompt_text, (
        "Behavior marker 'ANSWER-FIRST' is missing — the answer-first rule "
        "appears to have been stripped. Re-add the rule that mandates "
        "synthesizing an answer whenever any context chunk is relevant."
    )


def test_not_found_escape_hatch_preserved(prompt_text: str) -> None:
    """The 'không tìm thấy' escape hatch must survive the de-hardcode edit.

    The model must still know the canonical 'no relevant chunk' refusal
    phrase — this is the contract the no-fabrication rule references.
    """
    assert "không tìm thấy" in prompt_text.lower(), (
        "Behavior marker 'không tìm thấy' is missing — the model needs the "
        "canonical refusal phrase for zero-relevant-chunk cases."
    )
