"""PII detection flags for uploaded documents — ticket T04.

Reuses the guardrail's PII patterns rather than inventing a second set:
the guardrail already owns the regexes for phone / citizen-id / email
and the keyword list, and two definitions would drift apart.
"""
from __future__ import annotations

import re
from typing import Mapping

from src.services.guardrail import GuardrailService


def blocking_pii_flags(text: str, patterns: Mapping[str, "re.Pattern[str]" | None] | None = None) -> tuple[str, ...]:
    """PII flag names derived ONLY from ``GuardrailService.PII_PATTERNS``
    (email / phone / citizen_id).

    The caller stores the *flag* on the draft, never the matched value —
    the value belongs to the document, the flag is what the UI warns
    about. T05 owns the preview gate that a PII draft cannot be previewed
    from; this module only detects.

    Unlike :func:`pii_flags`, this variant excludes ``PII_KEYWORDS``:
    admission vocabulary such as ``otp`` appears in legitimate admissions
    text, and folding it in would block drafts that are otherwise clean.
    """
    if not text:
        return ()
    resolved = patterns or GuardrailService.PII_PATTERNS
    found: list[str] = []
    for name, pattern in resolved.items():
        if pattern is not None and pattern.search(text):
            found.append(name)
    return tuple(found)


def pii_flags(text: str, patterns: Mapping[str, "re.Pattern[str]" | None] | None = None) -> tuple[str, ...]:
    """Which PII markers ``text`` carries, as flag names (no values stored).

    The caller stores the *flag* on the draft, never the matched value —
    the value belongs to the document, the flag is what the UI warns
    about. T05 owns the preview gate that a PII draft cannot be previewed
    from; this module only detects.
    """
    if not text:
        return ()
    resolved = patterns or GuardrailService.PII_PATTERNS
    found: list[str] = []
    for name, pattern in resolved.items():
        if pattern is not None and pattern.search(text):
            found.append(name)
    for keyword in GuardrailService.PII_KEYWORDS:
        if keyword in text.lower():
            found.append(keyword)
    return tuple(found)


def pii_detected(text: str) -> bool:
    """Any PII marker present."""
    return bool(pii_flags(text))
