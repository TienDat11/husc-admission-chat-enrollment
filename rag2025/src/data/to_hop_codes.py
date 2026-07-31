"""Canonical HUSC tổ hợp codes — single source of truth.

All TO_HOP pattern usage in the codebase MUST import from this module.
Sync with llm_generator.py:_detect_english allowed set and generation_system_prompt.txt
"""
import re

ALL_TO_HOP_CODES: list[str] = sorted({
    "A00", "A01", "A02", "A04", "A09", "A15",
    "B00", "B02", "B03", "B08",
    "C00", "C01", "C02", "C03", "C04", "C14", "C19",
    "D01", "D06", "D07", "D09", "D10", "D14", "D66", "D78", "D84",
    "V00", "V01", "V02", "V12",
    "X01", "X02", "X06", "X21", "X25", "X26", "X70", "X78",
})

TO_HOP_PATTERN_RE = re.compile(
    r"\b(" + "|".join(ALL_TO_HOP_CODES) + r")\b",
    re.IGNORECASE,
)

# Alias for backward compatibility
_TO_HOP_PATTERN = TO_HOP_PATTERN_RE
