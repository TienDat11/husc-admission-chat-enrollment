# @spec(S13.5) year normalizer — replace relative time with explicit year
"""Replace relative-time phrases ("năm nay", "năm trước", ...) with explicit
years to make downstream retrieval / generation unambiguous.

Returns (normalized_query, replacements) so callers can also surface a
warning when a future-year phrase ("năm sau") was rewritten to a year that
has no data yet.
"""
from __future__ import annotations
import re
from typing import NamedTuple


_EXPLICIT_YEAR_RX = re.compile(r"\b(20[2-3]\d)\b")


class Replacement(NamedTuple):
    phrase: str
    replaced_with: str
    is_future: bool


_REL_RULES: list[tuple[re.Pattern[str], int, bool]] = [
    # (regex, year offset from current_year, is_future)
    (re.compile(r"năm\s+nay", re.IGNORECASE), 0, False),
    (re.compile(r"hiện\s+tại", re.IGNORECASE), 0, False),
    (re.compile(r"hiện\s+nay", re.IGNORECASE), 0, False),
    (re.compile(r"năm\s+hiện\s+tại", re.IGNORECASE), 0, False),
    (re.compile(r"năm\s+ngoái", re.IGNORECASE), -1, False),
    (re.compile(r"năm\s+trước", re.IGNORECASE), -1, False),
    (re.compile(r"năm\s+rồi", re.IGNORECASE), -1, False),
    (re.compile(r"năm\s+cũ", re.IGNORECASE), -1, False),
    (re.compile(r"năm\s+sau", re.IGNORECASE), 1, True),
    (re.compile(r"năm\s+tới", re.IGNORECASE), 1, True),
    (re.compile(r"sang\s+năm", re.IGNORECASE), 1, True),
]


def normalize_relative_year(query: str, current_year: int) -> tuple[str, list[Replacement]]:
    """Substitute relative-year phrases with explicit "năm <year>" form.

    Args:
        query: input string.
        current_year: anchor year (e.g. 2026).

    Returns:
        (normalized_query, replacements) — replacements list is empty when no
        substitution happened.
    """
    if not isinstance(query, str) or not query:
        return query, []

    # HIGH-1 fix: if query already contains an explicit year, do NOT substitute
    # relative phrases (avoids producing "năm 2026 2026" malformed output).
    if _EXPLICIT_YEAR_RX.search(query):
        return query, []

    out = query
    replacements: list[Replacement] = []
    for pattern, offset, is_future in _REL_RULES:
        target_year = current_year + offset
        replacement_text = f"năm {target_year}"

        def _sub(match: re.Match[str], _rt=replacement_text, _fut=is_future) -> str:
            # Default-arg capture makes the closure future-proof against lazy substitution.
            replacements.append(
                Replacement(phrase=match.group(0), replaced_with=_rt, is_future=_fut)
            )
            return _rt

        out = pattern.sub(_sub, out)
    return out, replacements
