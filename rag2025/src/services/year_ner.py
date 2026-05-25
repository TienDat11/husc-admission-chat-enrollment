# @spec(S13.5) year NER — pure regex extractor (no spaCy/heavy NER)
"""Extract year mentions from Vietnamese text.

Two extractors:
  - extract_years(text): list[int] of plausible years (1990-2039).
  - extract_relative_year_phrases(text): list of {phrase, kind} for relative
    expressions ("năm nay", "năm trước", "năm sau", ...).

False-positive guard: 4-digit numbers in major codes (start with 7) are NOT
year mentions and are filtered out. We accept the [1990, 2039] range so we
catch historical references; admissions data effectively starts ~2010 but
we keep the lower bound at 1990 for safety.
"""
from __future__ import annotations
import re
from typing import TypedDict


# Year range guard — outside this we treat as a non-year number.
YEAR_LO = 1990
YEAR_HI = 2039


class RelativePhrase(TypedDict):
    phrase: str
    kind: str  # one of: nay | truoc | ngoai | sau | toi | hien_tai


_YEAR_RX = re.compile(r"\b(\d{4})\b")
_REL_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"năm\s+nay", re.IGNORECASE), "nay"),
    (re.compile(r"năm\s+ngoái", re.IGNORECASE), "ngoai"),
    (re.compile(r"năm\s+trước", re.IGNORECASE), "truoc"),
    (re.compile(r"năm\s+sau", re.IGNORECASE), "sau"),
    (re.compile(r"năm\s+tới", re.IGNORECASE), "toi"),
    (re.compile(r"hiện\s+tại", re.IGNORECASE), "hien_tai"),
    (re.compile(r"hiện\s+nay", re.IGNORECASE), "hien_tai"),
    (re.compile(r"năm\s+cũ", re.IGNORECASE), "cu"),
    (re.compile(r"năm\s+rồi", re.IGNORECASE), "roi"),
    (re.compile(r"sang\s+năm", re.IGNORECASE), "sang_nam"),
]


def extract_years(text: str) -> list[int]:
    """Return all plausible 4-digit years (in YEAR_LO..YEAR_HI) from text.

    Filters out 4-digit numbers outside the year window (e.g. major codes
    like 7480201 don't match anyway because they're 7 digits, but defensive
    bounds keep us safe against edge inputs).
    """
    if not isinstance(text, str) or not text:
        return []
    out: list[int] = []
    for m in _YEAR_RX.finditer(text):
        n = int(m.group(1))
        if YEAR_LO <= n <= YEAR_HI:
            out.append(n)
    return out


def extract_relative_year_phrases(text: str) -> list[RelativePhrase]:
    """Return list of {phrase, kind} for relative-year expressions found."""
    if not isinstance(text, str) or not text:
        return []
    out: list[RelativePhrase] = []
    for pattern, kind in _REL_RULES:
        for m in pattern.finditer(text):
            out.append({"phrase": m.group(0), "kind": kind})
    return out
