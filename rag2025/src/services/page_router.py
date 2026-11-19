"""Per-page routing for PDFs (plan §PL-6.8).

Every page is classified from its pypdf text (and, when available, its count of
ruling operators) as ``text``, ``scan`` or ``table``. Only ``table`` pages are
sent to the Docling tier; ``text`` pages keep the pypdf output.

Pure: no I/O. ``count_ruling_ops`` is the only helper that touches pypdf, and
it is called by the worker, not by the router.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

PageKind = Literal["text", "scan", "table"]

SCAN_CHAR_THRESHOLD = 30
RULING_OPS_THRESHOLD = 10
SHORT_NUMERIC_RUN = 5

MAJOR_CODE_RX = re.compile(r"\b7\d{6}\b")
COMBINATION_RX = re.compile(r"\b[A-Z]\d{2}\b")
NUMERIC_TOKEN_RX = re.compile(r"\d+(?:[.,]\d+)*")
SHORT_NUMERIC_LINE_RX = re.compile(r"^[\d.,\s]{1,12}$")
TABLE_KEYWORD_RX = re.compile(r"chỉ tiêu|điểm chuẩn|học phí|mã ngành", re.IGNORECASE)


@dataclass(frozen=True)
class PageRoute:
    page: int
    kind: PageKind
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PageRouting:
    pages: tuple[PageRoute, ...]

    @property
    def scan_pages(self) -> tuple[int, ...]:
        return tuple(p.page for p in self.pages if p.kind == "scan")

    @property
    def table_pages(self) -> tuple[int, ...]:
        return tuple(p.page for p in self.pages if p.kind == "table")


def _numeric_column_lines(lines: Sequence[str]) -> int:
    """Lines that look like table rows: a major code, a combination, or two+ numbers."""
    count = 0
    for line in lines:
        if MAJOR_CODE_RX.search(line) or COMBINATION_RX.search(line):
            count += 1
        elif len(NUMERIC_TOKEN_RX.findall(line)) >= 2:
            count += 1
    return count


def _longest_short_numeric_run(lines: Sequence[str]) -> int:
    best = run = 0
    for line in lines:
        if SHORT_NUMERIC_LINE_RX.match(line.strip()) and line.strip():
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _route_page(page: int, text: str, ruling_ops: int) -> PageRoute:
    stripped = text.strip()
    if len(stripped) < SCAN_CHAR_THRESHOLD:
        return PageRoute(page, "scan", ("text_below_threshold",))

    lines = [line for line in stripped.splitlines() if line.strip()]
    reasons: list[str] = []
    if ruling_ops >= RULING_OPS_THRESHOLD:
        reasons.append("ruling_lines")
    if _numeric_column_lines(lines) >= 2:
        reasons.append("numeric_columns")
    if (
        MAJOR_CODE_RX.search(stripped)
        and TABLE_KEYWORD_RX.search(stripped)
        and not COMBINATION_RX.search(stripped)
    ):
        reasons.append("code_without_combination")
    if _longest_short_numeric_run(lines) >= SHORT_NUMERIC_RUN:
        reasons.append("short_numeric_run")

    return PageRoute(page, "table" if reasons else "text", tuple(reasons))


def route_pages(texts: Sequence[str], ruling_ops: Sequence[int] | None = None) -> PageRouting:
    """Classify each page. ``ruling_ops[i]`` is the ruling-operator count for page i+1."""
    ops = list(ruling_ops) if ruling_ops is not None else [0] * len(texts)
    if len(ops) != len(texts):
        raise ValueError("ruling_ops must have one entry per page")
    return PageRouting(
        tuple(_route_page(index + 1, text, ops[index]) for index, text in enumerate(texts))
    )


def count_ruling_ops(page) -> int:
    """Rectangle (``re``) and line (``l``) operators in one pypdf page's content stream."""
    data = page.get_contents()
    if data is None:
        return 0
    raw = data.get_data()
    return len(re.findall(rb"(?m)(?:^|\s)(?:re|l)\s*$", raw))
