"""Atomic gate: cross-check Docling table output against pypdf (plan §PL-6.8).

pypdf reads every number on a table page; Docling reads the same page with the
table structure. A number pypdf sees on a row that Docling does not is a cell
drop, and the upload is blocked for that page. Pure: no I/O.

Blocker codes:
- TABLE_CELL_DROP_SUSPECTED: a numeric atom on a pypdf row is absent from Docling
- TABLE_ROW_COUNT_MISMATCH: Docling shows fewer major-code rows than pypdf
- MAJOR_CODE_DROPPED: a major code pypdf reads is absent from Docling
- DOCLING_MISSING: a table page has no Docling output (fail closed)
- SCAN_LOW_CONFIDENCE: an OCR'd scan page still has too little text
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from src.services.page_router import MAJOR_CODE_RX, NUMERIC_TOKEN_RX, SCAN_CHAR_THRESHOLD


PAGE_GROUP = "__page__"


@dataclass(frozen=True)
class Blocker:
    code: str
    page: int | None
    payload: dict = field(default_factory=dict)


def _atoms(line: str) -> list[str]:
    """Numeric atoms on a line, codes removed, decimal comma normalized to dot."""
    without_codes = MAJOR_CODE_RX.sub(" ", line)
    return [token.replace(",", ".") for token in NUMERIC_TOKEN_RX.findall(without_codes)]


def _group_lines(text: str) -> dict[str, list[str]]:
    """Lines grouped by the major code they carry (or PAGE_GROUP)."""
    groups: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        code = MAJOR_CODE_RX.search(line)
        key = code.group(0) if code else PAGE_GROUP
        groups.setdefault(key, []).append(line)
    return groups


def _gate_table_page(page: int, pypdf_text: str, docling_text: str) -> list[Blocker]:
    blockers: list[Blocker] = []
    pypdf_groups = _group_lines(pypdf_text)
    docling_groups = _group_lines(docling_text)

    for key, lines in pypdf_groups.items():
        expected = Counter(atom for line in lines for atom in _atoms(line))
        seen = Counter(atom for line in docling_groups.get(key, []) for atom in _atoms(line))
        missing = expected - seen
        if not missing:
            continue
        missing_rows = [line.strip() for line in lines if any(atom in missing for atom in _atoms(line))]
        blockers.append(
            Blocker(
                "TABLE_CELL_DROP_SUSPECTED",
                page,
                {
                    "group": key,
                    "missing_atoms": sorted(missing),
                    "missing_pypdf_rows": missing_rows,
                },
            )
        )

    pypdf_codes = {key for key in pypdf_groups if key != PAGE_GROUP}
    docling_codes = {key for key in docling_groups if key != PAGE_GROUP}
    dropped = sorted(pypdf_codes - docling_codes)
    if dropped:
        blockers.append(Blocker("MAJOR_CODE_DROPPED", page, {"codes": dropped}))
    if len(docling_codes) < len(pypdf_codes):
        blockers.append(
            Blocker(
                "TABLE_ROW_COUNT_MISMATCH",
                page,
                {"pypdf_rows": len(pypdf_codes), "docling_rows": len(docling_codes)},
            )
        )
    return blockers


def atomic_gate(
    pypdf_pages: Sequence[str],
    table_pages: Iterable[int],
    docling_pages: Mapping[int, str],
    ocr_pages: Mapping[int, str] | None = None,
) -> list[Blocker]:
    """Return blockers. An empty list means the upload may proceed.

    ``pypdf_pages[i]`` is page i+1 text. ``docling_pages`` maps 1-based page
    numbers to Docling text for the table pages. ``ocr_pages`` maps scan pages
    to their OCR text.
    """
    blockers: list[Blocker] = []
    for page in table_pages:
        if page not in docling_pages:
            blockers.append(Blocker("DOCLING_MISSING", page))
            continue
        blockers.extend(_gate_table_page(page, pypdf_pages[page - 1], docling_pages[page]))

    for page, text in (ocr_pages or {}).items():
        if len(text.strip()) < SCAN_CHAR_THRESHOLD:
            blockers.append(Blocker("SCAN_LOW_CONFIDENCE", page, {"chars": len(text.strip())}))
    return blockers
