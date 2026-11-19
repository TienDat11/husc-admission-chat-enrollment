"""Atomic gate (plan §PL-6.8): a table page passes only when Docling keeps its cells."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.atomic_gate import atomic_gate  # noqa: E402

PYPDF = "7220201 Kỹ thuật phần mềm 120 25.5\n7340101 Quản trị kinh doanh 90 24.75"


def _codes(blockers) -> set[str]:
    return {b.code for b in blockers}


def test_identical_table_passes() -> None:
    assert atomic_gate([PYPDF], [1], {1: PYPDF}) == []


def test_decimal_comma_is_normalised() -> None:
    pypdf_text = PYPDF.replace("25.5", "25,5")
    assert atomic_gate([pypdf_text], [1], {1: PYPDF}) == []


def test_dropped_cell_is_flagged() -> None:
    docling = "7220201 Kỹ thuật phần mềm 120\n7340101 Quản trị kinh doanh 90 24.75"
    assert "TABLE_CELL_DROP_SUSPECTED" in _codes(atomic_gate([PYPDF], [1], {1: docling}))


def test_dropped_major_code_is_flagged() -> None:
    docling = "7220201 Kỹ thuật phần mềm 120 25.5"
    assert "MAJOR_CODE_DROPPED" in _codes(atomic_gate([PYPDF], [1], {1: docling}))


def test_missing_docling_fails_closed() -> None:
    assert _codes(atomic_gate([PYPDF], [1], {})) == {"DOCLING_MISSING"}
