# @spec(T0 / rich-markdown-generation-plan) test_markdown_table_validation
"""TDD: test the _validate_markdown_table post-guard for LLM-generated tables.

Three cases:
  1. Valid table → pass-through unchanged (idempotent).
  2. Mismatched columns → pad short rows to match expected column count.
  3. Completely broken (no separator row) → degrade to bullet list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

# Import the module-level function we will implement.
# (It does not exist yet — this test will FAIL until the implementation lands.)
from services.llm_generator import _validate_markdown_table


# ---------------------------------------------------------------------------
# Case 1: Valid table — idempotent pass-through
# ---------------------------------------------------------------------------

def test_valid_table_passes_through_unchanged() -> None:
    answer = (
        "Dưới đây là bảng học phí:\n\n"
        "| Khối | Đơn giá (VNĐ/tín chỉ) |\n"
        "|------|----------------------|\n"
        "| VII  | 490.000              |\n"
        "| IV   | 555.000              |\n"
        "| V    | 600.000              |\n"
    )
    result = _validate_markdown_table(answer)
    assert result == answer, "A valid, balanced table must pass through unchanged."
    assert "| Khối |" in result
    assert "|------|" in result


# ---------------------------------------------------------------------------
# Case 2: Mismatched columns — pad short rows
# ---------------------------------------------------------------------------

def test_mismatched_columns_are_padded() -> None:
    answer = (
        "Bảng so sánh:\n\n"
        "| Ngành       | Điểm chuẩn | Tổ hợp |\n"
        "|-------------|------------|--------|\n"
        "| CNTT        | 24.5       | A00    |\n"
        "| KHDL        | 23.0       |        |\n"  # missing last column
        "| KHMT        | 22.5       | A01    | B00 |\n"  # extra column (should not happen but handle)
    )
    result = _validate_markdown_table(answer)
    assert result is not None
    # After padding/fixing, all data rows should have the same column count as the header.
    lines = result.split("\n")
    # Find table rows
    data_rows = [l for l in lines if l.startswith("|") and "---" not in l and "Ngành" not in l and "Điểm" not in l]
    # All data rows should have same number of pipes
    pipe_counts = [l.count("|") for l in data_rows]
    if len(set(pipe_counts)) > 1:
        # If still unequal, the guard should have degraded it (Case 3 logic).
        # But for this test, padding should have normalized them.
        assert len(set(pipe_counts)) == 1, (
            f"After validation, all table rows must have equal pipe counts. "
            f"Got: {pipe_counts}"
        )


# ---------------------------------------------------------------------------
# Case 3: No separator row → degrade to bullet list
# ---------------------------------------------------------------------------

def test_no_separator_row_degraded_to_bullets() -> None:
    answer = (
        "Bảng:\n"
        "| CNTT | 24.5 |\n"
        "| KHDL | 23.0 |\n"
        "| KHMT | 22.5 |\n"
    )
    result = _validate_markdown_table(answer)
    assert result is not None
    # After degradation, the pipe characters should be gone (converted to bullets or plain text).
    assert "|" not in result, (
        "A table block without a separator row must be degraded — "
        "no pipe characters should remain."
    )
    # Content should still be present somehow.
    assert "CNTT" in result
    assert "KHDL" in result


# ---------------------------------------------------------------------------
# Edge case: No tables at all → pass-through unchanged
# ---------------------------------------------------------------------------

def test_no_tables_unchanged() -> None:
    answer = "Đây là câu trả lời không có bảng nào. Chỉ có **in đậm** và - gạch đầu dòng."
    result = _validate_markdown_table(answer)
    assert result == answer, "An answer with no table blocks must pass through unchanged."


# ---------------------------------------------------------------------------
# Edge case: Empty string
# ---------------------------------------------------------------------------

def test_empty_string_passes_through() -> None:
    assert _validate_markdown_table("") == ""
    assert _validate_markdown_table("   ") == "   "
