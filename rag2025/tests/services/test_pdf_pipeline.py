"""RED tests for three-tier PDF parsing router."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.pdf_pipeline import ParseResult, parse_pdf  # noqa: E402


def _healthy(parser: str, page_count: int, chars_per_page: int) -> ParseResult:
    return ParseResult(
        text="x" * (page_count * chars_per_page),
        tables_md=[],
        parser=parser,
        page_count=page_count,
        needs_review=False,
    )


def test_pypdf_tier0_happy(tmp_path: Path) -> None:
    pdf_path = tmp_path / "healthy.pdf"
    pdf_path.write_bytes(b"healthy")
    cache_dir = tmp_path / "cache"
    tier0_result = _healthy("pypdf", 5, 200)

    with (
        patch("src.services.pdf_pipeline._tier0_pypdf", return_value=tier0_result) as tier0,
        patch("src.services.pdf_pipeline._tier1_docling") as tier1,
        patch("src.services.pdf_pipeline._tier2_docling") as tier2,
    ):
        result = parse_pdf(pdf_path, cache_dir)

    assert result == tier0_result
    tier0.assert_called_once_with(pdf_path)
    tier1.assert_not_called()
    tier2.assert_not_called()


def test_escalate_on_empty_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"empty")
    cache_dir = tmp_path / "cache"
    tier0_result = _healthy("pypdf", 3, 3)
    tier1_result = _healthy("docling", 3, 200)

    with (
        patch("src.services.pdf_pipeline._tier0_pypdf", return_value=tier0_result) as tier0,
        patch("src.services.pdf_pipeline._tier1_docling", return_value=tier1_result) as tier1,
        patch("src.services.pdf_pipeline._tier2_docling") as tier2,
    ):
        result = parse_pdf(pdf_path, cache_dir)

    assert result == tier1_result
    tier0.assert_called_once_with(pdf_path)
    tier1.assert_called_once_with(pdf_path, ocr=False)
    tier2.assert_not_called()


def test_escalate_on_table_hint(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"table")
    cache_dir = tmp_path / "cache"
    tier0_result = _healthy("pypdf", 2, 4)
    tier1_result = _healthy("docling", 2, 180)

    with (
        patch("src.services.pdf_pipeline._tier0_pypdf", return_value=tier0_result),
        patch("src.services.pdf_pipeline._tier1_docling", return_value=tier1_result) as tier1,
        patch("src.services.pdf_pipeline._tier2_docling") as tier2,
    ):
        result = parse_pdf(pdf_path, cache_dir)

    assert result == tier1_result
    tier1.assert_called_once_with(pdf_path, ocr=False)
    tier2.assert_not_called()


def test_parse_cache_hit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cached.pdf"
    pdf_path.write_bytes(b"cached")
    cache_dir = tmp_path / "cache"
    tier0_result = _healthy("pypdf", 5, 200)

    with patch("src.services.pdf_pipeline._tier0_pypdf", return_value=tier0_result) as tier0:
        first = parse_pdf(pdf_path, cache_dir)
        second = parse_pdf(pdf_path, cache_dir)

    assert first == tier0_result
    assert second == tier0_result
    tier0.assert_called_once_with(pdf_path)
