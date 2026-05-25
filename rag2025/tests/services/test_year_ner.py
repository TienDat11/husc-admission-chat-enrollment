"""Tests for year_ner (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "year_ner.py"


@pytest.fixture
def yn():
    spec = importlib.util.spec_from_file_location("year_ner", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_year_from_text(yn):
    assert yn.extract_years("Tuyển sinh năm 2026 có gì mới?") == [2026]


def test_extract_multiple_years(yn):
    out = yn.extract_years("So sánh điểm chuẩn 2024 và 2025 với 2026")
    assert out == [2024, 2025, 2026]


def test_major_code_not_extracted_as_year(yn):
    # 7-digit major code like 7480201 — \b\d{4}\b doesn't match (7 digits)
    assert yn.extract_years("Mã ngành 7480201 thế nào?") == []


def test_three_digit_number_not_extracted(yn):
    assert yn.extract_years("Có 250 chỉ tiêu") == []


def test_year_outside_range_filtered(yn):
    # Years outside [1990, 2039] are filtered.
    assert yn.extract_years("Năm 1850 chưa có dữ liệu") == []
    assert yn.extract_years("Đến năm 2150 sẽ ra sao?") == []


def test_year_at_lower_bound(yn):
    assert yn.extract_years("Tài liệu từ năm 1990 đến 2039") == [1990, 2039]


def test_empty_text_returns_empty_list(yn):
    assert yn.extract_years("") == []


def test_non_string_returns_empty_list(yn):
    assert yn.extract_years(None) == []  # type: ignore[arg-type]


def test_extract_relative_phrases_nay(yn):
    out = yn.extract_relative_year_phrases("Học phí năm nay là?")
    assert len(out) == 1
    assert out[0]["kind"] == "nay"


def test_extract_relative_phrases_truoc(yn):
    out = yn.extract_relative_year_phrases("Năm trước điểm chuẩn?")
    assert len(out) == 1
    assert out[0]["kind"] == "truoc"


def test_extract_relative_phrases_multiple(yn):
    out = yn.extract_relative_year_phrases("Năm nay và năm trước có khác gì?")
    kinds = [p["kind"] for p in out]
    assert "nay" in kinds
    assert "truoc" in kinds


def test_extract_relative_phrases_hien_tai(yn):
    out = yn.extract_relative_year_phrases("Hiện tại có bao nhiêu ngành?")
    assert len(out) == 1
    assert out[0]["kind"] == "hien_tai"


def test_extract_relative_phrases_empty_returns_empty(yn):
    assert yn.extract_relative_year_phrases("Mã ngành 7480201") == []


def test_extract_phrases_nam_cu(yn):
    """HIGH-2 regression: 'năm cũ' is now recognized."""
    out = yn.extract_relative_year_phrases("Năm cũ điểm chuẩn?")
    kinds = [p["kind"] for p in out]
    assert "cu" in kinds


def test_extract_phrases_nam_roi(yn):
    out = yn.extract_relative_year_phrases("Năm rồi học phí?")
    kinds = [p["kind"] for p in out]
    assert "roi" in kinds


def test_extract_phrases_sang_nam(yn):
    out = yn.extract_relative_year_phrases("Sang năm có thay đổi?")
    kinds = [p["kind"] for p in out]
    assert "sang_nam" in kinds
