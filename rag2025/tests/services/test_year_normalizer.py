"""Tests for year_normalizer (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "year_normalizer.py"


@pytest.fixture
def yn():
    spec = importlib.util.spec_from_file_location("year_normalizer", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nam_nay_replaced_with_current_year(yn):
    out, rep = yn.normalize_relative_year("Học phí năm nay là?", 2026)
    assert out == "Học phí năm 2026 là?"
    assert len(rep) == 1
    assert rep[0].phrase.lower() == "năm nay"
    assert rep[0].replaced_with == "năm 2026"
    assert rep[0].is_future is False


def test_nam_truoc_replaced_with_previous_year(yn):
    out, rep = yn.normalize_relative_year("Năm trước điểm chuẩn?", 2026)
    assert "năm 2025" in out
    assert len(rep) == 1
    assert rep[0].is_future is False


def test_nam_sau_replaced_with_next_year_and_marked_future(yn):
    out, rep = yn.normalize_relative_year("Năm sau có gì mới?", 2026)
    assert "năm 2027" in out
    assert len(rep) == 1
    assert rep[0].is_future is True


def test_explicit_year_left_unchanged(yn):
    out, rep = yn.normalize_relative_year("Năm 2025 ra sao?", 2026)
    # No replacement should fire because the query already has explicit year text.
    assert out == "Năm 2025 ra sao?"
    assert rep == []


def test_multiple_phrases_in_one_query(yn):
    out, rep = yn.normalize_relative_year("So sánh năm nay với năm trước", 2026)
    assert "năm 2026" in out
    assert "năm 2025" in out
    assert len(rep) == 2


def test_empty_query(yn):
    out, rep = yn.normalize_relative_year("", 2026)
    assert out == ""
    assert rep == []


def test_unrelated_text_unchanged(yn):
    out, rep = yn.normalize_relative_year("Mã ngành 7480201 thế nào?", 2026)
    assert out == "Mã ngành 7480201 thế nào?"
    assert rep == []


def test_hien_tai_replaced(yn):
    out, rep = yn.normalize_relative_year("Hiện tại có bao nhiêu ngành?", 2026)
    assert "năm 2026" in out
    assert len(rep) == 1


def test_replacement_namedtuple_shape(yn):
    out, rep = yn.normalize_relative_year("Năm nay", 2026)
    r0 = rep[0]
    assert hasattr(r0, "phrase")
    assert hasattr(r0, "replaced_with")
    assert hasattr(r0, "is_future")


def test_returns_unchanged_for_non_string(yn):
    # Defensive: non-string input shouldn't crash.
    out, rep = yn.normalize_relative_year(None, 2026)  # type: ignore[arg-type]
    assert out is None
    assert rep == []


def test_double_year_guard_explicit_year_passes_through(yn):
    """HIGH-1 regression test: query with explicit year is not substituted."""
    out, rep = yn.normalize_relative_year("Năm nay 2026 học phí?", 2026)
    assert out == "Năm nay 2026 học phí?"
    assert rep == []


def test_double_year_guard_with_historical_explicit(yn):
    out, rep = yn.normalize_relative_year("Năm trước 2025 ra sao?", 2026)
    assert out == "Năm trước 2025 ra sao?"
    assert rep == []
