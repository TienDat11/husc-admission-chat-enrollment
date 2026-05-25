"""Tests for temporal_intent (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "temporal_intent.py"


@pytest.fixture
def ti():
    spec = importlib.util.spec_from_file_location("temporal_intent", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_explicit_year_2026_with_current_2026_returns_current(ti):
    assert ti.classify_temporal("Học phí năm 2026 là bao nhiêu?", 2026) == ti.TemporalIntent.current


def test_explicit_year_2025_with_current_2026_returns_historical(ti):
    assert ti.classify_temporal("Điểm chuẩn năm 2025?", 2026) == ti.TemporalIntent.historical


def test_explicit_year_future_year_2027_with_current_2026_returns_current(ti):
    # Future-year explicit (>= current) is treated as current per spec.
    assert ti.classify_temporal("Tuyển sinh 2027?", 2026) == ti.TemporalIntent.current


def test_nam_nay_returns_current(ti):
    assert ti.classify_temporal("Học phí năm nay là bao nhiêu?", 2026) == ti.TemporalIntent.current


def test_hien_tai_returns_current(ti):
    assert ti.classify_temporal("Hiện tại có bao nhiêu ngành?", 2026) == ti.TemporalIntent.current


def test_nam_truoc_returns_historical(ti):
    assert ti.classify_temporal("Năm trước điểm chuẩn ra sao?", 2026) == ti.TemporalIntent.historical


def test_nam_ngoai_returns_historical(ti):
    assert ti.classify_temporal("Năm ngoái có ngành A không?", 2026) == ti.TemporalIntent.historical


def test_so_sanh_returns_cross_year(ti):
    assert ti.classify_temporal("So sánh điểm chuẩn 2024 và 2026", 2026) == ti.TemporalIntent.cross_year


def test_qua_cac_nam_returns_cross_year(ti):
    assert ti.classify_temporal("Học phí qua các năm thay đổi thế nào", 2026) == ti.TemporalIntent.cross_year


def test_no_signal_returns_ambiguous(ti):
    assert ti.classify_temporal("Học phí ngành CNTT là bao nhiêu?", 2026) == ti.TemporalIntent.ambiguous


def test_empty_query_returns_ambiguous(ti):
    assert ti.classify_temporal("", 2026) == ti.TemporalIntent.ambiguous


def test_future_year_word_returns_ambiguous(ti):
    # "năm sau" → no future data → ambiguous so caller can warn
    assert ti.classify_temporal("Năm sau có thay đổi gì?", 2026) == ti.TemporalIntent.ambiguous


def test_cross_year_overrides_explicit_year(ti):
    # Cross-year keyword present + explicit year → cross_year wins.
    assert ti.classify_temporal("So sánh năm 2026 với 2025", 2026) == ti.TemporalIntent.cross_year


def test_extract_explicit_year_returns_first_match(ti):
    assert ti.extract_explicit_year("So sánh 2025 và 2026") == 2025


def test_extract_explicit_year_returns_none_for_major_code(ti):
    # 7480201 starts with 7 — outside the [2020, 2039] year range.
    assert ti.extract_explicit_year("Mã ngành 7480201 có gì?") is None


def test_extract_explicit_year_returns_none_for_no_year(ti):
    assert ti.extract_explicit_year("Học phí thế nào?") is None


@pytest.mark.asyncio
async def test_llm_fallback_only_runs_on_ambiguous(ti):
    calls = []

    async def fake_llm(q: str):
        calls.append(q)
        return ti.TemporalIntent.current

    # Non-ambiguous path: LLM not called.
    result = await ti.classify_with_llm_fallback("năm nay", 2026, llm_runner=fake_llm)
    assert result == ti.TemporalIntent.current
    assert calls == []

    # Ambiguous path: LLM called.
    result2 = await ti.classify_with_llm_fallback("Học phí thế nào", 2026, llm_runner=fake_llm)
    assert result2 == ti.TemporalIntent.current
    assert calls == ["Học phí thế nào"]


@pytest.mark.asyncio
async def test_llm_fallback_skipped_when_no_runner(ti):
    result = await ti.classify_with_llm_fallback("Học phí thế nào", 2026, llm_runner=None)
    assert result == ti.TemporalIntent.ambiguous


def test_relative_phrase_wins_over_explicit_year_historical(ti):
    """HIGH-3 regression: 'năm trước 2026' → historical, not current."""
    # User said "năm trước" (last year) — must be historical regardless of incidental 2026.
    assert ti.classify_temporal("Năm trước 2026 điểm chuẩn?", 2026) == ti.TemporalIntent.historical


def test_relative_phrase_wins_over_explicit_year_current(ti):
    """HIGH-3 regression: 'năm nay 2025' → current (user means current_year, not 2025)."""
    assert ti.classify_temporal("Năm nay 2025 học phí?", 2026) == ti.TemporalIntent.current


def test_year_2019_now_classified_as_historical(ti):
    """MED-6 regression: year range now covers [1990, 2039], so 2019 → historical."""
    assert ti.classify_temporal("Điểm chuẩn 2019 ra sao?", 2026) == ti.TemporalIntent.historical
