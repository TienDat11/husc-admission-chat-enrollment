"""Tests for yearly_rotation (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import asyncio
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "yearly_rotation.py"


@pytest.fixture
def yr():
    spec = importlib.util.spec_from_file_location("yearly_rotation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# detect_new_year_signal

def test_detect_year_2027_in_text(yr):
    assert yr.detect_new_year_signal("Tuyển sinh năm 2027 sắp mở", current_year=2026) == 2027


def test_no_signal_for_current_year_2026(yr):
    assert yr.detect_new_year_signal("Năm 2026 ổn định", current_year=2026) is None


def test_id_overflow_with_no_year_returns_anchor_plus_one(yr):
    out = yr.detect_new_year_signal(
        "thông báo mới",
        max_known_id=74,
        seen_id=75,
        current_year=2026,
    )
    assert out == 2027


def test_id_overflow_without_year_no_overflow(yr):
    assert yr.detect_new_year_signal(
        "thông báo cũ",
        max_known_id=74,
        seen_id=70,
        current_year=2026,
    ) is None


def test_content_year_takes_precedence_over_id_overflow(yr):
    out = yr.detect_new_year_signal(
        "năm 2028 thay đổi chỉ tiêu tuyển sinh",
        max_known_id=74,
        seen_id=75,
        current_year=2026,
    )
    assert out == 2028


def test_empty_text_no_id_returns_none(yr):
    assert yr.detect_new_year_signal("", current_year=2026) is None


def test_non_string_text_safe(yr):
    assert yr.detect_new_year_signal(None, current_year=2026) is None  # type: ignore[arg-type]


# rotate

def test_rotate_calls_runners_in_order(yr):
    calls: list[str] = []

    async def crawl(year: int) -> dict:
        calls.append(f"crawl:{year}")
        return {"ok": True}

    async def chunker(year: int) -> dict:
        calls.append(f"chunker:{year}")
        return {"ok": True}

    async def supersede(prior: int) -> None:
        calls.append(f"supersede:{prior}")

    async def reload(table: str) -> dict:
        calls.append(f"reload:{table}")
        return {"active_table": table}

    res = asyncio.run(yr.rotate(
        year=2027,
        crawl=crawl,
        chunker=chunker,
        supersede_prior_year=supersede,
        reload_table=reload,
    ))
    assert calls == ["crawl:2027", "chunker:2027", "supersede:2026", "reload:husc_v2027_blue"]
    assert res["year"] == 2027
    assert res["supersede"]["prior_year"] == 2026


def test_rotate_dry_run_skips_supersede_and_reload(yr):
    calls: list[str] = []

    async def crawl(year: int) -> dict:
        calls.append("crawl")
        return {}

    async def chunker(year: int) -> dict:
        calls.append("chunker")
        return {}

    async def supersede(prior: int) -> None:
        calls.append("supersede")

    async def reload(table: str) -> dict:
        calls.append("reload")
        return {}

    res = asyncio.run(yr.rotate(
        year=2027,
        crawl=crawl,
        chunker=chunker,
        supersede_prior_year=supersede,
        reload_table=reload,
        dry_run=True,
    ))
    assert calls == ["crawl", "chunker"]
    assert res["supersede"] == "skipped (dry_run)"
    assert res["reload"] == "skipped (dry_run)"


def test_rotate_rejects_non_int_year(yr):
    async def _noop(*_a, **_k): return {}

    with pytest.raises(TypeError):
        asyncio.run(yr.rotate(
            year="2027",  # type: ignore[arg-type]
            crawl=_noop, chunker=_noop, supersede_prior_year=_noop, reload_table=_noop,
        ))


def test_rotate_rejects_year_out_of_range(yr):
    async def _noop(*_a, **_k): return {}

    with pytest.raises(ValueError):
        asyncio.run(yr.rotate(
            year=2050,
            crawl=_noop, chunker=_noop, supersede_prior_year=_noop, reload_table=_noop,
        ))


def test_rotate_propagates_crawl_error(yr):
    async def crawl(year: int) -> dict:
        raise RuntimeError("crawl down")

    async def _noop(*_a, **_k): return {}

    with pytest.raises(RuntimeError, match="crawl down"):
        asyncio.run(yr.rotate(
            year=2027,
            crawl=crawl, chunker=_noop, supersede_prior_year=_noop, reload_table=_noop,
        ))


def test_historical_comparison_does_not_trigger(yr):
    """HIGH-1 regression: 'So sánh năm 2027 với năm 2026' MUST NOT trigger
    rotation when no admissions-context keyword is nearby — this is a
    retrospective comparison, not a new-cycle signal."""
    out = yr.detect_new_year_signal("So sánh số liệu năm 2027 với năm 2026", current_year=2026)
    assert out is None


def test_admissions_keyword_with_future_year_triggers(yr):
    """Sanity: legitimate admissions-context signals still detected."""
    out = yr.detect_new_year_signal(
        "Đề án tuyển sinh năm 2027 đã được phê duyệt",
        current_year=2026,
    )
    assert out == 2027


def test_admissions_context_thong_bao_keyword(yr):
    out = yr.detect_new_year_signal(
        "Thông báo tuyển sinh khóa 2027",
        current_year=2026,
    )
    assert out == 2027


def test_admissions_context_xet_tuyen_keyword(yr):
    out = yr.detect_new_year_signal(
        "Xét tuyển năm 2028 áp dụng phương thức mới",
        current_year=2026,
    )
    assert out == 2028
