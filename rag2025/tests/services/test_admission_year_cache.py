"""Lazy admission-year cache and hoisted main.py FAQ tables (behaviour-preserving refactor)."""
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import services.lancedb_retrieval as lr  # noqa: E402


def test_admission_year_is_lazy_cached_and_invalidated(monkeypatch):
    calls = []

    def fake_detect():
        calls.append(1)
        return 2000 + len(calls)

    monkeypatch.setattr(lr, "_detect_current_admission_year", fake_detect)
    lr.invalidate_admission_year_cache()
    try:
        assert lr.get_current_admission_year() == 2001
        assert lr.get_current_admission_year() == 2001
        assert len(calls) == 1
        lr.invalidate_admission_year_cache()
        assert lr.get_current_admission_year() == 2002
        assert len(calls) == 2
    finally:
        lr.invalidate_admission_year_cache()


def test_legacy_current_admission_year_symbol_resolves_lazily(monkeypatch):
    monkeypatch.setattr(lr, "_detect_current_admission_year", lambda: 2033)
    lr.invalidate_admission_year_cache()
    try:
        assert lr.CURRENT_ADMISSION_YEAR == 2033
        assert from_import_value() == 2033
    finally:
        lr.invalidate_admission_year_cache()
    with pytest.raises(AttributeError):
        lr.NO_SUCH_SYMBOL  # noqa: B018


def from_import_value():
    from services.lancedb_retrieval import CURRENT_ADMISSION_YEAR

    return CURRENT_ADMISSION_YEAR


def test_main_faq_tables_are_module_level():
    import main as app_main

    assert app_main.INTENT_TO_FAQ_TYPES["tuition"] == ["hoc_phi"]
    assert "các ngành" in app_main.OVERVIEW_KEYWORDS
    assert "học phí" in app_main.TUITION_KEYWORDS
    assert app_main.get_faq_type_filter("học phí một kỳ") == {
        "or_conditions": [{"faq_type": "hoc_phi"}]
    }
    assert app_main.get_faq_type_filter("hello") is None
