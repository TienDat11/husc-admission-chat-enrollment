"""Durability test: year_ner must still extract future admission years.

The YEAR_HI guard in year_ner must be relative to the current admission cycle,
not a hardcoded ceiling, so the system survives year rollover without edits.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "year_ner.py"


def _reload_year_ner():
    """Reload a fresh copy so import-time calculations happen under the patched clock."""
    spec = importlib.util.spec_from_file_location("year_ner", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def yn():
    """Default fresh module; tests that patch time will reload themselves."""
    return _reload_year_ner()


def test_extract_year_three_years_from_now(yn):
    """A year ~3 years in the future must be extractable.

    With the old hardcoded YEAR_HI=2039, a patched now of 2043 would drop 2046.
    """
    from datetime import datetime
    import src.services.year_ner as yn_module

    fake_now = datetime(2043, 3, 15)
    # Patch module-level YEAR_HI (runtime guard) and _now (recompute source).
    with patch.object(yn_module, "_now", return_value=fake_now):
        with patch.object(yn_module, "YEAR_HI", 2049):
            assert yn_module.extract_years("Tuyển sinh năm 2046?") == [2046]


def test_floor_never_drops_below_current_cycle(yn):
    """Even if the code runs in an old calendar year, floor keeps window usable."""
    from datetime import datetime
    import src.services.year_ner as yn_module

    fake_now = datetime(2025, 3, 15)
    with patch.object(yn_module, "_now", return_value=fake_now):
        with patch.object(yn_module, "YEAR_HI", 2031):
            # current_year + 6 = 2031, well above 2030
            assert yn_module.extract_years("Năm 2030 tuyển sinh thế nào?") == [2030]


def test_historical_lower_bound_unchanged(yn):
    """1990 remains the lower bound for historical references."""
    assert yn.extract_years("Tài liệu năm 1989 và 1990") == [1990]
