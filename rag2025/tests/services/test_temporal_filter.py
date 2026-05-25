"""Tests for temporal_filter (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services"
if str(SRC.parent) not in sys.path:
    sys.path.insert(0, str(SRC.parent))


@pytest.fixture
def ti_module():
    spec = importlib.util.spec_from_file_location("services.temporal_intent", SRC / "temporal_intent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["services.temporal_intent"] = module  # register BEFORE exec so relative imports resolve
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tf_module(ti_module):
    # ti_module must load first so services.temporal_intent is in sys.modules
    spec = importlib.util.spec_from_file_location("services.temporal_filter", SRC / "temporal_filter.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "services"  # enable relative imports within the module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_intent_returns_eq_filter(tf_module, ti_module):
    out = tf_module.apply_temporal_filter(ti_module.TemporalIntent.current, 2026)
    assert out == "data_year = 2026"


def test_historical_intent_returns_lt_filter(tf_module, ti_module):
    out = tf_module.apply_temporal_filter(ti_module.TemporalIntent.historical, 2026)
    assert out == "data_year < 2026"


def test_cross_year_returns_none(tf_module, ti_module):
    assert tf_module.apply_temporal_filter(ti_module.TemporalIntent.cross_year, 2026) is None


def test_ambiguous_returns_none(tf_module, ti_module):
    assert tf_module.apply_temporal_filter(ti_module.TemporalIntent.ambiguous, 2026) is None


def test_non_int_year_raises_type_error(tf_module, ti_module):
    with pytest.raises(TypeError):
        tf_module.apply_temporal_filter(ti_module.TemporalIntent.current, "2026; DROP TABLE husc--")  # type: ignore[arg-type]


def test_year_below_2020_raises(tf_module, ti_module):
    with pytest.raises(ValueError):
        tf_module.apply_temporal_filter(ti_module.TemporalIntent.current, 1999)


def test_year_above_2039_raises(tf_module, ti_module):
    with pytest.raises(ValueError):
        tf_module.apply_temporal_filter(ti_module.TemporalIntent.current, 2050)


def test_filter_uses_int_year_no_string_interpolation(tf_module, ti_module):
    out = tf_module.apply_temporal_filter(ti_module.TemporalIntent.current, 2025)
    # Output must NOT contain quotes around the year (proves int interpolation).
    assert "'" not in out
    assert '"' not in out
