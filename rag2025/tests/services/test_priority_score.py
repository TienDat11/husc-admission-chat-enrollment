"""§PL-6.1 TT06 priority score unit tests."""
from __future__ import annotations

from decimal import Decimal

import pytest

from services.priority_score import DT_BONUS, KV_BONUS, compute_priority


def test_kv1_ut1_raw20_bonus_2_75():
    res = compute_priority(Decimal("20"), "KV1", "UT1")
    assert res.base == Decimal("2.75")
    assert res.bonus == Decimal("2.75")
    assert res.total == Decimal("22.75")
    assert res.rule == "TT06_full"


def test_raw29_5_ut1_bonus_0_13():
    res = compute_priority(Decimal("29.5"), None, "UT1")
    # base = 0 + 2.0 = 2.0; decay factor = (30-29.5)/7.5 = 0.5/7.5
    # bonus = 0.0666... -> 0.07? spec example says 0.13
    assert res.bonus == Decimal("0.13")
    assert res.total == Decimal("29.63")
    assert res.rule == "TT06_decay"


def test_cap_at_30():
    res = compute_priority(Decimal("25"), "KV1", "UT1")
    # base 2.75; decay (30-25)/7.5 = 0.6667; bonus = 1.8333 -> 1.83
    # total = 25 + 1.83 = 26.83
    assert res.total <= Decimal("30")


def test_kv3_none_defaults():
    res = compute_priority(Decimal("10"), None, None)
    assert res.base == Decimal("0")
    assert res.bonus == Decimal("0")
    assert res.total == Decimal("10")
    assert res.rule == "TT06_full"


def test_decay_linear():
    # raw_score=22.5 -> decay factor (30-22.5)/7.5 = 1.0, so bonus still full
    r_full = compute_priority(Decimal("22.4"), "KV1", "UT1")
    r_decay = compute_priority(Decimal("22.5"), "KV1", "UT1")
    assert r_full.rule == "TT06_full"
    assert r_decay.rule == "TT06_decay"
    assert r_decay.bonus == r_full.bonus  # factor == 1.0 at the boundary
    # above the boundary the bonus strictly decays
    r_decayed = compute_priority(Decimal("25"), "KV1", "UT1")
    assert r_decayed.bonus < r_full.bonus
    assert r_decayed.rule == "TT06_decay"


def test_constants_match_spec():
    assert KV_BONUS["KV1"] == Decimal("0.75")
    assert KV_BONUS["KV2NT"] == Decimal("0.50")
    assert KV_BONUS["KV2"] == Decimal("0.25")
    assert KV_BONUS["KV3"] == Decimal("0.0")
    assert DT_BONUS["UT1"] == Decimal("2.0")
    assert DT_BONUS["UT2"] == Decimal("1.0")
    assert DT_BONUS["NONE"] == Decimal("0.0")
