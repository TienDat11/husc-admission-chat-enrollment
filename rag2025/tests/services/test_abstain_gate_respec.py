"""AB1 — TDD test for the abstain-gate re-spec.

Two pure functions live in ``results/ultraqa_metrics/compute_offline_metrics.py``:

* ``wilson_lower_bound(successes, n, z=1.96)`` — closed-form Wilson score
  lower bound at confidence ``z``. The standard 95% two-sided Z=1.96
  corresponds to a 95% lower bound (one-sided), which is what we want
  for a "do not falsely pass" gate.

* ``abstain_gate_verdict(successes, n, min_n=30, floor=0.85)`` — three-valued
  verdict: ``"informational"`` while ``n < min_n`` (under-powered, gate does
  not apply), then ``"pass"`` if ``wilson_lower_bound(s, n) >= floor`` else
  ``"fail"``.

Statistical constants (z=1.96, min_n=30, floor=0.85) are STATED, not tuned
to the 86Q set — they are textbook standard values (95% one-sided, normal-
approximation floor for central-limit theorem to apply, 85% as a quality
floor that is reachable but not trivially so).

Hand-computed Wilson values (Python 3, math.sqrt, computed 2026-06-08):
  wilson_lower_bound(7, 7)   = 0.645661  (±0.01)
  wilson_lower_bound(30, 30) = 0.886483  (±0.01)
  wilson_lower_bound(15, 20) = 0.531295  (±0.02)
  wilson_lower_bound(0, 0)   = 0.0        (explicit guard, no div-by-zero)

These are used for the floor=0.85 sanity check:
  7/7   → 0.6457 < 0.85, n=7<30 → "informational" (overrides the fail)
  30/30 → 0.8865 ≥ 0.85, n=30≥30 → "pass"
  0/30  → 0.0000 < 0.85, n=30≥30 → "fail" (clear under-performance)
  3/5   → ~0.36, n=5<30          → "informational"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO / "results" / "ultraqa_metrics"
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))

from compute_offline_metrics import (  # noqa: E402
    abst_corrected_from_triage,
    abstain_gate_verdict,
    wilson_lower_bound,
)


TRIAGE_PATH = REPO / "results" / "ultraqa_metrics" / "abstain_oos_triage.json"


# ---------------------------------------------------------------------------
# 0) Hand-computed Wilson lower-bound sanity (hand calc in module docstring).
# ---------------------------------------------------------------------------
def test_wilson_lower_bound_seven_of_seven():
    # p = 1.0, n = 7, z = 1.96 → 0.645661
    assert wilson_lower_bound(7, 7) == pytest.approx(0.646, abs=0.01)


def test_wilson_lower_bound_thirty_of_thirty():
    # p = 1.0, n = 30, z = 1.96 → 0.886483
    assert wilson_lower_bound(30, 30) == pytest.approx(0.886, abs=0.01)


def test_wilson_lower_bound_fifteen_of_twenty():
    # p = 0.75, n = 20, z = 1.96 → 0.531295
    assert wilson_lower_bound(15, 20) == pytest.approx(0.531, abs=0.02)


def test_wilson_lower_bound_zero_of_zero_guard():
    # No sample → no CI, return 0.0 (guard against div-by-zero).
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_clamps_negative_n_args():
    # Defensive: negative successes or n must not crash; treat as 0.0.
    assert wilson_lower_bound(-1, 5) == 0.0
    assert wilson_lower_bound(2, -1) == 0.0


# ---------------------------------------------------------------------------
# 1) abstain_gate_verdict — three-valued verdict logic.
# ---------------------------------------------------------------------------
def test_gate_verdict_informational_under_min_n_even_when_under_floor():
    # 7/7 → wilson_lo ≈ 0.646 < 0.85 (would be FAIL if n were ≥30)
    # BUT n=7 < 30 → gate is "informational" (under-powered), not "fail".
    assert abstain_gate_verdict(7, 7) == "informational"


def test_gate_verdict_informational_under_min_n_even_when_over_floor():
    # 5/5 → wilson_lo ≈ 0.566, n=5 < 30 → still informational.
    assert abstain_gate_verdict(5, 5) == "informational"


def test_gate_verdict_pass_at_min_n_boundary_when_floor_reached():
    # 30/30 → wilson_lo ≈ 0.886 ≥ 0.85, n=30 ≥ 30 → "pass".
    assert abstain_gate_verdict(30, 30) == "pass"


def test_gate_verdict_fail_at_min_n_when_floor_missed():
    # 0/30 → wilson_lo = 0.0 < 0.85, n=30 ≥ 30 → "fail".
    assert abstain_gate_verdict(0, 30) == "fail"


def test_gate_verdict_pass_with_high_correctness_over_min_n():
    # Hand-computed (z=1.96, n=30, default floor=0.85):
    #   27/30 → wilson_lo ≈ 0.744 → "fail"
    #   28/30 → wilson_lo ≈ 0.787 → "fail"
    #   29/30 → wilson_lo ≈ 0.833 → "fail"
    #   30/30 → wilson_lo ≈ 0.886 → "pass"
    # The gate is intentionally strict at small n — it must auto-tighten
    # as n grows, not just track the point estimate.
    assert abstain_gate_verdict(27, 30) == "fail"
    assert abstain_gate_verdict(28, 30) == "fail"
    assert abstain_gate_verdict(29, 30) == "fail"
    assert abstain_gate_verdict(30, 30) == "pass"


def test_gate_verdict_floor_is_configurable():
    # Same 7/7 sample, but if user sets floor=0.5 + min_n=1, the gate would
    # pass (7/7 → wilson_lo ≈ 0.646 ≥ 0.5, n=7 ≥ 1). The function must
    # respect the floor + min_n arguments.
    assert abstain_gate_verdict(7, 7, min_n=1, floor=0.5) == "pass"
    assert abstain_gate_verdict(7, 7, min_n=1, floor=0.9) == "fail"


def test_gate_verdict_zero_of_zero_is_informational_not_fail():
    # n=0 is not "fail" — there is simply no signal. Treat as informational.
    assert abstain_gate_verdict(0, 0) == "informational"


# ---------------------------------------------------------------------------
# 2) abst_corrected_from_triage — applies ONLY the gt_convention_FP reclass.
# ---------------------------------------------------------------------------
def test_corrected_abstain_counts_only_fp_reclassified():
    # Triage file: 7 OOS cases, 5 gt_convention_FP, 2 gt_correct.
    # The "correct" abstain accuracy counts the cases where pipeline behaviour
    # matches the (corrected) gold.  For these 7:
    #   * 5 FPs flipped from 'expected=abstain' to 'expected=answer' — they
    #     become CORRECT (pipeline did answer).
    #   * 2 gt_correct cases — gold stays 'abstain', pipeline abstained.
    # → 7/7 correct, n=7.
    out = abst_corrected_from_triage(TRIAGE_PATH)
    assert out["n_total"] == 7
    assert out["n_correct"] == 7
    assert out["n_genuine_miss"] == 0
    assert out["n_gt_convention_FP"] == 5
    assert out["n_gt_correct"] == 2
    assert out["point_estimate"] == pytest.approx(1.0)
    assert out["wilson_lower_bound"] == pytest.approx(wilson_lower_bound(7, 7), abs=1e-6)
    assert out["verdict"] == "informational"  # n=7 < min_n=30
    # Only the 5 FPs reclassified — gt_correct entries are NOT touched.
    by_id = {e["id"]: e for e in out["entries"]}
    assert by_id["msg026"]["reclassified"] is True
    assert by_id["msg028"]["reclassified"] is True
    assert by_id["msg034"]["reclassified"] is True
    assert by_id["msg044"]["reclassified"] is True
    assert by_id["msg059"]["reclassified"] is True
    assert by_id["msg027"]["reclassified"] is False
    assert by_id["msg055"]["reclassified"] is False
