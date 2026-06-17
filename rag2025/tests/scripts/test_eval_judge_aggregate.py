"""TDD tests for Slice S5b — judge verdict aggregator with bootstrap CI.

DETERMINISTIC arithmetic tests (NO LLM). Hand-built fake verdicts prove:

  1. float-mean round-trip: aggregator AVERAGES provided floats correctly
     (does NOT re-derive faithfulness from claims). Single verdict with
     faithfulness=0.6667 → mean ≈ 0.6667.
  2. mean correctness: [0.5, 1.0, 0.0] over 3 items → mean == 0.5 (exact).
  3. bootstrap CI determinism: bootstrap_ci(values, seed=42) called twice
     returns IDENTICAL (mean, lo, hi) and lo ≤ mean ≤ hi, lo ≥ 0, hi ≤ 1.
  4. bootstrap CI known-ish: constant [0.5]*100 → mean = lo = hi = 0.5
     (zero variance under resampling).
  5. null exclusion: faithfulness values [0.8, null, 0.6] with one item
     null → n == 2, mean == 0.7 (null EXCLUDED, NOT treated as 0).
  6. 3-pass per-id mean: same id scored 0.6 / 0.8 / 0.7 across three
     passes → per-id mean 0.7 → bootstrap over [0.7] → mean == 0.7.
  7. unmappable id: verdict with id not in expected_ids → ValueError.
  8. coverage: expected_ids has 3 ids, verdicts cover 2 → coverage 2/3.

The module under test is `rag2025/scripts/eval_judge_aggregate.py` —
loaded via importlib so this test file is independent of sys.path tweaks.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
from typing import Any, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "eval_judge_aggregate.py"


# ---------- Module loader ----------

@pytest.fixture(scope="module")
def ja_module():
    """Import eval_judge_aggregate.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("eval_judge_aggregate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Hand-built verdict builders (NO LLM) ----------

def _verdict(rid: str, *, faith=None, ar=None, cp=None, cr=None) -> Dict[str, Any]:
    """Build one verdict dict in the FLAT writeback shape (top-level scores)."""
    return {
        "id": rid,
        "faithfulness": faith,
        "answer_relevancy": ar,
        "context_precision": cp,
        "context_recall": cr,
        "claims": [],
        "hallucination_types": [],
        "rationale": "fake",
    }


def _batch(judge_pass: int, verdicts: List[Dict[str, Any]], batch_id="batch_000") -> Dict[str, Any]:
    """Build one batch payload (matches the WRITEBACK CONTRACT)."""
    return {
        "batch_id": batch_id,
        "judge_pass": judge_pass,
        "judge_model": "opus-subagent",
        "verdicts": list(verdicts),
    }


# =============================================================================
# 1. float-mean round-trip: aggregator AVERAGES provided floats
# =============================================================================

def test_aggregator_averages_provided_faithfulness_float(ja_module):
    """A single verdict with faithfulness=0.6667 → aggregate mean ≈ 0.6667."""
    batches = [_batch(1, [_verdict("msg001", faith=0.6667)])]
    result = ja_module.aggregate(batches)
    f = result["metrics"]["faithfulness"]
    assert f["n"] == 1, f
    assert f["mean"] == pytest.approx(0.6667, abs=1e-9), f
    # Single-value CI: lo == hi == mean (no variance to resample).
    assert f["ci_lo"] == pytest.approx(0.6667, abs=1e-9), f
    assert f["ci_hi"] == pytest.approx(0.6667, abs=1e-9), f


# =============================================================================
# 2. mean correctness — exact arithmetic
# =============================================================================

def test_mean_correctness_three_items(ja_module):
    """faithfulness [0.5, 1.0, 0.0] across 3 items → mean == 0.5 (EXACT)."""
    batches = [
        _batch(1, [
            _verdict("msg001", faith=0.5),
            _verdict("msg002", faith=1.0),
            _verdict("msg003", faith=0.0),
        ])
    ]
    result = ja_module.aggregate(batches)
    f = result["metrics"]["faithfulness"]
    assert f["n"] == 3, f
    # Exact arithmetic — (0.5 + 1.0 + 0.0) / 3 == 0.5.
    assert f["mean"] == pytest.approx(0.5, abs=1e-12), f
    # CI must bracket the mean and stay in [0, 1].
    assert f["ci_lo"] is not None and f["ci_hi"] is not None
    assert 0.0 <= f["ci_lo"] <= f["mean"] <= f["ci_hi"] <= 1.0, f


# =============================================================================
# 3. bootstrap CI determinism
# =============================================================================

def test_bootstrap_ci_is_deterministic_with_same_seed(ja_module):
    """bootstrap_ci(values, seed=42) called twice → IDENTICAL (mean, lo, hi)."""
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    a_mean, a_lo, a_hi = ja_module.bootstrap_ci(values, seed=42)
    b_mean, b_lo, b_hi = ja_module.bootstrap_ci(values, seed=42)
    # BYTE-identical, not just close — same seed must give same draws.
    assert a_mean == b_mean, (a_mean, b_mean)
    assert a_lo == b_lo, (a_lo, b_lo)
    assert a_hi == b_hi, (a_hi, b_hi)
    # Sanity: lo ≤ mean ≤ hi, both in [0, 1] (values are in [0, 1]).
    assert a_lo is not None and a_hi is not None and a_mean is not None
    assert a_lo <= a_mean <= a_hi, (a_lo, a_mean, a_hi)
    assert 0.0 <= a_lo <= 1.0, a_lo
    assert 0.0 <= a_hi <= 1.0, a_hi
    # Raw mean must be the arithmetic mean of `values`.
    assert a_mean == pytest.approx(sum(values) / len(values), abs=1e-12)


def test_bootstrap_ci_different_seeds_can_differ(ja_module):
    """Different seeds may produce different CI bounds (sanity for seeded RNG)."""
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    _, lo1, hi1 = ja_module.bootstrap_ci(values, seed=42)
    _, lo7, hi7 = ja_module.bootstrap_ci(values, seed=7)
    # We only assert that the seeded RNG is actually being used — at least
    # one bound should differ across seeds for a non-degenerate 5-value vector.
    assert (lo1, hi1) != (lo7, hi7), (lo1, hi1, lo7, hi7)


# =============================================================================
# 4. bootstrap CI known-ish: constant vector → zero variance
# =============================================================================

def test_bootstrap_ci_constant_vector_zero_variance(ja_module):
    """[0.5]*100 → mean == 0.5, lo == 0.5, hi == 0.5 (resample mean is always 0.5)."""
    values = [0.5] * 100
    mean, lo, hi = ja_module.bootstrap_ci(values, seed=42, n_resamples=500)
    assert mean == pytest.approx(0.5, abs=1e-12), mean
    assert lo == pytest.approx(0.5, abs=1e-12), lo
    assert hi == pytest.approx(0.5, abs=1e-12), hi


def test_bootstrap_ci_empty_list_returns_none(ja_module):
    """Empty values → (None, None, None) — graceful, not a crash."""
    mean, lo, hi = ja_module.bootstrap_ci([], seed=42)
    # Either None or NaN is acceptable; lo/hi must be None.
    assert lo is None and hi is None, (lo, hi)
    assert mean is None or (isinstance(mean, float) and math.isnan(mean)), mean


# =============================================================================
# 5. null exclusion
# =============================================================================

def test_null_values_are_excluded_from_metric_mean(ja_module):
    """faithfulness [0.8, null, 0.6] → n == 2, mean == 0.7 (null DROPPED, not 0)."""
    batches = [
        _batch(1, [
            _verdict("msg001", faith=0.8),
            _verdict("msg002", faith=None),  # null → must be excluded
            _verdict("msg003", faith=0.6),
        ])
    ]
    result = ja_module.aggregate(batches)
    f = result["metrics"]["faithfulness"]
    assert f["n"] == 2, f
    assert f["mean"] == pytest.approx(0.7, abs=1e-12), f
    # The spec calls this out separately — must be surfaced.
    assert result.get("faithfulness_n") == 2, result


# =============================================================================
# 6. 3-pass per-id-mean-then-bootstrap aggregation
# =============================================================================

def test_three_pass_aggregation_uses_per_id_mean_first(ja_module):
    """Same id scored 0.6 / 0.8 / 0.7 across 3 passes → per-id mean 0.7.

    The aggregator must FIRST collapse the 3 passes to a per-id mean,
    THEN bootstrap over the list of per-id means. This is the documented
    choice (passes are NOT independent samples — they replicate the same
    judgment per item).
    """
    batches = [
        _batch(1, [_verdict("msg001", faith=0.6)]),
        _batch(2, [_verdict("msg001", faith=0.8)]),
        _batch(3, [_verdict("msg001", faith=0.7)]),
    ]
    result = ja_module.aggregate(batches)
    f = result["metrics"]["faithfulness"]
    # One unique id → n == 1.
    assert f["n"] == 1, f
    # Per-id mean (0.6 + 0.8 + 0.7) / 3 == 0.7 exactly.
    assert f["mean"] == pytest.approx(0.7, abs=1e-12), f
    # Single-value bootstrap collapses to (μ, μ, μ).
    assert f["ci_lo"] == pytest.approx(0.7, abs=1e-12), f
    assert f["ci_hi"] == pytest.approx(0.7, abs=1e-12), f


def test_three_pass_aggregation_handles_null_in_some_passes(ja_module):
    """Same id with passes [0.6, null, 0.8] → per-id mean = (0.6+0.8)/2 = 0.7."""
    batches = [
        _batch(1, [_verdict("msg001", faith=0.6)]),
        _batch(2, [_verdict("msg001", faith=None)]),  # null dropped from pass-mean
        _batch(3, [_verdict("msg001", faith=0.8)]),
    ]
    result = ja_module.aggregate(batches)
    f = result["metrics"]["faithfulness"]
    assert f["n"] == 1, f
    assert f["mean"] == pytest.approx(0.7, abs=1e-12), f


# =============================================================================
# 7. unmappable id → ValueError
# =============================================================================

def test_unmappable_id_raises_value_error(ja_module):
    """Verdict id not in expected_ids → ValueError naming the bad id."""
    batches = [_batch(1, [_verdict("msgXXX", faith=0.8)])]
    expected_ids = {"msg001"}  # msgXXX is NOT in here
    with pytest.raises(ValueError) as excinfo:
        ja_module.aggregate(batches, expected_ids=expected_ids)
    # The error message must name the unmappable id so debugging is trivial.
    assert "msgXXX" in str(excinfo.value), str(excinfo.value)


# =============================================================================
# 8. coverage report (expected 3 / observed 2)
# =============================================================================

def test_coverage_reports_expected_vs_observed(ja_module):
    """expected_ids has 3 ids, verdicts cover 2 → coverage 2/3, 1 missing."""
    batches = [
        _batch(1, [
            _verdict("msg001", faith=0.9),
            _verdict("msg002", faith=0.6),
        ])
    ]
    expected_ids = {"msg001", "msg002", "msg003"}
    result = ja_module.aggregate(batches, expected_ids=expected_ids)
    cov = result.get("coverage")
    assert cov is not None, result
    assert cov["expected"] == 3, cov
    assert cov["covered"] == 2, cov
    assert cov["missing"] == ["msg003"], cov


# =============================================================================
# Bonus invariants (cheap, catch silent regressions)
# =============================================================================

def test_aggregate_metric_helper_matches_aggregate(ja_module):
    """aggregate_metric on a flat list returns the same {n, mean, ci_lo, ci_hi}."""
    verdicts = [
        _verdict("msg001", faith=0.5),
        _verdict("msg002", faith=1.0),
        _verdict("msg003", faith=0.0),
    ]
    direct = ja_module.aggregate_metric(verdicts, "faithfulness", seed=42)
    via_full = ja_module.aggregate([_batch(1, verdicts)])["metrics"]["faithfulness"]
    assert direct == via_full, (direct, via_full)


def test_supports_nested_scores_shape_from_s5a_rubric(ja_module):
    """Also accept S5a's nested {'items': [{'scores': {...}}]} shape."""
    payload = {
        "batch_id": "batch_000",
        "run_id": "fakerun",
        "items": [
            {
                "id": "msg001",
                "scores": {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.7,
                    "context_recall": 0.6,
                },
                "claims": [],
                "rationale": "fake",
            }
        ],
    }
    result = ja_module.aggregate([payload])
    assert result["metrics"]["faithfulness"]["n"] == 1
    assert result["metrics"]["faithfulness"]["mean"] == pytest.approx(0.8, abs=1e-12)
    assert result["metrics"]["context_recall"]["mean"] == pytest.approx(0.6, abs=1e-12)


def test_all_four_metrics_independently_aggregated(ja_module):
    """Each metric is aggregated INDEPENDENTLY — distinct n when nulls differ."""
    batches = [
        _batch(1, [
            # context_recall is null on msg001 (no reference) — must not poison
            # the other three metrics.
            _verdict("msg001", faith=0.8, ar=0.9, cp=0.7, cr=None),
            _verdict("msg002", faith=0.6, ar=0.7, cp=0.5, cr=0.4),
        ])
    ]
    result = ja_module.aggregate(batches)
    m = result["metrics"]
    assert m["faithfulness"]["n"] == 2
    assert m["answer_relevancy"]["n"] == 2
    assert m["context_precision"]["n"] == 2
    # context_recall has only one non-null → n == 1.
    assert m["context_recall"]["n"] == 1, m["context_recall"]
    assert m["context_recall"]["mean"] == pytest.approx(0.4, abs=1e-12)
