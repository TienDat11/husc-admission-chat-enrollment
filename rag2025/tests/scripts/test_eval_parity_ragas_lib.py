"""TDD tests for SLICE S-Parity — RAGAS-lib as ARBITER cross-check.

Two test tiers:

  TIER 1 — pure arithmetic (always runs, no ragas needed):
    1. compute_delta exact mean |Δ| arithmetic
    2. arbiter_verdict faithfulness tight (0.05)
    3. arbiter_verdict other-metric looser (0.10)
    4. run_parity insufficient_sample warning (n < 30) — uses monkeypatched
       `_ragas_score` seam so no real ragas is needed
    5. run_parity graceful skip when ragas import fails (monkeypatched)

  TIER 2 — requires ragas (skips if not installed):
    6. ragas library smoke: 1-item dataset produces faithfulness score

The script under test is `rag2025/scripts/eval_parity_ragas_lib.py` —
loaded via importlib to keep the test independent of sys.path tweaks.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "eval_parity_ragas_lib.py"


# ---------- Module loader ----------

@pytest.fixture(scope="module")
def parity_module():
    """Import eval_parity_ragas_lib.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location(
        "eval_parity_ragas_lib", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Helpers ----------

def _record(rid: str, *, q: str = "Q?", ans: str = "A.", ctx: str = "C.") -> Dict[str, Any]:
    """Build one minimal EvalRecord-style dict for parity sample."""
    return {"id": rid, "question": q, "answer": ans, "contexts": [ctx]}


# =============================================================================
# 1. compute_delta — pure arithmetic, mean |Δ| per metric on matched ids
# =============================================================================

def test_compute_delta_mean_abs_diff_faithfulness(parity_module):
    """opus {msg1:0.8, msg2:0.6} vs lib {msg1:0.82, msg2:0.55} → |Δ| = 0.035."""
    opus = {"msg1": {"faithfulness": 0.8}, "msg2": {"faithfulness": 0.6}}
    lib = {"msg1": {"faithfulness": 0.82}, "msg2": {"faithfulness": 0.55}}
    delta = parity_module.compute_delta(opus, lib)
    # (|0.02| + |0.05|) / 2 = 0.035
    assert "faithfulness" in delta, delta
    assert delta["faithfulness"] == pytest.approx(0.035, abs=1e-12), delta


def test_compute_delta_handles_missing_metric_in_one_side(parity_module):
    """If one side lacks the metric, exclude that id for that metric."""
    opus = {
        "a": {"faithfulness": 0.9, "answer_relevancy": 0.8},
        "b": {"faithfulness": 0.5},  # no answer_relevancy on opus side
    }
    lib = {
        "a": {"faithfulness": 0.85, "answer_relevancy": 0.7},
        "b": {"faithfulness": 0.6, "answer_relevancy": 0.5},
    }
    delta = parity_module.compute_delta(opus, lib)
    # faithfulness: (|0.05| + |0.1|) / 2 == 0.075
    assert delta["faithfulness"] == pytest.approx(0.075, abs=1e-12)
    # answer_relevancy: only id "a" is in both → 0.1
    assert delta["answer_relevancy"] == pytest.approx(0.1, abs=1e-12)


def test_compute_delta_ignores_non_overlapping_ids(parity_module):
    """An id present only on one side is not counted in any mean."""
    opus = {
        "shared": {"faithfulness": 0.8},
        "opus_only": {"faithfulness": 0.99},
    }
    lib = {
        "shared": {"faithfulness": 0.7},
        "lib_only": {"faithfulness": 0.0},
    }
    delta = parity_module.compute_delta(opus, lib)
    # Only "shared" → Δ == 0.1
    assert delta["faithfulness"] == pytest.approx(0.1, abs=1e-12)


def test_compute_delta_empty_input_returns_empty(parity_module):
    """No ids on either side → empty dict (no crash, no spurious zeros)."""
    assert parity_module.compute_delta({}, {}) == {}


# =============================================================================
# 2. arbiter_verdict — faithfulness tight (0.05)
# =============================================================================

def test_arbiter_agree_faithfulness_within_tight_tol(parity_module):
    """Δ faithfulness 0.04 (≤ 0.05) → agree True (no authoritative)."""
    verdict = parity_module.arbiter_verdict({"faithfulness": 0.04})
    f = verdict["per_metric"]["faithfulness"]
    assert f["agree"] is True, f
    assert f["delta"] == pytest.approx(0.04, abs=1e-12)
    assert f["tolerance"] == 0.05
    # No authoritative when agree.
    assert f.get("authoritative") in (None, "")
    # Overall agree: True if every metric agrees.
    assert verdict["overall_agree"] is True


def test_arbiter_disagree_faithfulness_beyond_tight_tol(parity_module):
    """Δ faithfulness 0.06 (> 0.05) → agree False, authoritative=='ragas_lib'."""
    verdict = parity_module.arbiter_verdict({"faithfulness": 0.06})
    f = verdict["per_metric"]["faithfulness"]
    assert f["agree"] is False, f
    assert f["authoritative"] == "ragas_lib"
    assert f["flagged_metric"] == "faithfulness"
    assert verdict["overall_agree"] is False


# =============================================================================
# 3. arbiter_verdict — other metrics use looser 0.10
# =============================================================================

def test_arbiter_agree_answer_relevancy_within_loose_tol(parity_module):
    """Δ answer_relevancy 0.08 (≤ 0.10) → agree True (looser tol)."""
    verdict = parity_module.arbiter_verdict({"answer_relevancy": 0.08})
    ar = verdict["per_metric"]["answer_relevancy"]
    assert ar["agree"] is True, ar
    assert ar["tolerance"] == 0.10
    assert verdict["overall_agree"] is True


def test_arbiter_disagree_answer_relevancy_beyond_loose_tol(parity_module):
    """Δ answer_relevancy 0.12 (> 0.10) → agree False, authoritative='ragas_lib'."""
    verdict = parity_module.arbiter_verdict({"answer_relevancy": 0.12})
    ar = verdict["per_metric"]["answer_relevancy"]
    assert ar["agree"] is False, ar
    assert ar["authoritative"] == "ragas_lib"
    assert ar["flagged_metric"] == "answer_relevancy"
    assert verdict["overall_agree"] is False


def test_arbiter_custom_tolerances(parity_module):
    """Passing custom tolerances should override the defaults."""
    verdict = parity_module.arbiter_verdict(
        {"faithfulness": 0.04, "answer_relevancy": 0.12},
        faithfulness_tol=0.02,  # tighter than default
        other_tol=0.20,         # looser than default
    )
    f = verdict["per_metric"]["faithfulness"]
    ar = verdict["per_metric"]["answer_relevancy"]
    # 0.04 > 0.02 → disagree on faithfulness
    assert f["agree"] is False and f["tolerance"] == 0.02
    # 0.12 ≤ 0.20 → agree on answer_relevancy
    assert ar["agree"] is True and ar["tolerance"] == 0.20
    assert verdict["overall_agree"] is False  # any disagree → False


def test_arbiter_multi_metric_overall_disagree_if_any_disagree(parity_module):
    """Two metrics, one agree + one disagree → overall disagree."""
    verdict = parity_module.arbiter_verdict(
        {"faithfulness": 0.01, "answer_relevancy": 0.99}
    )
    assert verdict["per_metric"]["faithfulness"]["agree"] is True
    assert verdict["per_metric"]["answer_relevancy"]["agree"] is False
    assert verdict["overall_agree"] is False


# =============================================================================
# 4. insufficient_sample — run_parity with n < 30 + monkeypatched _ragas_score
# =============================================================================

def _five_sample_records() -> List[Dict[str, Any]]:
    """Five hand-built records, well below the n_min=30 threshold."""
    return [
        _record(f"msg{i:03d}", q=f"Q{i}?", ans=f"A{i}.", ctx=f"C{i}.")
        for i in range(1, 6)
    ]


def test_run_parity_insufficient_sample_warning(parity_module, monkeypatch):
    """5-item sample (n < 30) → warning present, no crash even without ragas."""
    sample = _five_sample_records()
    # If we even REACH the lazy import, it should fail-soft via the seam.
    def _fake_ragas_score(records):
        # Canned lib scores so compute_delta has something to chew on.
        return {r["id"]: {"faithfulness": 0.8, "answer_relevancy": 0.7}
                for r in records}
    monkeypatch.setattr(parity_module, "_ragas_score", _fake_ragas_score)

    opus_verdicts = {
        r["id"]: {"faithfulness": 0.8, "answer_relevancy": 0.7}
        for r in sample
    }
    result = parity_module.run_parity(sample, opus_verdicts, n_min=30)
    # Insufficient sample warning must be set.
    assert result.get("insufficient_sample") is True, result
    assert result.get("n_sample") == 5
    assert result.get("n_min") == 30
    # We still get delta + verdict because the seam was monkeypatched.
    assert "delta" in result
    assert "verdict" in result


# =============================================================================
# 5. graceful skip when ragas is not installed
# =============================================================================

def test_run_parity_skips_gracefully_when_ragas_missing(parity_module, monkeypatch):
    """If `_ragas_score` raises ImportError → run_parity returns skipped dict."""
    def _fake_ragas_score(records):
        raise ImportError("ragas not installed (simulated)")

    monkeypatch.setattr(parity_module, "_ragas_score", _fake_ragas_score)

    # Build a 35-item sample (above the n_min) so we know skip is from
    # ImportError, NOT from insufficient_sample.
    sample = [_record(f"msg{i:03d}") for i in range(35)]
    opus_verdicts = {r["id"]: {"faithfulness": 0.7, "answer_relevancy": 0.6}
                     for r in sample}

    result = parity_module.run_parity(sample, opus_verdicts, n_min=30)
    assert result.get("skipped") is True, result
    assert "not installed" in result.get("reason", "").lower()
    # Sanity: no insufficient_sample flag (we hit the import error first).
    assert "insufficient_sample" not in result


# =============================================================================
# 6. real ragas smoke test — only runs when ragas is installed
# =============================================================================

def test_real_ragas_smoke_one_item(parity_module):
    """If ragas is installed: build 1-item dataset, expect lib score present.

    Skipped when ragas is unavailable (this is the TIER 2 test).
    """
    pytest.importorskip("ragas")
    # Monkeypatch to call the real `_ragas_score` (the default).
    sample = [_record("msg001", q="What is 1+1?", ans="2.", ctx="1+1=2.")]
    opus_verdicts = {"msg001": {"faithfulness": 0.9, "answer_relevancy": 0.9}}
    result = parity_module.run_parity(sample, opus_verdicts, n_min=1)
    # Even with n=1 (custom n_min=1 for the smoke), the delta block must
    # carry at least one of the two metrics present.
    if result.get("skipped"):
        pytest.skip(f"ragas skipped: {result.get('reason')}")
    assert "delta" in result
    # Lib may legitimately fail on tiny inputs; at minimum, the keys
    # the arbiter checks should be reachable.
    assert "verdict" in result
    assert "per_metric" in result["verdict"]


# =============================================================================
# Sanity: monkeypatchable seam exists
# =============================================================================

def test_ragas_score_seam_is_callable(parity_module):
    """The _ragas_score helper must exist on the module and be callable.

    Tests rely on monkeypatching it; if it's missing or not callable, the
    suite silently regresses into "can't run offline anymore".
    """
    assert hasattr(parity_module, "_ragas_score"), (
        "missing _ragas_score seam — tests cannot monkeypatch it"
    )
    assert callable(parity_module._ragas_score), (
        "_ragas_score must be callable"
    )
