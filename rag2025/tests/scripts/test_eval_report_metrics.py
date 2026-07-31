"""TDD tests for Slice S7 — final metrics report generator.

DETERMINISTIC tests (NO LLM, NO network). Hand-built inputs prove:

  1. percentile: 50th of [10,20,30,40,50] == 30 (mid); 50th of [10,20,30,40]
     == 25 (interpolated); p95 of a known vector matches a hand calculation.
  2. latency p50/p95: route_ms = [10,20,30,40,50] → p50 == 30, p95 ≈ 48.
  3. attribution caveat: when ALL records have null latency_attribution,
     attribution_available is False AND a non-empty attribution_caveat is
     present. When at least one record has attribution, no forced caveat.
  4. gate FAIL: faithfulness 0.62 (<0.90) → verdict FAIL, actual 0.62.
     context_recall 0.70 (<0.85) → FAIL.
  5. gate PASS: faithfulness 0.95 (≥0.90) → PASS.
  6. abstain accuracy: 4 refuse-expected items, 3 abstained → 0.75;
     gate (≥0.95) → FAIL.
  7. build_report shape: top-level keys for metrics, confusion_matrix,
     latency, gates, kappa, context_recall_caveat; the circularity caveat
     is a non-empty string.
  8. render_markdown: string contains "PASS"/"FAIL" tokens, metric names,
     "Cohen" (κ section), and the caveat text.

The module under test is `rag2025/scripts/eval_report_metrics.py` —
loaded via importlib so this test file is independent of sys.path tweaks.
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
SCRIPT_PATH = SCRIPTS_DIR / "eval_report_metrics.py"


# ---------- Module loader ----------

@pytest.fixture(scope="module")
def rep_module():
    """Import eval_report_metrics.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location(
        "eval_report_metrics", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Helpers ----------

def _agg(mean_faith: float, mean_cr: float) -> Dict[str, Any]:
    """Build a tiny aggregator-shaped dict for gate tests."""
    return {
        "metrics": {
            "faithfulness": {
                "n": 10, "mean": mean_faith, "ci_lo": mean_faith - 0.05,
                "ci_hi": mean_faith + 0.05,
            },
            "answer_relevancy": {
                "n": 10, "mean": 0.8, "ci_lo": 0.75, "ci_hi": 0.85,
            },
            "context_precision": {
                "n": 10, "mean": 0.8, "ci_lo": 0.75, "ci_hi": 0.85,
            },
            "context_recall": {
                "n": 10, "mean": mean_cr, "ci_lo": mean_cr - 0.05,
                "ci_hi": mean_cr + 0.05,
            },
        },
        "faithfulness_n": 10,
    }


def _halltype_pass() -> Dict[str, Any]:
    """A passing halltype dict (all type counts at or under gate)."""
    return {
        "per_type_counts": {
            "type1": 0, "type4": 0, "type5": 0, "type6": 0,
        },
        "gate": {
            k: {"actual": 0, "max": 3 if "1" in k or "5" in k else 0, "pass": True}
            for k in ("type1_max", "type4_max", "type5_max", "type6_max")
        },
    }


# =============================================================================
# 1. percentile
# =============================================================================

class TestPercentile:
    def test_p50_odd_count_midpoint(self, rep_module):
        # [10,20,30,40,50], p=0.5 → rank = 0.5*4 = 2.0 → sorted_values[2] = 30
        assert rep_module.percentile([10, 20, 30, 40, 50], 50) == 30

    def test_p50_even_count_interpolated(self, rep_module):
        # [10,20,30,40], p=0.5 → rank = 0.5*3 = 1.5
        # lo_idx=1, hi_idx=2, frac=0.5
        # 20 + 0.5*(30-20) = 25
        assert rep_module.percentile([10, 20, 30, 40], 50) == 25

    def test_p95_hand_calculated(self, rep_module):
        # [10,20,30,40,50,60,70,80,90,100], p=0.95
        # rank = 0.95 * 9 = 8.55
        # lo_idx=8, hi_idx=9, frac=0.55
        # 90 + 0.55 * (100-90) = 95.5
        # (use approx — linear interpolation carries IEEE-754 round-off)
        vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert rep_module.percentile(vals, 95) == pytest.approx(95.5, abs=1e-9)

    def test_p100_returns_max(self, rep_module):
        assert rep_module.percentile([1, 2, 3, 4, 5], 100) == 5

    def test_p0_returns_min(self, rep_module):
        assert rep_module.percentile([1, 2, 3, 4, 5], 0) == 1

    def test_empty_list_returns_nan(self, rep_module):
        import math
        assert math.isnan(rep_module.percentile([], 50))


# =============================================================================
# 2. latency p50/p95
# =============================================================================

class TestLatencyStats:
    def test_route_seam_p50_p95(self, rep_module):
        # 5 records with route_ms = [10,20,30,40,50]
        # p50 rank=0.5*4=2.0 → 30
        # p95 rank=0.95*4=3.8 → 40 + 0.8*(50-40) = 48.0
        records = []
        for v in [10, 20, 30, 40, 50]:
            records.append({
                "id": f"r{v}",
                "latency_ms": {
                    "route_ms": float(v),
                    "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0,
                    "total_ms": float(v) + 13.0,
                },
                "latency_attribution": None,
            })
        out = rep_module.latency_stats(records)
        assert out["route_ms"]["p50"] == 30
        assert out["route_ms"]["p95"] == 48.0
        assert out["route_ms"]["mean"] == 30.0
        # sanity on the other seams
        assert out["retrieval_loop_ms"]["mean"] == 5.0
        assert out["query_ms"]["mean"] == 8.0
        assert out["total_ms"]["mean"] == 43.0  # 30 + 5 + 8


# =============================================================================
# 3. attribution caveat
# =============================================================================

class TestAttributionCaveat:
    def test_all_none_triggers_caveat(self, rep_module):
        records = [
            {
                "id": "a",
                "latency_ms": {
                    "route_ms": 10.0, "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0, "total_ms": 23.0,
                },
                "latency_attribution": None,
            },
            {
                "id": "b",
                "latency_ms": {
                    "route_ms": 12.0, "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0, "total_ms": 25.0,
                },
                "latency_attribution": None,
            },
        ]
        out = rep_module.latency_stats(records)
        assert out["attribution_available"] is False
        assert "attribution_caveat" in out
        assert isinstance(out["attribution_caveat"], str)
        assert len(out["attribution_caveat"]) > 0

    def test_at_least_one_present_no_caveat(self, rep_module):
        records = [
            {
                "id": "a",
                "latency_ms": {
                    "route_ms": 10.0, "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0, "total_ms": 23.0,
                },
                "latency_attribution": None,
            },
            {
                "id": "b",
                "latency_ms": {
                    "route_ms": 12.0, "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0, "total_ms": 25.0,
                },
                "latency_attribution": {
                    "hyde_ms": 1.0, "router_inner_ms": 0.5,
                    "gen_ms": 3.0, "rerank_ms": 1.5,
                },
            },
        ]
        out = rep_module.latency_stats(records)
        assert out["attribution_available"] is True
        # No forced caveat when attribution is available.
        assert out.get("attribution_caveat") in (None, "")


# =============================================================================
# 4. gate FAIL
# =============================================================================

class TestGateFail:
    def test_faithfulness_below_threshold_fails(self, rep_module):
        agg = _agg(mean_faith=0.62, mean_cr=0.90)
        hall = _halltype_pass()
        abst = 1.0  # abstain gate is fine
        out = rep_module.eval_gates(agg, hall, abst)
        assert out["faithfulness"]["verdict"] == "FAIL"
        assert out["faithfulness"]["actual"] == 0.62
        assert out["faithfulness"]["threshold"] == 0.90

    def test_context_recall_below_threshold_fails(self, rep_module):
        agg = _agg(mean_faith=0.95, mean_cr=0.70)
        hall = _halltype_pass()
        out = rep_module.eval_gates(agg, hall, 1.0)
        assert out["context_recall"]["verdict"] == "FAIL"
        assert out["context_recall"]["actual"] == 0.70
        assert out["context_recall"]["threshold"] == 0.85


# =============================================================================
# 5. gate PASS
# =============================================================================

class TestGatePass:
    def test_faithfulness_at_threshold_passes(self, rep_module):
        agg = _agg(mean_faith=0.95, mean_cr=0.90)
        hall = _halltype_pass()
        out = rep_module.eval_gates(agg, hall, 1.0)
        assert out["faithfulness"]["verdict"] == "PASS"
        assert out["faithfulness"]["actual"] == 0.95

    def test_all_three_pass(self, rep_module):
        agg = _agg(mean_faith=0.95, mean_cr=0.90)
        hall = _halltype_pass()
        out = rep_module.eval_gates(agg, hall, 0.99)
        for k in ("faithfulness", "context_recall", "abstain_accuracy"):
            assert out[k]["verdict"] == "PASS", f"{k} should pass"


class TestEnforceGates:
    def test_failed_gate_is_detected(self, rep_module):
        assert rep_module.has_failed_gates({"faithfulness": {"verdict": "FAIL"}}) is True

    def test_all_pass_gates_are_not_detected(self, rep_module):
        assert rep_module.has_failed_gates({"faithfulness": {"verdict": "PASS"}}) is False


# =============================================================================
# 6. abstain accuracy
# =============================================================================

class TestAbstainAccuracy:
    def test_three_of_four_abstained(self, rep_module):
        # 4 refuse-expected items, 3 abstained (short-fallback), 1 hallucinated
        records = [
            {"id": "q1", "answer": "Tôi không thể tư vấn về ngành này.", "expected_behavior": "refuse"},
            {"id": "q2", "answer": "Chưa có thông tin chính thức về điểm chuẩn.", "expected_behavior": "refuse"},
            {"id": "q3", "answer": "Ngoài phạm vi tư vấn.", "expected_behavior": "refuse"},
            {"id": "q4", "answer": "Điểm chuẩn 2026 là 28.5.", "expected_behavior": "refuse"},
        ]
        gt = {r["id"]: {"id": r["id"], "expected_behavior": "refuse"} for r in records}
        halltype_classifications = [
            {"id": r["id"], "answer_text": r["answer"]} for r in records
        ]
        acc = rep_module.abstain_accuracy(records, gt, halltype_classifications)
        assert acc == 0.75

    def test_abstain_gate_fails_at_75(self, rep_module):
        agg = _agg(mean_faith=0.95, mean_cr=0.90)
        hall = _halltype_pass()
        out = rep_module.eval_gates(agg, hall, 0.75)
        assert out["abstain_accuracy"]["verdict"] == "FAIL"
        assert out["abstain_accuracy"]["actual"] == 0.75
        assert out["abstain_accuracy"]["threshold"] == 0.95


# =============================================================================
# 7. build_report shape
# =============================================================================

class TestBuildReport:
    def _stub_inputs(self) -> Dict[str, Any]:
        records = [
            {
                "id": "q1",
                "answer": "Điểm chuẩn ngành X năm 2026 là 24.0.",
                "expected_behavior": "answer",
                "latency_ms": {
                    "route_ms": 10.0, "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0, "total_ms": 23.0,
                },
                "latency_attribution": None,
            },
            {
                "id": "q2",
                "answer": "Chưa có thông tin chính thức.",
                "expected_behavior": "refuse",
                "latency_ms": {
                    "route_ms": 12.0, "retrieval_loop_ms": 5.0,
                    "query_ms": 8.0, "total_ms": 25.0,
                },
                "latency_attribution": None,
            },
        ]
        gt = {
            "q1": {"id": "q1", "expected_behavior": "answer", "critical_facts": []},
            "q2": {"id": "q2", "expected_behavior": "refuse", "critical_facts": []},
        }
        return {
            "agg": _agg(mean_faith=0.62, mean_cr=0.70),
            "kappa": {
                "per_metric_kappa": {
                    "faithfulness": 0.7, "answer_relevancy": 0.6,
                    "context_precision": 0.5, "context_recall": 0.65,
                },
                "n_compared": {"faithfulness": 50, "answer_relevancy": 50,
                               "context_precision": 50, "context_recall": 50},
                "mean_abs_diff": {"faithfulness": 0.05, "answer_relevancy": 0.06,
                                  "context_precision": 0.04, "context_recall": 0.07},
                "n_matched_ids": 50,
            },
            "halltype": {
                "per_type_counts": {
                    "type1": 2, "type4": 0, "type5": 4, "type6": 0,
                },
                "gate": {
                    "type1_max": {"actual": 2, "max": 3, "pass": True},
                    "type4_max": {"actual": 0, "max": 0, "pass": True},
                    "type5_max": {"actual": 4, "max": 3, "pass": False},
                    "type6_max": {"actual": 0, "max": 0, "pass": True},
                },
            },
            "records": records,
            "gt_by_id": gt,
        }

    def test_keys_present(self, rep_module):
        s = self._stub_inputs()
        report = rep_module.build_report(
            s["agg"], s["kappa"], s["halltype"], s["records"], s["gt_by_id"]
        )
        for k in (
            "metrics", "confusion_matrix", "latency", "gates",
            "kappa", "context_recall_caveat",
        ):
            assert k in report, f"missing key: {k}"

    def test_circularity_caveat_non_empty(self, rep_module):
        s = self._stub_inputs()
        report = rep_module.build_report(
            s["agg"], s["kappa"], s["halltype"], s["records"], s["gt_by_id"]
        )
        assert isinstance(report["context_recall_caveat"], str)
        assert len(report["context_recall_caveat"]) > 0


# =============================================================================
# 8. render_markdown
# =============================================================================

class TestRenderMarkdown:
    def test_contains_pass_fail_and_section_markers(self, rep_module):
        report = {
            "metrics": {
                "faithfulness": {
                    "n": 50, "mean": 0.62, "ci_lo": 0.58, "ci_hi": 0.66,
                },
                "context_recall": {
                    "n": 50, "mean": 0.70, "ci_lo": 0.66, "ci_hi": 0.74,
                },
            },
            "confusion_matrix": {
                "per_type_counts": {
                    "type1": 2, "type4": 0, "type5": 4, "type6": 0,
                },
                "gate": {
                    "type1_max": {"actual": 2, "max": 3, "pass": True},
                    "type4_max": {"actual": 0, "max": 0, "pass": True},
                    "type5_max": {"actual": 4, "max": 3, "pass": False},
                    "type6_max": {"actual": 0, "max": 0, "pass": True},
                },
            },
            "latency": {
                "route_ms": {"p50": 30.0, "p95": 48.0, "mean": 30.0},
                "retrieval_loop_ms": {"p50": 5.0, "p95": 8.0, "mean": 5.0},
                "query_ms": {"p50": 8.0, "p95": 12.0, "mean": 8.0},
                "total_ms": {"p50": 43.0, "p95": 68.0, "mean": 43.0},
                "attribution_available": False,
                "attribution_caveat": "HyDE vs generation latency cannot be isolated this round.",
            },
            "gates": {
                "faithfulness": {
                    "gate": "faithfulness_mean", "threshold": 0.90,
                    "actual": 0.62, "verdict": "FAIL",
                },
                "context_recall": {
                    "gate": "context_recall_mean", "threshold": 0.85,
                    "actual": 0.70, "verdict": "FAIL",
                },
                "abstain_accuracy": {
                    "gate": "out_of_scope_abstain_accuracy", "threshold": 0.95,
                    "actual": 0.75, "verdict": "FAIL",
                },
            },
            "kappa": {
                "per_metric_kappa": {
                    "faithfulness": 0.7, "answer_relevancy": 0.6,
                    "context_precision": 0.5, "context_recall": 0.65,
                },
            },
            "context_recall_caveat": (
                "context_recall measured fact-level; GT auto-generated from "
                "canonical sources — recall may be optimistic; see human spot-check."
            ),
        }
        md = rep_module.render_markdown(report)
        assert isinstance(md, str)
        assert "PASS" in md
        assert "FAIL" in md
        assert "faithfulness" in md
        assert "context_recall" in md
        assert "Cohen" in md  # κ section
        assert "HyDE vs generation" in md  # latency caveat
        assert "context_recall measured fact-level" in md  # circularity caveat
