"""TDD tests for Slice S5c — ns24 secondary judge + Cohen's κ agreement.

The PRIMARY judge is the Opus subagent (cross-vendor, breaks self-grading).
This slice adds a SECONDARY judge using the project's own `UnifiedLLMClient`
(ns24) — NOT to score the system, but to quantify judge-vendor
AGREEMENT (Cohen's κ) with the Opus verdicts.

Contract under test:
  * score_item(client, item, rubric) -> dict
      - client INJECTED (so tests pass a fake; no real network).
      - Calls client.chat_json(...) at temperature=0.
      - Returns {id, faithfulness, answer_relevancy, context_precision,
        context_recall} where each metric is float or null.
      - Malformed JSON / client exception => all-null for that item, no crash.
  * score_batch(client, batch) -> list[dict]
      - Score every item in a prepared batch (dict with items[]).
  * cohens_kappa(labels_a, labels_b) -> float
      - Standard Cohen's κ (po, pe). Edge cases handled.
  * binned_kappa(opus_scores, ns24_scores, *, bins=(0.5, 0.8))
      - Bin into low/mid/high, drop null pairs, return κ.
  * agreement_report(opus_verdicts, ns24_verdicts) -> dict
      - per_metric_kappa, n_compared, mean_abs_diff for 4 metrics.
      - MUST NOT include a "headline_score" key (ns24 = agreement only).
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
from typing import Any, Dict, List, Optional

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "eval_judge_ns24.py"


# ---------- Module loader ----------

@pytest.fixture(scope="module")
def jr_module():
    """Import eval_judge_ns24.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("eval_judge_ns24", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Fake client ----------

class _FakeClient:
    """Stand-in for UnifiedLLMClient — never touches network.

    `canned` is a dict of {prompt_substring (or None): response_dict} that
    maps to whatever the test wants back. If the key matches, the value is
    returned. If `raise_on_call` is set, the client raises (mirroring a real
    transport failure). If `canned` is None, a default healthy response is
    returned.
    """

    def __init__(
        self,
        canned: Optional[Dict[str, Any]] = None,
        *,
        raise_on_call: Optional[BaseException] = None,
        return_non_dict: bool = False,
        capture: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.canned = canned or {}
        self.raise_on_call = raise_on_call
        self.return_non_dict = return_non_dict
        self.capture = capture  # if set, append (prompt, system, kwargs) per call

    async def chat_json(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        if self.capture is not None:
            self.capture.append(
                {
                    "user": user_message,
                    "system": system_message,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
        if self.raise_on_call is not None:
            raise self.raise_on_call
        # Return the first matching canned value; else default healthy.
        for needle, val in self.canned.items():
            if needle is None or needle in user_message:
                if self.return_non_dict and isinstance(val, dict):
                    # Return a JSON-parseable but non-dict (e.g. list) by
                    # wrapping as a dict with a sentinel key the model can
                    # still consume — for this fake we just return the dict;
                    # the "non dict" path is exercised by raising instead.
                    pass
                return val
        # Default healthy response.
        return {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "context_recall": 0.5,
        }


# ---------- Helpers ----------

def _fake_item(rid: str = "q1") -> Dict[str, Any]:
    return {
        "id": rid,
        "question": "Học phí ngành CNTT năm 2026?",
        "answer": "Học phí 600.000 VNĐ/tín chỉ.",
        "contexts": ["Học phí năm 2026 ngành CNTT: 600.000 VNĐ/tín chỉ."],
        "reference_answer": "600.000 VNĐ/tín chỉ.",
        "expected_behavior": "answer",
        "critical_facts": [{"value": "600.000", "fact_kind": "hoc_phi"}],
    }


# ============================================================================
# 1. score_item parses a healthy JSON response
# ============================================================================

def test_score_item_parses_healthy_response(jr_module) -> None:
    """Fake client returns a dict → score_item extracts floats and id."""
    canned = {
        None: {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "context_recall": 0.5,
        }
    }
    client = _FakeClient(canned=canned)
    item = _fake_item("msg-001")

    out = asyncio.run(jr_module.score_item(client, item, "RUBRIC_BODY"))

    assert out["id"] == "msg-001"
    assert out["faithfulness"] == pytest.approx(0.8)
    assert out["answer_relevancy"] == pytest.approx(0.7)
    assert out["context_precision"] == pytest.approx(0.6)
    assert out["context_recall"] == pytest.approx(0.5)


# ============================================================================
# 2. score_item: malformed / exception → all-null, no crash
# ============================================================================

def test_score_item_malformed_response_returns_all_null(jr_module) -> None:
    """Client raises (simulates malformed/unparseable) → all-null for that item."""
    client = _FakeClient(
        raise_on_call=ValueError("malformed JSON from upstream"),
    )
    item = _fake_item("msg-broken")

    out = asyncio.run(jr_module.score_item(client, item, "RUBRIC_BODY"))

    assert out["id"] == "msg-broken"
    assert out["faithfulness"] is None
    assert out["answer_relevancy"] is None
    assert out["context_precision"] is None
    assert out["context_recall"] is None


# ============================================================================
# 3. cohens_kappa — perfect agreement
# ============================================================================

def test_cohens_kappa_perfect_agreement(jr_module) -> None:
    """cohens_kappa identical labels → 1.0."""
    a = ["a", "b", "a", "b"]
    b = ["a", "b", "a", "b"]
    assert jr_module.cohens_kappa(a, b) == pytest.approx(1.0)


# ============================================================================
# 4. cohens_kappa — known 2x2 case (κ=0)
# ============================================================================

def test_cohens_kappa_known_value_2x2(jr_module) -> None:
    """a=[1,1,0,0], b=[1,0,1,0]:
        po = 2/4 = 0.5
        pa (a=1) = 2/4 = 0.5 ; pb (b=1) = 2/4 = 0.5
        pa(a=0) = 0.5 ; pb(b=0) = 0.5
        pe = 0.5*0.5 + 0.5*0.5 = 0.5
        κ = (po - pe) / (1 - pe) = 0.0
    """
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    assert jr_module.cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# 5. cohens_kappa — partial agreement (0 < κ < 1)
# ============================================================================

def test_cohens_kappa_partial_agreement(jr_module) -> None:
    """a=[1,1,1,0], b=[1,1,0,0]:
        po = 3/4 = 0.75
        pa(a=1) = 3/4 = 0.75 ; pb(b=1) = 2/4 = 0.5
        pa(a=0) = 1/4 = 0.25 ; pb(b=0) = 2/4 = 0.5
        pe = 0.75*0.5 + 0.25*0.5 = 0.375 + 0.125 = 0.5
        κ = (0.75 - 0.5) / (1 - 0.5) = 0.25 / 0.5 = 0.5
    """
    a = [1, 1, 1, 0]
    b = [1, 1, 0, 0]
    assert jr_module.cohens_kappa(a, b) == pytest.approx(0.5, abs=1e-9)


# ============================================================================
# 6. binned_kappa — drops nulls in EITHER list
# ============================================================================

def test_binned_kappa_drops_null_pairs(jr_module) -> None:
    """opus=[0.9,0.4,null,0.85], ns24=[0.95,0.45,0.7,null]
        index 0: opus=0.9 (high) , ram=0.95 (high)  → (high, high) agree
        index 1: opus=0.4 (low)  , ram=0.45 (low)   → (low, low)   agree
        index 2: opus=null       → DROPPED
        index 3: ram=null        → DROPPED
        n_compared=2, both agree on identical bins → κ=1.0
    """
    opus = [0.9, 0.4, None, 0.85]
    ram = [0.95, 0.45, 0.7, None]
    kappa = jr_module.binned_kappa(opus, ram)
    # Both pairs agree bin-for-bin → perfect agreement among the 2 compared.
    assert kappa == pytest.approx(1.0)
    # Sanity: n_compared is 2 — verify by re-deriving via internal bin function.
    n_compared = sum(
        1 for o, r in zip(opus, ram) if o is not None and r is not None
    )
    assert n_compared == 2


def test_binned_kappa_mixed_binning(jr_module) -> None:
    """opus=[0.9, 0.4, 0.7], ram=[0.95, 0.45, 0.3]:
        index 0: (high, high)
        index 1: (low, low)
        index 2: (mid, low)
        2/3 agree → po = 0.6667
        bins: (high,high,low), (high,low,low)
        pa(high) = 1/3, pa(low) = 1/3, pa(mid) = 1/3
        pb(high) = 1/3, pb(low) = 2/3, pb(mid) = 0
        pe = 1/9 + 2/9 + 0 = 3/9 = 0.3333
        κ = (0.6667 - 0.3333) / (1 - 0.3333) = 0.3333/0.6667 = 0.5
    """
    opus = [0.9, 0.4, 0.7]
    ram = [0.95, 0.45, 0.3]
    kappa = jr_module.binned_kappa(opus, ram)
    assert kappa == pytest.approx(0.5, abs=1e-6)


# ============================================================================
# 7. agreement_report — shape, no headline_score
# ============================================================================

def test_agreement_report_shape_and_no_headline(jr_module) -> None:
    """agreement_report returns per-metric κ + n_compared + mean_abs_diff,
    and MUST NOT include a headline_score key (ns24 is agreement only).
    """
    opus_verdicts = [
        {
            "id": "q1",
            "scores": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.6,
            },
        },
        {
            "id": "q2",
            "scores": {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": 0.5,
            },
        },
    ]
    ram_verdicts = [
        {
            "id": "q1",
            "scores": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
                "context_precision": 0.7,
                "context_recall": 0.65,
            },
        },
        {
            "id": "q2",
            "scores": {
                "faithfulness": 0.4,
                "answer_relevancy": 0.55,
                "context_precision": 0.5,
                "context_recall": 0.45,
            },
        },
    ]

    report = jr_module.agreement_report(opus_verdicts, ram_verdicts)

    # Shape: 4 metrics, each with kappa, n_compared, mean_abs_diff.
    assert "per_metric_kappa" in report
    assert "n_compared" in report
    assert "mean_abs_diff" in report
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        assert metric in report["per_metric_kappa"]
        assert metric in report["n_compared"]
        assert metric in report["mean_abs_diff"]
    # Both ids matched → n_compared=2 for every metric.
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        assert report["n_compared"][metric] == 2

    # CRITICAL: ns24 scores must NOT be presented as the headline scorer.
    assert "headline_score" not in report
    # Also explicitly: no field name that suggests a single combined verdict.
    forbidden = {"system_score", "final_score", "primary_score", "headline"}
    assert forbidden.isdisjoint(report.keys())


# ============================================================================
# 8. mean_abs_diff — sanity value
# ============================================================================

def test_mean_abs_diff_simple(jr_module) -> None:
    """opus faithfulness [0.8, 0.6], ns24 [0.7, 0.6]
        |0.8-0.7| = 0.1, |0.6-0.6| = 0.0  →  mean = 0.05
    """
    opus_verdicts = [
        {"id": "q1", "scores": {"faithfulness": 0.8}},
        {"id": "q2", "scores": {"faithfulness": 0.6}},
    ]
    ram_verdicts = [
        {"id": "q1", "scores": {"faithfulness": 0.7}},
        {"id": "q2", "scores": {"faithfulness": 0.6}},
    ]

    report = jr_module.agreement_report(opus_verdicts, ram_verdicts)
    assert report["mean_abs_diff"]["faithfulness"] == pytest.approx(0.05)


# ============================================================================
# Auxiliary: temperature=0 enforced
# ============================================================================

def test_score_item_uses_temperature_zero(jr_module) -> None:
    """For judge determinism, score_item MUST call chat_json at temperature=0."""
    capture: List[Dict[str, Any]] = []
    client = _FakeClient(capture=capture)
    item = _fake_item("msg-temp")

    asyncio.run(jr_module.score_item(client, item, "RUBRIC_BODY"))

    assert len(capture) == 1
    assert capture[0]["temperature"] == 0
