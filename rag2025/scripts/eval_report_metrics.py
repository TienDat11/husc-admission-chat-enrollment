"""Slice S7 — Final metrics report generator (CI, confusion matrix, latency, gates).

Combines the aggregator output (S5b), the κ agreement (S5c), the hallucination
classifier (S6), and the raw records (S2, for latency) into ONE human-readable
+ machine-readable report. The report shows the TRUE baseline — including FAIL
verdicts. We expect the gates to fail; that IS the deliverable.

Public API
----------
* ``percentile(values, p) -> float`` — linear-interpolated percentile (stdlib only).
* ``latency_stats(records) -> dict`` — p50/p95/mean per seam + attribution caveat.
* ``eval_gates(agg, halltype, abstain_accuracy) -> dict`` — 3 baseline gates.
* ``abstain_accuracy(records, gt_by_id, halltype_classifications) -> float`` —
  fraction of refuse/abstain-expected items the system actually abstained on.
* ``build_report(agg, kappa, halltype, records, gt_by_id) -> dict`` — full machine
  report (metrics w/ CI, confusion matrix + gate counts, latency stats, gates,
  Cohen's κ, circularity caveat).
* ``render_markdown(report) -> str`` — render report dict to readable markdown.
* ``main(argv) -> int`` — argparse CLI: writes JSON + Markdown sidecars.

NO LLM. NO network. Pure arithmetic + string rendering. The "abstain
inference" rules below are the single source of truth for what counts as
abstention in the abstain-accuracy gate.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence


# ---------- Paths ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = RAG2025_ROOT / "results" / "eval_harness"


# ---------- Constants ----------

# Per-seam latency fields. Each is a top-level key under latency_ms.
LATENCY_SEAMS: tuple = ("route_ms", "retrieval_loop_ms", "query_ms", "total_ms")

# Baseline gates (per the deliverable spec). Each → verdict PASS iff actual ≥ threshold.
GATE_THRESHOLDS: Dict[str, float] = {
    "faithfulness_mean": 0.90,
    "context_recall_mean": 0.85,
    "out_of_scope_abstain_accuracy": 0.95,
}

# Short-fallback / refusal markers — what we count as a successful abstention
# when the question expected refuse/abstain. This MUST match the spirit of
# eval_judge_halltype._FALLBACK_MARKERS / refusal markers, but is re-declared
# here so the slice is self-contained and the gate stays deterministic.
ABSTAIN_MARKERS: tuple = (
    "chưa có thông tin",
    "chưa có thông tin chính thức",
    "chưa có",
    "chưa công bố",
    "chưa được công bố",
    "không tìm thấy thông tin",
    "không tìm thấy",
    "ngoài phạm vi",
    "tôi không thể",
    "không thể tư vấn",
    "xin lỗi, tôi không",
)

# MANDATORY latency attribution caveat — emitted verbatim when attribution is
# missing across ALL records. The test asserts the literal substring
# "HyDE vs generation" so the wording is fixed.
ATTRIBUTION_CAVEAT = (
    "HyDE vs generation latency cannot be isolated this round."
)

# Circularity caveat for context_recall — emitted verbatim into the report.
# The test asserts the literal substring "context_recall measured fact-level".
CONTEXT_RECALL_CAVEAT = (
    "context_recall measured fact-level; GT auto-generated from canonical "
    "sources — recall may be optimistic; see human spot-check."
)

# Per-type hallucination gate caps (Type-1≤3 / Type-4=0 / Type-5≤3 / Type-6=0).
HALLTYPE_GATE: Dict[str, int] = {
    "type1_max": 3,
    "type4_max": 0,
    "type5_max": 3,
    "type6_max": 0,
}


__all__ = [
    "percentile",
    "latency_stats",
    "eval_gates",
    "abstain_accuracy",
    "build_report",
    "render_markdown",
    "main",
]


# =============================================================================
# 1. percentile — linear-interpolated (numpy default).
# =============================================================================

def percentile(values: List[float], p: float) -> float:
    """Linear-interpolated percentile (stdlib only).

    Mirrors ``numpy.percentile`` default linear interpolation. ``p`` is given
    in [0, 100] (i.e. 50 = median, 95 = 95th percentile) — matching the
    conventional p-th percentile interface, NOT a [0, 1] fraction.

    Edge cases:
      * Empty list → NaN.
      * Single value → that value.
      * p <= 0 → min.
      * p >= 100 → max.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(values[0])
    if p <= 0.0:
        return float(min(values))
    if p >= 100.0:
        return float(max(values))
    sorted_vals = sorted(values)
    rank = (p / 100.0) * (n - 1)
    lo_idx = int(math.floor(rank))
    hi_idx = int(math.ceil(rank))
    if lo_idx == hi_idx:
        return float(sorted_vals[lo_idx])
    frac = rank - lo_idx
    lo_v = float(sorted_vals[lo_idx])
    hi_v = float(sorted_vals[hi_idx])
    return lo_v + (hi_v - lo_v) * frac


# =============================================================================
# 2. latency_stats
# =============================================================================

def _record_latency_ms(rec: Any) -> Optional[Dict[str, float]]:
    """Pull a ``{seam: float}`` dict out of a record (Pydantic or dict)."""
    if hasattr(rec, "model_dump"):
        try:
            d = rec.model_dump()
        except Exception:
            return None
    elif isinstance(rec, dict):
        d = rec
    else:
        return None
    lm = d.get("latency_ms")
    if not isinstance(lm, dict):
        return None
    out: Dict[str, float] = {}
    for seam in LATENCY_SEAMS:
        v = lm.get(seam)
        if v is None:
            continue
        try:
            out[seam] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _record_has_attribution(rec: Any) -> bool:
    """True iff this record's latency_attribution is a non-null dict."""
    if hasattr(rec, "model_dump"):
        try:
            d = rec.model_dump()
        except Exception:
            return False
    elif isinstance(rec, dict):
        d = rec
    else:
        return False
    la = d.get("latency_attribution")
    return isinstance(la, dict) and len(la) > 0


def latency_stats(records: List[dict]) -> Dict[str, Any]:
    """Compute p50 / p95 / mean per latency seam + attribution caveat.

    Args:
      records: list of dict-shaped or Pydantic records with ``latency_ms`` and
        (optionally) ``latency_attribution``.

    Returns:
      Dict with one key per seam → {p50, p95, mean}, plus
      ``attribution_available`` (bool) and (when unavailable) the mandatory
      ``attribution_caveat`` string. The caveat is NEVER omitted silently when
      attribution is unavailable — the deliverable is honest about the gap.
    """
    seam_values: Dict[str, List[float]] = {s: [] for s in LATENCY_SEAMS}
    any_attribution = False

    for rec in records or []:
        lm = _record_latency_ms(rec)
        if lm is None:
            continue
        for seam, v in lm.items():
            seam_values.setdefault(seam, []).append(v)
        if _record_has_attribution(rec):
            any_attribution = True

    out: Dict[str, Any] = {}
    for seam in LATENCY_SEAMS:
        vals = seam_values.get(seam, [])
        if not vals:
            out[seam] = {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
        else:
            out[seam] = {
                "p50": percentile(vals, 50.0),
                "p95": percentile(vals, 95.0),
                "mean": sum(vals) / len(vals),
            }

    out["attribution_available"] = any_attribution
    if not any_attribution:
        out["attribution_caveat"] = ATTRIBUTION_CAVEAT

    return out


# =============================================================================
# 3. eval_gates
# =============================================================================

# AB2: abstain gate re-spec constants — Wilson lower-bound + min-n informational.
# Stated statistical constants (NOT tuned to the 86Q set):
#   z=1.96  → 95% one-sided Wilson score lower bound (textbook).
#   min_n=30 → normal-approximation floor (central-limit theorem).
#   floor=0.85 → quality threshold reachable but not trivially so.
ABSTAIN_GATE_Z: float = 1.96
ABSTAIN_GATE_MIN_N: int = 30
ABSTAIN_GATE_FLOOR: float = 0.85


def _wilson_lower_bound(successes: int, n: int, z: float = ABSTAIN_GATE_Z) -> float:
    """Wilson score lower bound (closed-form, no data-fitted magic number).

    Guard: n<=0 or successes<0 → 0.0.
    """
    if n <= 0 or successes < 0:
        return 0.0
    s = min(int(successes), int(n))
    p = s / float(n)
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom)


def _abstain_gate_verdict(successes: int, n: int) -> str:
    """Three-valued verdict: 'informational' if n<min_n, else 'pass'/'fail'."""
    if n < ABSTAIN_GATE_MIN_N:
        return "informational"
    lo = _wilson_lower_bound(successes, n)
    return "pass" if lo >= ABSTAIN_GATE_FLOOR else "fail"


def _gate(actual: Optional[float], threshold: float, gate_name: str) -> Dict[str, Any]:
    """Render a single gate row."""
    if actual is None:
        return {
            "gate": gate_name,
            "threshold": threshold,
            "actual": None,
            "verdict": "FAIL",
        }
    return {
        "gate": gate_name,
        "threshold": threshold,
        "actual": float(actual),
        "verdict": "PASS" if float(actual) >= threshold else "FAIL",
    }


def _agg_metric(agg: Dict[str, Any], key: str) -> Optional[float]:
    """Read a metric's mean from the aggregator shape (None-safe)."""
    if not isinstance(agg, dict):
        return None
    metrics = agg.get("metrics")
    if not isinstance(metrics, dict):
        return None
    row = metrics.get(key)
    if not isinstance(row, dict):
        return None
    v = row.get("mean")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve_halltype_shape(
    halltype: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (per_type_counts, gate) from either the top level OR the nested
    ``confusion_matrix`` block.

    The S6 classifier (``eval_judge_halltype.py``) emits ``per_type_counts``
    and ``gate`` NESTED inside ``confusion_matrix``; older/stub shapes put
    them at the top level. Read top-level first, then fall back to the
    nested location so the report never silently drops the per-type gate.
    """
    if not isinstance(halltype, dict):
        return None, None
    per_type = halltype.get("per_type_counts")
    gate = halltype.get("gate")
    cm = halltype.get("confusion_matrix")
    if isinstance(cm, dict):
        if not isinstance(per_type, dict):
            per_type = cm.get("per_type_counts")
        if not isinstance(gate, dict):
            gate = cm.get("gate")
    return (
        per_type if isinstance(per_type, dict) else None,
        gate if isinstance(gate, dict) else None,
    )


def _halltype_gate(halltype: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Render the per-type halltype gate (Type-1≤3/4=0/5≤3/6=0).

    The function works on either the S6 ``gate`` shape (already has
    ``pass`` flag) or a raw ``per_type_counts`` shape — if neither, every
    cap is reported as actual=0/pass=True (vacuously passing) so the
    report always has a uniform structure. Both shapes are resolved from
    the top level OR the nested ``confusion_matrix`` block.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(halltype, dict):
        for k, cap in HALLTYPE_GATE.items():
            out[k] = {"actual": 0, "max": cap, "pass": True}
        return out

    per_type, existing_gate = _resolve_halltype_shape(halltype)

    for k, cap in HALLTYPE_GATE.items():
        kind = k.replace("_max", "")  # "type1", "type4", "type5", "type6"
        if existing_gate and k in existing_gate and isinstance(existing_gate[k], dict):
            row = dict(existing_gate[k])
            row.setdefault("max", cap)
            row.setdefault("pass", row.get("actual", 0) <= cap)
            out[k] = row
        elif per_type is not None and kind in per_type:
            actual = int(per_type[kind] or 0)
            out[k] = {"actual": actual, "max": cap, "pass": actual <= cap}
        else:
            out[k] = {"actual": 0, "max": cap, "pass": True}
    return out


def eval_gates(
    agg: Dict[str, Any],
    halltype: Dict[str, Any],
    abstain_accuracy_val: float,
    abstain_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Evaluate the 3 baseline gates.

    Args:
      agg: aggregator output (S5b shape).
      halltype: halltype classifier output (S6 shape, used for the
        per-type halltype sub-gate — included verbatim into the report).
      abstain_accuracy_val: scalar [0, 1] from ``abstain_accuracy``.
      abstain_counts: optional ``{"n_correct": int, "n_total": int}`` from
        ``abstain_accuracy_with_counts``. When supplied, the abstain gate
        uses the **Wilson lower-bound + min-n=30 informational** rule (AB2
        re-spec) instead of the raw 0.95 point-threshold. The Wilson bound
        is closed-form; the min_n=30 is the standard normal-approximation
        floor. Both are 10-year-stable by construction.

    Returns:
      Dict with three baseline gate rows (faithfulness / context_recall /
      abstain_accuracy) AND the four per-type halltype rows.
    """
    faith = _agg_metric(agg, "faithfulness")
    cr = _agg_metric(agg, "context_recall")

    # --- AB2: abstain gate re-spec ---
    # When counts are available, replace the raw 0.95 point-threshold with
    # the Wilson-LO + min-n informational rule. The "actual" point estimate
    # is still reported (informational; report-only) for transparency.
    if isinstance(abstain_counts, dict) and "n_total" in abstain_counts:
        n_correct = int(abstain_counts.get("n_correct", 0) or 0)
        n_total = int(abstain_counts.get("n_total", 0) or 0)
        point = (n_correct / n_total) if n_total else 0.0
        wlo = _wilson_lower_bound(n_correct, n_total)
        verdict = _abstain_gate_verdict(n_correct, n_total)
        gates: Dict[str, Dict[str, Any]] = {
            "faithfulness": _gate(faith, GATE_THRESHOLDS["faithfulness_mean"], "faithfulness_mean"),
            "context_recall": _gate(cr, GATE_THRESHOLDS["context_recall_mean"], "context_recall_mean"),
            "abstain_accuracy": {
                "gate": "out_of_scope_abstain_accuracy",
                "threshold": GATE_THRESHOLDS["out_of_scope_abstain_accuracy"],
                "actual": float(abstain_accuracy_val) if abstain_accuracy_val is not None else None,
                "point_estimate": point,
                "n_correct": n_correct,
                "n_total": n_total,
                "wilson_lower_bound": wlo,
                "wilson_z": ABSTAIN_GATE_Z,
                "min_n": ABSTAIN_GATE_MIN_N,
                "floor": ABSTAIN_GATE_FLOOR,
                "verdict": verdict,
            },
        }
    else:
        gates = {
            "faithfulness": _gate(faith, GATE_THRESHOLDS["faithfulness_mean"], "faithfulness_mean"),
            "context_recall": _gate(cr, GATE_THRESHOLDS["context_recall_mean"], "context_recall_mean"),
            "abstain_accuracy": _gate(
                float(abstain_accuracy_val) if abstain_accuracy_val is not None else None,
                GATE_THRESHOLDS["out_of_scope_abstain_accuracy"],
                "out_of_scope_abstain_accuracy",
            ),
        }
    gates["halltype"] = _halltype_gate(halltype)
    return gates


# =============================================================================
# 4. abstain_accuracy
# =============================================================================

def _looks_like_abstain(answer_text: str) -> bool:
    """True iff the answer reads as a graceful abstain / refusal."""
    a = (answer_text or "").lower().strip()
    if not a:
        return True
    return any(m in a for m in ABSTAIN_MARKERS)


def _expected_refuse_or_abstain(gt_entry: Optional[Dict[str, Any]]) -> bool:
    """True iff GT says the system SHOULD refuse or abstain on this item."""
    if not isinstance(gt_entry, dict):
        return False
    eb = gt_entry.get("expected_behavior")
    if isinstance(eb, str) and eb in ("refuse", "abstain"):
        return True
    return False


def _refuse_or_abstain_indices(
    records: List[dict],
    gt_by_id: Dict[str, Dict[str, Any]],
) -> List[int]:
    """Return indices of records whose expected_behavior ∈ {refuse, abstain}."""
    out: List[int] = []
    for i, rec in enumerate(records or []):
        rid: Optional[str] = None
        if isinstance(rec, dict):
            rid = rec.get("id")
        elif hasattr(rec, "model_dump"):
            try:
                rid = rec.model_dump().get("id")
            except Exception:
                rid = None
        if rid is None:
            continue
        gt = gt_by_id.get(rid)
        if _expected_refuse_or_abstain(gt):
            out.append(i)
    return out


def _record_answer_text(rec: Any) -> str:
    """Return the answer text from a record (Pydantic or dict)."""
    if hasattr(rec, "model_dump"):
        try:
            return str(rec.model_dump().get("answer", "") or "")
        except Exception:
            return ""
    if isinstance(rec, dict):
        return str(rec.get("answer", "") or "")
    return ""


def _classify_answer_text(item: Dict[str, Any]) -> str:
    """Read answer text from a classification (or fall back to GT)."""
    at = item.get("answer_text")
    if isinstance(at, str):
        return at
    return ""


def abstain_accuracy(
    records: List[dict],
    gt_by_id: Dict[str, Dict[str, Any]],
    halltype_classifications: List[dict],
) -> float:
    """Fraction of refuse/abstain-expected items the system actually abstained on.

    An item "abstained" iff its answer text matches a known fallback/refusal
    marker. Returns correct / total (0.0 when total == 0). The denominator
    ONLY counts items whose GT expected_behavior ∈ {refuse, abstain}.
    """
    indices = _refuse_or_abstain_indices(records, gt_by_id)
    total = len(indices)
    if total == 0:
        return 0.0

    # Build a quick id → classification lookup so we can read answer_text.
    cls_by_id: Dict[str, Dict[str, Any]] = {}
    for c in halltype_classifications or []:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            cls_by_id[c["id"]] = c

    correct = 0
    for i in indices:
        rec = records[i]
        rid: Optional[str] = None
        if isinstance(rec, dict):
            rid = rec.get("id")
        elif hasattr(rec, "model_dump"):
            try:
                rid = rec.model_dump().get("id")
            except Exception:
                rid = None
        # Prefer the answer_text from halltype (it may be enriched), else
        # fall back to the raw record's answer.
        ans_text = _classify_answer_text(cls_by_id.get(rid or "", {}))
        if not ans_text:
            ans_text = _record_answer_text(rec)
        if _looks_like_abstain(ans_text):
            correct += 1

    return correct / total


def abstain_accuracy_with_counts(
    records: List[dict],
    gt_by_id: Dict[str, Dict[str, Any]],
    halltype_classifications: List[dict],
) -> Dict[str, Any]:
    """Like :func:`abstain_accuracy` but also returns the (n_correct, n_total)
    counts so the AB2 gate can apply the Wilson lower-bound + min-n rule.

    Returns dict: {accuracy, n_correct, n_total}.
    """
    indices = _refuse_or_abstain_indices(records, gt_by_id)
    total = len(indices)
    if total == 0:
        return {"accuracy": 0.0, "n_correct": 0, "n_total": 0}

    cls_by_id: Dict[str, Dict[str, Any]] = {}
    for c in halltype_classifications or []:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            cls_by_id[c["id"]] = c

    correct = 0
    for i in indices:
        rec = records[i]
        rid: Optional[str] = None
        if isinstance(rec, dict):
            rid = rec.get("id")
        elif hasattr(rec, "model_dump"):
            try:
                rid = rec.model_dump().get("id")
            except Exception:
                rid = None
        ans_text = _classify_answer_text(cls_by_id.get(rid or "", {}))
        if not ans_text:
            ans_text = _record_answer_text(rec)
        if _looks_like_abstain(ans_text):
            correct += 1
    return {"accuracy": correct / total, "n_correct": correct, "n_total": total}


# =============================================================================
# 5. build_report
# =============================================================================

def build_report(
    agg: Dict[str, Any],
    kappa: Dict[str, Any],
    halltype: Dict[str, Any],
    records: List[dict],
    gt_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the full machine report.

    Args:
      agg: S5b aggregator output (per-metric {n, mean, ci_lo, ci_hi} +
        faithfulness_n + optional coverage/by_intent).
      kappa: S5c agreement report (per_metric_kappa + n_compared + mean_abs_diff).
      halltype: S6 hallucination classifier output (per_type_counts + gate).
      records: S2 EvalRecord list (JSONL-parsed dicts OR Pydantic objects).
      gt_by_id: {id: gt_entry} for the refuse/abstain lookup.

    Returns:
      JSON-serialisable dict ready to be written to disk and re-parsed by
      downstream consumers (NOT including the markdown rendering).
    """
    # Metrics block: copy the per-metric rows straight through, but normalize
    # so downstream renders always see the same shape.
    metrics_in = agg.get("metrics") if isinstance(agg, dict) else None
    metrics_out: Dict[str, Dict[str, Any]] = {}
    if isinstance(metrics_in, dict):
        for k, v in metrics_in.items():
            if isinstance(v, dict):
                metrics_out[k] = {
                    "n": v.get("n"),
                    "mean": v.get("mean"),
                    "ci_lo": v.get("ci_lo"),
                    "ci_hi": v.get("ci_hi"),
                }

    abst = abstain_accuracy_with_counts(
        records, gt_by_id,
        halltype.get("classifications", []) if isinstance(halltype, dict) else [],
    )
    abst_scalar = abst["accuracy"]
    abst_counts = {"n_correct": abst["n_correct"], "n_total": abst["n_total"]}

    gates = eval_gates(agg, halltype, abst_scalar, abstain_counts=abst_counts)
    latency = latency_stats(records)

    return {
        "metrics": metrics_out,
        "faithfulness_n": agg.get("faithfulness_n") if isinstance(agg, dict) else None,
        "coverage": agg.get("coverage") if isinstance(agg, dict) and "coverage" in agg else None,
        "by_intent": agg.get("by_intent") if isinstance(agg, dict) and "by_intent" in agg else None,
        "confusion_matrix": {
            "per_type_counts": _resolve_halltype_shape(halltype)[0],
            "gate": gates["halltype"],
        },
        "latency": latency,
        "gates": {
            "faithfulness": gates["faithfulness"],
            "context_recall": gates["context_recall"],
            "abstain_accuracy": gates["abstain_accuracy"],
            "halltype": gates["halltype"],
        },
        "kappa": {
            "per_metric_kappa": (kappa.get("per_metric_kappa") if isinstance(kappa, dict) else None),
            "n_compared": (kappa.get("n_compared") if isinstance(kappa, dict) else None),
            "mean_abs_diff": (kappa.get("mean_abs_diff") if isinstance(kappa, dict) else None),
            "n_matched_ids": (kappa.get("n_matched_ids") if isinstance(kappa, dict) else None),
        },
        "abstain_accuracy": abst,
        "context_recall_caveat": CONTEXT_RECALL_CAVEAT,
    }


# =============================================================================
# 6. render_markdown
# =============================================================================

def _fmt(x: Any, digits: int = 3) -> str:
    """Render a float / None safely in a fixed-width cell."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report dict as a human-readable Markdown document.

    The output is intentionally self-contained (no external CSS / no
    template engine) — it must read cleanly in a raw terminal and in
    a Figma/Slack/Notion preview.
    """
    parts: List[str] = []

    parts.append("# Eval metrics report")
    parts.append("")
    parts.append(
        "TRUE baseline — gates are reported PASS/FAIL as observed. We expect "
        "the system to fail the gates; that is the deliverable."
    )
    parts.append("")

    # ----- Metrics with CI -----
    parts.append("## Metrics (Opus primary, mean ± 95 % CI)")
    parts.append("")
    parts.append("| metric | n | mean | 95 % CI |")
    parts.append("|---|---:|---:|---|")
    metrics = report.get("metrics") or {}
    if not metrics:
        parts.append("| _(none)_ | — | — | — |")
    else:
        for name, row in metrics.items():
            if not isinstance(row, dict):
                continue
            n = row.get("n")
            mean = row.get("mean")
            lo = row.get("ci_lo")
            hi = row.get("ci_hi")
            ci = f"[{_fmt(lo)}, {_fmt(hi)}]"
            parts.append(
                f"| {name} | {n if n is not None else 'n/a'} | "
                f"{_fmt(mean)} | {ci} |"
            )
    parts.append("")

    # ----- Context recall caveat -----
    caveat = report.get("context_recall_caveat")
    if isinstance(caveat, str) and caveat:
        parts.append(f"> _Caveat — {caveat}_")
        parts.append("")

    # ----- Confusion matrix / per-type counts -----
    parts.append("## Hallucination type counts (vs gate)")
    parts.append("")
    cm = report.get("confusion_matrix") or {}
    ptc = cm.get("per_type_counts") if isinstance(cm, dict) else None
    hall_gate = (cm.get("gate") if isinstance(cm, dict) else None) or {}
    parts.append("| type | actual | max | verdict |")
    parts.append("|---|---:|---:|---|")
    if isinstance(ptc, dict) and hall_gate:
        for k in ("type1", "type4", "type5", "type6"):
            actual = ptc.get(k, 0)
            cap_key = f"{k}_max"
            row = hall_gate.get(cap_key) if isinstance(hall_gate, dict) else None
            cap = row.get("max") if isinstance(row, dict) else None
            verdict = "PASS" if (isinstance(row, dict) and row.get("pass")) else "FAIL"
            parts.append(f"| {k} | {actual} | {cap if cap is not None else 'n/a'} | {verdict} |")
    else:
        parts.append("| _(none)_ | — | — | — |")
    parts.append("")

    # ----- Latency -----
    parts.append("## Latency seams (p50 / p95 / mean, ms)")
    parts.append("")
    parts.append("| seam | p50 | p95 | mean |")
    parts.append("|---|---:|---:|---:|")
    lat = report.get("latency") or {}
    for seam in LATENCY_SEAMS:
        row = lat.get(seam) if isinstance(lat, dict) else None
        if isinstance(row, dict):
            parts.append(
                f"| {seam} | {_fmt(row.get('p50'), 1)} | "
                f"{_fmt(row.get('p95'), 1)} | {_fmt(row.get('mean'), 1)} |"
            )
        else:
            parts.append(f"| {seam} | n/a | n/a | n/a |")
    parts.append("")
    if isinstance(lat, dict) and not lat.get("attribution_available", True):
        ac = lat.get("attribution_caveat")
        if isinstance(ac, str) and ac:
            parts.append(f"> _Caveat — {ac}_")
            parts.append("")

    # ----- Baseline gates -----
    parts.append("## Baseline gates (PASS/FAIL)")
    parts.append("")
    parts.append("| gate | threshold | actual | verdict |")
    parts.append("|---|---:|---:|---|")
    gates = report.get("gates") or {}
    for k in ("faithfulness", "context_recall", "abstain_accuracy"):
        row = gates.get(k) if isinstance(gates, dict) else None
        if isinstance(row, dict):
            parts.append(
                f"| {row.get('gate', k)} | "
                f"{_fmt(row.get('threshold'), 2)} | "
                f"{_fmt(row.get('actual'), 3)} | "
                f"{row.get('verdict', 'FAIL')} |"
            )
        else:
            parts.append(f"| {k} | n/a | n/a | FAIL |")
    parts.append("")

    # ----- Cohen's κ -----
    parts.append("## Cohen's κ (Opus vs ramclouds)")
    parts.append("")
    kappa = report.get("kappa") or {}
    pmk = kappa.get("per_metric_kappa") if isinstance(kappa, dict) else None
    ncp = kappa.get("n_compared") if isinstance(kappa, dict) else None
    parts.append("| metric | κ | n_compared |")
    parts.append("|---|---:|---:|")
    if isinstance(pmk, dict):
        for k, v in pmk.items():
            n = ncp.get(k) if isinstance(ncp, dict) else None
            parts.append(f"| {k} | {_fmt(v, 3)} | {n if n is not None else 'n/a'} |")
    else:
        parts.append("| _(none)_ | — | — |")
    parts.append("")

    return "\n".join(parts)


# =============================================================================
# 7. CLI
# =============================================================================

def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a JSONL file → list of dicts (blank lines skipped)."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_gt_by_id(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"GT must be a JSON array, got: {type(data).__name__}")
    return {e["id"]: e for e in data if isinstance(e, dict) and "id" in e}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint — load JSON inputs, write JSON + Markdown report."""
    parser = argparse.ArgumentParser(
        description=(
            "Slice S7 — final metrics report. Combines the aggregator (S5b), "
            "κ agreement (S5c), hallucination classifier (S6), and raw records "
            "(S2, for latency) into ONE machine + human report."
        )
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument(
        "--records", required=True, type=pathlib.Path,
        help="Path to S2 EvalRecord JSONL.",
    )
    parser.add_argument(
        "--aggregate", required=True, type=pathlib.Path,
        help="Path to S5b judge_aggregate_<run_id>.json.",
    )
    parser.add_argument(
        "--kappa", required=True, type=pathlib.Path,
        help="Path to S5c judge_ramclouds_<run_id>.json.",
    )
    parser.add_argument(
        "--halltype", required=True, type=pathlib.Path,
        help="Path to S6 judge_halltype_<run_id>.json.",
    )
    parser.add_argument(
        "--gt", required=True, type=pathlib.Path,
        help="Path to GT JSON (array of {id, ...}).",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=None,
        help=(
            "Override output dir (default: "
            "rag2025/results/eval_harness/metrics_report/<run_id>/)."
        ),
    )
    args = parser.parse_args(argv)

    records = _read_jsonl(args.records)
    agg = _read_json(args.aggregate)
    kappa = _read_json(args.kappa)
    halltype = _read_json(args.halltype)
    gt_by_id = _read_gt_by_id(args.gt)

    report = build_report(agg, kappa, halltype, records, gt_by_id)
    report["run_id"] = args.run_id

    out_dir = args.out_dir or (RESULTS_ROOT / "metrics_report" / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"metrics_report_{args.run_id}.json"
    md_path = out_dir / f"metrics_report_{args.run_id}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"[eval_report_metrics] run_id={args.run_id} out={out_dir}")
    g = report.get("gates", {})
    for k in ("faithfulness", "context_recall", "abstain_accuracy"):
        row = g.get(k, {})
        print(f"  {k}: actual={row.get('actual')} verdict={row.get('verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
