"""SLICE S-Parity — RAGAS library as ARBITER on a ≥30Q sample.

We use the RAGAS library as an INDEPENDENT ARBITER against our
Opus-judge (subagent) faithfulness/answer_relevancy scores. On a
sample of ≥30 questions we:

  1. Build a RAGAS EvaluationDataset from the sample (question, answer,
     contexts, optionally ground truth).
  2. Run ``ragas.evaluate(metrics=[faithfulness, answer_relevancy])``.
  3. Compute the per-metric mean |Δ| between RAGAS-lib scores and
     Opus-judge scores (matched by id).
  4. Apply the ARBITER verdict per metric:
       * faithfulness  — TIGHT tolerance 0.05
       * other metrics — LOOSER tolerance 0.10
     If |Δ| ≤ tol → agree; else agree=False, "ragas_lib" is the
     AUTHORITATIVE value and the metric is flagged for review.

The module is LAZY by design. ``ragas`` is imported INSIDE
``_ragas_score`` (and only there) so that the core harness import
graph never sees it (mirrors the discipline in
``scripts/eval_judge_*.py``). If the import fails, ``run_parity``
returns ``{"skipped": True, ...}`` instead of crashing.

Public API
----------
``compute_delta(opus_scores, lib_scores) -> dict``
    Pure arithmetic: per-metric mean |Δ| on matched ids.
    No ragas needed → fully testable offline.

``arbiter_verdict(delta, *, faithfulness_tol=0.05, other_tol=0.10) -> dict``
    Pure decision: per-metric agree/disagree, with the
    tighter tolerance on faithfulness.

``run_parity(records_sample, opus_verdicts, *, n_min=30) -> dict``
    Orchestrator: lazy-imports ragas, builds dataset, scores, then
    delta + verdict. Returns:
      * ``{"skipped": True, "reason": "ragas not installed"}`` on ImportError
      * ``{"insufficient_sample": True, ...}`` when ``len(sample) < n_min``
      * otherwise full delta + verdict + raw scores

``_ragas_score(records_sample) -> dict``
    ISOLATED seam (lazy ragas import, dataset build, run). Tests
    monkeypatch this to run offline; the orchestrator only calls it.

``main()``
    CLI: ``--run-id --records --opus-aggregate --sample-n 30 --out``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ---------- Constants ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = RAG2025_ROOT / "results" / "eval_harness"

# The two RAGAS metrics this slice compares. Order is preserved in
# outputs for stable downstream comparison. `context_precision` and
# `context_recall` are EXCLUDED on purpose — they require gold-relevant
# chunk labels / gold answers that the 86Q set does not have (matches
# the honest subset in `scripts/ragas_lite_eval.py`).
PARITY_METRICS: tuple[str, ...] = ("faithfulness", "answer_relevancy")

# Default tolerances — faithfulness is the integrity-critical metric,
# so we use a TIGHT 0.05 there. Other RAGAS metrics are noisier on
# small samples, so we use a LOOSER 0.10 for them.
DEFAULT_FAITHFULNESS_TOL = 0.05
DEFAULT_OTHER_TOL = 0.10


__all__ = [
    "PARITY_METRICS",
    "DEFAULT_FAITHFULNESS_TOL",
    "DEFAULT_OTHER_TOL",
    "compute_delta",
    "arbiter_verdict",
    "run_parity",
    "_ragas_score",
    "main",
]


# =============================================================================
# Pure arithmetic helpers (no ragas, no I/O)
# =============================================================================

def _extract_metric(
    src: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> Dict[str, float]:
    """Return ``{id: metric_value}`` for every id whose metric is present.

    Drops entries whose metric is ``None``, NaN, or non-numeric.
    """
    out: Dict[str, float] = {}
    for rid, scores in src.items():
        if not isinstance(scores, Mapping):
            continue
        v = scores.get(metric)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        out[rid] = f
    return out


def compute_delta(
    opus_scores: Mapping[str, Mapping[str, Any]],
    lib_scores: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Per-metric mean |Δ| on ids present in BOTH sides.

    For each metric M in the UNION of metrics present on either side,
    restrict to ids that have M on BOTH sides, then average
    ``|opus[id][M] - lib[id][M]|``. Ids present on only one side are
    silently ignored (no overstate via partial overlap).

    Args:
      opus_scores: ``{id: {metric: value, ...}, ...}`` — typically the
        per-id Opus-judge scores from the aggregate JSON.
      lib_scores: same shape, from the RAGAS library.

    Returns:
      ``{metric: mean_abs_delta}`` for every metric that has ≥1 matched
      id. Empty input → ``{}``.
    """
    metrics: set[str] = set()
    for src in (opus_scores, lib_scores):
        for scores in src.values():
            if isinstance(scores, Mapping):
                metrics.update(scores.keys())

    out: Dict[str, float] = {}
    for m in sorted(metrics):
        opus_m = _extract_metric(opus_scores, m)
        lib_m = _extract_metric(lib_scores, m)
        matched_ids = set(opus_m) & set(lib_m)
        if not matched_ids:
            continue
        deltas = [abs(opus_m[i] - lib_m[i]) for i in matched_ids]
        out[m] = sum(deltas) / len(deltas)
    return out


def arbiter_verdict(
    delta: Mapping[str, float],
    *,
    faithfulness_tol: float = DEFAULT_FAITHFULNESS_TOL,
    other_tol: float = DEFAULT_OTHER_TOL,
) -> Dict[str, Any]:
    """Per-metric agree/disagree + overall agree.

    For each metric with a value in ``delta``:
      * ``agree = |Δ| ≤ tolerance``
      * on agree=True → ``tolerance`` only (no authoritative).
      * on agree=False → ``tolerance``, ``authoritative="ragas_lib"``,
        ``flagged_metric=metric``.

    faithfulness uses the TIGHT ``faithfulness_tol``; every other
    metric uses ``other_tol``. Metrics with ``None``/NaN delta are
    dropped from the verdict (not flagged).

    Returns::
        {
          "per_metric": {
            "<metric>": {
              "delta": float, "tolerance": float,
              "agree": bool, "authoritative": "ragas_lib" | None,
              "flagged_metric": "<metric>" | None,
            }, ...
          },
          "overall_agree": bool,   # True iff every metric agrees
        }
    """
    per_metric: Dict[str, Dict[str, Any]] = {}
    for m, d in delta.items():
        if d is None:
            continue
        try:
            df = float(d)
        except (TypeError, ValueError):
            continue
        if math.isnan(df):
            continue
        tol = faithfulness_tol if m == "faithfulness" else other_tol
        agree = abs(df) <= tol
        entry: Dict[str, Any] = {
            "delta": df,
            "tolerance": tol,
            "agree": agree,
        }
        if not agree:
            entry["authoritative"] = "ragas_lib"
            entry["flagged_metric"] = m
        per_metric[m] = entry

    overall_agree = all(e["agree"] for e in per_metric.values()) if per_metric else True
    return {"per_metric": per_metric, "overall_agree": overall_agree}


# =============================================================================
# Isolated seam: lazy ragas import + dataset build + run
# =============================================================================

def _ragas_score(records_sample: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Build a RAGAS dataset from ``records_sample`` and score it.

    ``records_sample`` is a list of dicts with at least::

        {"id": str, "question": str, "answer": str, "contexts": list[str]}

    Optional ``ground_truth`` (str) is forwarded to the dataset if
    present (some RAGAS metrics consume it).

    Returns ``{id: {metric: value}}`` for the metrics in ``PARITY_METRICS``.

    Lazy import: ``ragas`` is imported INSIDE this function. A
    ``ImportError`` propagates to the orchestrator, which returns
    ``{"skipped": True, ...}``. Any OTHER exception is RE-RAISED
    after logging — we never silently swallow real errors.
    """
    # Lazy import — the S0 discipline: never top-level-import ragas.
    from datasets import Dataset  # type: ignore
    from ragas import evaluate  # type: ignore
    from ragas.metrics import answer_relevancy, faithfulness  # type: ignore

    if not records_sample:
        return {}

    rows: List[Dict[str, Any]] = []
    ids: List[str] = []
    for r in records_sample:
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        question = r.get("question") or ""
        answer = r.get("answer") or ""
        contexts = r.get("contexts") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        row: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "contexts": [str(c) for c in contexts],
        }
        gt = r.get("ground_truth")
        if gt is not None:
            row["ground_truth"] = str(gt)
        rows.append(row)
        ids.append(rid)

    if not rows:
        return {}

    dataset = Dataset.from_list(rows)
    metrics_list = [faithfulness, answer_relevancy]
    result = evaluate(dataset, metrics=metrics_list)

    # RAGAS returns a ``Result`` object with ``scores`` keyed by metric,
    # each a list aligned with the dataset rows. We pull per-row values.
    raw_scores: Dict[str, List[Any]] = {}
    for m in PARITY_METRICS:
        col = getattr(result, "scores", {}).get(m)
        if col is None and hasattr(result, m):
            col = getattr(result, m)
        if col is None:
            continue
        raw_scores[m] = list(col)

    out: Dict[str, Dict[str, float]] = {rid: {} for rid in ids}
    for m, values in raw_scores.items():
        for i, rid in enumerate(ids):
            if i >= len(values):
                continue
            v = values[i]
            if v is None:
                continue
            try:
                out[rid][m] = float(v)
            except (TypeError, ValueError):
                continue
    return out


# =============================================================================
# Orchestrator: run_parity
# =============================================================================

def run_parity(
    records_sample: Sequence[Mapping[str, Any]],
    opus_verdicts: Mapping[str, Mapping[str, Any]],
    *,
    n_min: int = 30,
) -> Dict[str, Any]:
    """Run the parity cross-check between Opus-judge and RAGAS-library.

    Steps:
      1. Size check: if ``len(records_sample) < n_min``, attach an
         ``insufficient_sample`` warning but still attempt the
         comparison (so the result is still useful for offline tests).
      2. Call ``_ragas_score(records_sample)`` — the only seam that
         touches ragas. If it raises ``ImportError``, return
         ``{"skipped": True, "reason": "ragas not installed"}``.
      3. ``compute_delta`` + ``arbiter_verdict`` on the two score maps.
      4. Return a JSON-serialisable dict with delta, verdict, raw
         lib scores, and the sample-size metadata.

    Args:
      records_sample: list of dicts (id, question, answer, contexts, ...).
      opus_verdicts: ``{id: {metric: value}}`` from
        ``scripts/eval_judge_aggregate.py`` (or a per-id aggregate).
      n_min: minimum sample size before a warning is attached.
    """
    n_sample = len(records_sample)
    out: Dict[str, Any] = {
        "n_sample": n_sample,
        "n_min": n_min,
    }
    if n_sample < n_min:
        out["insufficient_sample"] = True

    try:
        lib_scores = _ragas_score(records_sample)
    except ImportError as exc:
        # The seam's lazy import failed → ragas not installed.
        return {
            "skipped": True,
            "reason": f"ragas not installed ({exc.__class__.__name__})",
            "n_sample": n_sample,
            "n_min": n_min,
        }
    # Any other exception: re-raise (never silently swallow real errors).

    delta = compute_delta(opus_verdicts, lib_scores)
    verdict = arbiter_verdict(delta)

    out.update({
        "delta": delta,
        "verdict": verdict,
        "lib_scores": lib_scores,
    })
    return out


# =============================================================================
# I/O + CLI
# =============================================================================

def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_aggregate(path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    """Load ``judge_aggregate_<run_id>.json`` → ``{id: {metric: value}}``.

    The aggregate file's payload shape is::

        {"metrics": {"faithfulness": {"mean": ...}, ...}, ...}

    but that's a SUMMARY, not per-id. For a true per-id map we need a
    different source. This helper is permissive: if the file contains
    a ``per_id`` key (a non-standard extension some slices add), use
    it directly. Otherwise we return an empty map and rely on the
    caller to provide per-id opus verdicts via another channel.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    per_id = data.get("per_id") if isinstance(data, dict) else None
    if isinstance(per_id, dict):
        # Coerce to {id: {metric: float}}.
        out: Dict[str, Dict[str, float]] = {}
        for rid, scores in per_id.items():
            if not isinstance(scores, Mapping):
                continue
            cleaned: Dict[str, float] = {}
            for k, v in scores.items():
                if v is None:
                    continue
                try:
                    cleaned[k] = float(v)
                except (TypeError, ValueError):
                    continue
            out[str(rid)] = cleaned
        return out
    return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: run the parity cross-check and write a JSON report.

    Args (positional via argparse)::

        --run-id         unique run identifier
        --records        path to EvalRecords JSONL (sample source)
        --opus-aggregate path to ``judge_aggregate_<run_id>.json``
                          (also needs a ``per_id`` block — see _load_aggregate)
        --sample-n 30    how many records to take from the top of the file
        --out            output path (default
                          ``rag2025/results/eval_harness/parity_<run_id>.json``)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run RAGAS-library-as-arbiter cross-check on a ≥30Q sample "
            "and emit a parity JSON report."
        ),
    )
    parser.add_argument("--run-id", required=True,
                        help="Unique run identifier.")
    parser.add_argument("--records", required=True, type=pathlib.Path,
                        help="Path to EvalRecords JSONL (sample source).")
    parser.add_argument("--opus-aggregate", required=False, type=pathlib.Path,
                        default=None,
                        help="Path to judge_aggregate_<run_id>.json "
                             "(with per_id block). Optional.")
    parser.add_argument("--sample-n", type=int, default=30,
                        help="Number of records to sample (default 30).")
    parser.add_argument("--out", required=False, type=pathlib.Path,
                        default=None,
                        help="Override output path.")
    args = parser.parse_args(argv)

    if not args.records.is_file():
        raise FileNotFoundError(f"records file not found: {args.records}")

    all_records = _read_jsonl(args.records)
    sample = all_records[: max(0, args.sample_n)]

    opus_verdicts: Dict[str, Dict[str, float]] = {}
    if args.opus_aggregate is not None and args.opus_aggregate.is_file():
        opus_verdicts = _load_aggregate(args.opus_aggregate)

    result = run_parity(sample, opus_verdicts, n_min=args.sample_n)
    result["run_id"] = args.run_id

    out_path = args.out or (
        RESULTS_ROOT / f"parity_{args.run_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if result.get("skipped"):
        print(f"[eval_parity_ragas_lib] run_id={args.run_id} SKIPPED: "
              f"{result.get('reason')}")
    else:
        v = result.get("verdict", {}).get("overall_agree")
        n = result.get("n_sample")
        print(f"[eval_parity_ragas_lib] run_id={args.run_id} n={n} "
              f"overall_agree={v} out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
