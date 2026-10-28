"""Slice S5c — ns24 gateway secondary judge + Cohen's κ agreement.

The PRIMARY judge is the Opus subagent (cross-vendor, breaks self-grading).
This module adds a SECONDARY judge using the project's own `UnifiedLLMClient`
(ns24 gateway) — NOT to score the system, but to quantify judge-vendor
AGREEMENT (Cohen's κ) with the Opus verdicts. The ns24 gateway is the AGREEMENT
BASELINE, NEVER the headline scorer. Scores produced here MUST NOT be
merged into the headline aggregate (that is Opus-only — Slice S5b).

Public API:
  * score_item(client, item, rubric) -> dict   (async — calls injected client)
  * score_batch(client, batch) -> list[dict]   (async)
  * cohens_kappa(labels_a, labels_b) -> float
  * binned_kappa(opus, ram, *, bins=(0.5, 0.8)) -> float
  * agreement_report(opus_verdicts, ns24_verdicts) -> dict
  * main() — argparse: --run-id --batches-dir --opus-verdicts-dir --out

Determinism:
  * Judge is called with temperature=0 to keep verdicts reproducible.
  * κ is a pure function of (labels_a, labels_b) — no randomness, no I/O.

The client is INJECTED so tests can pass a fake (no real network in tests).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger


# ---------- Paths ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = RAG2025_ROOT / "results" / "eval_harness"


# ---------- Constants ----------

# Metrics produced by every judge (the S5a verdict contract).
METRICS: Tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# Default bin edges: low/mid/high  →  3 categories.
DEFAULT_BINS: Tuple[float, float] = (0.5, 0.8)

# Judge temperature: 0 for determinism (per spec).
JUDGE_TEMPERATURE: float = 0.0

# Defensive ceiling for the judge call. Small because the verdict is
# just 4 floats + a short rationale.
JUDGE_MAX_TOKENS: int = 512


# ============================================================================
# Judge prompt
# ============================================================================

def _build_judge_prompt(rubric: str, item: Dict[str, Any]) -> str:
    """Render the per-item judge user message.

    The rubric is the SAME fixed contract used for the Opus subagent (S5a) so
    both judges reason over the same metric definitions. The item is the
    per-record shape produced by `prepare_batches` (id, question, answer,
    contexts, reference_answer, expected_behavior, critical_facts).
    """
    payload = {
        "id": item.get("id"),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "contexts": item.get("contexts", []),
        "reference_answer": item.get("reference_answer"),
        "expected_behavior": item.get("expected_behavior"),
        "critical_facts": item.get("critical_facts", []),
    }
    return (
        "You are the ns24 secondary judge. Score the item below using "
        "the EXACT rubric embedded in the system message.\n\n"
        "Return ONE JSON object with these keys (floats in [0.0, 1.0] or "
        "null where the rubric says not-applicable):\n"
        "  - faithfulness\n"
        "  - answer_relevancy\n"
        "  - context_precision\n"
        "  - context_recall\n\n"
        "Do not include any other keys. Do not wrap in markdown.\n\n"
        "ITEM:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _null_verdict(item_id: Any) -> Dict[str, Any]:
    """Return the all-null verdict shape (used on parse / call failure)."""
    return {
        "id": item_id,
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }


def _coerce_score(v: Any) -> Optional[float]:
    """Best-effort coerce to float in [0, 1] or None.

    Accepts int / float / str-parseable / None / 'null'. Anything else -> None.
    Out-of-range numeric values are clamped to [0, 1] so a model that
    occasionally drifts past the bound doesn't poison the κ computation.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        # bool is a subclass of int — explicitly reject to avoid silent 0/1.
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "null", "none", "n/a", "na"):
            return None
        try:
            f = float(s)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if f != f:  # NaN guard
        return None
    # Clamp into [0, 1] (defensive — rubric says closed interval).
    if f < 0.0:
        f = 0.0
    elif f > 1.0:
        f = 1.0
    return f


# ============================================================================
# score_item / score_batch
# ============================================================================

async def score_item(
    client: Any,
    item: Dict[str, Any],
    rubric: str,
) -> Dict[str, Any]:
    """Score a single prepared-batch item via the injected ns24 gateway client.

    The client is INJECTED so tests can pass a fake `chat_json`-only stub.
    For real runs, this is `UnifiedLLMClient` from
    `rag2025/src/services/llm_client.py`. We call it at temperature=0 with
    JSON mode for determinism.

    Returns:
        dict with keys: id, faithfulness, answer_relevancy, context_precision,
        context_recall. Any metric the model omits or returns malformed is
        `None`; the function never raises on a bad response (it logs and
        returns the all-null verdict).
    """
    rid = item.get("id")
    prompt = _build_judge_prompt(rubric, item)
    try:
        raw = await client.chat_json(
            prompt,
            system_message=rubric,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )
    except Exception as exc:  # network / parse / transport / any provider error
        logger.warning(
            f"[eval_judge_ns24] chat_json failed for id={rid!r}: {exc}"
        )
        return _null_verdict(rid)

    if not isinstance(raw, dict):
        logger.warning(
            f"[eval_judge_ns24] non-dict response for id={rid!r}: "
            f"type={type(raw).__name__}"
        )
        return _null_verdict(rid)

    out = _null_verdict(rid)
    for metric in METRICS:
        out[metric] = _coerce_score(raw.get(metric))
    return out


async def score_batch(client: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score every item in a prepared batch dict (has 'items' key)."""
    rubric = batch.get("rubric", "")
    items = batch.get("items", []) or []
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(await score_item(client, it, rubric))
    return out


# ============================================================================
# Cohen's κ
# ============================================================================

def cohens_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    """Standard Cohen's κ for two equal-length categorical label sequences.

    Returns:
        * 1.0 if the two sequences are identical and non-empty.
        * 1.0 if both sequences are empty (vacuously — no disagreement).
        * 0.0 if both sequences are non-empty but degenerate in a way that
          κ is undefined AND the two lists are NOT identical (i.e. one side
          is all-same-label A and the other all-same-label B — no overlap
          beyond random expectation, so κ → 0).
        * 0.0 if the sequences disagree on every item when one side is
          constant (pe == 1, denominator 0).
        * Standard (po - pe) / (1 - pe) otherwise.

    Raises:
        ValueError: if the two sequences have different lengths.
    """
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError(
            f"cohens_kappa: length mismatch (a={n}, b={len(labels_b)})"
        )

    # Empty case: vacuously perfect agreement.
    if n == 0:
        return 1.0

    # Perfect agreement shortcut.
    if list(labels_a) == list(labels_b):
        return 1.0

    # Count co-occurrences in the 2x2 / CxC contingency table.
    categories = sorted(set(labels_a) | set(labels_b))
    cat_index = {c: i for i, c in enumerate(categories)}
    c = len(categories)
    if c < 2:
        # Degenerate: only one category present. Since a != b, the other
        # side is also that single category (impossible — list equality was
        # already checked). Treat as undefined → 0.0.
        return 0.0

    table = [[0] * c for _ in range(c)]
    for a, b in zip(labels_a, labels_b):
        i = cat_index[a]
        j = cat_index[b]
        table[i][j] += 1

    # Marginals.
    row_totals = [sum(table[i]) for i in range(c)]
    col_totals = [sum(table[i][j] for i in range(c)) for j in range(c)]
    n_total = n

    # po = sum of diagonal / N
    po = sum(table[i][i] for i in range(c)) / n_total
    # pe = sum_i (row_i/N) * (col_i/N)
    pe = sum(
        (row_totals[i] / n_total) * (col_totals[i] / n_total)
        for i in range(c)
    )

    denom = 1.0 - pe
    if denom == 0.0:
        # pe == 1 means expected agreement == 1; if po != pe, κ is undefined.
        # We've already handled the "identical" case above, so this branch
        # means po < pe which is impossible — but defensively return 0.0.
        return 0.0
    return (po - pe) / denom


# ============================================================================
# Binning + binned_kappa
# ============================================================================

def _bin_score(score: Optional[float], bins: Tuple[float, float]) -> Optional[str]:
    """Bin a score into low/mid/high categories.

    Boundaries are interpreted as:
        score < bins[0]           → "low"
        bins[0] <= score < bins[1] → "mid"
        score >= bins[1]          → "high"
    """
    if score is None:
        return None
    if score < bins[0]:
        return "low"
    if score < bins[1]:
        return "mid"
    return "high"


def binned_kappa(
    opus_scores: Sequence[Optional[float]],
    ns24_scores: Sequence[Optional[float]],
    *,
    bins: Tuple[float, float] = DEFAULT_BINS,
) -> float:
    """Compute Cohen's κ on binned (low/mid/high) labels, dropping null pairs.

    Pairs where EITHER side is None are excluded. If after dropping no pairs
    remain, returns 1.0 (vacuously — no disagreement). If the binned labels
    are perfectly identical on the kept pairs, returns 1.0.
    """
    if len(opus_scores) != len(ns24_scores):
        raise ValueError(
            f"binned_kappa: length mismatch "
            f"(opus={len(opus_scores)}, ns24={len(ns24_scores)})"
        )

    a: List[str] = []
    b: List[str] = []
    for o, r in zip(opus_scores, ns24_scores):
        if o is None or r is None:
            continue
        ba = _bin_score(o, bins)
        br = _bin_score(r, bins)
        if ba is None or br is None:
            continue
        a.append(ba)
        b.append(br)

    return cohens_kappa(a, b)


# ============================================================================
# Agreement report
# ============================================================================

def _verdict_score_lookup(
    verdicts: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build a {id: {metric: score_or_None}} index from a verdict list.

    Accepts either the S5a verdict shape
    (`{"id": ..., "scores": {metric: ...}}`)
    OR the flattened score_item shape
    (`{"id": ..., metric: ...}` directly on the dict).
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for v in verdicts or []:
        rid = v.get("id")
        if rid is None:
            continue
        scores = v.get("scores")
        if isinstance(scores, dict):
            out[str(rid)] = {
                m: _coerce_score(scores.get(m)) for m in METRICS
            }
        else:
            # Flattened — each metric is a top-level key.
            out[str(rid)] = {m: _coerce_score(v.get(m)) for m in METRICS}
    return out


def agreement_report(
    opus_verdicts: Sequence[Dict[str, Any]],
    ns24_verdicts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the κ agreement report between two verdict lists.

    Per-metric: κ over binned (low/mid/high) labels, n_compared (number of
    matched, non-null pairs), and mean_abs_diff (|opus - ns24| average
    over the same non-null pairs).

    IMPORTANT: This function returns ONLY agreement statistics. It does NOT
    include a `headline_score` (or equivalent) key — ns24 verdicts are
    the AGREEMENT baseline, not the headline scorer. Headline scoring is
    Opus-only (S5b).
    """
    opus_idx = _verdict_score_lookup(opus_verdicts)
    ram_idx = _verdict_score_lookup(ns24_verdicts)
    common_ids = sorted(set(opus_idx.keys()) & set(ram_idx.keys()))

    per_metric_kappa: Dict[str, float] = {}
    n_compared: Dict[str, int] = {}
    mean_abs_diff: Dict[str, float] = {}

    for metric in METRICS:
        opus_pairs: List[Optional[float]] = []
        ram_pairs: List[Optional[float]] = []
        diffs: List[float] = []
        for rid in common_ids:
            o = opus_idx[rid].get(metric)
            r = ram_idx[rid].get(metric)
            if o is None or r is None:
                continue
            opus_pairs.append(o)
            ram_pairs.append(r)
            diffs.append(abs(o - r))

        n_compared[metric] = len(opus_pairs)
        per_metric_kappa[metric] = binned_kappa(opus_pairs, ram_pairs)
        mean_abs_diff[metric] = (sum(diffs) / len(diffs)) if diffs else 0.0

    return {
        "per_metric_kappa": per_metric_kappa,
        "n_compared": n_compared,
        "mean_abs_diff": mean_abs_diff,
        "n_matched_ids": len(common_ids),
    }


# ============================================================================
# CLI (skeleton — writes a placeholder output file; the real call-site is
# wired up by a later slice that owns the real ns24 gateway run).
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Slice S5c — score prepared batches via the ns24 gateway and write the "
            "Cohen's κ agreement report vs the Opus primary judge. The "
            "ns24 scores here are AGREEMENT BASELINE ONLY — they MUST "
            "NOT be merged into the headline aggregate."
        )
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument(
        "--batches-dir",
        required=True,
        type=pathlib.Path,
        help="Directory of batch_NNN.json (S5a output).",
    )
    parser.add_argument(
        "--opus-verdicts-dir",
        required=True,
        type=pathlib.Path,
        help="Directory of verdict_<run_id>_batch_NNN.json (S5b output).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=(
            "Override output path. Default: "
            "rag2025/results/eval_harness/judge_ns24_<run_id>.json"
        ),
    )
    args = parser.parse_args(argv)

    out_path = args.out or (RESULTS_ROOT / f"judge_ns24_{args.run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: a real invocation requires a live UnifiedLLMClient. Tests do not
    # exercise main(); a later slice wires the real client. We still write a
    # structured skeleton so the file surface is stable.
    skeleton = {
        "run_id": args.run_id,
        "batches_dir": str(args.batches_dir),
        "opus_verdicts_dir": str(args.opus_verdicts_dir),
        "out": str(out_path),
        "note": (
            "Skeleton — invoke via `await score_batch(client, batch)` and "
            "`agreement_report(...)` from a real client. This file only "
            "produces SECONDARY scores + κ agreement; it is NOT the "
            "headline scorer."
        ),
    }
    out_path.write_text(
        json.dumps(skeleton, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[eval_judge_ns24] skeleton written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
