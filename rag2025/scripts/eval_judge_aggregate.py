"""Slice S5b — Judge verdict aggregator with mean ± 95% bootstrap CI.

Aggregates Opus-judge verdicts (typically 3 passes per batch) into per-metric
mean and 95 % bootstrap confidence interval. NO LLM. NO networking. Pure
arithmetic + grouping + seeded percentile bootstrap.

WRITEBACK CONTRACT (input verdict files this slice consumes)
-----------------------------------------------------------
Each verdict file lives at::

    rag2025/results/eval_harness/judge_verdicts/<run_id>/
        verdict_<run_id>_batch_NNN.json            (judge_pass in payload)
        verdict_<run_id>_batch_NNN_run1.json       (or pass encoded in name)
        verdict_<run_id>_batch_NNN_run2.json
        verdict_<run_id>_batch_NNN_run3.json

and matches this shape::

    {
      "batch_id": "batch_000",
      "judge_pass": 1,                       # OR derive from filename `_runN`
      "judge_model": "opus-subagent",
      "verdicts": [
        {
          "id": "msg001",
          "faithfulness":      0.83 | null,  # null on empty-context items
          "answer_relevancy":  0.91,
          "context_precision": 0.75 | null,
          "context_recall":    0.80 | null,  # null when no reference answer
          "claims": [...],
          "hallucination_types": [...],
          "rationale": "..."
        }
      ]
    }

For compatibility with the S5a rubric the aggregator also accepts the
nested shape ``{"items": [{"id": "...", "scores": {"faithfulness": ...}}]}``.

PER-ID-MEAN-THEN-BOOTSTRAP (decision + rationale)
-------------------------------------------------
For each metric M and each id i with ≥ 1 non-null score across the 3 passes,
we compute::

    μ_i = mean(non-null scores of M for id i across all passes)

We then bootstrap over the *list of per-id means*  ``[μ_i : i ∈ ids]``,
NOT over the raw per-pass list. Rationale: the 3 passes are repeated
judgments of the *same* item, not independent IID samples. Treating them
as IID would over-state n by 3× and artificially shrink the CI; the
correct unit of analysis is the item.

Nulls (e.g. faithfulness on empty-context auto-answers) are EXCLUDED from
the per-id pass mean. An id whose every pass is null for M drops out of M
entirely — its ``μ_i`` is undefined. ``faithfulness_n`` is therefore
naturally smaller than the total record count (≈ 82/86 in S2 runs because
auto-answer items have null faithfulness by contract).

CI METHOD
---------
Plain percentile bootstrap (no BCa, no studentised). Seeded
``random.Random(seed)`` (NOT ``numpy.random.default_rng``, NOT the global
``random`` module — fully reproducible and side-effect-free). For
``n_resamples`` (default 1000) draws with replacement of size ``n``, take
the resample mean, then return the 2.5 / 97.5 percentiles of those means
via linear-interpolated rank.

Public API
----------
``bootstrap_ci(values, *, n_resamples=1000, seed=42, ci=0.95) ->
        (mean, ci_lo, ci_hi)``
    Empty input → ``(None, None, None)``. Single-value input →
    ``(v, v, v)``.

``aggregate_metric(verdicts_all_passes, metric, *, seed=42) ->
        {n, mean, ci_lo, ci_hi}``
    Collects non-null values of ``metric`` across all passes, per-id-means
    them, and returns the bootstrap stats. Nulls excluded.

``aggregate(verdict_files, *, expected_ids=None, intent_by_id=None,
            seed=42) -> dict``
    Loads all verdict files / payload dicts, validates against
    ``expected_ids`` (ValueError on unmappable id), computes per-metric
    aggregates plus optional by-intent breakdown, and reports coverage.

``main(argv=None) -> int``
    CLI entrypoint: ``--run-id``, ``--verdicts-dir``, ``--records``,
    ``--out``. Writes
    ``rag2025/results/eval_harness/judge_aggregate_<run_id>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# ---------- Constants ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = RAG2025_ROOT / "results" / "eval_harness"

# The four RAGAS-style metrics this slice aggregates. Order is preserved
# in the output payload for stable downstream comparison.
METRICS: Tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# Filename suffix encoding for judge passes (e.g. `..._run1.json`).
_PASS_SUFFIX_RX = re.compile(r"_run(\d+)(?:\.json)?$", re.IGNORECASE)


__all__ = [
    "METRICS",
    "bootstrap_ci",
    "aggregate_metric",
    "aggregate",
    "main",
]


# =============================================================================
# Bootstrap CI (seeded percentile method)
# =============================================================================

def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy.percentile default).

    `sorted_values` MUST already be sorted ascending. `p` is in [0, 1].
    """
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_values[0])
    if p <= 0.0:
        return float(sorted_values[0])
    if p >= 1.0:
        return float(sorted_values[-1])
    rank = p * (n - 1)
    lo_idx = int(math.floor(rank))
    hi_idx = int(math.ceil(rank))
    if lo_idx == hi_idx:
        return float(sorted_values[lo_idx])
    frac = rank - lo_idx
    lo_v = float(sorted_values[lo_idx])
    hi_v = float(sorted_values[hi_idx])
    return lo_v + (hi_v - lo_v) * frac


def bootstrap_ci(
    values: Sequence[Optional[float]],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return ``(mean, ci_lo, ci_hi)`` via seeded percentile bootstrap.

    Args:
      values: list of floats; ``None`` entries are dropped silently so the
        caller can pass a raw column without pre-filtering.
      n_resamples: number of bootstrap draws (default 1000).
      seed: RNG seed for ``random.Random(seed)``. SAME seed + SAME values
        → byte-identical output (the determinism contract this slice tests).
      ci: confidence level, default 0.95 → uses 2.5 / 97.5 percentiles.

    Returns:
      ``(mean, ci_lo, ci_hi)``. Empty input → ``(None, None, None)``.
      Single-value input → ``(v, v, v)`` (no variance to resample).
    """
    cleaned: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        cleaned.append(f)

    n = len(cleaned)
    if n == 0:
        return (None, None, None)
    if n == 1:
        v0 = cleaned[0]
        return (v0, v0, v0)

    raw_mean = sum(cleaned) / n

    # Seeded local RNG — never touches the global `random` state.
    rng = random.Random(seed)
    resample_means: List[float] = [0.0] * n_resamples
    for i in range(n_resamples):
        s = 0.0
        # Sample with replacement of size n.
        for _ in range(n):
            s += cleaned[rng.randrange(n)]
        resample_means[i] = s / n
    resample_means.sort()

    alpha = (1.0 - ci) / 2.0
    lo = _percentile(resample_means, alpha)
    hi = _percentile(resample_means, 1.0 - alpha)
    return (raw_mean, lo, hi)


# =============================================================================
# Verdict normalisation (handles both writeback shapes + both pass-encoding paths)
# =============================================================================

def _pass_from_filename(path: pathlib.Path) -> Optional[int]:
    """Return ``N`` from a filename matching ``..._runN[.json]``, else None."""
    m = _PASS_SUFFIX_RX.search(path.name)
    if not m:
        # Also accept run number that appears BEFORE `.json` somewhere in the stem.
        m = _PASS_SUFFIX_RX.search(str(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _resolve_pass(
    path_hint: Optional[pathlib.Path],
    payload: Dict[str, Any],
) -> int:
    """Resolve the judge-pass index from payload first, then filename, default 1."""
    jp = payload.get("judge_pass")
    if isinstance(jp, bool):  # `True`/`False` is technically int — reject explicitly.
        jp = None
    if isinstance(jp, (int, float)):
        try:
            return int(jp)
        except (TypeError, ValueError):
            pass
    if isinstance(path_hint, pathlib.Path):
        n = _pass_from_filename(path_hint)
        if n is not None:
            return n
    return 1


def _flatten_verdict(v: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict with the four metrics at top level.

    Accepts both:
      * S5b flat shape: ``{"id": "...", "faithfulness": 0.83, ...}``
      * S5a nested shape: ``{"id": "...", "scores": {"faithfulness": 0.83, ...}}``
    """
    if not isinstance(v, dict):
        return {}
    scores = v.get("scores")
    if isinstance(scores, dict):
        merged: Dict[str, Any] = dict(v)
        for k in METRICS:
            if k in scores:
                merged[k] = scores[k]
        return merged
    return dict(v)


def _normalize_array(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the per-item array from either ``verdicts`` or ``items`` key."""
    arr = payload.get("verdicts")
    if arr is None:
        arr = payload.get("items")
    if not isinstance(arr, list):
        return []
    return [_flatten_verdict(x) for x in arr if isinstance(x, dict)]


def _load_payload(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"verdict file is not a JSON object: {path}")
    return data


def _collect(
    verdict_files: Sequence[Union[pathlib.Path, str, Dict[str, Any]]],
) -> List[Tuple[int, Dict[str, Any]]]:
    """Load all verdicts, return ``[(judge_pass, flat_verdict), ...]``.

    Accepts a mix of:
      * ``pathlib.Path`` / ``str`` — read JSON from disk.
      * ``dict`` — treat as a pre-loaded batch payload (handy for tests).
    """
    out: List[Tuple[int, Dict[str, Any]]] = []
    for entry in verdict_files:
        if isinstance(entry, dict):
            payload = entry
            jp = _resolve_pass(None, payload)
        else:
            p = pathlib.Path(entry)
            payload = _load_payload(p)
            jp = _resolve_pass(p, payload)
        for v in _normalize_array(payload):
            out.append((jp, v))
    return out


# =============================================================================
# Per-id grouping (the "mean across passes per id" step)
# =============================================================================

def _per_id_means(
    flat_verdicts: Sequence[Tuple[int, Dict[str, Any]]],
    metric: str,
) -> Dict[str, float]:
    """Group by id; per-id, average non-null `metric` values across passes."""
    by_id: Dict[str, List[float]] = {}
    for _jp, v in flat_verdicts:
        rid = v.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        x = v.get(metric)
        if x is None:
            continue
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if math.isnan(xf):
            continue
        by_id.setdefault(rid, []).append(xf)
    return {rid: sum(vs) / len(vs) for rid, vs in by_id.items() if vs}


def _normalize_input(
    verdicts_all_passes: Union[
        Sequence[Dict[str, Any]],
        Sequence[Tuple[int, Dict[str, Any]]],
    ],
) -> List[Tuple[int, Dict[str, Any]]]:
    """Accept three input shapes, return the canonical [(pass, flat-verdict), ...].

      * batch payloads (dicts with ``verdicts``/``items`` key)
      * single-verdict dicts (treated as pass=judge_pass field, default 1)
      * pre-tagged tuples ``(pass, flat_verdict)``

    This is the public-facing helper that lets ``aggregate_metric`` be
    called with a list of verdict dicts in tests without wrapping them.
    """
    out: List[Tuple[int, Dict[str, Any]]] = []
    for entry in verdicts_all_passes:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
            try:
                out.append((int(entry[0]), _flatten_verdict(entry[1])))
            except (TypeError, ValueError):
                out.append((1, _flatten_verdict(entry[1])))
            continue
        if isinstance(entry, dict):
            # Two cases: a batch payload with `verdicts`/`items`, OR a single
            # already-flat verdict dict.
            if "verdicts" in entry or "items" in entry:
                jp = _resolve_pass(None, entry)
                for v in _normalize_array(entry):
                    out.append((jp, v))
            else:
                jp_raw = entry.get("judge_pass", 1)
                try:
                    jp = int(jp_raw) if not isinstance(jp_raw, bool) else 1
                except (TypeError, ValueError):
                    jp = 1
                out.append((jp, _flatten_verdict(entry)))
    return out


# =============================================================================
# Public: aggregate_metric (per-metric stats)
# =============================================================================

def aggregate_metric(
    verdicts_all_passes: Union[
        Sequence[Dict[str, Any]],
        Sequence[Tuple[int, Dict[str, Any]]],
    ],
    metric: str,
    *,
    seed: int = 42,
    n_resamples: int = 1000,
) -> Dict[str, Any]:
    """Aggregate one metric across all verdicts → ``{n, mean, ci_lo, ci_hi}``.

    Steps:
      1. Normalise input to ``[(pass, flat_verdict), ...]``.
      2. Group by id; per id, average non-null `metric` across passes.
      3. Bootstrap the LIST of per-id means.

    ``n`` is the count of UNIQUE ids that had at least one non-null value
    (the bootstrap sample size — NOT the raw verdict count).
    """
    normalized = _normalize_input(verdicts_all_passes)
    per_id = _per_id_means(normalized, metric)
    # Sort by id for deterministic resampling order (same seed → same draws).
    ids_sorted = sorted(per_id.keys())
    values: List[float] = [per_id[i] for i in ids_sorted]
    mean, lo, hi = bootstrap_ci(
        values, n_resamples=n_resamples, seed=seed
    )
    return {"n": len(values), "mean": mean, "ci_lo": lo, "ci_hi": hi}


# =============================================================================
# Public: aggregate (top-level — coverage + unmappable-id validation)
# =============================================================================

def aggregate(
    verdict_files: Sequence[Union[pathlib.Path, str, Dict[str, Any]]],
    *,
    expected_ids: Optional[Iterable[str]] = None,
    intent_by_id: Optional[Dict[str, str]] = None,
    seed: int = 42,
    n_resamples: int = 1000,
) -> Dict[str, Any]:
    """Aggregate verdicts into per-metric mean + 95 % CI, with coverage.

    Args:
      verdict_files: a list of file paths AND/OR pre-loaded batch payloads.
      expected_ids: if given, every verdict id MUST be a member — else
        ``ValueError`` is raised naming the unmappable id. The set also
        drives the coverage report.
      intent_by_id: optional ``{id: intent_string}`` mapping. When given,
        the output also contains a ``by_intent`` block with the same
        per-metric stats restricted to each intent.
      seed / n_resamples: passed through to ``bootstrap_ci``.

    Returns a JSON-serialisable dict::

      {
        "metrics": {<metric>: {n, mean, ci_lo, ci_hi}, ...},
        "faithfulness_n": int,
        "coverage": {"expected": int, "covered": int, "missing": [...]} | absent,
        "by_intent": {intent: {<metric>: {...}}} | absent
      }
    """
    expected_set: Optional[Set[str]] = (
        {str(x) for x in expected_ids} if expected_ids is not None else None
    )

    flat = _collect(verdict_files)

    # Hard-fail on unmappable ids (the writeback contract requires every
    # verdict id to be a known record id).
    if expected_set is not None:
        for _jp, v in flat:
            rid = v.get("id")
            if isinstance(rid, str) and rid not in expected_set:
                raise ValueError(
                    f"unmappable verdict id: {rid!r} (not in expected_ids)"
                )

    # Per-metric overall aggregates.
    overall: Dict[str, Dict[str, Any]] = {}
    for m in METRICS:
        overall[m] = aggregate_metric(
            flat, m, seed=seed, n_resamples=n_resamples
        )

    out: Dict[str, Any] = {
        "metrics": overall,
        # Surface faithfulness_n separately — the spec calls it out
        # because auto-answer items have null faithfulness, so it diverges
        # from the n of the other three metrics.
        "faithfulness_n": overall["faithfulness"]["n"],
    }

    # Coverage (only meaningful when expected_ids was given).
    if expected_set is not None:
        observed_ids: Set[str] = {
            v.get("id") for _jp, v in flat
            if isinstance(v.get("id"), str)
        }
        covered = observed_ids & expected_set
        missing = sorted(expected_set - covered)
        out["coverage"] = {
            "expected": len(expected_set),
            "covered": len(covered),
            "missing": missing,
        }

    # By-intent breakdown.
    if intent_by_id:
        by_intent: Dict[str, Dict[str, Dict[str, Any]]] = {}
        intents_seen: Set[str] = set()
        for _jp, v in flat:
            rid = v.get("id")
            if isinstance(rid, str) and rid in intent_by_id:
                intents_seen.add(intent_by_id[rid])
        for intent in sorted(intents_seen):
            subset: List[Tuple[int, Dict[str, Any]]] = [
                (jp, v)
                for jp, v in flat
                if intent_by_id.get(v.get("id")) == intent
            ]
            by_intent[intent] = {
                m: aggregate_metric(
                    subset, m, seed=seed, n_resamples=n_resamples
                )
                for m in METRICS
            }
        out["by_intent"] = by_intent

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


def _discover_verdict_files(
    verdicts_dir: pathlib.Path,
    run_id: str,
) -> List[pathlib.Path]:
    """Discover ``verdict_<run_id>_*.json`` files under ``verdicts_dir``.

    Falls back to ``verdict_*.json`` if no run-id-prefixed files are found.
    Returns a sorted list (deterministic ingestion order).
    """
    if not verdicts_dir.is_dir():
        return []
    prefix = f"verdict_{run_id}_"
    primary = sorted(
        p for p in verdicts_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name.startswith(prefix)
    )
    if primary:
        return primary
    return sorted(
        p for p in verdicts_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name.startswith("verdict_")
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: aggregate judge verdicts into ``judge_aggregate_<run_id>.json``."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Opus-judge verdicts into per-metric mean ± 95 % "
            "bootstrap CI. Reads judge_verdicts/<run_id>/*.json."
        ),
    )
    parser.add_argument(
        "--run-id", required=True, help="Unique run identifier (matches verdict filenames).",
    )
    parser.add_argument(
        "--verdicts-dir", required=True, type=pathlib.Path,
        help="Directory containing verdict_<run_id>_batch_NNN[_runN].json files.",
    )
    parser.add_argument(
        "--records", required=False, type=pathlib.Path, default=None,
        help=(
            "Optional EvalRecord JSONL. When provided, every verdict id "
            "MUST be a record id (else ValueError) and the coverage report "
            "is emitted."
        ),
    )
    parser.add_argument(
        "--out", required=False, type=pathlib.Path, default=None,
        help=(
            "Override output path "
            "(default: rag2025/results/eval_harness/judge_aggregate_<run_id>.json)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap seed (default 42 — must be deterministic).",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=1000,
        help="Bootstrap resample count (default 1000).",
    )
    args = parser.parse_args(argv)

    expected_ids: Optional[Set[str]] = None
    if args.records is not None:
        recs = _read_jsonl(args.records)
        expected_ids = {
            str(r["id"]) for r in recs
            if isinstance(r, dict) and "id" in r
        }

    verdict_files = _discover_verdict_files(args.verdicts_dir, args.run_id)
    result = aggregate(
        verdict_files,
        expected_ids=expected_ids,
        seed=args.seed,
        n_resamples=args.n_resamples,
    )
    result["run_id"] = args.run_id

    out_path = args.out or (
        RESULTS_ROOT / f"judge_aggregate_{args.run_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[eval_judge_aggregate] run_id={args.run_id} "
        f"files={len(verdict_files)} out={out_path}"
    )
    for m in METRICS:
        d = result["metrics"][m]
        print(
            f"  {m}: n={d['n']} mean={d['mean']} "
            f"ci=[{d['ci_lo']}, {d['ci_hi']}]"
        )
    if "coverage" in result:
        c = result["coverage"]
        print(
            f"  coverage: {c['covered']}/{c['expected']} "
            f"missing={len(c['missing'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
