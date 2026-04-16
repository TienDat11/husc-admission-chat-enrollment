"""eval_core - Core evaluation utilities for RAG pipeline testing.

Provides:
- load_test_questions: Load test questions with primary/fallback path support
- normalize_pipeline_output: Normalize API responses to standardized schema
- should_abort_after_smoke: Fail-fast rule for smoke test failure rate
- call_pipeline: Call the RAG pipeline API endpoint
"""
from __future__ import annotations

import json
import requests
from pathlib import Path
from typing import Any


class PipelineError(Exception):
    """Raised when call_pipeline fails due to connection or response errors."""
def load_test_questions(primary_path: str, fallback_path: str) -> tuple[list[dict[str, Any]], str]:
    """Load test questions from primary path, falling back to fallback_path on failure.

    Args:
        primary_path: Primary path to try loading JSON from.
        fallback_path: Fallback path used if primary fails.

    Returns:
        Tuple of (rows, used_path) where rows is the parsed JSON list and
        used_path is the path that was successfully loaded from.

    Raises:
        FileNotFoundError: If both primary_path and fallback_path fail to load.
    """
    # Try primary path first
    primary_p = Path(primary_path)
    if primary_p.exists():
        try:
            data = primary_p.read_text(encoding="utf-8")
            rows = json.loads(data)
            if isinstance(rows, list):
                return rows, str(primary_p.resolve())
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to fallback

    # Try fallback path
    fallback_p = Path(fallback_path)
    if fallback_p.exists():
        try:
            data = fallback_p.read_text(encoding="utf-8")
            rows = json.loads(data)
            if isinstance(rows, list):
                return rows, str(fallback_p.resolve())
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to error

    # Both failed
    raise FileNotFoundError(
        f"Could not load test questions from either primary path '{primary_path}' "
        f"or fallback path '{fallback_path}'. Both paths either missing or contained invalid JSON."
    )


def normalize_pipeline_output(raw: dict[str, Any], mode: str = "v2") -> dict[str, Any]:
    """Normalize API response to standardized output schema.

    Args:
        raw: The raw API response dictionary.
        mode: 'v2' (maps sources->source_ids, chunks->context_chunks) or
              'v1' (maps chunks->context_chunks only).

    Returns:
        Normalized dictionary with standardized keys:
        - answer: The answer string
        - context_chunks: List of chunk dicts with 'text' key
        - source_ids: List of source identifiers
        - confidence: Confidence score
        - groundedness_score: Score for answer groundedness
        - route: Routing label
        - raw: Original raw input
    """
    if mode == "v2":
        # v2 mode: sources -> source_ids, chunks -> context_chunks
        source_ids: list = []
        if "sources" in raw:
            sources_val = raw["sources"]
            if isinstance(sources_val, list):
                source_ids = sources_val

        context_chunks: list = []
        if "chunks" in raw:
            chunks_val = raw["chunks"]
            if isinstance(chunks_val, list):
                context_chunks = chunks_val

    else:
        # v1 mode: chunks -> context_chunks only, no source_ids mapping
        context_chunks = []
        if "chunks" in raw:
            chunks_val = raw["chunks"]
            if isinstance(chunks_val, list):
                context_chunks = chunks_val
        source_ids = []

    # Extract groundedness_score, default to 0.0 if not present
    groundedness_score: float = 0.0
    if "groundedness_score" in raw:
        groundedness_score = float(raw["groundedness_score"])

    # Extract route, default to empty string if not present
    route: str = ""
    if "route" in raw:
        route = str(raw["route"])

    return {
        "answer": raw.get("answer", ""),
        "context_chunks": context_chunks,
        "source_ids": source_ids,
        "confidence": raw.get("confidence", 0.0),
        "groundedness_score": groundedness_score,
        "route": route,
        "raw": raw,
    }

def should_abort_after_smoke(total: int, failures: int) -> bool:
    """Return True if smoke test suite should abort.

    Abort conditions:
    - failures/total > 0.5  (more than 50% failure rate)
    - total <= 0           (no tests run)

    Args:
        total: Total number of tests run.
        failures: Number of test failures.

    Returns:
        True if the suite should abort, False otherwise.
    """
    if total <= 0:
        return True
    return failures / total > 0.5


def call_pipeline(base_url: str, query: str, mode: str = "v2", top_k: int = 5) -> dict[str, Any]:
    """Call the RAG pipeline API endpoint.

    Args:
        base_url: Base URL of the pipeline server (e.g. "http://localhost:8000").
        query: Query string to send to the pipeline.
        mode: "v1" or "v2". v1 calls /query, v2 calls /v2/query.
        top_k: Number of top results to return (v2 mode only). Defaults to 5.

    Returns:
        JSON response from the pipeline as a dictionary.

    Raises:
        requests.HTTPError: If the response status is an error (4xx/5xx).
    """
    timeout = 120
    if mode == "v1":
        url = f"{base_url}/query"
        payload = {"query": query, "force_rag_only": False}
    else:
        url = f"{base_url}/v2/query"
        payload = {"query": query, "top_k": top_k}

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as e:
        raise PipelineError(
            f"Failed to connect to pipeline at {url}: {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise PipelineError(
            f"Pipeline at {url} returned non-JSON response: {e}"
        ) from e
# New functions to append to eval_core.py


# =============================================================================
# Text normalization
# =============================================================================

import hashlib
import platform
import datetime
import numpy as np
import re
import unicodedata
from typing import Any, Sequence


def normalize_text(s: str) -> str:
    """Lowercase, strip, collapse whitespace; remove punctuation [.,;:!?-()[]{}"']; keep underscore."""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Remove punctuation: period, comma, semicolon, colon, exclamation, question,
    # hyphen/dash, parentheses, brackets, braces, double and single quotes
    # Keep underscore
    for ch in ['.', ',', ';', ':', '!', '?', '-', '(', ')', '[', ']', '{', '}', '"', "'", '_']:
        s = s.replace(ch, "")
    return s


def exact_correctness(pred: str, gt: str) -> int:
    """Return 1 if normalized pred equals normalized gt, else 0."""
    return 1 if normalize_text(pred) == normalize_text(gt) else 0


# =============================================================================
# Retrieval recall proxy
# =============================================================================

from typing import Sequence


def retrieval_recall_proxy(source_ids: Sequence[str], gt_chunks: Sequence[str]) -> int:
    """Return 1 if source_ids and gt_chunks share at least one element and gt_chunks is non-empty.

    Returns 0 if gt_chunks is empty or if there is no overlap.
    """
    if not gt_chunks:
        return 0
    if not source_ids:
        return 0
    return 1 if set(source_ids).intersection(set(gt_chunks)) else 0


# =============================================================================
# Hallucination flag
# =============================================================================

def hallucination_flag(groundedness_score: float, threshold: float = 0.18) -> int:
    """Return 1 if groundedness_score < threshold, else 0."""
    return 1 if groundedness_score < threshold else 0


# =============================================================================
# Diagnostic report
# =============================================================================

def build_diagnostic_report(result: dict[str, Any]) -> str:
    """Build a markdown diagnostic report from eval results.

    Args:
        result: Dict with keys:
            - summary: dict with "accuracy", "avg_recall", "hallucination_rate" floats
            - per_category: dict with "simple" and "multihop" sub-dicts
            - errors: list of dicts with failed eval items (each may have error_type, score_exact, recall, hallucination, groundedness_score)

    Returns:
        Markdown string with report.
    """
    lines = ["# Diagnostic Report\n"]

    summary = result.get("summary", {})
    accuracy = summary.get("accuracy", 0.0)
    hallucination_rate = summary.get("hallucination_rate", 0.0)
    avg_recall = summary.get("avg_recall", 0.0)

    lines.append(f"**Overall Accuracy:** {accuracy:.2%}  ")
    lines.append(f"**Avg Recall:** {avg_recall:.2%}  ")
    lines.append(f"**Hallucination Rate:** {hallucination_rate:.2%}\n")

    # Per-category breakdown
    per_cat = result.get("per_category", {})
    if per_cat:
        lines.append("## Per-Category Breakdown\n")
        lines.append("| Category | Count | Accuracy | Recall | Hallucination Rate |")
        lines.append("|----------|-------|----------|--------|--------------------|")
        for cat in ["simple", "multihop"]:
            if cat in per_cat:
                c = per_cat[cat]
                count = c.get("count", 0)
                acc = c.get("accuracy", 0.0)
                rec = c.get("recall", 0.0)
                hall = c.get("hallucination_rate", 0.0)
                lines.append(f"| {cat} | {count} | {acc:.2%} | {rec:.2%} | {hall:.2%} |")
        lines.append("")

    # Top errors by type
    errors = result.get("errors", [])
    if errors:
        lines.append("## Top Errors by Type\n")
        error_groups = {}
        for err in errors:
            etype = err.get("error_type", "unknown")
            if etype not in error_groups:
                error_groups[etype] = []
            error_groups[etype].append(err)

        for etype, errs in error_groups.items():
            lines.append(f"### {etype.capitalize()} ({len(errs)})\n")
            for err in errs[:3]:
                q = err.get("question", "")[:80]
                a = str(err.get("answer", ""))[:80]
                lines.append(f"- **Q:** {q}...")
                lines.append(f"  **A:** {a}")
                if etype == "retrieval":
                    lines.append(f"  recall={err.get('recall', '?')}, gt_chunks={err.get('gt_chunks', [])}")
                elif etype == "hallucination":
                    lines.append(f"  groundedness={err.get('groundedness_score', '?')}")
                elif etype == "reasoning":
                    lines.append(f"  exact={err.get('score_exact', '?')}")
                elif etype == "format":
                    lines.append(f"  answer={err.get('answer', '')[:40]}")
                lines.append("")

        # Example bad questions
        lines.append("## Example Bad Questions\n")
        for err in errors[:5]:
            q = err.get("question", "")[:80]
            a = str(err.get("answer", ""))[:80]
            score_exact = err.get("score_exact", 0)
            recall = err.get("recall", 0)
            hall = err.get("hallucination", 0)
            lines.append(f"- **[acc={score_exact} recall={recall} hall={hall}]** Q: {q}...")
            lines.append(f"  A: {a}\n")
    else:
        lines.append("## Errors\n_No errors detected._\n")

    return "\n".join(lines)


# =============================================================================
# Constants
# =============================================================================

GROUNDING_THRESHOLD = 0.18  # hallucination flag threshold

# =============================================================================
# Latency stats
# =============================================================================

def latency_stats(times: list[float]) -> dict[str, float]:
    """Return median and p95 from a list of elapsed times in seconds.

    Args:
        times: List of elapsed times in seconds.
    Returns:
        dict with 'median' and 'p95' keys.
    """
    if not times:
        return {"median": 0.0, "p95": 0.0}
    sorted_times = sorted(times)
    n = len(sorted_times)
    median = sorted_times[(n - 1) // 2]
    p95_idx = int((n - 1) * 0.95)
    p95 = sorted_times[min(p95_idx, n - 1)]
    return {"median": median, "p95": p95}


# =============================================================================
# Tests
# =============================================================================

def test_latency_stats_median_and_p95():
    times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95, 1.0]
    stats = latency_stats(times)
    assert stats["median"] == 0.5
    assert stats["p95"] == 0.95
    assert latency_stats([]) == {"median": 0.0, "p95": 0.0}


# =============================================================================
# Decision Table (US4)
# =============================================================================

def get_decision_table() -> dict[str, dict[str, Any]]:
    """Return pre-registered decision table. Locked before eval runs.

    Margins are locked pre-run. Post-hoc adjustment = PROTOCOL_VIOLATION.
    """
    return {
        "accuracy":           {"gate_type": "must-pass",     "ni_margin": -0.01, "rationale": "Core quality — below this = user-visible regression"},
        "groundedness":       {"gate_type": "must-pass",     "ni_margin": -0.02, "rationale": "Faithfulness — below = hallucination risk"},
        "hallucination_rate": {"gate_type": "must-pass",     "ni_margin":  0.02, "rationale": "Safety — above = trust erosion"},
        "recall":            {"gate_type": "must-pass",     "ni_margin": -0.015,"rationale": "Retrieval coverage — below = missing facts"},
        "latency_p95":       {"gate_type": "support-only",  "ni_margin":  0.20, "rationale": "Performance — degradation noted but doesn't block"},
        "partial_credit":    {"gate_type": "support-only",  "ni_margin": -0.05, "rationale": "Informational — tracks nuance"},
    }


# =============================================================================
# Bootstrap CI95 (US4)
# =============================================================================

def bootstrap_ci95(values: list[float], n_bootstrap: int = 2000) -> tuple[float, float]:
    """Compute 95% bootstrap CI for a list of metric values.

    Args:
        values: list of binary correctness scores (0/1) or float ratios.
        n_bootstrap: number of bootstrap resamples.

    Returns:
        (ci_low, ci_high) tuple.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(np.mean(sample)))
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))
    return (ci_low, ci_high)


# =============================================================================
# Non-Inferiority Test (US4)
# =============================================================================

def non_inferiority_test(
    control: float, candidate: float, ni_margin: float, higher_is_better: bool = True
) -> tuple[bool, float]:
    """Return (passes, margin_delta) for NI test.

    Args:
        control: control arm metric value (0-1 scale for ratios).
        candidate: candidate arm metric value.
        ni_margin: NI margin (positive = acceptable loss).
        higher_is_better: if False, lower is better (e.g., hallucination_rate, latency).

    Returns:
        (passes_ni, margin_delta) where margin_delta = candidate - control.
    """
    margin_delta = candidate - control
    if higher_is_better:
        passes = margin_delta >= -ni_margin
    else:
        passes = margin_delta <= ni_margin
    return (passes, margin_delta)


# =============================================================================
# Route Parity (US2)
# =============================================================================

def compute_route_parity(
    controlled_results: list[dict[str, Any]],
    auto_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute route mismatch between controlled and auto-route lanes.

    Args:
        controlled_results: list of dicts with at least "question" and "route" keys from force_route call.
        auto_results: list of dicts with at least "question" and "route" keys from auto-route call.

    Returns:
        dict with global_mismatch_pct, per_slice_mismatches, total, mismatches.
    """
    if len(controlled_results) != len(auto_results):
        raise ValueError("Mismatched result lengths")
    total = len(controlled_results)
    mismatches = 0
    per_slice = []
    for c, a in zip(controlled_results, auto_results):
        c_route = c.get("route", "")
        a_route = a.get("route", "")
        if c_route != a_route:
            mismatches += 1
            per_slice.append({"question": c.get("question", ""), "controlled_route": c_route, "auto_route": a_route})
    global_pct = (mismatches / total * 100) if total > 0 else 0.0
    return {
        "global_mismatch_pct": global_pct,
        "global_threshold": 1.0,
        "global_pass": global_pct <= 1.0,
        "per_slice_mismatches": per_slice,
        "per_slice_threshold": 2.0,
        "total": total,
        "mismatches": mismatches,
    }


# =============================================================================
# Rerun Stability (US5)
# =============================================================================

def rerun_stability(
    seed_metrics: dict[str, list[float]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Multi-metric stability across seeds.

    Args:
        seed_metrics: dict of metric_name -> list of scalar values per seed.
                      e.g., {"accuracy": [0.82, 0.819, 0.821], "groundedness": [0.91, 0.909, 0.911]}
        thresholds: dict of metric -> max stddev in fraction. Defaults:
                    accuracy=0.005, groundedness=0.007, recall=0.005, hallucination_rate=0.005

    Returns:
        dict with "stable" (bool), "metrics" (dict per metric).
    """
    if thresholds is None:
        thresholds = {"accuracy": 0.005, "groundedness": 0.007, "recall": 0.005, "hallucination_rate": 0.005}
    all_stable = True
    metrics_result = {}
    for metric, values in seed_metrics.items():
        thr = thresholds.get(metric, 0.005)
        if len(values) < 3:
            metrics_result[metric] = {"stable": False, "stddev": None, "threshold": thr, "values": values, "reason": "min 3 reruns"}
            all_stable = False
            continue
        stddev = float(np.std(values, ddof=1))
        is_stable = stddev <= thr
        metrics_result[metric] = {"stable": is_stable, "stddev": stddev, "threshold": thr, "values": values}
        if not is_stable:
            all_stable = False
    return {"stable": all_stable, "metrics": metrics_result}


# =============================================================================
# Evidence Map (US5, US7)
# =============================================================================

def build_evidence_map(
    git_sha: str,
    dataset_hash: str,
    config_snapshot: dict[str, Any],
    n_queries: int,
    run_ids: list[str],
) -> dict[str, Any]:
    """Build evidence map pinning runtime context.

    Args:
        git_sha: current git commit SHA.
        dataset_hash: hash of test_questions.json used.
        config_snapshot: dict with embedding_model, force_route settings, etc.
        n_queries: number of queries evaluated.
        run_ids: list of run_ids in this eval session.

    Returns:
        dict with all evidence fields plus computed checksums.
    """
    runtime_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    config_str = json.dumps(config_snapshot, sort_keys=True)
    evidence_str = f"{git_sha}|{dataset_hash}|{config_str}"
    checksum = hashlib.sha256(evidence_str.encode()).hexdigest()[:16]

    return {
        "git_sha": git_sha,
        "dataset_hash": dataset_hash,
        "config_snapshot": config_snapshot,
        "runtime_info": runtime_info,
        "n_queries": n_queries,
        "run_ids": run_ids,
        "evidence_checksum": checksum,
    }


# =============================================================================
# Matrix Completeness (US1)
# =============================================================================

def validate_matrix_completeness(run_manifests: list[dict[str, Any]]) -> None:
    """Validate all 4 required matrix combos are present. Raise RuntimeError if any missing.

    Required combos: (BGE, padded_rag), (BGE, graph_rag), (Harrier, padded_rag), (Harrier, graph_rag)

    Each manifest dict must have "embedding_model" and "force_route" keys.
    """
    REQUIRED = {
        ("BGE", "padded_rag"), ("BGE", "graph_rag"),
        ("Harrier", "padded_rag"), ("Harrier", "graph_rag"),
    }
    found: set[tuple[str, str]] = set()
    for m in run_manifests:
        found.add((m.get("embedding_model", ""), m.get("force_route", "")))
    missing = REQUIRED - found
    if missing:
        missing_str = ", ".join(f"({em}, {fr})" for em, fr in sorted(missing))
        raise RuntimeError(f"MATRIX_INCOMPLETE: missing combos: {missing_str}. FAIL-FAST triggered.")


# =============================================================================
# Enriched Diagnostic Report (US6)
# =============================================================================

def build_enriched_diagnostic_report(result: dict[str, Any]) -> str:
    """Build an enriched markdown diagnostic report from eval results.

    Args:
        result: Dict with keys:
            - summary: dict with overall metrics
            - per_route: dict with padded_rag/graph_rag sub-dicts
            - per_embedding: dict with BGE/Harrier sub-dicts
            - cross_table: list of dicts with category x route x embedding results
            - errors: list of failed eval items
            - parity: dict with route parity results
            - gate_decision: dict with PASS/FAIL and reasons

    Returns:
        Markdown string with full report.
    """
    lines = ["# Diagnostic Report (v4)\n"]

    # ---- Executive Summary ----
    summary = result.get("summary", {})
    lines.append("## Executive Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---------|-------|")
    for key, val in summary.items():
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.2%} |")
        else:
            lines.append(f"| {key} | {val} |")
    lines.append("")

    # ---- Gate Decision ----
    gate = result.get("gate_decision", {})
    decision = gate.get("decision", "UNKNOWN")
    lines.append(f"## Gate Decision: **{decision}**\n")
    reasons = gate.get("reasons", [])
    if reasons:
        lines.append("| Check | Result |")
        lines.append("|-------|--------|")
        for r in reasons:
            lines.append(f"| {r.get('check','')} | {r.get('result','')} |")
    lines.append("")

    # ---- Per-Route Breakdown ----
    per_route = result.get("per_route", {})
    if per_route:
        lines.append("## Per-Route Breakdown\n")
        lines.append("| Route | Accuracy | Groundedness | Hallucination Rate | Recall | Latency p95 |")
        lines.append("|-------|----------|--------------|-------------------|--------|-------------|")
        for route, data in per_route.items():
            acc = data.get("accuracy", 0.0)
            grounded = data.get("groundedness", 0.0)
            hall = data.get("hallucination_rate", 0.0)
            rec = data.get("recall", 0.0)
            lat = data.get("latency_p95", 0.0)
            lines.append(f"| {route} | {acc:.2%} | {grounded:.2%} | {hall:.2%} | {rec:.2%} | {lat:.3f}s |")
        lines.append("")

    # ---- Per-Embedding Breakdown ----
    per_emb = result.get("per_embedding", {})
    if per_emb:
        lines.append("## Per-Embedding Breakdown\n")
        lines.append("| Embedding | Accuracy | Groundedness | Hallucination Rate | Recall |")
        lines.append("|-----------|----------|--------------|-------------------|--------|")
        for emb, data in per_emb.items():
            acc = data.get("accuracy", 0.0)
            grounded = data.get("groundedness", 0.0)
            hall = data.get("hallucination_rate", 0.0)
            rec = data.get("recall", 0.0)
            lines.append(f"| {emb} | {acc:.2%} | {grounded:.2%} | {hall:.2%} | {rec:.2%} |")
        lines.append("")

    # ---- Cross-Table ----
    cross = result.get("cross_table", [])
    if cross:
        lines.append("## Category × Route × Embedding Cross-Table\n")
        lines.append("| Category | Route | Embedding | Count | Accuracy | Recall |")
        lines.append("|----------|-------|-----------|-------|----------|--------|")
        for row in cross:
            lines.append(f"| {row.get('category','')} | {row.get('route','')} | {row.get('embedding','')} "
                        f"| {row.get('count',0)} | {row.get('accuracy',0.0):.2%} | {row.get('recall',0.0):.2%} |")
        lines.append("")

    # ---- Route Parity ----
    parity = result.get("parity", {})
    if parity:
        lines.append("## Route Parity Checklist\n")
        gp = parity.get("global_mismatch_pct", 0.0)
        gpass = "✅ PASS" if parity.get("global_pass", False) else "❌ FAIL"
        lines.append(f"- Global mismatch: {gp:.2f}% — {gpass} (threshold ≤ 1.0%)")
        lines.append(f"- Total queries: {parity.get('total', 0)}")
        lines.append(f"- Mismatches: {parity.get('mismatches', 0)}")
        if parity.get("per_slice_mismatches"):
            lines.append(f"- **First mismatch examples:**")
            for ms in parity["per_slice_mismatches"][:3]:
                lines.append(f"  - Q: {ms.get('question','')[:60]}... → controlled:{ms.get('controlled_route')} auto:{ms.get('auto_route')}")
        lines.append("")

    # ---- Top 5 Worst Queries ----
    errors = result.get("errors", [])
    if errors:
        lines.append("## Top 5 Worst Queries\n")
        lines.append("| # | Question | Answer | Score |")
        lines.append("|---|----------|--------|-------|")
        for i, err in enumerate(errors[:5], 1):
            q = err.get("question", "")[:50]
            a = str(err.get("answer", ""))[:40]
            score = err.get("score_exact", 0)
            lines.append(f"| {i} | {q}... | {a} | {score} |")
        lines.append("")

    # ---- Roadmap ----
    lines.append("## Roadmap\n")
    lines.append("### Quick Wins (1–3 days)\n")
    lines.append("- Review top-5 worst queries for retrieval pattern gaps\n")
    lines.append("- Tune groundedness threshold if hallucination false-positives high\n")
    lines.append("- Adjust top_k if recall is low for multihop\n")
    lines.append("")
    lines.append("### Medium (1–2 weeks)\n")
    lines.append("- Investigate multihop breakdown for reasoning chain breaks\n")
    lines.append("- Compare BGE vs Harrier per category to pick winner\n")
    lines.append("- Add hybrid retrieval tuning if recall < 0.80\n")
    lines.append("")
    lines.append("### Long-term (2–6 weeks)\n")
    lines.append("- Hard negative mining for multihop categories\n")
    lines.append("- Graph reasoning layer improvements\n")
    lines.append("- Benchmark expansion with additional test sets\n")
    lines.append("")

    return "\n".join(lines)

