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

import re
import unicodedata


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

