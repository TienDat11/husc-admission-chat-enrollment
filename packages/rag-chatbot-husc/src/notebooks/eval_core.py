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
