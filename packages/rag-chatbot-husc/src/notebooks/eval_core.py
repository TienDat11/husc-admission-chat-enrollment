"""eval_core - Core evaluation utilities for RAG pipeline testing.

Provides:
- load_test_questions: Load test questions with primary/fallback path support
- normalize_pipeline_output: Normalize API responses to standardized schema
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
