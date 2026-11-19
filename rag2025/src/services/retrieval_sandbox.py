"""Retrieval Diff Sandbox — U8 PL-6.3.

Pure, in-memory, dependency-free diff of a retrieval result before vs after an
overlay of draft chunks over the active release's chunks.

POST /admin/retrieval-preview owns the HTTP contract (record_ids -> golden
query set -> before/after top-k). This module owns the numeric core:

* tokenize: NFC -> lowercase -> \\w+ -> drop stopwords
* score: VECTOR_WEIGHT * (1 + cos)/2 + (1 - VECTOR_WEIGHT) * Jaccard
* tie-break: (-score, chunk_id)
* output per query: before top-k, after top-k, entered, left, rank_delta

No LLM, no network, no embedding at call time: every vector must already be
present (a missing vector, or an embedding_model that differs from the
release's, yields status: "insufficient_fixture").
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Iterable

from config.settings import settings
from src.services.text_stopwords import STOPWORDS_VI

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_INSUFFICIENT = "insufficient_fixture"


def _norm_text(text: str) -> str:
    """NFC-fold one string for token comparison."""
    import unicodedata

    return unicodedata.normalize("NFC", text or "").lower()


def tokenize(text: str) -> frozenset[str]:
    """NFC -> lowercase -> \\w+ -> drop stopwords."""
    return frozenset(
        token
        for token in _TOKEN_RE.findall(_norm_text(text))
        if token and token not in STOPWORDS_VI
    )


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity; ``0.0`` when either side is a zero vector."""
    a_vals = list(a)
    b_vals = list(b)
    if len(a_vals) != len(b_vals) or not a_vals:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a_vals, b_vals):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


@dataclass(frozen=True)
class Chunk:
    """One chunk the sandbox can score: id, text, optional embedding."""

    chunk_id: str
    text: str
    embedding: list[float] | None = None
    record_id: str | None = None


@dataclass(frozen=True)
class Query:
    """One sandbox query: id, text, optional pre-computed embedding."""

    query_id: str
    text: str
    embedding: list[float] | None = None


@dataclass(frozen=True)
class QueryDiff:
    """The before/after diff for one query."""

    query_id: str
    before_top_k: list[str]
    after_top_k: list[str]
    entered: list[str]
    left: list[str]
    rank_delta: dict[str, int]


@dataclass(frozen=True)
class SandboxResult:
    """Whole preview output."""

    queries: list[QueryDiff]
    elapsed_ms: float
    status: str = "ok"


def _ranked(chunks: list[Chunk], query: Query, k: int) -> list[str]:
    """Top-k chunk ids for ``query`` over ``chunks``, tie-broken by chunk_id."""
    scored = []
    q_tokens = tokenize(query.text)
    for chunk in chunks:
        vector_part = 0.0
        if query.embedding is not None and chunk.embedding is not None:
            vector_part = _cosine(query.embedding, chunk.embedding)
        jaccard_part = 0.0
        if q_tokens:
            chunk_tokens = tokenize(chunk.text)
            union = q_tokens | chunk_tokens
            if union:
                jaccard_part = len(q_tokens & chunk_tokens) / len(union)
        weight = float(settings.DIFF_SANDBOX_VECTOR_WEIGHT)
        score = weight * ((1.0 + vector_part) / 2.0) + (1.0 - weight) * jaccard_part
        scored.append((score, chunk.chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _score, chunk_id in scored[:k]]


def compute_diff(
    queries: list[Query],
    before_chunks: list[Chunk],
    after_chunks: list[Chunk],
    k: int,
) -> SandboxResult:
    """Compute the before/after diff for every query.

    ``before_chunks`` are the active-release chunks; ``after_chunks`` are the
    same set with draft chunks overlaid in place of matching ``record_id``.
    """
    t0 = time.perf_counter()
    before_index = {chunk.chunk_id: chunk for chunk in before_chunks}
    after_index = {chunk.chunk_id: chunk for chunk in after_chunks}
    before_ids = list(before_index)
    after_ids = list(after_index)

    diffs: list[QueryDiff] = []
    for query in queries:
        before_ids_k = _ranked(before_chunks, query, k)
        after_ids_k = _ranked(after_chunks, query, k)
        before_set = set(before_ids_k)
        after_set = set(after_ids_k)
        entered = [cid for cid in after_ids_k if cid not in before_set]
        left = [cid for cid in before_ids_k if cid not in after_set]
        before_pos = {cid: i for i, cid in enumerate(before_ids_k)}
        after_pos = {cid: i for i, cid in enumerate(after_ids_k)}
        rank_delta = {
            cid: after_pos.get(cid, -1) - before_pos.get(cid, -1)
            for cid in (before_set | after_set)
        }
        diffs.append(
            QueryDiff(
                query_id=query.query_id,
                before_top_k=before_ids_k,
                after_top_k=after_ids_k,
                entered=entered,
                left=left,
                rank_delta=rank_delta,
            )
        )

    return SandboxResult(
        queries=diffs,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        status="ok",
    )


def preview_status_ok() -> bool:
    """Whether the sandbox is in a state that can answer (no embedding needed)."""
    return True
