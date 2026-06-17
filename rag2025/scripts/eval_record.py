"""EvalRecord — instrumentation contract for full-evidence RAG evaluation.

Pydantic v2 schema that persists the *full* evidence per evaluated question:
retrieved chunk TEXT (not just IDs), full answer, latency seams, model versions,
and judge scores. A later RAGAS-grade eval can be derived from this without
re-running the pipeline.

This is a PURE data module — no I/O, no LLM, no ragas/deepeval imports.

NOTE: do NOT add `from __future__ import annotations` here — Pydantic v2.10
struggles to resolve stringified `Optional[...]` / `list[...]` annotations
in this nested-schema configuration.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """A single retrieved chunk, with FULL text (never truncated)."""

    chunk_id: str
    source: str
    score: float
    text: str
    data_year: Optional[int] = None
    is_superseded: bool = False
    # S15.1 diagnostic: True when this chunk was admitted by the GraphRAG
    # expander (PPR-only, not in the vector baseline). Lets the report
    # quantify how often graph expansion actually contributed context.
    # Optional + default False so pre-S15 records still parse under extra="forbid".
    graph_injected: bool = False


class LatencyMs(BaseModel):
    """Honest latency seams + total. The three seams cover the whole pipeline."""

    route_ms: float = Field(
        ..., description="Router stage — includes HyDE (fused in router.route())."
    )
    retrieval_loop_ms: float = Field(
        ..., description="Retrieval loop — includes embed (interleaved loop)."
    )
    query_ms: float = Field(
        ..., description="Query stage — includes rerank + gen (fused in pipeline.query())."
    )
    total_ms: float = Field(..., description="End-to-end total.")


class LatencyAttribution(BaseModel):
    """Best-effort, optional sub-fields for finer-grained attribution."""

    hyde_ms: Optional[float] = None
    router_inner_ms: Optional[float] = None
    gen_ms: Optional[float] = None
    rerank_ms: Optional[float] = None


class ModelVersion(BaseModel):
    """Pinned model versions for a given run (reproducibility)."""

    gen: str
    ner: str
    hyde: str
    embedding: str


class EvalRecord(BaseModel):
    """One evaluated question, with full evidence persisted."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    run_id: str
    seed: int
    harness_version: str
    timestamp: str  # ISO string

    model_version: ModelVersion

    # Question
    id: str
    question: str
    intent: Optional[str] = None
    complexity: Optional[int] = None
    route: Optional[str] = None
    expected_behavior: Optional[Literal["answer", "abstain", "refuse"]] = None

    # Retrieval
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)

    # Answer (FULL, never truncated)
    answer: str

    # Latency
    latency_ms: LatencyMs
    latency_attribution: Optional[LatencyAttribution] = None

    # Judge
    judge_scores: Optional[dict[str, float]] = None
    judge_rationale: Optional[dict[str, str]] = None

    # ---- helpers ----

    def to_jsonl_line(self) -> str:
        """Serialize to a single-line JSON string (one record per line)."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> "EvalRecord":
        """Parse a single JSONL line back into an EvalRecord."""
        return cls.model_validate_json(line)
