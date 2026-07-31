"""S15.10 — Comparison E2E (Opus-authored, /ultraqa-grade).

Contract under test
-------------------
For an "A vs B" comparison query, a chunk that is **only reachable via the
graph** (PPR-only, NOT in the vector baseline) for entity B must SURVIVE both:

  1. S15.1 GraphRAG EXPANDER injection (chunk_fetcher → `score=0.0`
     neutralization + `metadata["graph_injected"]=True`).
  2. S15.5 Precision CUTOFF at the pipeline boundary
     (`apply_rerank_cutoff(ratio=0.3, min_keep=2)`), which would otherwise
     drop the PPR-only chunk on the strength of its `β·ppr` alone.

This is the "comparison misses one side's context" defect that CF-1 was created
to fix: the expander must admit the chunk, AND the cutoff must let it through.

This is a *test-only* module. It does NOT edit any src file. If any assertion
reveals a real bug, the test reports the failing assertion and the file:line
of the defect — it does not attempt a fix.

Setup
-----
- Fake `_NERService` yields 2 NGANH seeds (`NGANH:cntt`, `NGANH:truyenthong`)
  from `extract_from_query` — this triggers the comparison-detection bump
  (CF-4: M 5→8, top_k +3).
- Fake `KnowledgeGraph.ppr_scores_by_chunk` returns scores for:
    * `cA`     (also in baseline, also in PPR — entity A)
    * `cB`     (NOT in baseline, only in PPR — entity B, the test chunk)
    * 6 noise PPR-only chunks so M=8 is observably consumed
- Fake `chunk_fetcher` returns real `RetrievedDocument` for `cB` (with the
  fetch_by_id sentinel `score=1.0`) and 6 noise docs.
- `baseline_docs` is the vector baseline = [`cA` + 2 noise docs], does NOT
  contain `cB`.
- We exercise:
    (a) `GraphRAGRetriever.retrieve(...)` directly, and
    (b) `apply_rerank_cutoff(...)` on the reranked output to simulate the
        pipeline-boundary step (S15.5, MUST-FIX #4, applied inside
        `UnifiedRAGPipeline.query()`).

We deliberately do NOT instantiate `UnifiedRAGPipeline.from_disk()` here —
that pulls a real LanceDB singleton and an LLM client. The contract under
test is the retriever + cutoff, not the wiring. The S15.5 boundary is
simulated by calling `apply_rerank_cutoff` with the env defaults
(`RERANK_KEEP_RATIO=0.3`, `RERANK_MIN_KEEP=2`).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.graphrag_retriever import (  # noqa: E402
    GraphRAGRetriever,
    apply_rerank_cutoff,
)
from src.services.lancedb_retrieval import RetrievedDocument  # noqa: E402
from src.services.query_router import QueryRoute, RouterResult  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fakes (mirroring test_graphrag_expander.py / test_graphrag_multiseed.py)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeNER:
    """NER stub: `extract_from_query` returns the fixed seed list. The async
    `extract` is never invoked because regex returns non-empty."""

    def __init__(self, seeds: List[str]) -> None:
        self._seeds = list(seeds)
        self.extract_calls = 0

    def extract_from_query(self, query: str) -> List[str]:
        return list(self._seeds)

    async def extract(self, chunk):
        self.extract_calls += 1

        class _R:
            is_success = False
            entities: list = []

        return _R()


class _FakeGraph:
    """Graph stub returning a controllable `ppr_scores_by_chunk` mapping."""

    def __init__(self, ppr_map: Dict[str, float]) -> None:
        self._ppr_map = dict(ppr_map)
        self.calls: List[List[str]] = []

    async def ppr_scores_by_chunk(self, seeds, alpha: float = 0.85):
        self.calls.append(list(seeds))
        return dict(self._ppr_map)

    def stats(self):
        return {"nodes": 0, "edges": 0}


class _RecordingFetcher:
    """`chunk_fetcher` fake: returns a doc per chunk_id from the fixture map.
    Records every call (for M-cap assertion). Returns None for unknown ids
    (mirrors `fetch_by_id` miss behavior)."""

    def __init__(self, doc_map: Dict[str, RetrievedDocument]) -> None:
        self._doc_map = dict(doc_map)
        self.calls: List[str] = []

    def __call__(self, chunk_id: str) -> Optional[RetrievedDocument]:
        self.calls.append(chunk_id)
        return self._doc_map.get(chunk_id)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────


# Two NGANH seeds → comparison-detection fires (CF-4): M 5→8, top_k +3.
SEEDS = ["NGANH:cntt", "NGANH:truyenthong"]

# entity A (cA) is in baseline AND in PPR.
# entity B (cB) is the test chunk — only in PPR, NOT in baseline.
# c_inj2..c_inj7 are 6 noise PPR-only chunks — they let us observe M=8 being
# fully consumed AND let the cutoff have realistic low-score filler to drop.
PPR_MAP: Dict[str, float] = {
    "cA":     0.50,   # in baseline
    "cB":     0.40,   # the test chunk — PPR-only
    "c_inj2": 0.35,
    "c_inj3": 0.30,
    "c_inj4": 0.20,
    "c_inj5": 0.15,
    "c_inj6": 0.10,
    "c_inj7": 0.05,
    "c_inj8": 0.04,
}

BASELINE = [
    # cA is entity A — in baseline AND in PPR, dominates the top.
    ("cA",     2.0,  "CNTT context — điểm chuẩn, học phí, chương trình"),
    # noise1 + noise2 are above the 0.3·top_score cutoff threshold after fusion
    # → they survive the cutoff (no exemption needed).
    ("noise1", 1.5,  "vector noise 1"),
    ("noise2", 1.0,  "vector noise 2"),
    # noise3 is a baseline doc whose fused score (0.6·(0.5/2.0) = 0.15) lands
    # BELOW the 0.3·top_score cutoff threshold. It is NOT graph_injected
    # (it was in baseline, not fetched by the expander) and NOT booster_injected
    # — so the cutoff MUST drop it. This is the doc that proves the cutoff
    # branch actually fired (otherwise the cB-survival assertion is vacuous).
    ("noise3", 0.5,  "vector noise 3 — low score, no exemption, must be cut"),
]

# `fetch_by_id` returns score=1.0 sentinel (lancedb_retrieval.py:537 "injected
# priority"). Expander MUST neutralize to 0.0 (MUST-FIX #1).
def _build_fetcher() -> _RecordingFetcher:
    doc_map: Dict[str, RetrievedDocument] = {}
    # cB (the test chunk) — fetch_by_id would return score=1.0; we mirror that
    # here so the test proves the EXPANDER neutralizes it (not the fetcher).
    doc_map["cB"] = _make_doc("cB", 1.0, "Truyền thông số context — điểm chuẩn, học phí")
    for cid in ("c_inj2", "c_inj3", "c_inj4", "c_inj5", "c_inj6", "c_inj7", "c_inj8"):
        doc_map[cid] = _make_doc(cid, 1.0, f"injected noise {cid}")
    return _RecordingFetcher(doc_map=doc_map)


def _make_doc(
    cid: str,
    score: float,
    text: str = "",
    metadata: Optional[dict] = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        text=text or f"text-{cid}",
        source=f"src-{cid}",
        chunk_id=cid,
        metadata=metadata or {},
        score=score,
        point_id=cid,
    )


def _router(query: str = "so sánh ngành CNTT và Truyền thông số") -> RouterResult:
    return RouterResult(
        original_query=query,
        step_back_query=query,
        hypothetical_doc="hyde-placeholder",
        hyde_variants=[],
        route=QueryRoute.GRAPH_RAG,
        complexity=4,
        intent="so_sanh",
        reasoning="S15.10 comparison E2E fixture",
    )


def _build_retriever(
    ppr_map: Optional[Dict[str, float]] = None,
    fetcher: Optional[_RecordingFetcher] = None,
) -> GraphRAGRetriever:
    ner = _FakeNER(seeds=SEEDS)
    graph = _FakeGraph(ppr_map=ppr_map or PPR_MAP)
    return GraphRAGRetriever(
        ner_service=ner,
        graph=graph,
        alpha=0.6,
        beta=0.4,
        chunk_fetcher=fetcher or _build_fetcher(),
    )


def _baseline_docs() -> List[RetrievedDocument]:
    return [
        _make_doc(cid, score, text)
        for cid, score, text in BASELINE
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — assertion shortcuts
# ─────────────────────────────────────────────────────────────────────────────


def _by_id(docs: List[RetrievedDocument]) -> Dict[str, RetrievedDocument]:
    return {d.chunk_id: d for d in docs if d.chunk_id}


# ─────────────────────────────────────────────────────────────────────────────
# The ONE high-value E2E test
# ─────────────────────────────────────────────────────────────────────────────


def test_s15_10_comparison_ppr_only_chunk_survives_expander_and_cutoff():
    """S15.10 — The full expander-fusion contract for an A-vs-B comparison.

    Setup:
      - 2 NGANH seeds → CF-4 comparison bump: M 5→8, top_k +3.
      - PPR-only chunk `cB` (entity B) with PPR=0.4, fetchable via
        `chunk_fetcher`, NOT in vector baseline.
      - `cA` (entity A) is in baseline AND in PPR.

    Assertions (the contract):
      1. After `retrieve()`, `cB` is in the reranked output and is tagged
         `metadata["graph_injected"]=True`.
      2. `cB`'s fused score is NOT the auto-#1 (sentinel 1.0 was neutralized
         to 0.0; rank is by β·ppr, not the fetch sentinel).
      3. After `apply_rerank_cutoff(ratio=0.3, min_keep=2)` (simulating
         `UnifiedRAGPipeline.query()` boundary), `cB` SURVIVES even though
         its β·ppr score is well below 0.3×top_score — the CF-1 fix works.
      4. BOTH `cA` and `cB` are in the final post-cutoff set → the
         comparison has both sides' context.
      5. Comparison bump fired: M==8 (8 fetcher calls for PPR-only chunks
         — cA is in baseline and skipped) and the reranked list length ==
         top_k + 3 = 8.
    """
    fetcher = _build_fetcher()
    retriever = _build_retriever(fetcher=fetcher)
    baseline = _baseline_docs()
    top_k = 5

    # ── Stage 1: GraphRAG expander-fusion (S15.1) ─────────────────────────
    reranked, ppr_scores = asyncio.run(
        retriever.retrieve(
            "so sánh ngành CNTT và Truyền thông số",
            _router(),
            baseline_docs=baseline,
            top_k=top_k,
        )
    )
    by_id = _by_id(reranked)

    # Assert 5 (first half): comparison bump fired.
    # M=8: 8 fetcher calls for PPR-only chunks (cA is in baseline_ids so it's
    # skipped before the fetcher is even called). We seeded 8 PPR-only
    # chunk_ids (cB + c_inj2..c_inj8), so all 8 should be fetched.
    assert len(fetcher.calls) == 8, (
        f"comparison bump must set M=8 (CF-4); got {len(fetcher.calls)} "
        f"fetches: {fetcher.calls}"
    )
    # M=8 also drives the final output: top_k+3 = 8.
    assert len(reranked) == top_k + 3 == 8, (
        f"comparison bump must bump top_k 5→8 (top_k+3); got {len(reranked)} "
        f"reranked docs"
    )

    # Assert 1: cB is in reranked output, tagged graph_injected=True.
    assert "cB" in by_id, (
        f"entity-B PPR-only chunk MUST reach reranked output; "
        f"got chunk_ids={[d.chunk_id for d in reranked]}"
    )
    cB_out = by_id["cB"]
    assert (cB_out.metadata or {}).get("graph_injected") is True, (
        f"entity-B chunk must be tagged graph_injected=True (CF-1); "
        f"got metadata={cB_out.metadata!r}"
    )

    # Assert 2: cB is NOT auto-#1 (sentinel neutralized to 0.0).
    # cA is the strong baseline doc (score=2.0) and also in PPR → must outrank.
    assert "cA" in by_id, "entity-A chunk must be in reranked output"
    cA_out = by_id["cA"]
    assert cA_out.score > cB_out.score, (
        f"cA (rrf-dominant + PPR) must outrank cB (sentinel-neutralized); "
        f"cA.score={cA_out.score}, cB.score={cB_out.score}"
    )
    # And cB must not be at the very top of the list (it shouldn't be #1).
    rank_of = {cid: i for i, cid in enumerate(d.chunk_id for d in reranked)}
    assert rank_of["cB"] != 0, (
        f"cB must NOT auto-rank #1 from fetch sentinel; "
        f"got order={[d.chunk_id for d in reranked]}"
    )

    # Sanity: the injected chunks that ARE in baseline (cA) are NOT marked
    # graph_injected (the tag is only set on chunks the EXPANDER fetched).
    assert not (cA_out.metadata or {}).get("graph_injected"), (
        "cA is in baseline → must NOT carry graph_injected marker"
    )

    # ── Stage 2: S15.5 precision cutoff at the pipeline boundary ─────────
    # Simulate `UnifiedRAGPipeline.query()`'s env defaults:
    #   RERANK_KEEP_RATIO=0.3, RERANK_MIN_KEEP=2.
    final_docs = apply_rerank_cutoff(
        reranked,
        ratio=0.3,
        min_keep=2,
    )
    final_by_id = _by_id(final_docs)

    # Assert 3: cB survives the cutoff. Its β·ppr fused score is well below
    # 0.3×top_score (the baseline cA is the new top); without the CF-1
    # exemption it would be dropped. THIS is the CF-1 fix.
    top_score = max(d.score for d in reranked)
    cutoff_threshold = 0.3 * top_score
    assert cB_out.score < cutoff_threshold, (
        f"test setup error: cB.score={cB_out.score} should be < "
        f"cutoff={cutoff_threshold} (top_score={top_score}); "
        f"otherwise the cutoff-survival assertion is vacuous"
    )
    assert "cB" in final_by_id, (
        f"BUG?: entity-B PPR-only chunk was DROPPED by apply_rerank_cutoff "
        f"despite graph_injected=True. CF-1 exemption may be broken. "
        f"cB.score={cB_out.score}, threshold={cutoff_threshold}, "
        f"top_score={top_score}, final={[d.chunk_id for d in final_docs]}"
    )
    # And the cut must have actually done SOMETHING — prove we exercised the
    # cutoff branch (not the ratio=0 short-circuit). We assert that at least
    # one non-exempt doc below the threshold was dropped.
    dropped_non_exempt_below_threshold = [
        d for d in reranked
        if d.score < cutoff_threshold
        and not (d.metadata or {}).get("graph_injected")
        and not (d.metadata or {}).get("booster_injected")
    ]
    dropped_non_exempt_ids_actually_dropped = [
        d.chunk_id for d in dropped_non_exempt_below_threshold
        if d.chunk_id not in final_by_id
    ]
    assert dropped_non_exempt_ids_actually_dropped, (
        "test setup error: cutoff did not drop any non-exempt low-score doc; "
        "the cB-survival assertion is vacuous. reranked scores: "
        + ", ".join(f"{d.chunk_id}={d.score:.3f}" for d in reranked)
    )

    # Assert 4: BOTH cA and cB are in the final post-cutoff set.
    assert "cA" in final_by_id, (
        f"entity-A chunk must survive the cutoff; "
        f"final={[d.chunk_id for d in final_docs]}"
    )
    assert "cB" in final_by_id, (
        f"BUG?: entity-B chunk missing from final post-cutoff set — "
        f"comparison has only ONE side's context. "
        f"final={[d.chunk_id for d in final_docs]}"
    )
    # Both sides' context present → the comparison query has the structural
    # context required to answer "A vs B" honestly.
    final_chunk_ids = {d.chunk_id for d in final_docs}
    assert {"cA", "cB"}.issubset(final_chunk_ids), (
        f"BUG?: comparison missing one side. Expected both cA and cB; "
        f"got final={sorted(final_chunk_ids)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Belt-and-suspenders: separate, focused checks for the two seams.
# These pin the sub-contracts so a future refactor can't silently break one
# without the other.
# ─────────────────────────────────────────────────────────────────────────────


def test_s15_10_seam1_expander_admits_cB_and_tags_it_graph_injected():
    """Seam 1: the GraphRAG expander must admit cB and tag it graph_injected."""
    fetcher = _build_fetcher()
    retriever = _build_retriever(fetcher=fetcher)
    reranked, _ = asyncio.run(
        retriever.retrieve(
            "so sánh ngành CNTT và Truyền thông số",
            _router(),
            baseline_docs=_baseline_docs(),
            top_k=5,
        )
    )
    by_id = _by_id(reranked)
    assert "cB" in by_id
    assert (by_id["cB"].metadata or {}).get("graph_injected") is True
    # Original metadata preserved (the expander merges, not replaces).
    # The fetched doc for cB had no extra metadata → just the graph_injected
    # marker should be present.
    assert "graph_injected" in (by_id["cB"].metadata or {})


def test_s15_10_seam2_cutoff_exempts_cB_below_threshold():
    """Seam 2: even with cB's score *clearly* below the cutoff threshold and
    no other exemption mechanism at play, the CF-1 exemption keeps cB in.

    This bypasses the retriever and tests `apply_rerank_cutoff` directly with
    a synthetic reranked list so the seam is provable in isolation.
    """
    docs = [
        _make_doc("cA",     1.00),  # top
        _make_doc("noise1", 0.50),
        _make_doc("noise2", 0.40),  # above threshold (0.30) — kept
        _make_doc("cB",     0.10, metadata={"graph_injected": True}),  # below
        _make_doc("c_drop", 0.10),  # same score, no exemption → dropped
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.3, min_keep=2)
    kept_ids = {d.chunk_id for d in kept}

    assert "cB" in kept_ids, (
        f"BUG?: graph_injected chunk dropped by cutoff; "
        f"kept={sorted(kept_ids)}"
    )
    assert "c_drop" not in kept_ids, (
        f"non-exempt low-score chunk must be dropped to prove the cutoff "
        f"branch fired; kept={sorted(kept_ids)}"
    )
    assert "cA" in kept_ids
