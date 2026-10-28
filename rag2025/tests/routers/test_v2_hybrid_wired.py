"""Ticket 35 — hybrid retrieval must serve the production `/v2` path.

The bug this file locks down: `HybridSearchService` was constructed at startup
but **no answering route called it** — `/v2` (the path the FE uses) went
straight to dense LanceDB. This file proves the wiring through the HTTP seam
(`POST /v2/query`), never by calling `_compute_baseline`/`HybridSearchService`
directly, per `spec.md`'s seam principle.

The trap this file exists to catch (ticket §"RÀNG BUỘC SỐ 1"): RRF fusion
scores live on a different scale (max `0.6/61 + 0.4/61 = 0.01639`) than the
adaptive rerank threshold (`RERANKER_ADAPTIVE_CONFIDENCE=0.5`). Wiring hybrid
naively would make `confidence < threshold` true for essentially every query,
so the fast path would rerank everything at ~+1.8s. Resolution (A) keeps the
RRF score as the RANKING and feeds the gate a separate, calibrated
`retrieval_confidence` (dense top-1 cosine). `test_high_confidence_query_skips_
rerank` is the regression test for that; removing the gate's `confidence <
threshold` condition turns it red.

The fakes below deliberately model the scale split: the hybrid service returns
RRF-scale scores (0.0164 max) while the dense retriever returns cosine-scale
scores (~0.8). A naive wiring therefore trips the gate in the high-confidence
test — which is exactly the failure mode being guarded.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Real dataclasses, not look-alikes: the /v2 seam calls `replace()`, reads the
# `is_success` property and `to_dict()`. A stand-in with the same field names
# would silently diverge from the contract under test.
from services.lancedb_retrieval import RetrievalResult, RetrievedDocument  # noqa: E402

# RRF fusion scale: rank-0 in BOTH arms, k=60, weights 0.6/0.4 (.env:70-71).
# The hybrid service's own `confidence` is mean(d.score) on this same scale.
RRF_MAX = 0.6 / 61 + 0.4 / 61  # 0.016393...
# The calibrated threshold the gate compares against (rag2025/.env:55).
ADAPTIVE_CONFIDENCE = 0.5
# Cosine-scale scores the dense arm actually produces (measured on the real
# table: 0.6744 .. 0.7973).
HIGH_COSINE = 0.7973
LOW_COSINE = 0.41


def _doc(text: str, chunk_id: str, score: float, *, year: int = 2026) -> RetrievedDocument:
    return RetrievedDocument(
        text=text,
        source="doc.pdf",
        chunk_id=chunk_id,
        metadata={"summary": f"summary {chunk_id}", "data_year": year},
        score=score,
        point_id=chunk_id,
    )


def _build_main_module(monkeypatch):
    """Load `src/main.py` with the heavy env-driven singletons defused."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    repo_root = MAIN_PATH.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    spec = importlib.util.spec_from_file_location("v2_hybrid_wired_test", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _router_result(*, route_value: str = "padded_rag", in_scope: bool = True):
    return SimpleNamespace(
        skip_retrieval=False,
        auto_answer=None,
        step_back_query=None,
        intent="factoid",
        complexity=1,
        reasoning="test",
        hyde_variants=[],
        route=SimpleNamespace(value=route_value),
        is_in_scope=in_scope,
        sub_queries=None,
    )


def _dense_docs(*, top_score: float, n: int = 4):
    return [_doc(f"dense text {i}", f"dense-{i}", top_score - i * 0.01) for i in range(n)]


def _rrf_docs(*, n: int = 4):
    """Documents on the RRF scale, as `HybridSearchService.retrieve()` returns."""
    return [_doc(f"fused text {i}", f"fused-{i}", RRF_MAX - i * 0.0005) for i in range(n)]


def _install_fakes(
    main_module,
    *,
    hybrid_on: bool,
    dense_top_score: float = HIGH_COSINE,
    hybrid_docs=None,
    hybrid_n: int = 4,
    dense_docs=None,
    dense_empty: bool = False,
    bm25_ready: bool = True,
):
    """Wire the /v2 seam with fakes that model the RRF-vs-cosine scale split.

    Returns a dict of captured handles so tests assert on observable effects
    (what the hybrid service was asked, what reached the generator) rather
    than on internal call plumbing.
    """
    dense = list(dense_docs if dense_docs is not None else _dense_docs(top_score=dense_top_score))
    if dense_empty:
        dense = []
    fused = list(hybrid_docs if hybrid_docs is not None else _rrf_docs(n=hybrid_n))

    captured: dict = {
        "hybrid_calls": [],
        "dense_calls": [],
        "generator_chunks": None,
        "rerank_calls": [],
        "pipeline_baseline": None,
    }

    retriever = MagicMock()

    def _dense_retrieve(*args, **kwargs):
        captured["dense_calls"].append(kwargs)
        if dense_empty:
            return RetrievalResult(
                documents=[], error_type="EMPTY_RESULT", error_message="none", confidence=0.0
            )
        return RetrievalResult(
            documents=list(dense), error_type=None, confidence=dense[0].score
        )

    retriever.retrieve = MagicMock(side_effect=_dense_retrieve)
    retriever.fetch_by_id = MagicMock(return_value=None)
    # The sparse arm routes BM25 candidates through the retriever's post-fetch
    # rule; identity models "no temporal filter active".
    retriever.apply_post_fetch_filters = MagicMock(side_effect=lambda rows, *a, **k: rows)
    main_module.lancedb_retriever_service = retriever

    embedding = MagicMock()
    embedding.encode_query = MagicMock(
        return_value=SimpleNamespace(tolist=lambda: [0.0] * 4)
    )
    main_module.embedding_service = embedding

    # ── hybrid service (the thing under test) ───────────────────────────
    hybrid = None
    if hybrid_on:
        hybrid = MagicMock()

        async def _hybrid_retrieve(*, query, query_vector, top_k, metadata_filter=None):
            captured["hybrid_calls"].append(
                {
                    "query": query,
                    "query_vector": query_vector,
                    "top_k": top_k,
                    "metadata_filter": metadata_filter,
                }
            )
            if not bm25_ready:
                # Mirrors the real service's documented dense-only fallback.
                return RetrievalResult(documents=list(fused), confidence=fused[0].score)
            return RetrievalResult(documents=list(fused), confidence=RRF_MAX * 0.8)

        hybrid.retrieve = AsyncMock(side_effect=_hybrid_retrieve)
        hybrid.build_bm25_index = MagicMock(return_value=True)
    main_module.hybrid_search_service = hybrid

    guardrail = MagicMock()
    guardrail.precheck = AsyncMock(
        return_value=SimpleNamespace(
            is_in_scope=True,
            internal_code="SUCCESS",
            reason="ok",
            public_code="SUCCESS",
            short_answer="",
            pii_detected=False,
            data_gap_hints=[],
        )
    )
    guardrail.expose_internal = MagicMock(return_value=False)
    guardrail._pii_warning_answer = MagicMock(return_value="Vui lòng ẩn PII.")
    guardrail._contains_sensitive_pii = MagicMock(return_value=False)
    guardrail.classify_no_result = AsyncMock(
        return_value=SimpleNamespace(internal_code="SUCCESS", reason="ok", data_gap_hints=[])
    )
    guardrail.public_status = MagicMock(side_effect=lambda code: code)
    main_module.guardrail_service = guardrail
    main_module.query_cache = None

    async def _pipeline_query(user_query, baseline_docs, top_k, **kwargs):
        captured["pipeline_baseline"] = list(baseline_docs)
        # The real pipeline derives confidence from the ranking scores. On the
        # hybrid path those are RRF-scale — this is the exact quantity that
        # would poison the gate without resolution (A).
        return SimpleNamespace(
            query=user_query,
            route="padded_rag",
            documents=list(baseline_docs[:top_k]),
            router_result=_router_result(),
            ppr_scores={},
            latency_ms=1.0,
            confidence=(baseline_docs[0].score if baseline_docs else 0.0),
        )

    pipeline = MagicMock()
    pipeline.query = AsyncMock(side_effect=_pipeline_query)
    pipeline._graphrag = MagicMock()
    pipeline._graphrag.graph_stats = {"nodes": 0, "edges": 0}
    pipeline._graphrag.retrieve = AsyncMock(
        return_value=(list(fused), {"dense-0": 0.5})
    )
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=_router_result())
    main_module.unified_pipeline = pipeline

    # ── reranker: record whether the adaptive gate fired ────────────────
    reranker = MagicMock()
    reranker.enabled = True
    reranker.adaptive_confidence = ADAPTIVE_CONFIDENCE
    reranker.pool = 5

    def _rerank(*, query, chunks, top_k, apply_lost_in_middle=True):
        captured["rerank_calls"].append({"query": query, "chunks": list(chunks), "top_k": top_k})
        return [dict(c) for c in chunks]

    reranker.rerank = MagicMock(side_effect=_rerank)
    main_module.reranker_service = reranker

    async def _generate_answer(*, query, chunks, confidence, is_program_list_query, history=None):
        captured["generator_chunks"] = list(chunks)
        captured["generator_confidence"] = confidence
        return {
            "answer": "test answer",
            "sources": [
                {
                    "id": c.get("chunk_id") or "?",
                    "title": c.get("chunk_id") or "?",
                    "url": None,
                    "snippet": (c.get("text") or "")[:120],
                    "data_year": "2026",
                }
                for c in chunks
            ],
            "confidence": confidence,
            "provider": "fake",
            "chunks_used": len(chunks),
        }

    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=_generate_answer)
    main_module.llm_generator_service = generator

    return {"captured": captured, "retriever": retriever, "hybrid": hybrid, "reranker": reranker}


async def _post_v2(main_module, query: str, *, top_k: int = 5):
    """Drive the real HTTP seam (`POST /v2/query`) through the ASGI app."""
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v2/query", json={"query": query, "top_k": top_k})


@pytest.fixture
def main_module(monkeypatch):
    return _build_main_module(monkeypatch)


# ─────────────────────────────────────────────────────────────────────────
# C1 — hybrid actually serves /v2 (not merely constructed)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c1_hybrid_serves_v2_when_flag_on(main_module):
    """With the hybrid service wired, `/v2/query` must retrieve THROUGH it and
    the fused documents must be the ones the generator sees. Asserting only
    "the service was built" is what let the original bug ship.
    """
    fakes = _install_fakes(main_module, hybrid_on=True)
    resp = await _post_v2(main_module, "Học phí ngành Công nghệ thông tin là bao nhiêu?")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["hybrid_calls"]) == 1, (
        "REGRESSION (C1): /v2 did not call HybridSearchService.retrieve() — "
        "the hybrid service is constructed but no answering route uses it."
    )
    call = captured["hybrid_calls"][0]
    assert call["query"] == "Học phí ngành Công nghệ thông tin là bao nhiêu?"
    # top_k * 3 = 15, matching the dense call it replaces (ticket §Cạm bẫy 5).
    assert call["top_k"] == 15, f"expected top_k*3=15, got {call['top_k']}"
    # Constraint 2: hybrid injects `query` into metadata_filter itself; passing
    # query= would be a TypeError and dropping the temporal hint.
    assert call["metadata_filter"] is None

    # The result the student sees must come from the hybrid arm, not dense.
    gen_ids = [c["chunk_id"] for c in captured["generator_chunks"]]
    assert gen_ids and all(cid.startswith("fused-") for cid in gen_ids), (
        f"expected fused (hybrid) chunks to reach the generator, got {gen_ids!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# C2 — flag off == the old dense behavior
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c2_flag_off_is_dense_only(main_module):
    """`USE_HYBRID_RETRIEVAL=false` → `build_hybrid_service()` returns None →
    the else-branch runs, i.e. exactly one dense retrieval and no hybrid call.
    This is ticket 09 criterion 5's rollback path.
    """
    fakes = _install_fakes(main_module, hybrid_on=False)
    resp = await _post_v2(main_module, "Điểm chuẩn ngành Kỹ thuật phần mềm năm 2025")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert captured["hybrid_calls"] == [], "hybrid must not be consulted when the flag is off"
    # Exactly one dense retrieval — the historical single call.
    assert len(captured["dense_calls"]) == 1, (
        f"expected exactly 1 dense retrieve() on the dense path, got {len(captured['dense_calls'])}"
    )
    assert captured["dense_calls"][0]["top_k"] == 15
    assert captured["dense_calls"][0]["query"] == "Điểm chuẩn ngành Kỹ thuật phần mềm năm 2025"

    gen_ids = [c["chunk_id"] for c in captured["generator_chunks"]]
    assert gen_ids and all(cid.startswith("dense-") for cid in gen_ids), (
        f"expected dense chunks on the flag-off path, got {gen_ids!r}"
    )
    # Dense path already yields a calibrated confidence; the response must
    # carry that value (not a rewritten one).
    assert resp.json()["confidence"] == pytest.approx(HIGH_COSINE, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# C3 — the fast path survives (mutation-checked)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c3_high_confidence_query_skips_rerank(main_module):
    """A high-confidence query must NOT rerank.

    This is the regression test for the RRF-vs-threshold scale mismatch: the
    hybrid documents carry RRF scores (max 0.0164) while the gate compares
    against 0.5, so without resolution (A) this test fails and rerank fires on
    every query (+~1.8s). Mutation check: deleting the gate's
    `rag_result.confidence < reranker_service.adaptive_confidence` condition
    makes this test red.
    """
    fakes = _install_fakes(main_module, hybrid_on=True, dense_top_score=HIGH_COSINE)
    resp = await _post_v2(main_module, "Học phí ngành Công nghệ thông tin là bao nhiêu?")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["hybrid_calls"]) == 1, "hybrid path not exercised — test is vacuous"

    assert captured["rerank_calls"] == [], (
        "REGRESSION (C3): rerank fired on a HIGH-confidence query. The gate "
        "received an RRF-scale confidence (max 0.0164) instead of the "
        "calibrated cosine value, so `confidence < 0.5` was always true and "
        "the fast path is broken for every query."
    )
    # The calibrated value must survive to the response, on the cosine scale.
    body = resp.json()
    assert body["confidence"] == pytest.approx(HIGH_COSINE, abs=1e-6), (
        f"response confidence {body['confidence']!r} is not the calibrated "
        f"cosine value; an RRF score (max {RRF_MAX:.5f}) would mean the "
        f"ranking scale leaked into the confidence field."
    )
    assert body["confidence"] > ADAPTIVE_CONFIDENCE


# ─────────────────────────────────────────────────────────────────────────
# C4 — rerank still runs when it should, with the right pool
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c4_low_confidence_query_reranks_with_correct_pool(main_module):
    """A low-confidence query MUST rerank, with pool = min(RERANKER_POOL, len(docs))."""
    fakes = _install_fakes(
        main_module, hybrid_on=True, dense_top_score=LOW_COSINE, dense_docs=_dense_docs(top_score=LOW_COSINE, n=4)
    )
    resp = await _post_v2(main_module, "Ký túc xá có chỗ cho tân sinh viên không?")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["rerank_calls"]) == 1, (
        "REGRESSION (C4): a LOW-confidence query did not rerank — the gate is "
        "over-suppressing."
    )
    # 4 docs available, RERANKER_POOL=5 → pool = min(5, 4) = 4.
    assert captured["rerank_calls"][0]["top_k"] == 4
    assert len(captured["rerank_calls"][0]["chunks"]) == 4


@pytest.mark.asyncio
async def test_c4b_pool_capped_by_reranker_pool(main_module):
    """With more documents than RERANKER_POOL, the pool is capped at RERANKER_POOL."""
    # 9 documents reach the pipeline; the gate must still hand the reranker
    # exactly RERANKER_POOL=5, not 9.
    fakes = _install_fakes(
        main_module,
        hybrid_on=True,
        dense_top_score=LOW_COSINE,
        dense_docs=_dense_docs(top_score=LOW_COSINE, n=9),
        hybrid_n=9,
    )
    resp = await _post_v2(main_module, "Thủ tục nhập học cần chuẩn bị giấy tờ gì?", top_k=9)
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["pipeline_baseline"]) == 9, (
        f"expected 9 baseline docs, got {len(captured['pipeline_baseline'])}"
    )
    assert len(captured["rerank_calls"]) == 1, "low-confidence query should rerank"
    assert captured["rerank_calls"][0]["top_k"] == 5, "pool must be capped at RERANKER_POOL=5"


# ─────────────────────────────────────────────────────────────────────────
# C5 — filter symmetry through /v2
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c5_hybrid_receives_no_asymmetric_filter(main_module):
    """Through `/v2`, the hybrid arm must not be handed a filter that the dense
    arm would not also apply. `metadata_filter=None` makes hybrid inject only
    the temporal `query` hint; both arms then share the retriever's post-fetch
    rule, so published/year filtering stays symmetric (ticket 09 criterion 4).
    """
    fakes = _install_fakes(main_module, hybrid_on=True)
    resp = await _post_v2(main_module, "so sánh học phí 2024 và 2025 ngành CNTT")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["hybrid_calls"]) == 1, "hybrid path not exercised"

    call = captured["hybrid_calls"][0]
    assert call["metadata_filter"] is None, (
        "the /v2 hybrid arm must pass metadata_filter=None so the service "
        "injects only {'query': ...}; a pre-filtered dict here would let the "
        "dense arm and the sparse arm diverge."
    )
    # `query` reaches the service as a first-class argument (it re-injects it
    # for the temporal classifier) — never smuggled in as a filter key.
    assert call["query"] == "so sánh học phí 2024 và 2025 ngành CNTT"


# ─────────────────────────────────────────────────────────────────────────
# C7 — additional regression coverage on the seam
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c7_hybrid_reuses_the_encoded_vector(main_module):
    """The query vector handed to hybrid must be the one `_compute_baseline`
    encoded — re-encoding costs ~1.1s (ticket §Cạm bẫy 1)."""
    fakes = _install_fakes(main_module, hybrid_on=True)
    resp = await _post_v2(main_module, "Trường Đại học Khoa học có những ngành nghề gì?")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["hybrid_calls"]) == 1
    assert captured["hybrid_calls"][0]["query_vector"] == [0.0] * 4
    # encode_query must run exactly once for the whole request.
    assert main_module.embedding_service.encode_query.call_count == 1


@pytest.mark.asyncio
async def test_c7_degenerate_probe_uses_cosine_floor_not_rrf(main_module):
    """When the dense probe finds nothing, the gate must NOT be handed the RRF
    score.

    A sparse-only pool (BM25 found documents the dense arm missed) has no
    calibrated cosine value. Falling through to `rag_result.confidence` would
    feed the gate an RRF number (max 0.0164) — always below 0.5, i.e. the
    scale-mismatch defect again, just on a rarer path. The implementation pins
    the cosine floor (0.0) instead, so the gate reranks on a real
    cosine-scale value and the response never carries an RRF number.
    """
    fakes = _install_fakes(main_module, hybrid_on=True, dense_empty=True)
    resp = await _post_v2(main_module, "một câu hỏi không có tài liệu")
    assert resp.status_code == 200, resp.text

    captured = fakes["captured"]
    assert len(captured["hybrid_calls"]) == 1, "hybrid path not exercised"
    # Conservative: unknown/absent dense evidence → rerank.
    assert len(captured["rerank_calls"]) == 1, (
        "the degenerate path must rerank (no calibrated evidence of a good match)"
    )
    body = resp.json()
    assert body["confidence"] == 0.0, (
        f"degenerate path leaked a non-cosine confidence {body['confidence']!r}; "
        f"an RRF score (max {RRF_MAX:.5f}) here would mean the gate consumed a "
        f"ranking score as a relevance score"
    )
