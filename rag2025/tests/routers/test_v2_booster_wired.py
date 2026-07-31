"""TDD: /v2/query must inject aggregation chunks (booster) on enumeration/superlative
queries — mirroring the /v3 path — so aggregate summary chunks (`liet_ke_28_nganh_*`,
`bang_xep_hang_chi_tieu_*`, `tuyensinh_overview_*`) survive `apply_rerank_cutoff`.

Regression: the /v2 path built `baseline_docs` from LanceDB and passed them straight
to `unified_pipeline.query()`. Inside the pipeline, `apply_rerank_cutoff` drops
low-score docs (ratio * top_score). Aggregate summary chunks had low fused scores
and were dropped, leaving the generator with only 3 `nganh_detail_*` chunks → the
chatbot listed 3 majors instead of 28.

Approach: helper-refactor (not TestClient on the heavy main.app). We import
`src/main.py` via importlib, monkeypatch the module-level singletons
(`unified_pipeline`, `llm_generator_service`, `lancedb_retriever_service`,
`embedding_service`, `query_cache`, `guardrail_service`) and `boost_with_aggregation`
itself, then drive the FastAPI handler with `AsyncClient(app=app)`. We assert the
SEAM:
  - For an enumeration query, `boost_with_aggregation` is called by /v2 and the
    aggregate chunk_ids reach the generator's `chunks` arg (after fusion + cutoff).
  - For a non-enumeration query, `boost_with_aggregation` is still called (it is
    a no-op for non-enumeration queries) but no aggregate chunks are injected,
    so the generator sees only the baseline docs.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Build a real RetrievedDocument-like dataclass so `dataclasses.replace`
# works inside the booster (booster calls `replace(doc, metadata=...)`).
@dataclass
class _FakeRetrievedDoc:
    text: str
    source: str
    chunk_id: str
    metadata: dict
    score: float = 0.9
    point_id: str = ""


SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
TESTS_DIR = Path(__file__).resolve().parents[1]


def _build_main_module(monkeypatch):
    """Load src/main.py with heavy env-driven singletons defused."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    # Also expose 'src' package resolution for relative imports inside main.py
    repo_root = MAIN_PATH.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    spec = importlib.util.spec_from_file_location("v2_booster_wired_test", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────

def _fake_router_result(*, route_value: str = "graph_rag", intent: str = "enumeration"):
    return SimpleNamespace(
        skip_retrieval=False,
        auto_answer=None,
        step_back_query="step",
        intent=intent,
        complexity=2,
        reasoning="enumeration query",
        hyde_variants=[],
        route=SimpleNamespace(value=route_value),
        is_in_scope=True,
    )


def _fake_doc(chunk_id: str, text: str, *, score: float = 0.9, booster: bool = False):
    meta = {"summary": f"summary-{chunk_id}", "source": "doc.pdf", "data_year": 2026}
    if booster:
        meta["booster_injected"] = True
    return _FakeRetrievedDoc(
        text=text,
        source="doc.pdf",
        chunk_id=chunk_id,
        metadata=meta,
        score=score,
        point_id=chunk_id,
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def main_module(monkeypatch):
    return _build_main_module(monkeypatch)


def _install_fakes(
    main_module,
    monkeypatch,
    *,
    baseline_chunk_ids: list[str],
    booster_chunk_ids: list[str],
    is_enumeration: bool,
):
    """Install fakes for unified_pipeline, llm_generator_service,
    lancedb_retriever_service, embedding_service, query_cache, guardrail_service.
    Returns a dict of all the mocks so the test can assert on them.
    """
    # baseline_docs the lancedb retriever will return
    baseline_docs = [_fake_doc(cid, f"baseline text {cid}", score=0.9) for cid in baseline_chunk_ids]
    # booster docs the booster will return (only when query is an enumeration query)
    booster_docs = [_fake_doc(cid, f"AGGREGATE text {cid}", score=0.9, booster=True)
                   for cid in booster_chunk_ids]

    # lancedb_retriever_service: retrieve() → baseline docs
    lancedb_retriever = MagicMock()
    lancedb_retriever.retrieve = MagicMock(
        return_value=SimpleNamespace(is_success=True, documents=list(baseline_docs))
    )
    # fetch_by_id(cid) → booster doc (so the booster succeeds for enumeration queries)
    def _fetch_by_id(cid):
        for d in booster_docs:
            if d.chunk_id == cid:
                return d
        return None
    lancedb_retriever.fetch_by_id = MagicMock(side_effect=_fetch_by_id)
    main_module.lancedb_retriever_service = lancedb_retriever

    # embedding_service: encode_query → fake vector
    embedding = MagicMock()
    embedding.encode_query = MagicMock(return_value=SimpleNamespace(tolist=lambda: [0.0] * 4))
    main_module.embedding_service = embedding

    # guardrail_service: precheck.is_in_scope = True, classify_no_result is unused
    guardrail = MagicMock()
    guardrail.precheck = AsyncMock(
        return_value=SimpleNamespace(is_in_scope=True, internal_code="SUCCESS",
                                     reason="ok", public_code="SUCCESS", short_answer="",
                                     pii_detected=False, data_gap_hints=[])
    )
    guardrail._pii_warning_answer = MagicMock(return_value="Vui lòng ẩn thông tin cá nhân.")
    guardrail.expose_internal = MagicMock(return_value=False)
    guardrail._contains_sensitive_pii = MagicMock(return_value=False)
    main_module.guardrail_service = guardrail

    # query_cache disabled (avoid TTL coupling)
    main_module.query_cache = None

    # unified_pipeline: query() returns a RAGResult whose `documents` is what
    # the booster produced (post-fusion, post-cutoff). We mirror the real
    # pipeline behavior: docs that come out of the pipeline are exactly what
    # we passed in as `baseline_docs` to the booster, so the seam is observable
    # by inspecting what reached the pipeline vs the generator.
    captured = {"pipeline_baseline_docs": None, "generator_chunks": None}

    async def _fake_pipeline_query(user_query, baseline_docs, top_k, **kwargs):
        # Snapshot what reached the pipeline (i.e. post-booster) so the test
        # can assert booster docs are present.
        captured["pipeline_baseline_docs"] = list(baseline_docs)
        # The pipeline hands the docs to fusion+cutoff; the real pipeline
        # preserves the booster_injected tag through fusion. Mirror that
        # by simply returning the docs we were given (no real fusion here).
        return SimpleNamespace(
            query=user_query,
            route="graph_rag",
            documents=list(baseline_docs),
            router_result=_fake_router_result(),
            ppr_scores={},
            latency_ms=1.0,
            confidence=baseline_docs[0].score if baseline_docs else 0.0,
        )

    pipeline = MagicMock()
    pipeline.query = AsyncMock(side_effect=_fake_pipeline_query)
    pipeline._graphrag = MagicMock()
    pipeline._graphrag.graph_stats = {"nodes": 0, "edges": 0}
    # SLICE-D2.LATENCY: /v2 now calls unified_pipeline._router.route(query)
    # CONCURRENTLY with baseline retrieval, then passes the result INTO
    # pipeline.query(router_result=...). Mock the router so the concurrent
    # await resolves to a valid RouterResult.
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=_fake_router_result())
    main_module.unified_pipeline = pipeline

    # llm_generator_service: capture the chunks passed to generate_answer
    async def _fake_generate_answer(*, query, chunks, confidence, is_program_list_query, history=None):
        captured["generator_chunks"] = list(chunks)
        # Emit SourceChip-shaped dicts so UnifiedQueryResponse validation passes
        sources = []
        for c in chunks:
            cid = c.get("chunk_id", "?")
            sources.append({
                "id": cid,
                "title": cid,
                "url": None,
                "snippet": (c.get("text", "") or "")[:120],
                "data_year": "2026",
            })
        return {
            "answer": "test answer",
            "sources": sources,
            "confidence": confidence,
            "provider": "fake",
            "chunks_used": len(chunks),
        }

    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=_fake_generate_answer)
    main_module.llm_generator_service = generator

    # Patch the booster at the seam: /v2 will import
    # `services.aggregation_booster.boost_with_aggregation` (and we will also
    # patch the module-level helper to be observable).
    from services import aggregation_booster as _booster_mod

    real_boost = _booster_mod.boost_with_aggregation

    booster_called = {"n": 0, "queries": []}

    def _spy_boost(query, baseline_docs, lancedb_retriever, top_k=5, max_inject=2):
        booster_called["n"] += 1
        booster_called["queries"].append(query)
        out = real_boost(query=query, baseline_docs=list(baseline_docs),
                         lancedb_retriever=lancedb_retriever,
                         top_k=top_k, max_inject=max_inject)
        return out

    # Force the /v2 path to use a stubbed detect_aggregation_chunks so we
    # control whether the booster actually injects.
    real_detect = _booster_mod.detect_aggregation_chunks

    def _spy_detect(query):
        if is_enumeration:
            return list(booster_chunk_ids)
        return []

    # Patch both the import-inside-main.py path and the import-inside-test path
    monkeypatch.setattr(_booster_mod, "boost_with_aggregation", _spy_boost)
    monkeypatch.setattr(_booster_mod, "detect_aggregation_chunks", _spy_detect)
    # Also patch the `services.aggregation_booster` import location used by
    # main.py (it does `from services.aggregation_booster import ...`).
    if "services.aggregation_booster" in sys.modules:
        sys.modules["services.aggregation_booster"].boost_with_aggregation = _spy_boost
        sys.modules["services.aggregation_booster"].detect_aggregation_chunks = _spy_detect

    return {
        "lancedb_retriever": lancedb_retriever,
        "pipeline": pipeline,
        "generator": generator,
        "booster_called": booster_called,
        "captured": captured,
        "real_detect": real_detect,
    }


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_v2_enumeration_query_injects_booster_chunks(monkeypatch, main_module):
    """ENUMERATION query → /v2 must call boost_with_aggregation, the
    aggregate chunk_ids (liet_ke_28_nganh_2026_v2, etc.) must reach the
    pipeline's baseline_docs AND the generator's chunks list.
    """
    booster_ids = ["liet_ke_28_nganh_2026_v2", "bang_xep_hang_chi_tieu_2026_v2"]
    fakes = _install_fakes(
        main_module, monkeypatch,
        baseline_chunk_ids=["nganh_detail_cntt_2026", "nganh_detail_kt_2026"],
        booster_chunk_ids=booster_ids,
        is_enumeration=True,
    )

    query = "Trường Đại học Khoa học những ngành nghề gì?"
    request = main_module.UnifiedQueryRequest(query=query, top_k=5)

    response = await main_module.unified_query(request, raw_request=MagicMock())

    # 1. The booster was invoked
    assert fakes["booster_called"]["n"] == 1, (
        "REGRESSION: /v2 path did NOT call boost_with_aggregation for an "
        "enumeration query — booster chunks will be dropped by "
        "apply_rerank_cutoff and the model will only see per-ngành chunks."
    )

    # 2. Booster chunk_ids reached the pipeline (so they survive the cutoff)
    pipeline_doc_ids = [d.chunk_id for d in fakes["captured"]["pipeline_baseline_docs"]]
    for cid in booster_ids:
        assert cid in pipeline_doc_ids, (
            f"booster chunk {cid!r} missing from pipeline baseline_docs; "
            f"got {pipeline_doc_ids!r}"
        )

    # 3. Booster chunk_ids reached the generator (i.e. the user-visible answer)
    gen_chunk_ids = [c.get("chunk_id") for c in fakes["captured"]["generator_chunks"]]
    for cid in booster_ids:
        assert cid in gen_chunk_ids, (
            f"booster chunk {cid!r} missing from generator chunks; "
            f"got {gen_chunk_ids!r}"
        )


@pytest.mark.asyncio
async def test_v2_non_enumeration_query_does_not_inject(monkeypatch, main_module):
    """NON-ENUMERATION query → /v2 must not inject any aggregate chunks.
    The generator sees only the baseline docs (the per-ngành chunks).
    """
    booster_ids = ["liet_ke_28_nganh_2026_v2", "bang_xep_hang_chi_tieu_2026_v2"]
    fakes = _install_fakes(
        main_module, monkeypatch,
        baseline_chunk_ids=["nganh_detail_cntt_2026", "nganh_detail_kt_2026"],
        booster_chunk_ids=booster_ids,
        is_enumeration=False,
    )

    query = "Học phí ngành CNTT là bao nhiêu?"
    request = main_module.UnifiedQueryRequest(query=query, top_k=5)

    response = await main_module.unified_query(request, raw_request=MagicMock())

    # 1. Booster was called (the /v2 path always tries; it's a no-op
    # when detect_aggregation_chunks returns empty).
    assert fakes["booster_called"]["n"] == 1
    # 2. But NO aggregate chunks were injected.
    gen_chunk_ids = [c.get("chunk_id") for c in fakes["captured"]["generator_chunks"]]
    for cid in booster_ids:
        assert cid not in gen_chunk_ids, (
            f"non-enumeration query leaked booster chunk {cid!r} into the "
            f"generator — booster is meant to be a no-op for non-enum queries."
        )
    # 3. Baseline chunks are still present.
    assert "nganh_detail_cntt_2026" in gen_chunk_ids


@pytest.mark.asyncio
async def test_v2_booster_safe_when_lancedb_retriever_none(monkeypatch, main_module):
    """If lancedb_retriever_service is None, /v2 must still succeed (not 500).
    The booster seam must be guarded, not crash.
    """
    main_module.lancedb_retriever_service = None
    main_module.embedding_service = MagicMock()
    main_module.embedding_service.encode_query = MagicMock(
        return_value=SimpleNamespace(tolist=lambda: [0.0] * 4)
    )
    guardrail = MagicMock()
    guardrail.precheck = AsyncMock(
        return_value=SimpleNamespace(is_in_scope=True, internal_code="SUCCESS",
                                     reason="ok", public_code="SUCCESS",
                                     short_answer="", pii_detected=False, data_gap_hints=[])
    )
    guardrail._pii_warning_answer = MagicMock(return_value="Vui lòng ẩn thông tin cá nhân.")
    guardrail.expose_internal = MagicMock(return_value=False)
    guardrail._contains_sensitive_pii = MagicMock(return_value=False)
    main_module.guardrail_service = guardrail
    main_module.query_cache = None

    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=SimpleNamespace(
        query="x", route="padded_rag", documents=[],
        router_result=_fake_router_result(route_value="padded_rag"),
        ppr_scores={}, latency_ms=1.0, confidence=0.0,
    ))
    pipeline._graphrag = MagicMock(graph_stats={"nodes": 0, "edges": 0})
    # SLICE-D2.LATENCY: mock the now-external router.route call.
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=_fake_router_result(route_value="padded_rag"))
    main_module.unified_pipeline = pipeline

    async def _gen(**kwargs):
        return {"answer": "ok", "sources": [], "confidence": 0.0,
                "provider": "fake", "chunks_used": 0}
    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=_gen)
    main_module.llm_generator_service = generator

    request = main_module.UnifiedQueryRequest(
        query="Trường những ngành gì?", top_k=5
    )
    # Must not raise
    response = await main_module.unified_query(request, raw_request=MagicMock())
    assert response.answer == "ok"
