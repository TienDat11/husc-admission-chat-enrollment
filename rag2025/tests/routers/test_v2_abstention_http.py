# @spec(T11) abstention through the real HTTP seam
"""Ticket 11 acceptance at the surface students see: POST /v2/query and
/v2/query/stream.

The policy unit tests (tests/services/test_abstention_policy.py) prove the
rules; the generator-seam tests prove ``generate_answer`` refuses before any
provider call. This file closes the last gap named in the plan's testing
rule ("test citation/abstention ... qua HTTP layer"): it mounts the REAL
``/v2/query`` + ``/v2/query/stream`` handlers from ``src/main.py`` (FIX-05
harness pattern of test_v2_status_contract.py) over a REAL ``LLMGenerator``
whose only fake is the network client. So a refusal here travels
router -> payload builder -> _synthesize_v2_answer -> generate_answer ->
abstention gate -> response model -> (SSE frames), exactly as in production.

Every refusal test asserts the two things the ticket cares about:
  * the model was never called (``chat.await_count == 0``) — no fabricated
    number can exist in the answer;
  * the refusal is never empty and names what WAS verified / the related
    sources (criterion 6).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"

# Sys-path bootstrap mirrors test_v2_status_contract.py.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services import abstention_policy as ap  # noqa: E402
from services.abstention_policy import ABSTAIN_STRING  # noqa: E402


# ---------------------------------------------------------------------------
# Harness: real main.py handlers + real LLMGenerator, mocked client
# ---------------------------------------------------------------------------

def _load_main(monkeypatch):
    """Load src/main.py stand-alone with the provider keys blanked (the
    generator we build below is wired by hand, so startup never runs)."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")

    spec = importlib.util.spec_from_file_location("v2_abstention_http_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_doc(text: str, *, year: str, source: str | None = None,
              extra_meta: dict | None = None) -> SimpleNamespace:
    """RetrievedDocument-shaped object; ``main._build_v2_chunks`` reads
    exactly .metadata / .text / .score / .chunk_id / .source."""
    meta: dict = {"data_year": year}
    if source is not None:
        meta["source"] = source
    if extra_meta:
        meta.update(extra_meta)
    return SimpleNamespace(
        text=text,
        metadata=meta,
        score=0.7,
        chunk_id="chunk-1",
        source=source,
    )


def _make_rag_result(query: str, documents: list) -> SimpleNamespace:
    return SimpleNamespace(
        query=query,
        route="padded_rag",
        documents=documents,
        router_result=SimpleNamespace(
            skip_retrieval=False,
            auto_answer=None,
            step_back_query="step",
            intent="general",
            complexity=1,
            reasoning="ok",
            hyde_variants=[],
            is_in_scope=True,
        ),
        ppr_scores={},
        latency_ms=12.5,
        confidence=0.5,
    )


def _make_guardrail() -> MagicMock:
    """PII precheck clean; the empty-retrieval classification answers
    SUCCESS so the abstention — not a guardrail status — is what the student
    sees on the zero-evidence path."""
    decision = SimpleNamespace(
        is_in_scope=True,
        internal_code="SUCCESS",
        reason="",
        short_answer="",
        data_gap_hints=[],
        pii_detected=False,
    )
    svc = MagicMock()
    svc.precheck = AsyncMock(return_value=decision)
    svc.classify_no_result = AsyncMock(return_value=decision)
    svc.public_status = MagicMock(side_effect=lambda code: code)
    svc.expose_internal = MagicMock(return_value=False)
    svc._pii_warning_answer = MagicMock(return_value="pii")
    svc._contains_sensitive_pii = MagicMock(return_value=False)
    return svc


def _make_generator(answer_text: str) -> tuple:
    """REAL LLMGenerator; only the network client is faked.
    _resolve_admission_context is stubbed to (2026, IN_SEASON, True) so the
    gate sees a published current year — the same seam the generator-season
    tests use."""
    from services.llm_generator import LLMGenerator
    from services.season import SeasonPhase

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST SYSTEM PROMPT"
    gen.has_any_provider = True
    gen.gen_model = "test-model"
    gen.gen_fallback_model = "test-fallback"
    gen.unified_fallback_client = None

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "test-model"
    resp.provider = "test"
    fake_unified.chat = AsyncMock(return_value=resp)
    gen.unified_client = fake_unified

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer

    gen._enforce_vietnamese = _passthrough_vi
    gen._resolve_admission_context = lambda chunks: (2026, SeasonPhase.IN_SEASON, True)
    return gen, fake_unified


def _build_app(main_module, *, guardrail, pipeline, generator):
    """Mount BOTH /v2 surfaces on a fresh app. Retriever/embedding/hybrid/
    reranker/cache stay None: the baseline+booster legs skip, so the only
    moving parts are the real handler plumbing and the real gate."""
    main_module.guardrail_service = guardrail
    main_module.unified_pipeline = pipeline
    main_module.lancedb_retriever_service = None
    main_module.embedding_service = None
    main_module.hybrid_search_service = None
    main_module.reranker_service = None
    main_module.query_cache = None
    main_module.llm_generator_service = generator

    app = FastAPI()
    app.post("/v2/query", response_model=main_module.UnifiedQueryResponse)(
        main_module.unified_query
    )
    app.post("/v2/query/stream")(main_module.unified_query_stream)
    return app


def _client(monkeypatch, *, documents: list, answer_text: str = "Canned model answer goes here."):
    """Wire one full HTTP stack for a query scenario; returns
    (TestClient, the fake unified chat AsyncMock)."""
    main_module = _load_main(monkeypatch)
    gen, fake_unified = _make_generator(answer_text)

    def _query_side_effect(**kwargs):
        return _make_rag_result(kwargs.get("user_query", ""), documents)

    pipeline = MagicMock()
    pipeline.query = AsyncMock(side_effect=_query_side_effect)
    pipeline._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 0})
    pipeline._router = MagicMock()
    # The router's merged L3.1 call is irrelevant here (in-scope, no
    # auto-answer); the handler awaits it concurrently with the baseline.
    pipeline._router.route = AsyncMock(
        return_value=SimpleNamespace(
            skip_retrieval=False,
            auto_answer=None,
            step_back_query="step",
            intent="general",
            complexity=1,
            reasoning="ok",
            hyde_variants=[],
            is_in_scope=True,
        )
    )
    app = _build_app(
        main_module, guardrail=_make_guardrail(), pipeline=pipeline, generator=gen
    )
    return TestClient(app), fake_unified.chat


def _sse_frames(body: str) -> list[tuple[str, dict]]:
    """Parse 'event: X\\ndata: {...}\\n\\n' blocks from an SSE body."""
    frames = []
    for block in body.split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None:
            frames.append((event, data or {}))
    return frames


# ---------------------------------------------------------------------------
# Criterion 1 — zero evidence -> abstain (HTTP surface)
# ---------------------------------------------------------------------------

def test_zero_evidence_refuses_via_http(monkeypatch):
    client, chat = _client(monkeypatch, documents=[])
    r = client.post(
        "/v2/query", json={"query": "Điểm chuẩn năm 2026 là bao nhiêu?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert chat.await_count == 0, "model ran on empty evidence"
    # In-season current-year factual query with nothing retrieved: the
    # standard no-fabrication refusal, verbatim — now carrying the T10
    # trailer (hotline + official link) the plan requires on every refusal.
    answer = body["answer"]
    assert answer.startswith(ABSTAIN_STRING)
    assert "Hotline" in answer, "T10: a refusal must name the hotline"
    assert "tuyensinh.husc.edu.vn" in answer
    assert answer.strip(), "refusal must never be empty"
    # An intentional refusal is NOT a degraded generation (ticket 44 status).
    assert body["status_code"] == "SUCCESS"


# ---------------------------------------------------------------------------
# Criterion 2 — only other-year evidence -> refuse the current-year question
# ---------------------------------------------------------------------------

def test_year_mismatch_refuses_via_http(monkeypatch):
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2025 là 14.000.000 đồng.",
            year="2025",
            source="TB59 2025",
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    r = client.post(
        "/v2/query", json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"}
    )
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert chat.await_count == 0, "model extrapolated across years"
    assert "2026" in answer and "2025" in answer
    # Criterion 6: the refusal keeps the verified year + related source...
    assert "TB59 2025" in answer
    assert r.json()["sources"], "refusal must still carry related sources"
    # ...and never echoes the old-year NUMBER as if it answered 2026.
    assert "14.000.000" not in answer


# ---------------------------------------------------------------------------
# Criterion 3 — unresolved same-year conflict -> conflict state, never a pick
# ---------------------------------------------------------------------------

def test_unresolved_conflict_returns_conflict_state_via_http(monkeypatch):
    conflict = SimpleNamespace(
        admission_year=2026,
        ma_nganh="",
        field="hoc_phi_tin_chi_vnd",
        statements={"TB59 2025": "14.000.000", "TB60 2026": "16.000.000"},
        status="open",
    )
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2026 là 14.000.000 đồng.",
            year="2026",
            source="TB59 2025",
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    ap.configure_conflict_source(lambda: [conflict])
    try:
        r = client.post(
            "/v2/query", json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"}
        )
    finally:
        ap.configure_conflict_source(None)
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert chat.await_count == 0, "model self-selected past an open conflict"
    assert "mâu thuẫn" in answer
    # Both sides surface; the system never picks one.
    assert "14.000.000" in answer and "16.000.000" in answer
    assert "TB59 2025" in answer and "TB60 2026" in answer


# ---------------------------------------------------------------------------
# Criterion 4 — citation lineage missing -> abstain
# ---------------------------------------------------------------------------

def test_lineage_missing_refuses_via_http(monkeypatch):
    # Canonical-store chunk (record_id present) with NO traceable source
    # document: an uncitable number is not an answer.
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2026 là 18.000.000 đồng.",
            year="2026",
            extra_meta={"record_id": "rec-1", "record_version_key": "rvk-1"},
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    r = client.post(
        "/v2/query", json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"}
    )
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert chat.await_count == 0, "untraceable number was generated over"
    assert "không truy vết được" in answer
    assert answer.strip()


# ---------------------------------------------------------------------------
# Criterion 5 — major outside the current admissions scope -> abstain
# ---------------------------------------------------------------------------

def test_out_of_scope_major_refuses_via_http(monkeypatch):
    # 7729001 (Y khoa) is not in the 2026 HUSC registry (measured:
    # get_whitelist(2026) -> 35 codes, member False). The chunk is otherwise
    # perfectly answerable evidence — the SCOPE rule must still win.
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2026 là 18.000.000 đồng.",
            year="2026",
            source="TB60 2026",
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    r = client.post(
        "/v2/query", json={"query": "Học phí 2026 của ngành 7729001 là bao nhiêu?"}
    )
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert chat.await_count == 0, "answered for an out-of-scope major"
    assert "7729001" in answer
    assert "không thuộc phạm vi tuyển sinh" in answer


# ---------------------------------------------------------------------------
# Control — sufficient evidence: the model DOES run and its answer ships
# ---------------------------------------------------------------------------

def test_sufficient_evidence_answers_via_http(monkeypatch):
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2026 là 18.000.000 đồng.",
            year="2026",
            source="TB60 2026",
        )
    ]
    client, chat = _client(
        monkeypatch,
        documents=docs,
        answer_text="Học phí 2026 ngành CNTT là 18.000.000 đồng theo TB60 2026.",
    )
    r = client.post(
        "/v2/query", json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"}
    )
    assert r.status_code == 200
    body = r.json()
    # Gate must NOT over-refuse: the model ran exactly once and its number
    # reached the student.
    assert chat.await_count == 1
    assert "18.000.000" in body["answer"]
    assert body["status_code"] == "SUCCESS"


# ---------------------------------------------------------------------------
# Criterion 6 on the streamed surface — the refusal reaches the SSE frames
# ---------------------------------------------------------------------------

def test_stream_carries_refusal_without_model_call(monkeypatch):
    """SSE refusal travels the REAL /v2/query/stream handler.

    Honest scope: src/main.py's SSE route (docstring at main.py:2102-2104
    claims it uses ``generate_answer_stream`` but it does NOT) calls
    ``_build_v2_query_payload`` -> ``generate_answer_sync`` and then just
    re-chunks the already-guarded answer string via
    ``_chunk_text_for_stream``. So this test exercises the SYNC gate
    (llm_generator.py:718), not the stream-variant gate
    (llm_generator.py:1103). The stream-variant gate is covered separately by
    ``test_stream_variant_gate_refuses_without_model_call``. (The HTTP-stream
    mutation witness is STILL-GREEN precisely because the SSE route never
    calls generate_answer_stream today — logged as a ticket-11 follow-up.)"""
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2025 là 14.000.000 đồng.",
            year="2025",
            source="TB59 2025",
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    r = client.post(
        "/v2/query/stream",
        json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"},
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    events = [e for e, _ in frames]
    assert "meta" in events and "done" in events
    done_answer = next(d for e, d in frames if e == "done")["answer"]
    meta = next(d for e, d in frames if e == "meta")
    assert chat.await_count == 0, "stream generated over thin evidence"
    assert "2025" in done_answer and "TB59 2025" in done_answer
    assert "14.000.000" not in done_answer
    assert done_answer.strip()
    assert meta["status_code"] == "SUCCESS"

# ---------------------------------------------------------------------------
# Stream-variant gate (generator seam) — the branch the SSE route does NOT
# call today, but which must stay correct so a future wiring switch does not
# ship an unguarded streamer. Mutating llm_generator.py:1103 to
# `if False and ...` makes THIS test go RED (the HTTP-stream witness stays
# green only because the route bypasses generate_answer_stream).
# ---------------------------------------------------------------------------

def _stream_variant_generator(answer_text: str):
    """A real LLMGenerator with ONLY the network mocked; _build_context is
    stubbed so the gate (which returns before context building) short-circuits
    the settings/tiktoken import path entirely."""
    from services.llm_generator import LLMGenerator
    from services.season import SeasonPhase

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST SYSTEM PROMPT"
    gen.has_any_provider = True
    gen.gen_model = "test-model"
    gen.gen_fallback_model = "test-fallback"
    gen.unified_fallback_client = None
    gen._resolve_admission_context = lambda chunks: (2026, SeasonPhase.IN_SEASON, True)
    gen._build_context = lambda chunks: "CONTEXT"
    async def _vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _vi

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "test-model"
    resp.provider = "test"
    fake_unified.chat = AsyncMock(return_value=resp)
    fake_unified.chat_stream = AsyncMock(return_value=iter([]))
    gen.unified_client = fake_unified
    return gen, fake_unified, fake_unified.chat


async def _collect_stream(gen, query, chunks):
    pieces = []
    async for piece in gen.generate_answer_stream(
        query=query, chunks=chunks, confidence=0.5
    ):
        pieces.append(piece)
    return pieces


@pytest.mark.asyncio
async def test_stream_variant_gate_refuses_without_model_call(monkeypatch):
    """The generate_answer_stream gate must refuse before any chunk hits the
    model — covering llm_generator.py:1103 (the line the HTTP-stream mutation
    cannot reach)."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    gen, _, chat = _stream_variant_generator("phony")
    chunks = [
        {
            "text": "Học phí ngành CNTT năm 2025 là 14.000.000 đồng.",
            "metadata": {"data_year": "2025", "source": "TB59 2025"},
            "score": 0.7,
            "chunk_id": "chunk-1",
            "source": "TB59 2025",
        }
    ]
    pieces = await _collect_stream(
        gen,
        "Học phí 2026 của ngành CNTT là bao nhiêu?",
        chunks,
    )
    assert chat.await_count == 0, "stream variant generated over thin evidence"
    answer = "".join(pieces)
    assert "2026" in answer and "2025" in answer and "TB59 2025" in answer
    assert "14.000.000" not in answer
    assert answer.strip()


# ---------------------------------------------------------------------------
# Ticket 17 — the abstention must be CLASSIFIABLE at the HTTP boundary
#
# The gate has always set provider="evidence-policy:{reason_code}", but the
# value died at the final mile: the /v2 response builder had no `provider`
# key, UnifiedQueryResponse had no field, and the SSE meta dict omitted it.
# A policy refusal therefore reached the FE as status_code SUCCESS plus
# ordinary prose — indistinguishable from a real answer — so the FE had no
# choice but to classify by prose, which ticket 17 forbids.
#
# These tests assert on the TRANSPORT payload, not the generator's return
# value: asserting the generator's own dict is exactly the blindness that let
# this bug survive. Mutation witness: delete the "provider" line from
# response_payload in main.py and both tests go RED.
# ---------------------------------------------------------------------------

def test_json_payload_carries_the_abstention_provider(monkeypatch):
    """A policy refusal must be self-identifying on the JSON surface."""
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2025 là 14.000.000 đồng.",
            year="2025",
            source="TB59 2025",
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    r = client.post(
        "/v2/query", json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert chat.await_count == 0, "policy refusal must not call the model"
    assert body["provider"] == "evidence-policy:year-mismatch", (
        "the refusal must carry its reason_code so the FE can classify it "
        "without reading the prose"
    )


def test_model_answer_does_not_look_like_an_abstention(monkeypatch):
    """The complement: a real answer must NOT carry an evidence-policy tag.

    Guards the classifier's other branch — if a model answer also carried a
    policy tag (or a bare null that the FE read as "abstain"), every normal
    answer would render as a refusal.
    """
    docs = [
        _make_doc(
            "Học phí 2026 ngành CNTT là 18.000.000 đồng theo TB60 2026.",
            year="2026",
            source="TB60 2026",
        )
    ]
    client, chat = _client(
        monkeypatch,
        documents=docs,
        answer_text="Học phí 2026 ngành CNTT là 18.000.000 đồng theo TB60 2026.",
    )
    r = client.post(
        "/v2/query", json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert chat.await_count == 1
    provider = body["provider"]
    assert not (
        isinstance(provider, str) and provider.startswith("evidence-policy:")
    ), f"a model answer must not be tagged as a policy refusal (got {provider!r})"


def test_stream_meta_carries_the_abstention_provider(monkeypatch):
    """The SSE surface must classify the refusal exactly like the JSON one."""
    docs = [
        _make_doc(
            "Học phí ngành CNTT năm 2025 là 14.000.000 đồng.",
            year="2025",
            source="TB59 2025",
        )
    ]
    client, chat = _client(monkeypatch, documents=docs)
    r = client.post(
        "/v2/query/stream",
        json={"query": "Học phí 2026 của ngành CNTT là bao nhiêu?"},
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    meta = next(d for e, d in frames if e == "meta")
    assert chat.await_count == 0
    assert meta["provider"] == "evidence-policy:year-mismatch", (
        "a streamed refusal that cannot be told apart from an answer forces "
        "the FE to guess from prose"
    )
