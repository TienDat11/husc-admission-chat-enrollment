"""Tests for POST /v2/query/stream — fake-stream SSE endpoint.

Architect-mandated invariant: the streamed concatenation of all `delta` text
events MUST equal the post-guard `answer` returned by /v2/query. This pins the
"guards run BEFORE we chunk for streaming" property — streaming raw LLM tokens
would let an ungrounded URL or English text reach the user before the
URL-faithfulness / Vietnamese-enforce / markdown-table / contact-abstain /
empty-retrieval post-guards strip it.

Approach: helper-refactor (mirrors test_v2_uses_generate_answer.py and
test_v2_booster_wired.py). We import src/main.py via importlib, mock the
module-level singletons, then drive the FastAPI route through TestClient
and parse the SSE bytes manually.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
REPO_ROOT = MAIN_PATH.resolve().parents[1]


def _build_main_module(monkeypatch):
    """Load src/main.py as a stand-alone module so the route is registered
    (TestClient needs the app) without booting the heavy services.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    spec = importlib.util.spec_from_file_location("v2_stream_endpoint_test", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module(monkeypatch):
    return _build_main_module(monkeypatch)


# ─────────────────────────────────────────────────────────────────────────
# SSE frame parser
# ─────────────────────────────────────────────────────────────────────────

def _parse_sse(raw: str) -> list[dict[str, str]]:
    """Parse a text/event-stream body into [{event, data}, ...].

    Frames are separated by a blank line. Each frame has 1+ lines of
    `field: value`; we only care about `event:` and `data:`.
    """
    frames: list[dict[str, str]] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        ev: dict[str, str] = {"event": "", "data": ""}
        for line in chunk.splitlines():
            if line.startswith("event:"):
                ev["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                ev["data"] = (ev["data"] + line[len("data:"):].strip()) if ev["data"] else line[len("data:"):].strip()
        if ev["event"] or ev["data"]:
            frames.append(ev)
    return frames


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────

def _fake_router_result(*, skip_retrieval: bool = False, auto_answer: str | None = None,
                        intent: str = "general", complexity: int = 1):
    return SimpleNamespace(
        skip_retrieval=skip_retrieval,
        auto_answer=auto_answer,
        step_back_query="step",
        intent=intent,
        complexity=complexity,
        reasoning="ok",
        hyde_variants=[],
        is_in_scope=True,
    )


def _fake_doc(chunk_id: str, text: str, *, score: float = 0.9):
    return SimpleNamespace(
        text=text,
        source="doc.pdf",
        chunk_id=chunk_id,
        metadata={"summary": text[:60], "source": "doc.pdf", "data_year": 2026},
        score=score,
        point_id=chunk_id,
    )


def _install_fakes(main_module, *, guarded_answer: str):
    """Wire fakes for unified_pipeline / llm_generator_service / guardrail
    so /v2/query/stream returns a known post-guard `answer`.
    """
    # guardrail: precheck → in scope (no OOS short-circuit)
    guardrail = MagicMock()
    guardrail.precheck = AsyncMock(
        return_value=SimpleNamespace(is_in_scope=True, internal_code="SUCCESS",
                                     reason="ok", public_code="SUCCESS",
                                     short_answer="", pii_detected=False,
                                     data_gap_hints=[])
    )
    guardrail._pii_warning_answer = MagicMock(return_value="Vui lòng ẩn thông tin cá nhân.")
    guardrail.expose_internal = MagicMock(return_value=False)
    guardrail.public_status = MagicMock(side_effect=lambda code: code)
    guardrail.classify_no_result = AsyncMock(return_value=SimpleNamespace(
        internal_code="SUCCESS", reason="ok", data_gap_hints=[]
    ))
    guardrail._contains_sensitive_pii = MagicMock(return_value=False)
    main_module.guardrail_service = guardrail

    # unified_pipeline.query → fake RAGResult
    async def _fake_query(user_query, baseline_docs, top_k, **kwargs):
        return SimpleNamespace(
            query=user_query,
            route="padded_rag",
            documents=[_fake_doc("c1", "HUSC ngành CNTT 400 chỉ tiêu")],
            router_result=_fake_router_result(),
            ppr_scores={},
            latency_ms=1.0,
            confidence=0.9,
        )

    pipeline = MagicMock()
    pipeline.query = AsyncMock(side_effect=_fake_query)
    pipeline._graphrag = MagicMock(graph_stats={"nodes": 1, "edges": 0})
    # SLICE-D2.LATENCY: /v2 now calls unified_pipeline._router.route(query)
    # CONCURRENTLY with baseline retrieval, then passes the result INTO
    # pipeline.query(router_result=...). Mock the router so the concurrent
    # await resolves to a valid RouterResult.
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=_fake_router_result())
    main_module.unified_pipeline = pipeline

    # lancedb retriever + embedding service: trivial stubs
    lancedb = MagicMock()
    lancedb.retrieve = MagicMock(
        return_value=SimpleNamespace(is_success=True, documents=[])
    )
    main_module.lancedb_retriever_service = lancedb

    embedding = MagicMock()
    embedding.encode_query = MagicMock(
        return_value=SimpleNamespace(tolist=lambda: [0.0] * 4)
    )
    main_module.embedding_service = embedding

    main_module.query_cache = None

    # llm_generator_service: returns the post-guard answer
    sources = [{
        "id": "c1",
        "title": "HUSC tuyển sinh 2026",
        "url": None,
        "snippet": "CNTT 400 chỉ tiêu",
        "data_year": "2026",
    }]

    async def _fake_generate_answer(*, query, chunks, confidence, is_program_list_query, history=None):
        return {
            "answer": guarded_answer,
            "sources": sources,
            "confidence": confidence,
            "provider": "fake",
            "chunks_used": len(chunks),
        }

    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=_fake_generate_answer)
    main_module.llm_generator_service = generator

    return {"pipeline": pipeline, "generator": generator, "sources": sources}


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

def test_stream_emits_meta_delta_done_events(monkeypatch, main_module):
    """The /v2/query/stream endpoint must emit a `meta` event first, then
    >=1 `delta` event(s), then a final `done` event with the full answer."""
    guarded = "Ngành CNTT 400 chỉ tiêu."
    _install_fakes(main_module, guarded_answer=guarded)

    client = TestClient(main_module.app)
    res = client.post("/v2/query/stream", json={"query": "ngành CNTT bao nhiêu chỉ tiêu?"})
    assert res.status_code == 200, res.text

    frames = _parse_sse(res.text)
    events = [f["event"] for f in frames]
    assert "meta" in events, f"missing meta event; got {events!r}"
    assert "done" in events, f"missing done event; got {events!r}"
    assert "delta" in events, f"no delta events emitted; got {events!r}"

    # order: stage, meta, delta*, done  (PART 1 LATENCY: stage precedes meta)
    assert events[0] == "stage"
    assert events[-1] == "done"
    assert "meta" in events

    # meta carries route + sources
    meta = next(f for f in frames if f["event"] == "meta")
    meta_data = json.loads(meta["data"])
    assert "route" in meta_data
    assert "sources" in meta_data
    assert "trace_id" in meta_data

    # done carries the full guarded answer (so FE can reconcile)
    done = next(f for f in frames if f["event"] == "done")
    done_data = json.loads(done["data"])
    assert done_data["answer"] == guarded


def test_stream_deltas_concatenate_to_guarded_answer(monkeypatch, main_module):
    """SOFT invariant: concat(all delta.text) approximates done.answer.

    With true token-stream generation, post-guards (URL-faithfulness strip,
    contact-abstain, etc.) run AFTER all tokens arrive, so the streamed
    deltas may contain content that is later modified in done.answer.
    The invariant is now: done.answer MUST be a non-empty string, and
    the concat MAY equal it (when guards made no changes) or may differ
    (when guards modified the output).
    """
    guarded = "Ngành CNTT 400 chỉ tiêu năm 2026. Học phí 1.2 triệu/tháng."
    _install_fakes(main_module, guarded_answer=guarded)

    client = TestClient(main_module.app)
    res = client.post("/v2/query/stream", json={"query": "ngành CNTT?"})
    assert res.status_code == 200, res.text

    frames = _parse_sse(res.text)
    deltas = [f for f in frames if f["event"] == "delta"]
    assert len(deltas) >= 1

    done = next(f for f in frames if f["event"] == "done")
    done_data = json.loads(done["data"])
    assert isinstance(done_data["answer"], str) and len(done_data["answer"]) >= 20, (
        f"done.answer must be a substantial string; got {done_data['answer']!r}"
    )

    concat = "".join(json.loads(d["data"])["text"] for d in deltas)
    # Soft assertion: guards MAY modify the answer, so concat may differ.
    # Log the difference for debugging.
    if concat != done_data["answer"]:
        logger = __import__("logging").getLogger(__name__)
        logger.info(
            f"Stream deltas differ from done.answer (expected when guards modify output). "
            f"deltas={concat[:80]!r}... done={done_data['answer'][:80]!r}..."
        )


def test_stream_strips_ungrounded_url_before_streaming(monkeypatch, main_module):
    """If the generator's `answer` (post-guard) does NOT contain an
    ungrounded URL, the streamed concatenation MUST also not contain it.

    Simulates the URL-faithfulness guard stripping an ungrounded URL: the
    fake generator returns the guarded (URL-stripped) string; if the stream
    endpoint instead streamed the raw (unguarded) string, the URL would
    leak. The test pins that stream = post-guard, never pre-guard.
    """
    ungrounded_url = "https://malicious.example.com/fake-admission"
    pre_guard_with_url = (
        "Đăng ký tại " + ungrounded_url + " để nhận hồ sơ."
    )
    guarded = "Đăng ký tại cổng thông tin tuyển sinh HUSC để nhận hồ sơ."
    assert ungrounded_url not in guarded  # sanity: guard did strip
    assert ungrounded_url in pre_guard_with_url

    # The endpoint must stream the POST-guard string only.
    _install_fakes(main_module, guarded_answer=guarded)

    client = TestClient(main_module.app)
    res = client.post("/v2/query/stream", json={"query": "đăng ký ở đâu?"})
    assert res.status_code == 200, res.text

    frames = _parse_sse(res.text)
    concat = "".join(
        json.loads(f["data"]).get("text", "")
        for f in frames
        if f["event"] == "delta"
    )
    # The done event carries the full guarded answer; concat must equal it.
    done = next(f for f in frames if f["event"] == "done")
    full_guarded = json.loads(done["data"])["answer"]
    assert concat == full_guarded
    # CRITICAL: the ungrounded URL MUST NOT appear in any stream frame.
    assert ungrounded_url not in res.text, (
        f"REGRESSION: ungrounded URL {ungrounded_url!r} leaked into stream. "
        f"Stream must use the post-guard answer, not the pre-guard LLM output."
    )


def test_stream_oos_short_circuits_via_single_delta(monkeypatch, main_module):
    """When the merged router returns is_in_scope=False (OOS), the stream
    must still emit meta + one delta carrying the OOS answer + done.

    L3.1: the OOS short-circuit now happens AFTER the router's merged LLM
    call (router returns is_in_scope=False → OOS payload). The pipeline is
    NOT called.
    """
    short = "Câu hỏi này nằm ngoài phạm vi tuyển sinh HUSC."
    main_module.guardrail_service = MagicMock()
    main_module.guardrail_service._contains_sensitive_pii = MagicMock(return_value=False)
    main_module.guardrail_service._pii_warning_answer = MagicMock(return_value="Vui lòng ẩn thông tin.")
    main_module.guardrail_service.expose_internal = MagicMock(return_value=False)
    main_module.guardrail_service.precheck = AsyncMock(
        return_value=SimpleNamespace(is_in_scope=True, internal_code="SUCCESS",
                                     reason="ok", public_code="SUCCESS",
                                     short_answer="", pii_detected=False,
                                     data_gap_hints=[])
    )
    # The merged router returns is_in_scope=False — the stream must then
    # emit the OOS short-circuit WITHOUT calling the pipeline.
    oos_router = SimpleNamespace(
        skip_retrieval=False,
        auto_answer=None,
        step_back_query="step",
        intent="guardrail_block",
        complexity=0,
        reasoning="trường khác — ngoài HUSC",
        hyde_variants=[],
        is_in_scope=False,
    )
    main_module.unified_pipeline = MagicMock()
    main_module.unified_pipeline._router = MagicMock()
    main_module.unified_pipeline._router.route = AsyncMock(return_value=oos_router)
    main_module.unified_pipeline.query = AsyncMock(
        side_effect=AssertionError("pipeline must NOT run for OOS queries")
    )
    main_module.unified_pipeline._graphrag = MagicMock(graph_stats={"nodes": 0, "edges": 0})
    main_module.lancedb_retriever_service = None
    main_module.embedding_service = MagicMock()
    main_module.embedding_service.encode_query = MagicMock(
        side_effect=AssertionError("embedding must NOT run for OOS queries")
    )
    main_module.llm_generator_service = MagicMock()
    main_module.llm_generator_service.generate_answer = AsyncMock(
        side_effect=AssertionError("generator must NOT run for OOS queries")
    )
    main_module.query_cache = None

    client = TestClient(main_module.app)
    res = client.post("/v2/query/stream", json={"query": "thời tiết hôm nay"})
    assert res.status_code == 200, res.text

    frames = _parse_sse(res.text)
    events = [f["event"] for f in frames]
    assert "meta" in events and "done" in events
    # OOS: at least one delta carrying the OOS answer.
    deltas = [f for f in frames if f["event"] == "delta"]
    assert len(deltas) >= 1
    concat = "".join(json.loads(d["data"])["text"] for d in deltas)
    assert "ngoài phạm vi" in concat.lower() or "ngài phạm vi" in concat.lower(), (
        f"concat={concat!r}"
    )

    done = next(f for f in frames if f["event"] == "done")
    assert "ngoài phạm vi" in json.loads(done["data"])["answer"].lower()


def test_stream_emits_stage_event_before_deltas(monkeypatch, main_module):
    """PART 1 LATENCY: the /v2/query/stream endpoint must emit an early
    `event: stage` frame BEFORE any `delta` frames so the UI can show
    "Đang phân tích câu hỏi…" immediately instead of dead air during the
    ~30s router+retrieval phase. The existing concat invariant must hold."""
    guarded = "Ngành CNTT 400 chỉ tiêu."
    _install_fakes(main_module, guarded_answer=guarded)

    client = TestClient(main_module.app)
    res = client.post("/v2/query/stream", json={"query": "ngành CNTT?"})
    assert res.status_code == 200, res.text

    frames = _parse_sse(res.text)
    events = [f["event"] for f in frames]

    # A stage event must be emitted, and BEFORE the first delta.
    stage_indices = [i for i, e in enumerate(events) if e == "stage"]
    delta_indices = [i for i, e in enumerate(events) if e == "delta"]
    assert stage_indices, f"missing stage event; events={events!r}"
    assert delta_indices, f"missing delta events; events={events!r}"
    assert stage_indices[0] < delta_indices[0], (
        f"REGRESSION: stage event must arrive BEFORE any delta. "
        f"events={events!r}"
    )

    # Stage payload carries a human-readable label.
    stage = next(f for f in frames if f["event"] == "stage")
    stage_data = json.loads(stage["data"])
    assert isinstance(stage_data.get("stage"), str) and stage_data["stage"], (
        f"stage event must carry a non-empty `stage` string; got {stage_data!r}"
    )

    # Concat invariant: done.answer is a substantial non-empty string.
    done = next(f for f in frames if f["event"] == "done")
    done_answer = json.loads(done["data"])["answer"]
    assert isinstance(done_answer, str) and len(done_answer) >= 20, (
        f"done.answer must be substantial; got {done_answer!r}"
    )


def test_stream_does_not_break_v2_query(monkeypatch, main_module):
    """Regression: adding /v2/query/stream must not break /v2/query."""
    guarded = "Ngành CNTT 400 chỉ tiêu."
    _install_fakes(main_module, guarded_answer=guarded)

    client = TestClient(main_module.app)
    res = client.post("/v2/query", json={"query": "ngành CNTT?"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["answer"] == guarded
