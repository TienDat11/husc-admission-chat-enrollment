"""Tests for /v2/query answer-synthesis path.

Regression: prior /v2 handler bypassed LLMGenerator.generate_answer() and
called llm.chat() inline, which skipped the S16.5 URL-faithfulness guard,
the season-aware GAP_DISCLAIMER, and the S15.6 contact-keyword abstain
guard. It also swallowed RouterResult.auto_answer (CONTACT_BLOCK) with a
generic "không có thông tin" string.

These tests pin the fix: a small extracted helper `_synthesize_v2_answer`
that routes through `generator.generate_answer()` for normal queries and
returns `router_result.auto_answer` verbatim for `skip_retrieval` cases.

Approach: helper-refactor (not TestClient on the heavy main.app). The
helper is imported directly from src/main.py via importlib, then unit-
tested with sentinel generator and fake RAGResult/RouterResult. This
avoids standing up the full GraphRAG pipeline in CI.
"""
from __future__ import annotations
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"


def _build_main_module(monkeypatch):
    """Load src/main.py as a stand-alone module so the helper can be imported
    without registering the FastAPI app globally (or pulling in the
    production config). Returns the freshly-loaded module.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location("v2_synthesize_helper_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module(monkeypatch):
    # Defuse heavy env-driven singletons that are not under test.
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    return _build_main_module(monkeypatch)


# ────────────────────────────────────────────────────────────────────────
# Fake RAGResult / RouterResult builders
# ────────────────────────────────────────────────────────────────────────

def _fake_router_result(*, skip_retrieval: bool = False, auto_answer: str | None = None,
                       step_back: str = "step", intent: str = "general",
                       complexity: int = 1, reasoning: str = "ok",
                       hyde_variants: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        skip_retrieval=skip_retrieval,
        auto_answer=auto_answer,
        step_back_query=step_back,
        intent=intent,
        complexity=complexity,
        reasoning=reasoning,
        hyde_variants=hyde_variants or [],
    )


def _fake_rag_result(*, documents: list | None = None, route: str = "padded_rag",
                     confidence: float = 0.0, router_result: SimpleNamespace | None = None
                    ) -> SimpleNamespace:
    return SimpleNamespace(
        query="test query",
        route=route,
        documents=documents if documents is not None else [],
        router_result=router_result or _fake_router_result(),
        ppr_scores={},
        latency_ms=1.0,
        confidence=confidence,
    )


def _fake_doc(text: str = "chunk text", *, summary: str = "summary",
              source: str = "doc.pdf", chunk_id: str = "c1",
              metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        source=source,
        chunk_id=chunk_id,
        metadata=metadata or {"summary": summary, "source": source, "data_year": 2026},
        score=0.9,
        point_id=chunk_id,
    )


def _run(coro):
    """Run an async coroutine in tests without requiring pytest-asyncio.

    Uses a FRESH event loop per call — a shared global loop gets closed/polluted
    by other async tests in the suite (the v2 tests passed alone but failed
    after tests/services/ ran), so never reuse asyncio.get_event_loop().
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────

def test_normal_query_routes_through_generate_answer(main_module):
    """A non-skip_retrieval RAGResult must call generator.generate_answer
    exactly once and forward the produced answer string.
    """
    helper = main_module._synthesize_v2_answer

    doc = _fake_doc(text="HUSC ngành CNTT học phí 1.2 triệu/tháng", chunk_id="c1")
    rag_result = _fake_rag_result(
        documents=[doc], confidence=0.87, route="graph_rag",
        router_result=_fake_router_result(),
    )

    sentinel = {
        "answer": "Ngành CNTT học phí 1.2 triệu/tháng.",
        "sources": ["TB100", "doc.pdf"],
        "confidence": 0.87,
        "provider": "mimo-v2.5-pro (test)",
        "chunks_used": 1,
    }
    generator = MagicMock()
    generator.generate_answer = AsyncMock(return_value=sentinel)

    out = _run(helper(rag_result=rag_result, query="học phí ngành CNTT?", generator=generator))

    # 1. generator.generate_answer was called exactly once
    generator.generate_answer.assert_awaited_once()
    # 2. answer flows to the helper output
    assert out["answer"] == sentinel["answer"]
    assert out["provider"] == sentinel["provider"]
    assert out["confidence"] == rag_result.confidence
    # 3. chunks were built (text + summary + metadata)
    call_kwargs = generator.generate_answer.await_args.kwargs
    chunks_passed = call_kwargs["chunks"]
    assert len(chunks_passed) == 1
    assert chunks_passed[0]["text"] == doc.text
    assert chunks_passed[0]["summary"] == "summary"
    assert chunks_passed[0]["metadata"] is not None
    # 4. query passed through
    assert call_kwargs["query"] == "học phí ngành CNTT?"
    # 5. confidence is from rag_result, not from sentinel
    assert call_kwargs["confidence"] == rag_result.confidence


def test_skip_retrieval_auto_answer_is_not_swallowed(main_module):
    """When the router sets skip_retrieval=True and provides auto_answer
    (e.g. CONTACT_BLOCK), the helper MUST return that string verbatim
    and MUST NOT call generator.generate_answer. This is the regression:
    the old inline llm.chat() path emitted a generic "Không có thông tin
    phù hợp." instead of the auto_answer.
    """
    helper = main_module._synthesize_v2_answer

    contact_block = (
        "Để được tư vấn trực tiếp về tuyển sinh HUSC, vui lòng liên hệ:\n"
        "- Hotline: 0234.xxx.xxxx\n"
        "- Cổng thông tin: https://tuyensinh.husc.edu.vn"
    )
    rag_result = _fake_rag_result(
        documents=[],  # skip_retrieval path returns empty docs
        confidence=1.0,  # confidence forced to 1.0 by pipeline.py
        route="padded_rag",
        router_result=_fake_router_result(
            skip_retrieval=True,
            auto_answer=contact_block,
            intent="hyde_contact_block",
        ),
    )

    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=AssertionError(
        "generate_answer MUST NOT be called when router has auto_answer"
    ))

    out = _run(helper(rag_result=rag_result, query="hotline tư vấn?", generator=generator))

    # 1. auto_answer string flows to the helper output verbatim
    assert out["answer"] == contact_block
    # 2. sources empty (no chunks retrieved on skip path)
    assert out["sources"] == []
    # 3. confidence preserved
    assert out["confidence"] == 1.0
    # 4. provider signals the auto-answer path so observability can tell
    assert out["provider"] == "auto_answer"


def test_helper_returns_expected_field_shape(main_module):
    """The helper output must always carry the fields the /v2 response
    consumes (answer, sources, confidence, provider). This is the
    contract that downstream `UnifiedQueryResponse(**payload)` relies on.
    """
    helper = main_module._synthesize_v2_answer

    rag_result = _fake_rag_result(
        documents=[_fake_doc()], confidence=0.5,
        router_result=_fake_router_result(),
    )
    generator = MagicMock()
    generator.generate_answer = AsyncMock(return_value={
        "answer": "abc",
        "sources": ["s1"],
        "confidence": 0.5,
        "provider": "p1",
        "chunks_used": 1,
    })

    out = _run(helper(rag_result=rag_result, query="x", generator=generator))
    for key in ("answer", "sources", "confidence", "provider"):
        assert key in out, f"missing {key!r} in helper output"
    assert isinstance(out["sources"], list)


def test_v2_endpoint_does_not_contain_inline_llm_chat(monkeypatch):
    """Pin the regression at the file level: the /v2 handler MUST NOT
    call `llm.chat(...)` inline. If a future refactor reintroduces the
    bug, this test will fail loudly.
    """
    main_module = _build_main_module(monkeypatch)
    source = MAIN_PATH.read_text(encoding="utf-8")

    # Find the /v2/query handler block (between the @app.post and the next
    # @app.post or non-indented def).
    start = source.find('@app.post("/v2/query"')
    assert start != -1, "/v2/query endpoint not found in src/main.py"
    # End the block at the next top-level decorator or def
    after = source.find('\n@app.', start + 1)
    after2 = source.find('\ndef ', start + 1)
    candidates = [x for x in (after, after2) if x != -1]
    end = min(candidates) if candidates else len(source)
    v2_block = source[start:end]

    # The old bug: llm.chat() inside the /v2 handler
    assert "llm.chat(" not in v2_block, (
        "REGRESSION: /v2/query handler still contains inline llm.chat(...). "
        "Use _synthesize_v2_answer() which routes through LLMGenerator."
    )
    # The fix: the handler MUST route through the helper either directly
    # (legacy `_synthesize_v2_answer(...)`) or via the shared /v2 core
    # (`_build_v2_query_payload(...)`) introduced for /v2/query/stream
    # (SLICE-B.LATENCY). Either pattern keeps the URL-faithfulness +
    # season-gap + contact-abstain guards active.
    assert (
        "_synthesize_v2_answer(" in v2_block
        or "_build_v2_query_payload(" in v2_block
    ), (
        "REGRESSION: /v2/query handler no longer routes through the "
        "LLMGenerator helper. Must call _synthesize_v2_answer() (direct) "
        "or _build_v2_query_payload() (shared /v2 core) so the post-guards run."
    )

