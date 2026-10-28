"""Tests for eval_run_instrumented — S2 instrumented 86Q runner.

TDD: fakes-only, no real LLM/network. Mirrors the existing test pattern in
this repo (importlib spec_from_file_location + asyncio.run() inside sync tests).
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np
import pytest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "eval_run_instrumented.py"
)
EVAL_RECORD_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "eval_record.py"
)


# =============================================================================
# FAKES — defined ONLY in this test file. The runner module MUST NOT define any
# Fake* class at module level (C6 guard). If you copy fakes into the runner,
# the C6 test will fail.
# =============================================================================


class FakeDoc:
    """Mimics a RetrievedDocument for the runner's getattr-based extraction."""

    def __init__(
        self,
        chunk_id: str,
        source: str,
        score: float,
        text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.source = source
        self.score = score
        self.text = text
        self.metadata = metadata or {}


class FakeRouterResult:
    def __init__(
        self,
        *,
        all_queries_for_embedding: List[str] | None = None,
        complexity: int = 2,
        intent: str = "factual_lookup",
        skip_retrieval: bool = False,
        auto_answer: Any = None,
        route: str = "graph_rag",
    ) -> None:
        self.all_queries_for_embedding = all_queries_for_embedding or ["var1", "var2"]
        self.complexity = complexity
        self.intent = intent
        self.skip_retrieval = skip_retrieval
        self.auto_answer = auto_answer
        self.route = route


class _FakeSubRouter:
    def __init__(self, result: FakeRouterResult) -> None:
        self._result = result

    async def route(self, query: str) -> FakeRouterResult:
        return self._result


class FakeRagResult:
    def __init__(
        self,
        route: str,
        router_result: FakeRouterResult,
        documents: List[FakeDoc],
    ) -> None:
        self.route = route
        self.router_result = router_result
        self.documents = documents


class FakePipeline:
    def __init__(self, router_result: FakeRouterResult, rag_result: FakeRagResult) -> None:
        self._router = _FakeSubRouter(router_result)
        self._rag_result = rag_result

    async def query(self, *, user_query, baseline_docs, top_k, router_result):
        return self._rag_result


class _FakeRetrieveResult:
    def __init__(self, documents: List[FakeDoc], is_success: bool = True) -> None:
        self.documents = documents
        self.is_success = is_success


class FakeRetriever:
    def __init__(self, documents: List[FakeDoc]) -> None:
        self._documents = documents

    def retrieve(self, vec, top_k: int = 10) -> _FakeRetrieveResult:
        return _FakeRetrieveResult(self._documents)

    def fetch_by_id(self, chunk_id: str):
        return None


class FakeEmbedding:
    """encode_single returns a numpy vector with .tolist()."""

    def encode_single(self, text: str):
        return np.zeros(8, dtype=np.float32)


class FakeGenerator:
    def __init__(self, answer_text: str | None = None) -> None:
        self._answer = answer_text if answer_text is not None else ("A" * 1500)
        self.called = False

    async def generate_answer(self, *, query, chunks, confidence, is_program_list_query):
        self.called = True
        return {"answer": self._answer, "provider": "fake"}


class _SettingsStub:
    """Stand-in for RAGSettings; provides only the model-version fields."""

    NS24_GEN_MODEL = "stub-gen"
    NS24_NER_MODEL = "stub-ner"
    NS24_HYDE_MODEL = "stub-hyde"
    EMBEDDING_MODEL = "stub-emb"


# =============================================================================
# Module loader + fixture
# =============================================================================


def _load_via_spec(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register BEFORE exec so the runner's `import eval_record` resolves to THIS
    # module instance (avoids class-identity drift between the runner's _er_mod
    # and the test's `er`).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def er():
    """EvalRecord schema module — same instance the runner imports."""
    return _load_via_spec("eval_record", EVAL_RECORD_PATH)


@pytest.fixture
def runner(er):
    """Load the runner via spec_from_file_location; patch boost passthrough."""
    # er is loaded first (fixture order) and registered in sys.modules.
    module = _load_via_spec("eval_run_instrumented", SCRIPT_PATH)
    # Replace the real boost_with_aggregation with a passthrough — the real one
    # needs a LanceDBRetriever with a real table, which we don't have here.
    module.boost_with_aggregation = (
        lambda query, baseline_docs, *args, **kwargs: baseline_docs
    )
    return module


# =============================================================================
# 1. Full text persisted (no truncation)
# =============================================================================


def test_full_text_persisted_in_retrieved_chunks(runner, er):
    full_text = "DOC_BODY_" * 200  # 1800 chars
    doc = FakeDoc(
        chunk_id="c-1",
        source="src-A",
        score=0.91,
        text=full_text,
        metadata={"data_year": 2025, "is_superseded": False},
    )
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(
        route="graph_rag", router_result=router_result, documents=[doc]
    )
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="answer" * 100)

    q = {"id": "msg001", "question": "Câu hỏi thử?"}
    record = asyncio.run(
        runner.run_one_instrumented(
            q,
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1",
            seed=42,
            settings=_SettingsStub(),
        )
    )

    assert len(record.retrieved_chunks) == 1
    rc = record.retrieved_chunks[0]
    assert rc.text == full_text
    assert len(rc.text) == len(full_text)
    assert rc.data_year == 2025
    assert rc.is_superseded is False
    assert rc.chunk_id == "c-1"
    assert rc.source == "src-A"
    assert rc.score == 0.91


def test_full_answer_not_truncated(runner, er):
    long_answer = "Z" * 1500
    doc = FakeDoc("c-1", "src", 0.5, "body", {"data_year": 2025})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text=long_answer)

    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "q", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r", seed=1, settings=_SettingsStub(),
        )
    )
    assert record.answer == long_answer
    assert len(record.answer) == 1500


# =============================================================================
# 2. All 4 latency fields present, total >= max(seams)
# =============================================================================


def test_latency_seams_all_present_and_total_dominates(runner, er):
    doc = FakeDoc("c-1", "src", 0.5, "body", {"data_year": 2025})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="x" * 200)

    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg001", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
        )
    )

    lat = record.latency_ms
    assert lat.route_ms >= 0
    assert lat.retrieval_loop_ms >= 0
    assert lat.query_ms >= 0
    assert lat.total_ms >= 0
    assert lat.total_ms >= max(lat.route_ms, lat.retrieval_loop_ms, lat.query_ms)


# =============================================================================
# 3. Identity (run_id, seed, model_version)
# =============================================================================


def test_identity_fields_populated(runner, er):
    doc = FakeDoc("c-1", "src", 0.5, "body", {})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="x" * 200)

    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg001", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r-2026-06-02-A", seed=42, settings=_SettingsStub(),
        )
    )

    assert record.run_id == "r-2026-06-02-A"
    assert record.seed == 42
    assert record.model_version.gen == "stub-gen"
    assert record.model_version.ner == "stub-ner"
    assert record.model_version.hyde == "stub-hyde"
    assert record.model_version.embedding == "stub-emb"
    assert record.harness_version == "s2-v1"
    assert isinstance(record.timestamp, str) and len(record.timestamp) > 0


# =============================================================================
# 4. Auto-answer short-circuit
# =============================================================================


def test_auto_answer_path_emits_hyde_auto_answer_record(runner, er):
    auto = "Xin chào! Đây là câu trả lời tự động."
    router_result = FakeRouterResult(
        skip_retrieval=True, auto_answer=auto, intent="intro"
    )
    rag_result = FakeRagResult(
        route="graph_rag", router_result=router_result, documents=[]
    )
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([])
    embedding = FakeEmbedding()
    generator = FakeGenerator()  # MUST NOT be called

    q = {"id": "msg002", "question": "Trường ở đâu?"}
    record = asyncio.run(
        runner.run_one_instrumented(
            q,
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
        )
    )

    assert record.route == "hyde_auto_answer"
    assert record.retrieved_chunks == []
    assert record.answer == auto
    # retrieval + gen are skipped in short-circuit
    assert record.latency_ms.retrieval_loop_ms == 0
    assert record.latency_ms.query_ms == 0
    assert record.latency_ms.route_ms >= 0
    # generator MUST not be invoked on the short-circuit path
    assert generator.called is False


# =============================================================================
# 5. JSONL run_all writes one record per question
# =============================================================================


def test_run_all_writes_jsonl_one_line_per_question(runner, er, tmp_path):
    doc = FakeDoc("c-1", "src", 0.5, "body", {"data_year": 2025})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="ans" * 50)

    questions = [
        {"id": "q1", "question": "Câu 1?"},
        {"id": "q2", "question": "Câu 2?"},
        {"id": "q3", "question": "Câu 3?"},
    ]
    out_path = tmp_path / "out.jsonl"

    asyncio.run(
        runner.run_all(
            questions,
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
            out_path=str(out_path),
        )
    )

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(questions)
    seen_ids = set()
    for line in lines:
        rec = er.EvalRecord.from_jsonl_line(line)
        seen_ids.add(rec.id)
    assert seen_ids == {"q1", "q2", "q3"}


# =============================================================================
# 6. C6 guard — no top-level mocks/fakes in the runner module
# =============================================================================


def test_c6_no_top_level_mocks_in_runner_module(runner):
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    # Top-level scope = lines that are not indented. Comments are skipped.
    top_level_lines: List[str] = []
    for line in src.splitlines():
        if not line:
            continue
        if line[0].isspace():
            continue
        if line.lstrip().startswith("#"):
            continue
        top_level_lines.append(line)
    joined = "\n".join(top_level_lines)
    assert "class Fake" not in joined, (
        "Runner must not define Fake* classes at module level (C6)"
    )
    assert "Mock(" not in joined, (
        "Runner must not instantiate unittest.mock.Mock at module level (C6)"
    )
    assert "unittest.mock" not in joined, (
        "Runner must not import unittest.mock at module level (C6)"
    )


# =============================================================================
# 7. expected_behavior from gt_lookup
# =============================================================================


def test_expected_behavior_from_gt_lookup(runner, er):
    doc = FakeDoc("c-1", "src", 0.5, "body", {"data_year": 2025})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="x" * 200)

    gt = {"msg001": "answer", "msg999": "refuse"}
    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg001", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
            gt_lookup=gt,
        )
    )
    assert record.expected_behavior == "answer"

    # For an id NOT in gt_lookup, expected_behavior is None.
    record2 = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg002", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
            gt_lookup=gt,
        )
    )
    assert record2.expected_behavior is None


# =============================================================================
# 8. (Bonus) No gt_lookup → expected_behavior is None
# =============================================================================


def test_expected_behavior_none_when_no_gt_lookup(runner, er):
    doc = FakeDoc("c-1", "src", 0.5, "body", {"data_year": 2025})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="x" * 200)

    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg001", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
        )
    )
    assert record.expected_behavior is None


# =============================================================================
# 9. (Bonus) Attribution defensive: must NEVER break the run
# =============================================================================


def test_attribution_failure_does_not_break_run(runner, er):
    """If attribution monkeypatch fails (no _generate_hyde_doc on fakes), the
    run still completes. Attribution is allowed to be:
      - None (if NOTHING was attributable on the fakes), OR
      - a LatencyAttribution with at least one populated field (e.g. gen_ms if
        the fake's `generate_answer` was wrappable).
    The point of the test: the run MUST NOT crash on attribution setup.
    """
    doc = FakeDoc("c-1", "src", 0.5, "body", {"data_year": 2025})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="x" * 200)

    # FakePipeline has no _generate_hyde_doc/_step_back/_classify on _router.
    # The runner's attribution code must swallow that gracefully.
    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg003", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
        )
    )
    # The main 4 seams are populated regardless.
    assert record.latency_ms.route_ms >= 0
    assert record.latency_ms.retrieval_loop_ms >= 0
    assert record.latency_ms.query_ms >= 0
    assert record.latency_ms.total_ms >= 0
    # Attribution is either None or a LatencyAttribution — never anything else,
    # never a crash. If present, the FakeGenerator's generate_answer was
    # wrappable so gen_ms may be populated; the router sub-methods are absent
    # on fakes so hyde_ms / router_inner_ms stay None.
    if record.latency_attribution is not None:
        assert record.latency_attribution.hyde_ms is None
        assert record.latency_attribution.router_inner_ms is None


# =============================================================================
# 10. T12 — per-question LLM token accounting
# =============================================================================


class FakeLedger:
    """Mimics TokenLedger — snapshot() + reset() for the drain pattern."""

    def __init__(self, calls):
        self._calls = list(calls)

    def snapshot(self):
        return list(self._calls)

    def reset(self):
        self._calls.clear()


class FakeFakeGeneratorWithTokens:
    """Generator carrying two LLM clients with known token usage."""

    def __init__(self, primary_calls, fallback_calls):
        self.unified_client = _client_with_ledger(primary_calls)
        self.unified_fallback_client = _client_with_ledger(fallback_calls)

    async def generate_answer(self, *, query, chunks, confidence, is_program_list_query):
        return {"answer": "x" * 200, "provider": "fake"}


def _client_with_ledger(calls):
    class C:
        def __init__(self, calls):
            self.token_ledger = FakeLedger(calls)

    return C(calls)


def _make_call(provider: str, model: str, p: int, c: int):
    class Call:
        pass
    call = Call()
    call.provider = provider
    call.model = model
    call.prompt_tokens = p
    call.completion_tokens = c
    return call


def test_drain_collects_and_resets_ledgers(runner, er):
    """T12 helper contract: drain returns current calls AND resets ledgers."""
    primary_calls = [
        _make_call("ns24", "gemini-3.5-flash-lite", 50, 10),
        _make_call("ns24", "gemini-3.5-flash-lite", 100, 20),
    ]
    fallback_calls = [_make_call("ns24", "deepseek-v4.1-flash", 10, 2)]
    generator = FakeFakeGeneratorWithTokens(primary_calls, fallback_calls)
    pipeline = FakePipeline(
        FakeRouterResult(),
        FakeRagResult(route="graph_rag", router_result=FakeRouterResult(), documents=[]),
    )

    calls = runner._drain_token_ledgers(pipeline, generator)
    assert len(calls) == 3
    assert {c["provider"] for c in calls} == {"ns24"}
    # ledger reset after drain — second drain is empty
    assert runner._drain_token_ledgers(pipeline, generator) == []


class _RecordingGenerator:
    """Generator that records fake LLM calls into its own ledger during
    generate_answer — mimics real pipeline usage where calls happen
    BETWEEN the runner's initial drain and final drain."""

    def __init__(self, calls):
        self.unified_client = _client_with_ledger([])
        self._to_record = list(calls)

    async def generate_answer(self, *, query, chunks, confidence, is_program_list_query):
        for call in self._to_record:
            self.unified_client.token_ledger._calls.append(call)
        return {"answer": "x" * 200, "provider": "fake"}


def test_llm_tokens_total_per_question(runner, er):
    """T12 acceptance: record carries per-call entries + totals for the question."""
    doc = FakeDoc("c-1", "src", 0.5, "body", {})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()

    generator = _RecordingGenerator([
        _make_call("ns24", "gemini-3.5-flash-lite", 50, 10),
        _make_call("ns24", "deepseek-v4.1-flash", 10, 2),
    ])

    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg012", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
        )
    )

    assert record.llm_tokens is not None
    assert record.llm_tokens.call_count == 2
    assert record.llm_tokens.prompt_tokens == 60
    assert record.llm_tokens.completion_tokens == 12
    assert record.llm_tokens.total_tokens == 72
    assert {c.provider for c in record.llm_token_calls} == {"ns24"}


def test_no_llm_clients_means_no_tokens(runner, er):
    """No LLM clients on generator/pipeline → llm_tokens stays None."""
    doc = FakeDoc("c-1", "src", 0.5, "body", {})
    router_result = FakeRouterResult()
    rag_result = FakeRagResult(route="graph_rag", router_result=router_result, documents=[doc])
    pipeline = FakePipeline(router_result, rag_result)
    retriever = FakeRetriever([doc])
    embedding = FakeEmbedding()
    generator = FakeGenerator(answer_text="x" * 200)  # bare FakeGenerator, no clients

    record = asyncio.run(
        runner.run_one_instrumented(
            {"id": "msg013", "question": "Q?"},
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id="r1", seed=42, settings=_SettingsStub(),
        )
    )
    assert record.llm_tokens is None
    assert record.llm_token_calls == []
