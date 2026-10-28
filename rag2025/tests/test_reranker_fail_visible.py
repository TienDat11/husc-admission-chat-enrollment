"""Ticket 40: a dead reranker must never fail silently.

Contract pinned by these tests:
  1. `RerankerService` exposes a THREE-state load status
     (`loaded` / `load_failed` / `disabled_by_config`) — a nonexistent
     model name + RERANKER_ENABLED=true yields `load_failed`, NOT
     `disabled_by_config`.
  2. GET /health derives `reranker_load_ok`, `services.reranker` and
     `ready` from the REAL load state, not from object existence.
  3. The flat `_metrics` key `reranker_bypass_load_failed` (no Prometheus)
     is declared, snapshotted into /health `metrics` and /metrics, and
     bumped at ALL THREE rerank call sites (legacy /query, /v2 gate,
     /v3 gate) — the same counter, once per bypass.
  4. `rerank_skipped_reason` ("load_failed" | "disabled_by_config" | None)
     reaches the caller on POST /v2/query, on the SSE `meta` frame, and on
     /v3 via `pipeline_stages`.
  5. Regression: healthy or intentionally-off reranker → no counter bump,
     correct reason values.

Harness: same importlib trick as test_health_ready.py /
test_v2_generation_failed_status.py — load src/main.py as a stand-alone
module, wire fake service globals, drive the REAL app. The dead-reranker
fixtures use the REAL RerankerService class with CrossEncoder patched to
raise, so the tri-state logic under test is the production one.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = SRC / "main.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
assert MAIN_PATH.is_file(), f"src/main.py not found at {MAIN_PATH}"

from config.settings import RAGSettings  # noqa: E402
from services.reranker import (  # noqa: E402
    LOAD_STATE_DISABLED_BY_CONFIG,
    LOAD_STATE_LOAD_FAILED,
    LOAD_STATE_LOADED,
    RerankerService,
)

COUNTER_KEY = "reranker_bypass_load_failed"


# ────────────────────────────────────────────────────────────────────────
# Fixtures / factories
# ────────────────────────────────────────────────────────────────────────


def _load_main(monkeypatch) -> object:
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    spec = importlib.util.spec_from_file_location("t40_reranker_main", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dead_reranker(monkeypatch) -> RerankerService:
    """REAL service whose model load raises — the production defect path."""

    def _boom(*_a, **_k):
        raise RuntimeError("cross-encoder model unavailable (test)")

    monkeypatch.setattr("services.reranker.CrossEncoder", _boom)
    return RerankerService(
        RAGSettings(RERANKER_ENABLED=True, RERANKER_MODEL="DeadModel-XYZ")
    )


def _disabled_reranker() -> RerankerService:
    return RerankerService(RAGSettings(RERANKER_ENABLED=False))


def _healthy_reranker(monkeypatch, adaptive: float = 0.5) -> RerankerService:
    """REAL service with a stubbed CrossEncoder returning fixed scores."""
    model = MagicMock()
    model.predict = MagicMock(
        side_effect=lambda pairs, **_k: np.array(
            [0.9 - 0.05 * i for i in range(len(pairs))]
        )
    )
    monkeypatch.setattr("services.reranker.CrossEncoder", lambda *_a, **_k: model)
    return RerankerService(
        RAGSettings(
            RERANKER_ENABLED=True,
            RERANKER_MODEL="FakeLive-XYZ",
            RERANKER_ADAPTIVE_CONFIDENCE=adaptive,
        )
    )


def _doc(text: str = "Học phí ngành CNTT là 15 triệu đồng/năm.") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        score=0.83,
        chunk_id="c1",
        source="hocphi.pdf",
        metadata={"summary": text, "data_year": "2026"},
    )


def _legacy_doc() -> SimpleNamespace:
    d = SimpleNamespace(
        chunk_id="c1",
        source="hocphi.pdf",
        text="Học phí ngành CNTT là 15 triệu đồng/năm.",
        score=0.83,
    )
    d.to_dict = lambda: {
        "chunk_id": d.chunk_id,
        "source": d.source,
        "text": d.text,
        "score": d.score,
    }
    return d


def _fake_guardrail():
    decision = SimpleNamespace(
        is_in_scope=True,
        internal_code="SUCCESS",
        reason="ok",
        short_answer="",
        data_gap_hints=[],
        pii_detected=False,
    )
    svc = MagicMock()
    svc.precheck = AsyncMock(return_value=decision)
    svc.classify_no_result = AsyncMock(return_value=decision)
    svc.public_status = MagicMock(return_value="SUCCESS")
    svc.expose_internal = MagicMock(return_value=True)
    svc._contains_sensitive_pii = MagicMock(return_value=False)
    svc._pii_warning_answer = MagicMock(return_value="An information.")
    return svc


def _fake_generator(answer: str = "Hoc phi 15 trieu."):
    payload = {
        "answer": answer,
        "sources": [{"id": "s1", "title": "Nguồn 1", "url": None,
                     "snippet": "trich doan", "data_year": "2026"}],
        "confidence": 0.83,
        "provider": "fake-model",
        "chunks_used": 1,
    }
    gen = MagicMock()
    gen.generate_answer = AsyncMock(return_value=payload)
    return gen


def _fake_v2_pipeline(documents: list):
    router_result = SimpleNamespace(
        skip_retrieval=False,
        auto_answer=None,
        step_back_query="step",
        intent="general",
        complexity=1,
        reasoning="ok",
        hyde_variants=[],
        is_in_scope=True,
    )
    rag_result = SimpleNamespace(
        query="hoc phi nganh CNTT?",
        route="padded_rag",
        documents=documents,
        router_result=router_result,
        ppr_scores={},
        latency_ms=12.5,
        confidence=0.83,
    )
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=rag_result)
    pipeline._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 0})
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=router_result)
    return pipeline


def _fake_lancedb_retriever(documents: list):
    r = MagicMock()
    r.check_collection = MagicMock(return_value={"exists": True, "vectors_count": 3})
    r.retrieve = MagicMock(
        return_value=SimpleNamespace(
            documents=documents, is_success=True,
            error_type=None, error_message=None,
        )
    )
    return r


def _wire_v2(main, reranker) -> None:
    main.guardrail_service = _fake_guardrail()
    main.unified_pipeline = _fake_v2_pipeline([_doc()])
    main.lancedb_retriever_service = _fake_lancedb_retriever([_doc()])
    main.embedding_service = MagicMock()
    main.llm_generator_service = _fake_generator()
    main.hybrid_search_service = None
    main.reranker_service = reranker
    main.query_cache = None
    main.app.state.warmup_stage = "ready"


def _wire_legacy(main, reranker) -> None:
    enhancer = MagicMock()
    enhancer.enhance_query = AsyncMock(return_value={
        "variants": ["hoc phi nganh CNTT?"],
        "original_query": "hoc phi nganh CNTT?",
        "detected_intent": "tuition",
        "query_type": "specific",
        "top_k": 5,
    })
    embedding = MagicMock()
    embedding.encode_batch = MagicMock(return_value=np.zeros((1, 4), dtype=np.float32))
    main.guardrail_service = _fake_guardrail()
    main.query_enhancer_service = enhancer
    main.embedding_service = embedding
    main.lancedb_retriever_service = _fake_lancedb_retriever([_legacy_doc()])
    main.llm_generator_service = _fake_generator()
    main.hybrid_search_service = None
    main.reranker_service = reranker
    main.query_cache = None
    main.app.state.warmup_stage = "ready"


def _wire_v3(main, reranker) -> None:
    embedding = MagicMock()
    embedding.encode_query = MagicMock(return_value=np.zeros(4, dtype=np.float32))
    main.embedding_service = embedding
    main.lancedb_retriever_service = _fake_lancedb_retriever([_doc()])
    main.llm_generator_service = _fake_generator()
    main.hybrid_search_service = None
    main.reranker_service = reranker
    main.query_cache = None
    main.app.state.warmup_stage = "ready"


def _parse_sse(raw: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        ev = {"event": "", "data": ""}
        for line in chunk.splitlines():
            if line.startswith("event:"):
                ev["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                payload = line[len("data:"):].strip()
                ev["data"] = f"{ev['data']}{payload}" if ev["data"] else payload
        if ev["event"] or ev["data"]:
            frames.append(ev)
    return frames


@pytest.fixture
def main_module(monkeypatch):
    return _load_main(monkeypatch)


# ────────────────────────────────────────────────────────────────────────
# 1. Three-state load status on the REAL class
# ────────────────────────────────────────────────────────────────────────


def test_nonexistent_model_yields_load_failed_not_disabled(monkeypatch):
    """Ticket observation: nonexistent model name + RERANKER_ENABLED=true
    -> load_failed. Uses the REAL name (no patch) so the exception is the
    production one; HF_HUB_OFFLINE=1 makes the failed load deterministic
    (offline error) instead of network-dependent retries."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    svc = RerankerService(
        RAGSettings(RERANKER_ENABLED=True, RERANKER_MODEL="DefinitelyNotARealModel-XYZ-12345")
    )
    assert svc.load_state == LOAD_STATE_LOAD_FAILED
    assert svc.load_state != LOAD_STATE_DISABLED_BY_CONFIG
    assert svc.skip_reason == "load_failed"
    assert svc.enabled is False


def test_disabled_by_config_state():
    svc = _disabled_reranker()
    assert svc.load_state == LOAD_STATE_DISABLED_BY_CONFIG
    assert svc.skip_reason == "disabled_by_config"
    assert svc.enabled is False


def test_loaded_state(monkeypatch):
    svc = _healthy_reranker(monkeypatch)
    assert svc.load_state == LOAD_STATE_LOADED
    assert svc.skip_reason is None
    assert svc.enabled is True


# ────────────────────────────────────────────────────────────────────────
# 2. /health reads the real load state, not object existence
# ────────────────────────────────────────────────────────────────────────


def test_health_dead_reranker_not_ready(main_module, monkeypatch):
    main = main_module
    _wire_v2(main, _dead_reranker(monkeypatch))
    client = TestClient(main.app)

    body = client.get("/health").json()
    assert body["reranker_load_ok"] is False, body
    assert body["services"]["reranker"] is False, body
    assert body["ready"] is False, body


def test_health_disabled_reranker_is_ok(main_module):
    main = main_module
    _wire_v2(main, _disabled_reranker())
    body = TestClient(main.app).get("/health").json()
    assert body["reranker_load_ok"] is True, body
    assert body["services"]["reranker"] is True, body
    assert body["ready"] is True, body


def test_health_loaded_reranker_is_ok(main_module, monkeypatch):
    main = main_module
    _wire_v2(main, _healthy_reranker(monkeypatch))
    body = TestClient(main.app).get("/health").json()
    assert body["reranker_load_ok"] is True, body
    assert body["ready"] is True, body


# ────────────────────────────────────────────────────────────────────────
# 3+4. One flat counter, bumped at all three call sites
# ────────────────────────────────────────────────────────────────────────


def test_all_three_call_sites_bump_the_same_counter(main_module, monkeypatch):
    """(a) legacy /query, (b) /v2 gate, (c) /v3 gate — each rerank bypass
    caused by a dead model increments `reranker_bypass_load_failed` once."""
    main = main_module
    dead = _dead_reranker(monkeypatch)

    _wire_legacy(main, dead)
    assert TestClient(main.app).post(
        "/query", json={"query": "a1 hoc phi?", "force_rag_only": False}
    ).status_code == 200
    assert main._metrics[COUNTER_KEY] == 1, "legacy /query did not bump"

    _wire_v2(main, dead)
    assert TestClient(main.app).post(
        "/v2/query", json={"query": "a2 hoc phi?"}
    ).status_code == 200
    assert main._metrics[COUNTER_KEY] == 2, "/v2 gate did not bump"

    _wire_v3(main, dead)
    assert TestClient(main.app).post(
        "/v3/query", json={"query": "a3 hoc phi?"}
    ).status_code == 200
    assert main._metrics[COUNTER_KEY] == 3, "/v3 gate did not bump"


def test_counter_visible_in_health_and_metrics(main_module, monkeypatch):
    """The metric is declared in `_metrics`, and snapshotted into
    GET /health `metrics` and GET /metrics — no Prometheus anywhere."""
    main = main_module
    _wire_v2(main, _dead_reranker(monkeypatch))
    client = TestClient(main.app)

    assert client.get("/health").json()["metrics"][COUNTER_KEY] == 0
    client.post("/v2/query", json={"query": "hoc phi nganh CNTT?"})

    health = client.get("/health").json()
    assert health["metrics"][COUNTER_KEY] == 1, health["metrics"]
    metrics = client.get("/metrics").json()
    assert metrics["metrics"][COUNTER_KEY] == 1, metrics


# ────────────────────────────────────────────────────────────────────────
# 6+7. /v2 self-reports the skip — non-stream AND SSE meta
# ────────────────────────────────────────────────────────────────────────


def test_v2_non_stream_reports_load_failed(main_module, monkeypatch):
    main = main_module
    _wire_v2(main, _dead_reranker(monkeypatch))
    body = TestClient(main.app).post(
        "/v2/query", json={"query": "hoc phi nganh CNTT?"}
    ).json()
    assert body["rerank_skipped_reason"] == "load_failed", body


def test_v2_reports_load_failed_on_high_confidence_path(main_module, monkeypatch):
    """The gate is enabled-only: with a dead model BOTH the low- and
    high-confidence paths must surface the reason, not just one."""
    main = main_module
    _wire_v2(main, _dead_reranker(monkeypatch))
    # pipeline fake carries confidence 0.83 >= default adaptive threshold
    # 0.5 — the "easy query" path.
    body = TestClient(main.app).post(
        "/v2/query", json={"query": "de thi cac nganh?"}
    ).json()
    assert body["confidence"] >= 0.5
    assert body["rerank_skipped_reason"] == "load_failed", body


def test_v2_stream_meta_carries_load_failed(main_module, monkeypatch):
    main = main_module
    _wire_v2(main, _dead_reranker(monkeypatch))
    res = TestClient(main.app).post(
        "/v2/query/stream", json={"query": "hoc phi nganh CNTT?"}
    )
    assert res.status_code == 200, res.text
    frames = _parse_sse(res.text)
    meta = json.loads(next(f for f in frames if f["event"] == "meta")["data"])
    assert meta["rerank_skipped_reason"] == "load_failed", meta
    # stage/delta/done shapes unchanged
    assert {f["event"] for f in frames} >= {"stage", "meta", "delta", "done"}
    done = json.loads(next(f for f in frames if f["event"] == "done")["data"])
    assert set(done) == {"answer"}


# ────────────────────────────────────────────────────────────────────────
# 8. /v3 pipeline_stages tells the same truth
# ────────────────────────────────────────────────────────────────────────


def test_v3_pipeline_stages_expose_cause(main_module, monkeypatch):
    main = main_module
    _wire_v3(main, _dead_reranker(monkeypatch))
    body = TestClient(main.app).post(
        "/v3/query", json={"query": "hoc phi nganh CNTT?"}
    ).json()
    stages = body["pipeline_stages"]
    assert stages["reranker_used"] is False, stages
    assert stages["rerank_skipped_reason"] == "load_failed", stages


# ────────────────────────────────────────────────────────────────────────
# 9. Regression: alive or intentionally-off rerankers are untouched
# ────────────────────────────────────────────────────────────────────────


def test_disabled_config_reports_reason_without_counter(main_module):
    main = main_module
    _wire_v2(main, _disabled_reranker())
    client = TestClient(main.app)
    body = client.post("/v2/query", json={"query": "hoc phi nganh CNTT?"}).json()
    assert body["rerank_skipped_reason"] == "disabled_by_config", body
    assert main._metrics[COUNTER_KEY] == 0, "intentional off must not bump"
    assert client.get("/health").json()["reranker_load_ok"] is True


def test_healthy_reranker_no_reason_no_counter(main_module, monkeypatch):
    """Adaptive gate actually runs rerank (confidence 0.83 < threshold 0.9):
    reason None, counter 0, reranker_used True on /v3."""
    main = main_module
    live = _healthy_reranker(monkeypatch, adaptive=0.9)
    _wire_v2(main, live)
    body = TestClient(main.app).post(
        "/v2/query", json={"query": "hoc phi nganh CNTT?"}
    ).json()
    assert body["rerank_skipped_reason"] is None, body
    assert main._metrics[COUNTER_KEY] == 0

    _wire_v3(main, live)
    v3 = TestClient(main.app).post("/v3/query", json={"query": "thi gi?"}).json()
    assert v3["pipeline_stages"]["reranker_used"] is True, v3["pipeline_stages"]
    assert v3["pipeline_stages"]["rerank_skipped_reason"] is None
    assert main._metrics[COUNTER_KEY] == 0


def test_v3_disabled_config_reason_no_counter(main_module):
    main = main_module
    _wire_v3(main, _disabled_reranker())
    body = TestClient(main.app).post(
        "/v3/query", json={"query": "hoc phi nganh CNTT?"}
    ).json()
    assert body["pipeline_stages"]["rerank_skipped_reason"] == "disabled_by_config"
    assert body["pipeline_stages"]["reranker_used"] is False
    assert main._metrics[COUNTER_KEY] == 0
