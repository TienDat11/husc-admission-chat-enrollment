"""Tests for eval_record — the instrumentation contract for full-evidence eval.

Pydantic v2 schema persistence tests. TDD: schema-first.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Any, Dict

import pytest
from pydantic import ValidationError


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "eval_record.py"
)


@pytest.fixture
def er():
    spec = importlib.util.spec_from_file_location("eval_record", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _full_record(er) -> Any:
    """Build a fully-populated EvalRecord for round-trip tests."""
    chunk_a = er.RetrievedChunk(
        chunk_id="c-001",
        source="husc_v2025_blue",
        score=0.8732,
        text="A" * 4000,  # long text to exercise no-truncation invariant
        data_year=2025,
        is_superseded=False,
    )
    chunk_b = er.RetrievedChunk(
        chunk_id="c-002",
        source="husc_v2025_blue",
        score=0.5121,
        text="B" * 250,
        data_year=2025,
        is_superseded=True,
    )
    latency = er.LatencyMs(
        route_ms=12.4,
        retrieval_loop_ms=87.6,
        query_ms=345.9,
        total_ms=445.9,
    )
    attribution = er.LatencyAttribution(
        hyde_ms=4.1,
        router_inner_ms=8.3,
        gen_ms=300.0,
        rerank_ms=12.5,
    )
    mv = er.ModelVersion(
        gen="gemini-2.5-pro",
        ner="underthesea-1.3.4",
        hyde="gemini-2.5-pro",
        embedding="AITeamVN/bge-m3",
    )
    return er.EvalRecord(
        run_id="run-2026-06-02-001",
        seed=42,
        harness_version="0.1.0",
        timestamp="2026-06-02T10:15:00+00:00",
        model_version=mv,
        id="q-014",
        question="Điểm chuẩn ngành CNTT năm 2025 là bao nhiêu?",
        intent="factual_lookup",
        complexity=2,
        route="graph_rag",
        expected_behavior="answer",
        retrieved_chunks=[chunk_a, chunk_b],
        answer="X" * 5000,  # long answer to exercise no-truncation invariant
        latency_ms=latency,
        latency_attribution=attribution,
        judge_scores={"faithfulness": 0.91, "answer_relevancy": 0.88},
        judge_rationale={
            "faithfulness": "Anchored to retrieved chunks.",
            "answer_relevancy": "Directly addresses the question.",
        },
    )


# ---------- 1. Lossless round-trip ----------

def test_lossless_round_trip_through_jsonl(er):
    r = _full_record(er)
    line = r.to_jsonl_line()
    r2 = er.EvalRecord.from_jsonl_line(line)
    assert r2.model_dump() == r.model_dump()


def test_to_jsonl_line_returns_string(er):
    r = _full_record(er)
    line = r.to_jsonl_line()
    assert isinstance(line, str)
    # JSON-lines: must be single-line (no embedded newline in payload).
    assert "\n" not in line


# ---------- 2. Fail-fast on missing required field ----------

def test_missing_run_id_raises(er):
    payload: Dict[str, Any] = _full_record(er).model_dump()
    payload.pop("run_id")
    with pytest.raises(ValidationError):
        er.EvalRecord.model_validate(payload)


def test_missing_seed_raises(er):
    payload: Dict[str, Any] = _full_record(er).model_dump()
    payload.pop("seed")
    with pytest.raises(ValidationError):
        er.EvalRecord.model_validate(payload)


def test_latency_ms_missing_total_raises(er):
    payload = _full_record(er).model_dump()
    payload["latency_ms"].pop("total_ms")
    with pytest.raises(ValidationError):
        er.EvalRecord.model_validate(payload)


# ---------- 3. No truncation ----------

def test_no_truncation_on_long_answer(er):
    r = _full_record(er)
    r2 = er.EvalRecord.from_jsonl_line(r.to_jsonl_line())
    assert r2.answer == "X" * 5000
    assert len(r2.answer) == 5000


def test_no_truncation_on_long_chunk_text(er):
    r = _full_record(er)
    r2 = er.EvalRecord.from_jsonl_line(r.to_jsonl_line())
    assert r2.retrieved_chunks[0].text == "A" * 4000
    assert len(r2.retrieved_chunks[0].text) == 4000


# ---------- 4. extra=forbid ----------

def test_extra_top_level_kwarg_forbidden(er):
    with pytest.raises(ValidationError):
        er.EvalRecord(
            **_full_record(er).model_dump(),
            foo="bar",
        )


# ---------- 5. expected_behavior literal ----------

def test_expected_behavior_rejects_unknown_value(er):
    payload: Dict[str, Any] = _full_record(er).model_dump()
    payload["expected_behavior"] = "maybe"
    with pytest.raises(ValidationError):
        er.EvalRecord.model_validate(payload)


@pytest.mark.parametrize("value", ["answer", "abstain", "refuse", None])
def test_expected_behavior_accepts_valid_values(er, value):
    payload: Dict[str, Any] = _full_record(er).model_dump()
    payload["expected_behavior"] = value
    r = er.EvalRecord.model_validate(payload)
    assert r.expected_behavior == value


# ---------- 6. latency_attribution optional ----------

def test_latency_attribution_optional(er):
    payload: Dict[str, Any] = _full_record(er).model_dump()
    payload["latency_attribution"] = None
    r = er.EvalRecord.model_validate(payload)
    assert r.latency_attribution is None


# ---------- bonus: nested schema sanity ----------

def test_retrieved_chunk_defaults(er):
    c = er.RetrievedChunk(
        chunk_id="c-x",
        source="src",
        score=0.1,
        text="hi",
    )
    assert c.data_year is None
    assert c.is_superseded is False


def test_model_version_round_trip(er):
    mv = er.ModelVersion(
        gen="g", ner="n", hyde="h", embedding="e",
    )
    mv2 = er.ModelVersion.model_validate_json(mv.model_dump_json())
    assert mv2 == mv


def test_latency_attribution_all_optional(er):
    a = er.LatencyAttribution()
    assert a.hyde_ms is None
    assert a.router_inner_ms is None
    assert a.gen_ms is None
    assert a.rerank_ms is None
