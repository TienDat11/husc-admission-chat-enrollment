"""Tests for scripts/eval_records_to_results_json.py — EvalRecord JSONL -> legacy results.json.

The two report generators (gen_thi_sinh_pdf.py, gen_thi_sinh_reports.py) read a
legacy `results/86q_thi_sinh_results.json` schema. The fixed-pipeline eval writes
EvalRecord JSONL with a different schema. This converter bridges the two so any
EvalRecord run can be turned into the report-consumable `results.json`.

TDD — schema + has_answer semantics are validated here BEFORE the converter is
written. The has_answer logic in convert() must replicate
gen_thi_sinh_pdf._has_fallback exactly: empty OR len<50 OR startswith any of
"tôi không tìm thấy"/"không tìm thấy"/"tôi không cung cấp" OR contains
"không tìm thấy thông tin này trong tài liệu".
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval_records_to_results_json.py"
)


# ─────────────────────────── helpers ───────────────────────────


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "eval_records_to_results_json", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conv():
    return _load_module()


def _make_record(
    *,
    id: str = "q-001",
    question: str = "Đại học Khoa học có những ngành nào?",
    answer: str = "HUSC năm 2026 tuyển sinh 28 ngành đại học chính quy với tổng chỉ tiêu dự kiến là 2.079.",
    intent: str | None = "liet_ke",
    complexity: int | None = 4,
    route: str | None = "graph_rag",
    gen: str = "gemini-3.5-flash-low",
    ner: str = "gemini-3.5-flash-low",
    hyde: str = "gemini-3.5-flash-low",
    embedding: str = "Qwen/Qwen3-Embedding-0.6B",
    latency_total_ms: float = 42755.1,
    chunk_ids: list[str] | None = None,
    chunk_sources: list[str] | None = None,
) -> dict:
    chunk_ids = chunk_ids if chunk_ids is not None else ["c-001", "c-002", "c-003"]
    chunk_sources = (
        chunk_sources
        if chunk_sources is not None
        else ["pdf_1", "tuyensinh.husc.edu.vn", "pdf_3"]
    )
    chunks = [
        {
            "chunk_id": cid,
            "source": cs,
            "score": 0.5,
            "text": f"text for {cid}",
            "data_year": 2026,
            "is_superseded": False,
        }
        for cid, cs in zip(chunk_ids, chunk_sources)
    ]
    return {
        "run_id": "s15",
        "seed": 42,
        "harness_version": "s2-v1",
        "timestamp": "2026-06-03T03:36:58.010441",
        "model_version": {
            "gen": gen,
            "ner": ner,
            "hyde": hyde,
            "embedding": embedding,
        },
        "id": id,
        "question": question,
        "intent": intent,
        "complexity": complexity,
        "route": route,
        "expected_behavior": "answer",
        "retrieved_chunks": chunks,
        "answer": answer,
        "latency_ms": {
            "route_ms": 1000.0,
            "retrieval_loop_ms": 2000.0,
            "query_ms": 3000.0,
            "total_ms": latency_total_ms,
        },
        "latency_attribution": None,
        "judge_scores": None,
        "judge_rationale": None,
    }


# ───────────────────────── tests ─────────────────────────


def test_convert_normal_long_answer_has_answer_true(conv):
    record = _make_record(
        answer="HUSC năm 2026 tuyển sinh 28 ngành đại học chính quy với tổng chỉ tiêu dự kiến là 2.079.",
        latency_total_ms=42755.1,
    )
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    assert len(out["results"]) == 1
    r = out["results"][0]
    assert r["id"] == "q-001"
    assert r["has_answer"] is True
    # latency_s = total_ms / 1000.0, rounded to 1 decimal
    assert r["latency_s"] == round(42755.1 / 1000.0, 1)
    assert r["latency_s"] == 42.8
    assert r["chunk_ids"] == ["c-001", "c-002", "c-003"]
    assert r["chunk_sources"] == ["pdf_1", "tuyensinh.husc.edu.vn", "pdf_3"]
    assert r["intent"] == "liet_ke"
    assert r["complexity"] == 4
    assert r["route"] == "graph_rag"
    # question + answer round-trip verbatim
    assert r["question"] == record["question"]
    assert r["answer"] == record["answer"]


def test_convert_explicit_not_found_sentinel_has_answer_false(conv):
    record = _make_record(
        answer="Tôi không tìm thấy thông tin này trong tài liệu hiện có."
    )
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    r = out["results"][0]
    assert r["has_answer"] is False


def test_convert_empty_answer_has_answer_false(conv):
    record = _make_record(answer="")
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    r = out["results"][0]
    assert r["has_answer"] is False


def test_convert_short_answer_under_50_chars_has_answer_false(conv):
    record = _make_record(answer="abc " * 8)  # 40 chars (incl. trailing spaces stripped after .strip().lower())
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    a = "abc " * 8
    assert len(a) < 50
    r = out["results"][0]
    assert r["has_answer"] is False


def test_convert_startswith_khong_tim_thay_has_answer_false(conv):
    # gen_thi_sinh_pdf._has_fallback uses `startswith` (not `in`) for
    # "không tìm thấy" — only answers that OPEN with the sentinel are
    # counted as fallback, to avoid false-positives on legit answers that
    # mention "không tìm thấy" mid-sentence.
    record = _make_record(answer="Không tìm thấy thông tin ngành học em hỏi.")
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    r = out["results"][0]
    assert r["has_answer"] is False


def test_convert_startswith_toi_khong_cung_cap_has_answer_false(conv):
    record = _make_record(
        answer="Tôi không cung cấp thông tin về điểm chuẩn năm 2024 vì ngoài phạm vi dữ liệu."
    )
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    r = out["results"][0]
    assert r["has_answer"] is False


def test_convert_intent_complexity_defaults(conv):
    record = _make_record(intent=None, complexity=None)
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    r = out["results"][0]
    assert r["intent"] == "general"
    assert r["complexity"] == 1


def test_convert_metadata_derives_models_from_record(conv):
    record = _make_record(
        gen="my-gen",
        ner="my-ner",
        hyde="my-hyde",
        embedding="my-emb",
    )
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    meta = out["metadata"]
    # The converter pulls model_version from the FIRST record so the report
    # header reflects the real models that produced the run, ignoring the
    # kwarg defaults the caller might pass.
    assert meta["gen_model"] == "my-gen"
    assert meta["hyde_model"] == "my-hyde"
    assert meta["ner_model"] == "my-ner"
    assert meta["embedding_model"] == "my-emb"
    assert meta["total"] == 1
    assert meta["completed"] == 1


def test_convert_metadata_counts_match_records_length(conv):
    records = [_make_record(id=f"q-{i:03d}") for i in range(5)]
    out = conv.convert(records, gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    assert out["metadata"]["total"] == 5
    assert out["metadata"]["completed"] == 5
    assert len(out["results"]) == 5


def test_convert_vietnamese_diacritics_preserved_verbatim(conv):
    vn_question = "Đại học Khoa học, tổ hợp, điểm chuẩn năm 2026?"
    vn_answer = "Trường Đại học Khoa học - Đại học Huế (HUSC) — mã DHT. Tổ hợp xét tuyển: A00, B00, C00, D01. Điểm chuẩn 2025 ngành CNTT: 24.5."
    record = _make_record(question=vn_question, answer=vn_answer)
    out = conv.convert([record], gen_model="g", hyde_model="h", ner_model="n", embedding_model="e")
    r = out["results"][0]
    assert r["question"] == vn_question
    assert r["answer"] == vn_answer
    # Round-trip via JSON to assert survive serialization
    blob = json.dumps(out, ensure_ascii=False)
    restored = json.loads(blob)
    assert restored["results"][0]["question"] == vn_question
    assert restored["results"][0]["answer"] == vn_answer
