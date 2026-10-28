"""Ticket 08 — per-intent eval harness (reporting + manifest diff).

Proves observable behavior, not wiring:

  * per-intent recall@k / latency p50+p95 with hand-computed numbers;
  * a run with NO LLM provider still yields retrieval metrics and marks the
    generation-dependent metrics ``blocked_by_provider`` with the verbatim
    provider error (never 0, never a bare null);
  * byte-identical machine-readable re-runs on the same input;
  * two-manifest compare mode printing real delta math with confidence
    intervals;
  * a deterministic, intent-stratified CI subset;
  * the intent classifier's stated rule (dataset label first, else derived).

Fakes live ONLY in this test file. No network, no real pipeline.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np
import pytest
from loguru import logger

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts"
INTENT_PATH = SCRIPTS_DIR / "eval_intent.py"
REPORT_PATH = SCRIPTS_DIR / "eval_per_intent_report.py"
RUNNER_PATH = SCRIPTS_DIR / "eval_run_instrumented.py"
EVAL_RECORD_PATH = SCRIPTS_DIR / "eval_record.py"


# =============================================================================
# Module loading (repo convention: spec_from_file_location)
# =============================================================================


def _load_via_spec(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def intent_mod():
    return _load_via_spec("eval_intent", INTENT_PATH)


@pytest.fixture(scope="module")
def report_mod():
    return _load_via_spec("eval_per_intent_report", REPORT_PATH)


@pytest.fixture
def runner_mod():
    _load_via_spec("eval_record", EVAL_RECORD_PATH)
    module = _load_via_spec("eval_run_instrumented", RUNNER_PATH)
    module.boost_with_aggregation = (
        lambda query, baseline_docs, *args, **kwargs: baseline_docs
    )
    return module


# =============================================================================
# Fixtures on disk
# =============================================================================

# 7 questions: 3 simple, 2 multihop, 1 comparative (+1 comparative).
# Recall outcomes are chosen so the per-intent rates are hand-computable.
QUESTIONS: List[Dict[str, Any]] = [
    {"id": "s1", "category": "simple", "question": "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?"},
    {"id": "s2", "category": "simple", "question": "Mã ngành Kỹ thuật phần mềm là gì?"},
    {"id": "s3", "category": "simple", "question": "Điểm chuẩn ngành Toán học HUSC năm 2026 bao nhiêu?"},
    {"id": "m1", "category": "multihop", "question": "Liệt kê tất cả các ngành đào tạo của HUSC"},
    {"id": "m2", "category": "multihop", "question": "Những ngành nào sử dụng tổ hợp A01?"},
    {"id": "c1", "category": "comparison", "question": "So sánh học phí ngành Sinh học và Vật lý"},
    {"id": "c2", "category": "temporal", "question": "So sánh điểm chuẩn ngành CNTT và Toán học năm 2026"},
]

GT: List[Dict[str, Any]] = [
    {"id": "s1", "must_include": ["600000", "tín chỉ"]},
    {"id": "s2", "must_include": ["7480103"]},
    {"id": "s3", "must_include": ["25.5"]},
    {"id": "m1", "must_include": ["CNTT", "Toán"]},
    {"id": "m2", "must_include": ["A01"]},
    {"id": "c1", "must_include": ["500000", "600000"]},
    {"id": "c2", "must_include": ["27.5", "26.0"]},
]

# (chunk text, answer, total_ms) per id — baseline.
BASELINE_ROWS = {
    "s1": ("Học phí 600000 đồng/tín chỉ", "Học phí là 600.000 đồng/tín chỉ.", 100.0),
    "s2": ("Mã ngành 7480103", "Mã ngành là 7480103.", 200.0),
    "s3": ("Chưa có dữ liệu điểm chuẩn", "Chưa có thông tin.", 300.0),
    "m1": ("Các ngành: CNTT, Toán", "Gồm CNTT và Toán.", 400.0),
    "m2": ("Không tìm thấy tổ hợp", "Không tìm thấy thông tin.", 500.0),
    "c1": ("Sinh học 500000, Vật lý 600000", "Sinh học 500.000, Vật lý 600.000.", 600.0),
    "c2": ("CNTT 27.5", "CNTT 27.5.", 700.0),
}

# Candidate run: every question retrieved; slower simple, faster multihop.
CANDIDATE_ROWS = {
    "s1": ("Học phí 600000 đồng/tín chỉ", "Học phí là 600.000 đồng/tín chỉ.", 150.0),
    "s2": ("Mã ngành 7480103", "Mã ngành là 7480103.", 250.0),
    "s3": ("Điểm chuẩn 25.5", "Điểm chuẩn là 25.5.", 350.0),
    "m1": ("Các ngành: CNTT, Toán", "Gồm CNTT và Toán.", 400.0),
    "m2": ("Tổ hợp A01 gồm Toán, Lý, Anh", "Tổ hợp A01.", 500.0),
    "c1": ("Sinh học 500000, Vật lý 600000", "Sinh học 500.000, Vật lý 600.000.", 600.0),
    "c2": ("CNTT 27.5, Toán 26.0", "CNTT 27.5, Toán 26.0.", 700.0),
}


def _record(qid: str, rows: Dict[str, Any]) -> Dict[str, Any]:
    text, answer, total = rows[qid]
    return {
        "run_id": "fixture",
        "seed": 42,
        "harness_version": "s2-v1",
        "timestamp": "2026-01-01T00:00:00",
        "model_version": {"gen": "g", "ner": "n", "hyde": "h", "embedding": "e"},
        "id": qid,
        "question": next(q["question"] for q in QUESTIONS if q["id"] == qid),
        "intent": "general",
        "complexity": 2,
        "route": "hybrid",
        "expected_behavior": None,
        "retrieved_chunks": [
            {
                "chunk_id": f"chunk-{qid}",
                "source": "fixture",
                "score": 1.0,
                "text": text,
                "data_year": 2026,
                "is_superseded": False,
                "graph_injected": False,
            }
        ],
        "answer": answer,
        "latency_ms": {
            "route_ms": total / 4,
            "retrieval_loop_ms": total / 4,
            "query_ms": total / 2,
            "total_ms": total,
        },
        "latency_attribution": None,
        "judge_scores": None,
        "judge_rationale": None,
    }


def _write_jsonl(path: pathlib.Path, rows: Dict[str, Any]) -> pathlib.Path:
    path.write_text(
        "\n".join(json.dumps(_record(q["id"], rows), ensure_ascii=False) for q in QUESTIONS)
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fixture_paths(tmp_path):
    records = _write_jsonl(tmp_path / "baseline.jsonl", BASELINE_ROWS)
    cand_records = _write_jsonl(tmp_path / "candidate.jsonl", CANDIDATE_ROWS)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(GT, ensure_ascii=False), encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(QUESTIONS, ensure_ascii=False), encoding="utf-8")
    return {
        "records": records,
        "candidate_records": cand_records,
        "gt": gt,
        "dataset": dataset,
        "tmp": tmp_path,
    }


def _build(report_mod, records_path, gt_path, dataset_path, **kwargs):
    return report_mod.build_manifest(
        report_mod.load_records(records_path),
        records_path=records_path,
        gt_lookup=report_mod.load_gt_lookup(gt_path),
        label_lookup=report_mod.load_label_lookup(dataset_path),
        gt_path=gt_path,
        dataset_path=dataset_path,
        **kwargs,
    )


# =============================================================================
# 1. Intent source + classifier rule
# =============================================================================


class TestIntentClassifier:
    def test_dataset_label_wins_when_it_names_a_taxonomy_class(self, intent_mod):
        assert intent_mod.classify_intent("bất kỳ", "simple") == ("simple", "dataset_label")
        assert intent_mod.classify_intent("bất kỳ", "multihop") == ("multihop", "dataset_label")
        assert intent_mod.classify_intent("bất kỳ", "comparison") == ("comparative", "dataset_label")
        assert intent_mod.classify_intent("bất kỳ", "Comparative") == ("comparative", "dataset_label")

    def test_non_taxonomy_label_falls_through_to_derived_rule(self, intent_mod):
        # `temporal` is a dataset label but NOT a taxonomy class -> derived rule.
        intent, source = intent_mod.classify_intent(
            "So sánh điểm chuẩn ngành CNTT và Toán học năm 2026", "temporal"
        )
        assert (intent, source) == ("comparative", "derived_rule")

    def test_derived_comparative(self, intent_mod):
        for question in (
            "So sánh học phí ngành Sinh học và Vật lý",
            "Học phí ngành CNTT khác nhau thế nào so với ngành Toán",
            "Học phí HUSC tăng bao nhiêu từ 2025 sang 2026?",
        ):
            assert intent_mod.derive_intent(question) == "comparative", question

    def test_derived_multihop(self, intent_mod):
        for question in (
            "Liệt kê tất cả các ngành đào tạo của HUSC",
            "Có bao nhiêu ngành tuyển sinh tại HUSC năm 2026?",
            "Ngành nào có chỉ tiêu cao nhất?",
            "Em muốn hỏi học phí và điểm chuẩn ngành CNTT",
        ):
            assert intent_mod.derive_intent(question) == "multihop", question

    def test_derived_simple(self, intent_mod):
        for question in (
            "Mã ngành Kỹ thuật phần mềm là gì?",
            "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?",
        ):
            assert intent_mod.derive_intent(question) == "simple", question

    def test_classifier_reuses_the_router_patterns_not_a_fork(self, intent_mod):
        """The derived rule must be built on the router's own regexes."""
        from services.query_router import (
            _COMPARISON_PATTERNS,
            _ENUMERATION_PATTERNS,
        )

        assert intent_mod._COMPARISON_PATTERNS is _COMPARISON_PATTERNS
        assert intent_mod._ENUMERATION_PATTERNS is _ENUMERATION_PATTERNS

    def test_datasets_without_labels_get_derived_intents(self, intent_mod):
        """husc_thi_sinh_thuc.json carries no label — the rule must still bucket."""
        project = SCRIPTS_DIR.parent
        questions = json.loads(
            (project / "data" / "eval" / "husc_thi_sinh_thuc.json").read_text(
                encoding="utf-8"
            )
        )
        assert all("category" not in q and "intent" not in q for q in questions)
        sources = {
            intent_mod.classify_intent(q["question"], None)[1] for q in questions
        }
        assert sources == {"derived_rule"}
        counts = intent_mod.intent_counts(questions)
        assert sum(counts.values()) == len(questions)
        assert set(counts) == set(intent_mod.INTENT_TAXONOMY)


# =============================================================================
# 2. CI subset — deterministic + intent-stratified
# =============================================================================


class TestCiSubset:
    def test_subset_is_deterministic_and_spans_every_intent(self, intent_mod):
        project = SCRIPTS_DIR.parent
        questions = json.loads(
            (project / "data" / "eval" / "husc_86q_test.json").read_text(encoding="utf-8")
        )
        first = intent_mod.select_ci_subset(questions, 9)
        second = intent_mod.select_ci_subset(questions, 9)
        assert [q["id"] for q in first] == [q["id"] for q in second]
        assert len(first) == 9
        counts = intent_mod.intent_counts(first)
        assert counts == {"simple": 3, "multihop": 3, "comparative": 3}

    def test_subset_is_a_real_subset_of_the_input(self, intent_mod):
        questions = [{"id": f"q{i}", "question": "Mã ngành là gì?"} for i in range(5)]
        subset = intent_mod.select_ci_subset(questions, 3)
        assert [q["id"] for q in subset] == ["q0", "q1", "q2"]

    def test_n_larger_than_input_returns_everything(self, intent_mod):
        questions = [{"id": "q0", "question": "Mã ngành là gì?"}]
        assert intent_mod.select_ci_subset(questions, 50) == questions

    def test_subset_uses_the_dataset_label_lookup_when_records_lack_labels(
        self, intent_mod
    ):
        # Records carry no `category`; every text derives the SAME bucket, so
        # only the dataset label lookup can stratify them.
        text = "Học phí ngành Công nghệ thông tin là bao nhiêu?"
        assert intent_mod.derive_intent(text) == "simple"
        records = [{"id": i, "question": text} for i in ("a", "b", "c", "d")]
        lookup = {"a": "simple", "b": "multihop", "c": "multihop", "d": "comparative"}
        subset = intent_mod.select_ci_subset(records, 3, label_lookup=lookup)
        assert [r["id"] for r in subset] == ["a", "b", "d"]
        assert [r["id"] for r in intent_mod.select_ci_subset(records, 3)] == [
            "a",
            "b",
            "c",
        ]
        # A record's ROUTER intent (`general`/`so_sanh`) is not a dataset label:
        # the dataset lookup must still win over the derived rule.
        routed = [dict(r, intent="general") for r in records]
        routed[1]["intent"] = "so_sanh"  # router vocabulary, NOT a label
        subset = intent_mod.select_ci_subset(routed, 3, label_lookup=lookup)
        assert [r["id"] for r in subset] == ["a", "b", "d"]

    def test_report_cli_ci_subset_keeps_one_question_per_intent(
        self, report_mod, fixture_paths
    ):
        out = fixture_paths["tmp"] / "ci_subset_manifest.json"
        rc = report_mod.main(
            [
                "--records", str(fixture_paths["records"]),
                "--gt", str(fixture_paths["gt"]),
                "--dataset", str(fixture_paths["dataset"]),
                "--ci-subset", "3",
                "--out", str(out),
            ]
        )
        assert rc == 0
        manifest = json.loads(out.read_text(encoding="utf-8"))
        assert manifest["aggregate"]["question_count"] == 3
        for intent in ("simple", "multihop", "comparative"):
            assert manifest["by_intent"][intent]["question_count"] == 1

    def test_runner_cli_ci_subset_runs_only_the_stratified_subset(
        self, runner_mod, tmp_path, monkeypatch
    ):
        """End-to-end through the runner's real CLI entry point."""
        import services.embedding as embedding_mod
        import services.graphrag_retriever as graphrag_mod
        import services.lancedb_retrieval as lancedb_mod
        import services.llm_generator as generator_mod

        docs = [
            _FakeDoc(f"chunk-{q['id']}", BASELINE_ROWS[q["id"]][0]) for q in QUESTIONS
        ]
        monkeypatch.setattr(
            graphrag_mod.UnifiedRAGPipeline,
            "from_disk",
            classmethod(lambda cls: _FakePipeline(docs)),
        )
        monkeypatch.setattr(
            lancedb_mod.LanceDBRetriever,
            "from_env",
            classmethod(lambda cls: _FakeRetriever(docs)),
        )
        monkeypatch.setattr(
            embedding_mod, "EmbeddingService", lambda settings: _FakeEmbedding()
        )
        monkeypatch.setattr(
            generator_mod,
            "LLMGenerator",
            lambda: _DeadProviderGenerator(PROVIDER_ERROR),
        )

        questions_file = tmp_path / "questions.json"
        questions_file.write_text(json.dumps(QUESTIONS, ensure_ascii=False), encoding="utf-8")
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(GT, ensure_ascii=False), encoding="utf-8")
        out = tmp_path / "ci_records.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_run_instrumented.py",
                "--run-id", "ci",
                "--questions", str(questions_file),
                "--gt", str(gt_file),
                "--ci-subset", "3",
                "--out", str(out),
            ],
        )
        runner_mod.main()

        ids = [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()]
        assert ids == ["s1", "m1", "c1"]  # one per intent, dataset order preserved


# =============================================================================
# 3. Per-intent metrics — hand-computed
# =============================================================================


class TestPerIntentMetrics:
    def test_recall_and_latency_per_intent(self, report_mod, fixture_paths):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )

        simple = manifest["by_intent"]["simple"]
        assert simple["question_count"] == 3
        assert simple["recall_at_k"]["scored_questions"] == 3
        assert simple["recall_at_k"]["all_present"]["hits"] == 2
        assert simple["recall_at_k"]["all_present"]["rate"] == pytest.approx(0.6667, abs=1e-4)
        assert simple["latency_ms"]["total_ms"]["p50"] == 200.0
        assert simple["latency_ms"]["total_ms"]["p95"] == 290.0

        multihop = manifest["by_intent"]["multihop"]
        assert multihop["question_count"] == 2
        assert multihop["recall_at_k"]["all_present"]["rate"] == pytest.approx(0.5)
        assert multihop["latency_ms"]["total_ms"]["p50"] == 450.0
        assert multihop["latency_ms"]["total_ms"]["p95"] == 495.0

        comparative = manifest["by_intent"]["comparative"]
        assert comparative["question_count"] == 2
        assert comparative["recall_at_k"]["all_present"]["rate"] == pytest.approx(0.5)
        assert comparative["recall_at_k"]["any_present"]["rate"] == pytest.approx(1.0)
        assert comparative["latency_ms"]["total_ms"]["p50"] == 650.0
        assert comparative["latency_ms"]["total_ms"]["p95"] == 695.0

    def test_aggregate_matches_hand_computation(self, report_mod, fixture_paths):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        agg = manifest["aggregate"]
        assert agg["question_count"] == 7
        assert agg["recall_at_k"]["all_present"]["hits"] == 4
        assert agg["recall_at_k"]["all_present"]["rate"] == pytest.approx(0.5714, abs=1e-4)
        assert agg["recall_at_k"]["any_present"]["rate"] == pytest.approx(0.7143, abs=1e-4)
        assert agg["latency_ms"]["total_ms"]["p50"] == 400.0
        assert agg["latency_ms"]["total_ms"]["p95"] == 670.0

    def test_every_intent_reports_p50_and_p95(self, report_mod, fixture_paths):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        for intent in ("simple", "multihop", "comparative"):
            total = manifest["by_intent"][intent]["latency_ms"]["total_ms"]
            assert total["p50"] is not None and total["p95"] is not None
            assert len(total["ci95_p50"]) == 2 and len(total["ci95_p95"]) == 2
        assert manifest["aggregate"]["recall_at_k"]["all_present"]["ci95"] is not None

    def test_intent_source_is_reported(self, report_mod, fixture_paths):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        assert manifest["by_intent"]["simple"]["intent_source"] == {"dataset_label": 3}
        # c2 carries the non-taxonomy label `temporal` -> derived.
        assert manifest["by_intent"]["comparative"]["intent_source"] == {
            "dataset_label": 1,
            "derived_rule": 1,
        }
        assert manifest["intent"]["dataset_labels_not_in_taxonomy"] == ["temporal"]
        assert manifest["intent"]["rule"] == report_mod.INTENT_RULE_TEXT


# =============================================================================
# 4. No provider -> retrieval metrics survive, generation metrics blocked
# =============================================================================


class _FakeRouterResult:
    def __init__(self) -> None:
        self.all_queries_for_embedding = ["variant"]
        self.complexity = 2
        self.intent = "general"
        self.skip_retrieval = False
        self.auto_answer = None
        self.route = "hybrid"


class _FakeSubRouter:
    async def route(self, query: str) -> _FakeRouterResult:
        return _FakeRouterResult()


class _FakeDoc:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.source = "fixture"
        self.score = 1.0
        self.text = text
        self.metadata = {"data_year": 2026, "is_superseded": False}


class _FakeRagResult:
    def __init__(self, docs: List[_FakeDoc]) -> None:
        self.route = "hybrid"
        self.documents = docs
        self.router_result = _FakeRouterResult()


class _FakePipeline:
    def __init__(self, docs: List[_FakeDoc]) -> None:
        self._router = _FakeSubRouter()
        self._docs = docs

    async def query(self, *, user_query, baseline_docs, top_k, router_result):
        return _FakeRagResult(self._docs)


class _FakeRetrieveResult:
    is_success = True

    def __init__(self, docs: List[_FakeDoc]) -> None:
        self.documents = docs


class _FakeRetriever:
    def __init__(self, docs: List[_FakeDoc]) -> None:
        self._docs = docs

    def retrieve(self, vec, top_k: int = 10, query: str = ""):
        return _FakeRetrieveResult(self._docs)


class _FakeEmbedding:
    def encode_single(self, text: str):
        return np.zeros(8, dtype=np.float32)


class _DeadProviderGenerator:
    """Simulates the dead-provider reality: logs the raw error, then raises."""

    def __init__(self, message: str) -> None:
        self.message = message

    async def generate_answer(self, *, query, chunks, confidence, is_program_list_query):
        logger.error(f"LLM generation failed: {self.message}")
        raise RuntimeError(self.message)


class _SettingsStub:
    NS24_GEN_MODEL = "stub-gen"
    NS24_NER_MODEL = "stub-ner"
    NS24_HYDE_MODEL = "stub-hyde"
    EMBEDDING_MODEL = "stub-emb"


PROVIDER_ERROR = "ramclouds 403 insufficient_user_quota; groq 401 invalid_api_key"


class TestDeadProviderRun:
    def _run(self, runner_mod, fixture_paths, tmp_path):
        docs = [
            _FakeDoc(f"chunk-{q['id']}", BASELINE_ROWS[q["id"]][0]) for q in QUESTIONS
        ]
        out = tmp_path / "dead_provider_records.jsonl"
        completed = asyncio.run(
            runner_mod.run_all(
                QUESTIONS,
                pipeline=_FakePipeline(docs),
                retriever=_FakeRetriever(docs),
                embedding=_FakeEmbedding(),
                generator=_DeadProviderGenerator(PROVIDER_ERROR),
                run_id="dead-provider",
                seed=42,
                settings=_SettingsStub(),
                out_path=str(out),
            )
        )
        assert completed == len(QUESTIONS)
        return out

    def test_blocked_metrics_marked_with_verbatim_provider_error(
        self, runner_mod, report_mod, fixture_paths, tmp_path
    ):
        out = self._run(runner_mod, fixture_paths, tmp_path)
        meta = json.loads(runner_mod.run_meta_path(out).read_text(encoding="utf-8"))
        assert any(
            PROVIDER_ERROR in entry["text"] for entry in meta["provider_errors"]
        ), meta["provider_errors"]

        manifest = _build(
            report_mod, out, fixture_paths["gt"], fixture_paths["dataset"]
        )

        assert manifest["provider"]["status"] == "blocked"
        assert manifest["provider"]["generation_blocked"] == len(QUESTIONS)
        assert any(
            PROVIDER_ERROR in entry["text"] for entry in manifest["provider"]["errors"]
        )

        for intent in ("simple", "multihop", "comparative"):
            for key in ("semantic_accuracy", "groundedness"):
                block = manifest["by_intent"][intent][key]
                assert block["value"] is None, "must NOT fabricate a score"
                assert block["value"] != 0
                assert block["status"] == "blocked_by_provider"
                assert block["why"]
                assert any(
                    PROVIDER_ERROR in entry["text"]
                    for entry in block["provider_errors"]
                ), (intent, key, block["provider_errors"])

        assert manifest["aggregate"]["semantic_accuracy"]["status"] == "blocked_by_provider"
        assert manifest["aggregate"]["groundedness"]["status"] == "blocked_by_provider"

    def test_retrieval_metrics_still_produced_without_a_provider(
        self, runner_mod, report_mod, fixture_paths, tmp_path
    ):
        out = self._run(runner_mod, fixture_paths, tmp_path)
        manifest = _build(
            report_mod, out, fixture_paths["gt"], fixture_paths["dataset"]
        )
        simple = manifest["by_intent"]["simple"]
        assert simple["recall_at_k"]["all_present"]["rate"] == pytest.approx(
            0.6667, abs=1e-4
        )
        assert simple["latency_ms"]["total_ms"]["p50"] is not None
        assert manifest["aggregate"]["recall_at_k"]["scored_questions"] == len(QUESTIONS)

    def test_judge_ran_but_no_generation_is_reported_as_unmeasured_not_blocked(
        self, report_mod, fixture_paths
    ):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        block = manifest["by_intent"]["simple"]["semantic_accuracy"]
        assert block["status"] == "unmeasured_no_judge"
        assert block["value"] is None


# =============================================================================
# 5. Reproducibility — byte-identical re-runs
# =============================================================================


class TestReproducibility:
    def test_cli_re_run_is_byte_identical(self, report_mod, fixture_paths):
        out_a = fixture_paths["tmp"] / "manifest_a.json"
        out_b = fixture_paths["tmp"] / "manifest_b.json"
        for out in (out_a, out_b):
            rc = report_mod.main(
                [
                    "--records", str(fixture_paths["records"]),
                    "--gt", str(fixture_paths["gt"]),
                    "--dataset", str(fixture_paths["dataset"]),
                    "--out", str(out),
                ]
            )
            assert rc == 0
        assert out_a.read_bytes() == out_b.read_bytes()

    def test_manifest_has_no_wall_clock_by_default(self, report_mod, fixture_paths):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        assert "provenance" not in manifest
        assert manifest["content_hash"]

    def test_with_timestamp_is_excluded_from_content_hash(self, report_mod, fixture_paths):
        a = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"],
            generated_at="2026-01-01T00:00:00",
        )
        b = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"],
            generated_at="2026-09-09T09:09:09",
        )
        assert a["provenance"] != b["provenance"]
        assert a["content_hash"] == b["content_hash"]

    def test_bootstrap_is_deterministic(self, report_mod):
        values = [float(v) for v in range(1, 40)]
        first = report_mod.bootstrap_percentile_interval(values, 95.0)
        second = report_mod.bootstrap_percentile_interval(values, 95.0)
        assert first == second
        assert first[0] <= first[1]

    def test_markdown_table_stays_intact_with_the_label_note(self, report_mod, fixture_paths):
        manifest = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        lines = [ln for ln in report_mod.render_manifest_markdown(manifest).splitlines() if ln]
        sep = lines.index("|---|---|---|---|---|---|---|")
        assert [
            ln.split("|")[1].strip() for ln in lines[sep + 1 : sep + 5]
        ] == ["simple", "multihop", "comparative", "aggregate"]


# =============================================================================
# 6. Compare mode — delta + confidence intervals
# =============================================================================


class TestCompareMode:
    def _manifests(self, report_mod, fixture_paths):
        base = _build(
            report_mod, fixture_paths["records"], fixture_paths["gt"], fixture_paths["dataset"]
        )
        cand = _build(
            report_mod, fixture_paths["candidate_records"], fixture_paths["gt"],
            fixture_paths["dataset"],
        )
        return base, cand

    def test_recall_delta_math_and_ci(self, report_mod, fixture_paths):
        base, cand = self._manifests(report_mod, fixture_paths)
        diff = report_mod.build_diff(base, cand)

        simple = diff["by_intent"]["simple"]["recall_at_k_all_present"]
        assert simple["baseline"] == pytest.approx(0.6667, abs=1e-4)
        assert simple["candidate"] == pytest.approx(1.0)
        assert simple["delta"] == pytest.approx(0.3333, abs=1e-4)
        assert simple["ci95_method"] == report_mod.CI_METHODS["rate_delta"]
        assert simple["ci95"][0] <= simple["delta"] <= simple["ci95"][1]

        multihop = diff["by_intent"]["multihop"]["recall_at_k_all_present"]
        assert multihop["delta"] == pytest.approx(0.5)

        agg = diff["by_intent"]["aggregate"]["recall_at_k_all_present"]
        assert agg["delta"] == pytest.approx(3 / 7, abs=1e-4)
        assert agg["n"] == {"baseline": 7, "candidate": 7}

    def test_diff_warns_when_manifests_cover_different_populations(
        self, report_mod, fixture_paths
    ):
        base, cand = self._manifests(report_mod, fixture_paths)
        # Different record files, SAME questions -> no population warning.
        warnings = report_mod.build_diff(base, cand)["warnings"]
        assert not [w for w in warnings if "DIFFERENT question sets" in w]

        other = json.loads(json.dumps(cand))
        other["inputs"]["questions"] = {"count": 5, "sha256": "deadbeef"}
        other["inputs"]["dataset"] = {"name": "other_dataset.json", "sha256": "cafe"}
        warnings = report_mod.build_diff(base, other)["warnings"]
        assert any("DIFFERENT question sets" in w for w in warnings)
        assert any("different datasets" in w for w in warnings)

    def test_identical_manifests_produce_a_zero_delta(self, report_mod, fixture_paths):
        base, _ = self._manifests(report_mod, fixture_paths)
        diff = report_mod.build_diff(base, json.loads(json.dumps(base)))
        agg = diff["by_intent"]["aggregate"]
        assert agg["recall_at_k_all_present"]["delta"] == 0.0
        assert agg["latency_p50_ms"]["delta"] == 0.0
        assert agg["latency_p95_ms"]["delta"] == 0.0

    def test_newcombe_flags_a_clear_difference_and_not_a_small_one(self, report_mod):
        # Interval is for p1 - p2 (candidate first, matching the reported delta).
        wide = report_mod.newcombe_diff_interval(100, 100, 0, 100)
        assert wide is not None and wide[0] > 0  # 100% vs 0% on n=100 is clear
        wide_negative = report_mod.newcombe_diff_interval(0, 100, 100, 100)
        assert wide_negative is not None and wide_negative[1] < 0
        narrow = report_mod.newcombe_diff_interval(1, 3, 2, 3)
        assert narrow is not None and narrow[0] < 0 < narrow[1]

    def test_latency_delta_and_ci(self, report_mod, fixture_paths):
        base, cand = self._manifests(report_mod, fixture_paths)
        diff = report_mod.build_diff(base, cand)
        p50 = diff["by_intent"]["simple"]["latency_p50_ms"]
        assert p50["baseline"] == 200.0
        assert p50["candidate"] == 250.0
        assert p50["delta"] == 50.0
        assert len(p50["ci95"]) == 2 and p50["ci95"][0] <= p50["ci95"][1]
        assert p50["ci95_method"] == report_mod.CI_METHODS["percentile_delta"]

    def test_diff_marks_generation_metrics_blocked_not_fabricated(
        self, report_mod, fixture_paths
    ):
        base, cand = self._manifests(report_mod, fixture_paths)
        diff = report_mod.build_diff(base, cand)
        for intent in ("simple", "multihop", "comparative", "aggregate"):
            block = diff["by_intent"][intent]["semantic_accuracy"]
            assert block["delta"] is None
            assert block["status"] in {"blocked_by_provider", "unmeasured_no_judge"}

    def test_compare_cli_prints_deltas_with_intervals(self, report_mod, fixture_paths, capsys):
        base, cand = self._manifests(report_mod, fixture_paths)
        base_path = fixture_paths["tmp"] / "base.json"
        cand_path = fixture_paths["tmp"] / "cand.json"
        out_path = fixture_paths["tmp"] / "diff.json"
        base_path.write_text(report_mod.canonical_json(base), encoding="utf-8")
        cand_path.write_text(report_mod.canonical_json(cand), encoding="utf-8")

        rc = report_mod.main(
            ["--compare", str(base_path), str(cand_path), "--out", str(out_path)]
        )
        assert rc == 0
        printed = capsys.readouterr().out
        assert "recall@k all-present" in printed
        assert "latency p50 ms" in printed
        assert "multihop" in printed and "comparative" in printed
        assert "[" in printed and "]" in printed  # CI bracket rendered

        diff = json.loads(out_path.read_text(encoding="utf-8"))
        assert diff["schema"] == report_mod.DIFF_SCHEMA
        assert diff["by_intent"]["simple"]["recall_at_k_all_present"]["delta"] == pytest.approx(
            0.3333, abs=1e-4
        )

    def test_diff_is_reproducible(self, report_mod, fixture_paths):
        base, cand = self._manifests(report_mod, fixture_paths)
        first = report_mod.canonical_json(report_mod.build_diff(base, cand))
        second = report_mod.canonical_json(report_mod.build_diff(base, cand))
        assert first == second


# =============================================================================
# 7. Real frozen baseline cross-check (frozen numbers, no provider needed)
# =============================================================================


class TestFrozenBaselineCrossCheck:
    def test_aggregate_matches_the_frozen_baseline_artifact(self, report_mod):
        project = SCRIPTS_DIR.parent
        records = (
            project
            / "results"
            / "eval_harness"
            / "86q_records_bf01_86q_20260912T152147Z.jsonl"
        )
        frozen = (
            project.parent
            / ".scratch"
            / "husc-rag-admin-upgrade"
            / "baseline"
            / "metrics_bf01_86q.json"
        )
        if not records.exists() or not frozen.exists():
            pytest.skip("frozen baseline artifacts not present")

        expected = json.loads(frozen.read_text(encoding="utf-8"))
        manifest = _build(
            report_mod, records, project / "data" / "eval" / "husc_86q_test.json",
            project / "data" / "eval" / "husc_86q_test.json",
        )
        agg = manifest["aggregate"]
        assert agg["recall_at_k"]["all_present"]["rate"] == expected[
            "retrieval_context_recall_must_include"
        ]["all_present_rate"]
        assert agg["latency_ms"]["total_ms"]["p50"] == expected["latency_ms"]["p50_total"]
        assert agg["latency_ms"]["total_ms"]["p95"] == expected["latency_ms"]["p95_total"]
        assert manifest["provider"]["generation_blocked"] == expected["generation_failures"]

    def test_multihop_recall_is_visible_below_simple(self, report_mod):
        """The ticket's whole point: multihop must not hide behind simple."""
        project = SCRIPTS_DIR.parent
        records = (
            project
            / "results"
            / "eval_harness"
            / "86q_records_bf01_86q_20260912T152147Z.jsonl"
        )
        if not records.exists():
            pytest.skip("frozen baseline records not present")
        manifest = _build(
            report_mod, records, project / "data" / "eval" / "husc_86q_test.json",
            project / "data" / "eval" / "husc_86q_test.json",
        )
        simple = manifest["by_intent"]["simple"]["recall_at_k"]["all_present"]["rate"]
        multihop = manifest["by_intent"]["multihop"]["recall_at_k"]["all_present"]["rate"]
        assert simple is not None and multihop is not None
        assert multihop < simple
        assert manifest["by_intent"]["multihop"]["question_count"] > 0
