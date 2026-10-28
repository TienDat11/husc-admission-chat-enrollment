"""Ticket 09 — retrieval-only A/B harness (``eval_retrieval_ab.py``).

What is actually load-bearing here, and therefore tested:

  * the RECORD contract — the arm records are ``EvalRecord``-shaped, carry the
    retrieved chunk TEXT (recall@k is a text-substring test), and are readable
    by the existing reporter unchanged. A record that dropped the text would
    silently score 0 on every question and still "run".
  * the PAIRING — both arms see ONE query vector per question. If each arm
    embedded independently, a recall delta could come from the encoder rather
    than from fusion, and the whole artifact would be worthless.
  * the CANDIDATE-POOL MEASUREMENT — the proxy that records the dense pool must
    delegate ``_adapter`` / ``apply_post_fetch_filters`` to the real retriever,
    or ``build_bm25_index()`` would build an empty index and the hybrid arm
    would silently degrade to dense-only. That failure mode produces a
    plausible-looking artifact, so it is pinned.
  * the VERDICT — a bucket too small to resolve a regression must NOT be
    reported as a pass. This is the repo's first honest note of that kind.

Fakes live ONLY in this file; no network, no real model, no real LanceDB.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts"
AB_PATH = SCRIPTS_DIR / "eval_retrieval_ab.py"
EVAL_RECORD_PATH = SCRIPTS_DIR / "eval_record.py"
REPORT_PATH = SCRIPTS_DIR / "eval_per_intent_report.py"
INTENT_PATH = SCRIPTS_DIR / "eval_intent.py"


def _load_via_spec(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ab_mod():
    _load_via_spec("eval_record", EVAL_RECORD_PATH)
    _load_via_spec("eval_intent", INTENT_PATH)
    _load_via_spec("eval_report_metrics", SCRIPTS_DIR / "eval_report_metrics.py")
    _load_via_spec("eval_per_intent_report", REPORT_PATH)
    module = _load_via_spec("eval_retrieval_ab", AB_PATH)
    return module


# =============================================================================
# Fakes — one retriever with a real (tiny) BM25-buildable table shape.
# =============================================================================


class FakeDoc:
    """Mimics ``RetrievedDocument`` for the harness's getattr-based extraction."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        score: float = 1.0,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.source = "fake"
        self.score = score
        self.metadata = metadata or {}
        self.point_id = None


class FakeRetrievalResult:
    def __init__(self, documents: List[FakeDoc], error_type: Any = None) -> None:
        self.documents = documents
        self.error_type = error_type
        self.error_message = None
        self.confidence = 0.5
        self.degraded_filter = False


class _FakeTable:
    """The column-selective ``to_pandas()`` shape ``build_bm25_index`` needs."""

    def __init__(self, docs: List[FakeDoc]) -> None:
        import pandas as pd

        self.schema = type("S", (), {"names": ["text", "chunk_id", "metadata_json"]})()
        self._df = pd.DataFrame(
            {
                "text": [d.text for d in docs],
                "chunk_id": [d.chunk_id for d in docs],
                "metadata_json": [json.dumps(d.metadata) for d in docs],
            }
        )

    def select(self, cols: List[str]) -> Any:
        return type("Sel", (), {"to_pandas": lambda _self: self._df})()

    def to_pandas(self) -> Any:
        return self._df

    def count_rows(self) -> int:
        return len(self._df)


class FakeAdapter:
    def __init__(self, table: _FakeTable) -> None:
        self._table = table


class FakeRetriever:
    """Dense arm: returns the docs whose text matches, in a fixed order.

    ``apply_post_fetch_filters`` is a REAL pass-through (not a bare MagicMock,
    which returns an empty iterable and would silently drop every BM25
    candidate — the exact weakness ``tests/test_hybrid_search.py`` documents).
    """

    def __init__(self, docs: List[FakeDoc], dense_order: List[str]) -> None:
        self._docs = {d.chunk_id: d for d in docs}
        self._dense_order = dense_order
        self._adapter = FakeAdapter(_FakeTable(docs))
        self.calls: List[Dict[str, Any]] = []

    def retrieve(self, query_vector, top_k=None, metadata_filter=None, query=None):
        self.calls.append(
            {
                "top_k": top_k,
                "metadata_filter": metadata_filter,
                "query_vector_len": len(query_vector),
            }
        )
        docs = [self._docs[cid] for cid in self._dense_order[:top_k]]
        return FakeRetrievalResult(docs)

    def apply_post_fetch_filters(self, rows, metadata_filter=None, query=None):
        return rows

    def get_admission_context(self):
        return {"year": 2026, "has_data": True, "season": "regular"}


class FakeEmbedding:
    """Records how many times it was called — the pairing invariant."""

    def __init__(self, vector: List[float] | None = None) -> None:
        self.calls = 0
        self._vector = vector or [0.1] * 4

    def encode_query(self, text: str):
        self.calls += 1

        class _Arr:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return self._values

        return _Arr(self._vector)


class FakeSettings:
    NS24_GEN_MODEL = "gen"
    NS24_NER_MODEL = "ner"
    NS24_HYDE_MODEL = "hyde"
    EMBEDDING_MODEL = "fake-embed"
    USE_HYBRID_RETRIEVAL = False
    HYBRID_FUSION_DENSE_WEIGHT = 0.6
    HYBRID_FUSION_SPARSE_WEIGHT = 0.4


CORPUS = [
    FakeDoc("d1", "học phí ngành công nghệ thông tin là 600000 đồng một tín chỉ"),
    FakeDoc("d2", "mã ngành kỹ thuật phần mềm là 7480103"),
    FakeDoc("d3", "điểm chuẩn ngành toán học năm 2026 là 25.5"),
    FakeDoc("d4", "tổ hợp xét tuyển a01 gồm toán lý anh"),
    FakeDoc("d5", "thời gian nộp hồ sơ xét tuyển đại học"),
]


def _hybrid_for(ab_mod, retriever: FakeRetriever, dense_order: List[str]):
    """Build the same explicit hybrid service ``main()`` builds."""
    from services.hybrid_search import HybridSearchService

    recorder = ab_mod.PoolRecorder(retriever)
    hybrid = HybridSearchService(lancedb_retriever=recorder, settings=FakeSettings())
    assert hybrid.build_bm25_index() is True
    ab_mod._wrap_sparse_recorder(recorder, hybrid)
    return recorder, hybrid


def _run_pair(ab_mod, *, question: str, dense_order: List[str], corpus=None):
    corpus = corpus or CORPUS
    retriever = FakeRetriever(corpus, dense_order)
    recorder, hybrid = _hybrid_for(ab_mod, retriever, dense_order)
    embedding = FakeEmbedding()
    import asyncio

    return (
        asyncio.run(
            ab_mod.run_pair(
                {"id": "q1", "question": question},
                retriever=retriever,
                recorder=recorder,
                hybrid=hybrid,
                embedding=embedding,
                top_k=3,
                run_id="test",
                seed=42,
                settings=FakeSettings(),
            )
        ),
        retriever,
        embedding,
    )


# =============================================================================
# 1. Record contract — what the reporter needs to compute recall@k at all
# =============================================================================


class TestRecordContract:
    def test_record_is_evalrecord_shaped_with_full_chunk_text(self, ab_mod):
        (dense_rec, hybrid_rec, _pool), _, _ = _run_pair(
            ab_mod,
            question="học phí công nghệ thông tin",
            dense_order=["d1", "d2", "d3"],
        )
        import eval_record

        # Round-trips through the SAME schema the instrumented runner writes, so
        # the existing reporter consumes it with no adapter.
        reparsed = eval_record.EvalRecord.from_jsonl_line(hybrid_rec.to_jsonl_line())
        assert reparsed.id == "q1"
        assert reparsed.route == "hybrid_retrieval"
        assert dense_rec.route == "dense_retrieval"

        # FULL text, not ids: recall@k is a substring test over the chunk text.
        assert reparsed.retrieved_chunks
        assert reparsed.retrieved_chunks[0].text.startswith("học phí")
        assert "600000" in reparsed.retrieved_chunks[0].text

        # No generator ran, so the record must not claim an answer.
        assert reparsed.answer == ""

    def test_reporter_computes_recall_from_the_records_it_is_handed(self, ab_mod, tmp_path):
        """End-to-end through the EXISTING reporter: text in -> recall@k out."""
        (dense_rec, hybrid_rec, _pool), _, _ = _run_pair(
            ab_mod,
            question="học phí công nghệ thông tin",
            dense_order=["d1", "d2", "d3"],
        )
        records = tmp_path / "records_hybrid.jsonl"
        records.write_text(hybrid_rec.to_jsonl_line() + "\n", encoding="utf-8")
        dataset = tmp_path / "dataset.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "category": "simple",
                        "question": "học phí công nghệ thông tin",
                    }
                ]
            ),
            encoding="utf-8",
        )
        gt = tmp_path / "gt.json"
        gt.write_text(
            json.dumps([{"id": "q1", "must_include": ["600000", "tín chỉ"]}]),
            encoding="utf-8",
        )

        manifest = json.loads(
            ab_mod.invoke_reporter(
                [
                    "--records",
                    str(records),
                    "--dataset",
                    str(dataset),
                    "--gt",
                    str(gt),
                ]
            )
        )
        recall = manifest["aggregate"]["recall_at_k"]["all_present"]
        assert recall["scored_questions"] == 1
        assert recall["hits"] == 1

    def test_generation_dependent_metrics_are_unmeasured_not_zero(self, ab_mod, tmp_path):
        (_, hybrid_rec, _), _, _ = _run_pair(
            ab_mod, question="học phí", dense_order=["d1", "d2", "d3"]
        )
        records = tmp_path / "r.jsonl"
        records.write_text(hybrid_rec.to_jsonl_line() + "\n", encoding="utf-8")
        dataset = tmp_path / "d.json"
        dataset.write_text(
            json.dumps([{"id": "q1", "category": "simple", "question": "học phí"}]),
            encoding="utf-8",
        )
        manifest = json.loads(
            ab_mod.invoke_reporter(
                ["--records", str(records), "--dataset", str(dataset), "--gt", str(dataset)]
            )
        )
        assert manifest["aggregate"]["semantic_accuracy"]["status"] == "unmeasured_no_judge"
        assert manifest["aggregate"]["semantic_accuracy"]["value"] is None


# =============================================================================
# 2. Pairing invariant — one embedding, two arms
# =============================================================================


class TestPairing:
    def test_both_arms_share_one_query_vector(self, ab_mod):
        (_, _, _), retriever, embedding = _run_pair(
            ab_mod, question="học phí", dense_order=["d1", "d2", "d3"]
        )
        # Exactly one encode for the pair: a per-arm embedding would let the
        # encoder move the delta instead of fusion.
        assert embedding.calls == 1

    def test_dense_arm_receives_the_query_as_temporal_hint(self, ab_mod):
        (_, _, _), retriever, _ = _run_pair(
            ab_mod, question="học phí", dense_order=["d1", "d2", "d3"]
        )
        # The hybrid path injects {"query": <text>} into its own dense call;
        # the dense arm must carry the same hint or the delta would include the
        # temporal wiring as well as fusion.
        assert retriever.calls[0]["metadata_filter"] == {"query": "học phí"}

    def test_both_arms_request_the_same_k(self, ab_mod):
        (dense_rec, hybrid_rec, _), _, _ = _run_pair(
            ab_mod, question="học phí", dense_order=["d1", "d2", "d3"]
        )
        assert len(dense_rec.retrieved_chunks) == 3
        assert len(hybrid_rec.retrieved_chunks) <= 3


# =============================================================================
# 3. PoolRecorder must not break BM25 indexing (silent dense-only degradation)
# =============================================================================


class TestPoolRecorder:
    def test_proxy_delegates_the_attributes_bm25_build_needs(self, ab_mod):
        retriever = FakeRetriever(CORPUS, ["d1", "d2"])
        recorder = ab_mod.PoolRecorder(retriever)
        # build_bm25_index reaches through _adapter._table; a proxy that hid
        # these would build an EMPTY index, hybrid would fall back to
        # dense-only, and the artifact would still look plausible.
        assert recorder._adapter is retriever._adapter
        assert recorder.apply_post_fetch_filters([{"x": 1}]) == [{"x": 1}]

    def test_bm25_index_is_actually_built_through_the_proxy(self, ab_mod):
        from services.hybrid_search import HybridSearchService

        retriever = FakeRetriever(CORPUS, ["d1", "d2"])
        recorder = ab_mod.PoolRecorder(retriever)
        hybrid = HybridSearchService(lancedb_retriever=recorder, settings=FakeSettings())
        assert hybrid.build_bm25_index() is True
        assert hybrid.corpus_version["chunk_count"] == len(CORPUS)

    def test_recorder_captures_both_candidate_pools(self, ab_mod):
        (_, _, pool), _, _ = _run_pair(
            ab_mod, question="học phí công nghệ thông tin", dense_order=["d1", "d2", "d3"]
        )
        # The hybrid arm asks each side for top_k*2 = 6 — asserted on the
        # recorded REQUEST, since this corpus only has 5 docs and a count-based
        # assertion would pass for the wrong reason.
        assert pool["top_k"] == 3
        assert pool["dense_requested_k"] == 6
        assert pool["sparse_requested_k"] == 6
        # The fused union is the criterion-3 quantity: at least as wide as final.
        assert pool["fused_union"] >= pool["final"]
        assert pool["final"] == 3


# =============================================================================
# 4. Verdict logic — a bucket that cannot resolve a regression is not a pass
# =============================================================================


class TestVerdictHonesty:
    def test_unresolved_ci_is_not_reported_as_no_change(self, ab_mod):
        # delta 0 with a wide CI: real point estimate, no resolution.
        block = {"n": {"candidate": 30}, "delta": 0.0, "ci95": [-0.24, 0.24]}
        assert ab_mod._verdict_code(block) == "unresolved"
        assert "no change" not in ab_mod._verdict(block)

    def test_significant_regression_is_flagged(self, ab_mod):
        block = {"n": {"candidate": 100}, "delta": -0.2, "ci95": [-0.3, -0.1]}
        assert ab_mod._verdict_code(block) == "regression"
        assert "REGRESSION" in ab_mod._verdict(block)

    def test_significant_improvement_is_flagged(self, ab_mod):
        block = {"n": {"candidate": 100}, "delta": 0.2, "ci95": [0.1, 0.3]}
        assert ab_mod._verdict_code(block) == "improvement"

    def test_empty_bucket_is_not_measurable_not_a_pass(self, ab_mod):
        block = {"n": {"candidate": 0}, "delta": None, "ci95": None}
        assert ab_mod._verdict_code(block) == "not_measurable"

    def test_summary_states_criterion_1_is_not_certified_when_a_bucket_is_unresolved(
        self, ab_mod
    ):
        """The repo's first honest note: unresolved buckets must not read as a pass."""
        manifest = {
            "by_intent": {"simple": {}, "multihop": {}},
            "aggregate": {},
        }
        delta = {
            "by_intent": {
                "simple": {
                    "recall_at_k_all_present": {
                        "n": {"candidate": 148},
                        "delta": 0.0068,
                        "ci95": [-0.104, 0.1173],
                    }
                },
                "multihop": {
                    "recall_at_k_all_present": {
                        "n": {"candidate": 125},
                        "delta": 0.016,
                        "ci95": [-0.1056, 0.1369],
                    }
                },
                "aggregate": {
                    "recall_at_k_all_present": {
                        "n": {"candidate": 303},
                        "delta": 0.0099,
                        "ci95": [-0.0692, 0.0889],
                    }
                },
            }
        }
        meta = {
            "run_id": "t",
            "dataset": {"questions": 303, "name": "d.json", "sha256": "x" * 64},
            "top_k": 5,
            "embedding": {"model": "m", "dim": 1024},
            "lancedb": {"table": "t", "uri": "u", "rows": 10},
            "hybrid": {
                "bm25_build_ms": 1.0,
                "env_use_hybrid_retrieval": False,
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
                "rrf_k": 60,
                "candidate_pool_per_arm": 10,
            },
            "pools": {
                "dense_candidates": {"n": 303, "p50": 10},
                "sparse_candidates": {"n": 303, "p50": 10},
                "sparse_after_post_fetch": {"n": 303, "p50": 10},
                "fused_union": {"n": 303, "p50": 18},
                "final": {"n": 303, "p50": 5},
                "final_from_sparse_only": {"n": 303, "p50": 0},
                "questions_with_union_gt_final": 303,
                "questions_with_final_from_sparse_only": 0,
            },
            "artifacts": {},
        }
        text = ab_mod.render_summary(
            run_id="t",
            meta=meta,
            dense_manifest=manifest,
            hybrid_manifest=manifest,
            delta=delta,
            historical=None,
            dense_vs_historical=None,
        )
        assert "does not certify criterion 1" in text
        # And it must not claim a pass anywhere.
        assert "Criterion 1 is met" not in text
        # The CI half-width is named as the smallest detectable regression.
        assert "0.1106" in text


# =============================================================================
# 5. Output discipline
# =============================================================================


class TestOutputDiscipline:
    def test_refuses_to_overwrite_an_existing_run_directory(self, ab_mod, tmp_path):
        run_dir = tmp_path / "retrieval_ab" / "fixed_id"
        run_dir.mkdir(parents=True)
        (run_dir / "records_dense.jsonl").write_text("x", encoding="utf-8")
        rc = ab_mod.main(
            [
                "--run-id",
                "fixed_id",
                "--out-root",
                str(tmp_path / "retrieval_ab"),
                "--limit",
                "1",
            ]
        )
        assert rc == 2
        assert (run_dir / "records_dense.jsonl").read_text(encoding="utf-8") == "x"

    def test_summary_only_rerenders_without_retrieval(self, ab_mod, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        manifest = {
            "by_intent": {"simple": {}},
            "aggregate": {},
        }
        delta = {
            "by_intent": {
                "simple": {"recall_at_k_all_present": {"n": {"candidate": 0}}},
                "aggregate": {"recall_at_k_all_present": {"n": {"candidate": 0}}},
            }
        }
        meta = {
            "run_id": "r",
            "dataset": {"questions": 0, "name": "d", "sha256": "x" * 64},
            "top_k": 5,
            "embedding": {"model": "m", "dim": 1},
            "lancedb": {"table": "t", "uri": "u", "rows": 0},
            "hybrid": {
                "bm25_build_ms": 0.0,
                "env_use_hybrid_retrieval": False,
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
                "rrf_k": 60,
                "candidate_pool_per_arm": 10,
            },
            "pools": {},
            "artifacts": {},
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (run_dir / "manifest_dense.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "manifest_hybrid.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "delta_hybrid_vs_dense.json").write_text(json.dumps(delta), encoding="utf-8")
        assert ab_mod.main(["--summary-only", str(run_dir)]) == 0
        assert (run_dir / "summary.md").exists()
