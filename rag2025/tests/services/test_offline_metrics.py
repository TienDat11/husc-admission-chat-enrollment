"""G3-T4: TDD test for the offline metrics math.

We import the pure primitives from `compute_offline_metrics` and assert
their behavior on a hand-checked 3-question mini fixture. We DO NOT touch
real records here — the goal is to pin the metric math so future refactors
of the script can't silently regress precision/recall/MRR.

Bootstraps `sys.path` the same way as `tests/services/test_abstain_hardening.py`:
the metrics script lives under `results/ultraqa_metrics/`, NOT under `src/`,
so we add that dir to sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO / "results" / "ultraqa_metrics"
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))

from compute_offline_metrics import (  # noqa: E402
    fact_level_context_recall,
    hit_rate_at_k,
    is_abstain,
    mrr,
    percentile,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
    route_distribution,
)


# ---------------------------------------------------------------------------
# Hand-checked mini fixture — 3 questions, retrieved chunk_ids, GT facts.
# Q1: 1 GT fact, retrieved=[A, B, C] — relevant in top1 → p@1=1, r@1=1, mrr=1
# Q2: 2 GT facts, retrieved=[A, X, Y, B, Z] — relevant(A) at rank 1, relevant(B) at rank 4
#     → p@5=2/5=0.4, r@5=2/2=1, mrr=1
# Q3: 1 GT fact, retrieved=[X, Y, Z, W, V] — none relevant → p@5=0, r@5=0, mrr=0
# ---------------------------------------------------------------------------
MINI_RECORDS = [
    {
        "id": "mini1",
        "route": "padded_rag",
        "expected_behavior": "answer",
        "retrieved_chunks": [
            {"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"},
        ],
        "answer": "Some long answer " * 5,
        "latency_ms": {"route_ms": 1000, "retrieval_loop_ms": 200, "query_ms": 800, "total_ms": 2000},
        "latency_attribution": {"hyde_ms": 300, "router_inner_ms": 500, "gen_ms": 200, "rerank_ms": None},
    },
    {
        "id": "mini2",
        "route": "hybrid",
        "expected_behavior": "answer",
        "retrieved_chunks": [
            {"chunk_id": "A"}, {"chunk_id": "X"}, {"chunk_id": "Y"},
            {"chunk_id": "B"}, {"chunk_id": "Z"},
        ],
        "answer": "Another long answer " * 5,
        "latency_ms": {"route_ms": 1500, "retrieval_loop_ms": 300, "query_ms": 900, "total_ms": 2700},
        "latency_attribution": {"hyde_ms": 400, "router_inner_ms": 600, "gen_ms": 300, "rerank_ms": None},
    },
    {
        "id": "mini3",
        "route": "graph_rag",
        "expected_behavior": "answer",
        "retrieved_chunks": [
            {"chunk_id": "X"}, {"chunk_id": "Y"}, {"chunk_id": "Z"},
            {"chunk_id": "W"}, {"chunk_id": "V"},
        ],
        "answer": "Yet another long answer " * 5,
        "latency_ms": {"route_ms": 2000, "retrieval_loop_ms": 400, "query_ms": 1000, "total_ms": 3400},
        "latency_attribution": {"hyde_ms": 500, "router_inner_ms": 700, "gen_ms": 400, "rerank_ms": None},
    },
]


MINI_GT = [
    {
        "id": "mini1",
        "critical_facts": [
            {"value": "fact-A", "fact_kind": "test", "supporting_chunk_ids": ["A"]},
        ],
    },
    {
        "id": "mini2",
        "critical_facts": [
            {"value": "fact-A", "fact_kind": "test", "supporting_chunk_ids": ["A"]},
            {"value": "fact-B", "fact_kind": "test", "supporting_chunk_ids": ["B"]},
        ],
    },
    {
        "id": "mini3",
        "critical_facts": [
            {"value": "fact-MISSING", "fact_kind": "test", "supporting_chunk_ids": ["MISSING"]},
        ],
    },
]


# ---------------------------------------------------------------------------
# 0) is_abstain — lead-deferral detection
# ---------------------------------------------------------------------------
def test_is_abstain_pins_contract():
    # Standard string
    assert is_abstain("Tôi không tìm thấy thông tin này trong tài liệu hiện có.") is True
    # Lead-deferral in first 240 chars then fallback content (still abstaining)
    head_defer = (
        "Hiện tại, điểm chuẩn năm 2026 của HUSC chưa được công bố. "
        "Tài liệu tuyển sinh hiện có không cung cấp số liệu của các năm trước đó. "
    ) + "Tham khảo 2025: 15-22 điểm. " * 30
    assert is_abstain(head_defer) is True
    # Deferral buried in mid-paragraph after 240 chars → NOT abstain
    mid = "A long answer " * 50 + " chưa được công bố " + " more text " * 10
    assert is_abstain(mid) is False
    # Empty / very short
    assert is_abstain("") is True
    assert is_abstain("    ") is True
    assert is_abstain("ngắn") is True  # < 80 chars
    # Genuine long answer with no deferral → NOT abstain
    assert is_abstain("Điểm chuẩn ngành CNTT năm 2025 là 24.5 theo phương thức THPT. " * 10) is False


# ---------------------------------------------------------------------------
# 1) precision@k / recall@k / MRR / hit-rate@k — hand-checked
# ---------------------------------------------------------------------------
def test_precision_at_k_basic():
    # Q1: retrieved=[A, B, C], relevant={A} → p@1=1, p@3=1/3
    assert precision_at_k(["A", "B", "C"], {"A"}, 1) == 1.0
    assert precision_at_k(["A", "B", "C"], {"A"}, 3) == pytest.approx(1 / 3)
    # Q3: retrieved=[X, Y, Z], relevant={MISSING} → p@5=0
    assert precision_at_k(["X", "Y", "Z", "W", "V"], {"MISSING"}, 5) == 0.0
    # k=0 → 0
    assert precision_at_k(["A"], {"A"}, 0) == 0.0
    # empty retrieved
    assert precision_at_k([], {"A"}, 5) == 0.0


def test_recall_at_k_basic():
    # Q2: relevant={A, B}, retrieved=[A, X, Y, B, Z] → r@1=1/2, r@3=1/2, r@5=2/2
    assert recall_at_k(["A", "X", "Y", "B", "Z"], {"A", "B"}, 1) == 0.5
    assert recall_at_k(["A", "X", "Y", "B", "Z"], {"A", "B"}, 3) == 0.5
    assert recall_at_k(["A", "X", "Y", "B", "Z"], {"A", "B"}, 5) == 1.0
    # empty relevant
    assert recall_at_k(["A", "B"], set(), 5) == 0.0


def test_mrr_basic():
    # Q1: relevant {A} at rank 1 → mrr=1
    assert mrr(["A", "B", "C"], {"A"}) == 1.0
    # Q2: relevant {A,B}; A at rank 1 → mrr=1 (first hit)
    assert mrr(["A", "X", "Y", "B", "Z"], {"A", "B"}) == 1.0
    # Q2: relevant only {B} (drop A) → mrr=1/4
    assert mrr(["A", "X", "Y", "B", "Z"], {"B"}) == pytest.approx(0.25)
    # miss
    assert mrr(["X", "Y"], {"A"}) == 0.0


def test_hit_rate_at_k_basic():
    # Q1
    assert hit_rate_at_k(["A", "B", "C"], {"A"}, 1) == 1.0
    assert hit_rate_at_k(["A", "B", "C"], {"A"}, 5) == 1.0
    # Q3
    assert hit_rate_at_k(["X", "Y", "Z", "W", "V"], {"MISSING"}, 5) == 0.0
    # boundary k
    assert hit_rate_at_k(["A", "X", "Y"], {"A"}, 3) == 1.0
    assert hit_rate_at_k(["X", "A", "Y"], {"A"}, 1) == 0.0
    assert hit_rate_at_k(["X", "A", "Y"], {"A"}, 2) == 1.0


# ---------------------------------------------------------------------------
# 2) percentile — hand-checked
# ---------------------------------------------------------------------------
def test_percentile_basic():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert percentile([1, 2, 3, 4, 5], 0) == 1.0
    assert percentile([1, 2, 3, 4, 5], 100) == 5.0
    # 95th percentile of [1..100] ≈ 95.05 (linear interp)
    p = percentile(list(range(1, 101)), 95)
    assert 94 <= p <= 96
    # empty
    assert percentile([], 50) == 0.0
    # single
    assert percentile([42], 50) == 42


# ---------------------------------------------------------------------------
# 3) route_distribution — counter correctness
# ---------------------------------------------------------------------------
def test_route_distribution():
    rd = route_distribution(MINI_RECORDS)
    assert rd == {"padded_rag": 1, "hybrid": 1, "graph_rag": 1}


# ---------------------------------------------------------------------------
# 4) fact_level_context_recall — full pipeline math
# ---------------------------------------------------------------------------
def test_fact_level_context_recall_on_mini():
    cr = fact_level_context_recall(MINI_RECORDS, MINI_GT)
    # mini1: 1/1 hits; mini2: 2/2 hits; mini3: 0/1 hits
    # total: 3 facts, 3 hits → recall = 1.0
    assert cr["n_facts"] == 4  # 1+2+1
    assert cr["n_hits"] == 3
    assert cr["n_questions"] == 3
    assert cr["context_recall"] == pytest.approx(0.75)  # 3/4
    by_id = {x["id"]: x for x in cr["per_question"]}
    assert by_id["mini1"]["recall"] == 1.0
    assert by_id["mini2"]["recall"] == 1.0
    assert by_id["mini3"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# 5) retrieval_metrics — full pipeline math
# ---------------------------------------------------------------------------
def test_retrieval_metrics_on_mini():
    rm = retrieval_metrics(MINI_RECORDS, MINI_GT)
    # precision_at_k divides by k (not len(retrieved)) — this is standard IR convention.
    # Q1: 1 fact {A} at rank 1, retrieved=[A,B,C] (3 items) → p@1=1/1=1, p@5=1/5=0.2, r@1=1, mrr=1
    # Q2: 2 facts {A,B}; A@1, B@4, retrieved=[A,X,Y,B,Z] → p@1=1, p@5=2/5=0.4, r@5=1, mrr=1
    # Q3: 1 fact {MISSING}; none retrieved, retrieved=[X,Y,Z,W,V] → p@5=0, r@5=0, mrr=0
    # n=3 questions
    assert rm["n_evaluated"] == 3
    # p@1: (1+1+0)/3 = 0.6667
    assert rm["precision_at_k"]["p@1"] == pytest.approx(2 / 3, rel=1e-3)
    # p@5: (0.2 + 0.4 + 0) / 3
    assert rm["precision_at_k"]["p@5"] == pytest.approx(0.6 / 3, rel=1e-3)
    # r@5: (1 + 1 + 0) / 3 = 0.6667
    assert rm["recall_at_k"]["r@5"] == pytest.approx(2 / 3, rel=1e-3)
    # MRR: (1 + 1 + 0) / 3 = 0.6667
    assert rm["mrr"] == pytest.approx(2 / 3, rel=1e-3)
    # hit@5: (1+1+0)/3 = 0.6667
    assert rm["hit_rate_at_k"]["hit@5"] == pytest.approx(2 / 3, rel=1e-3)
