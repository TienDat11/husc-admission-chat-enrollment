"""30-question synthetic temporal classification gate (TDD V5-R030).

Acceptance: ≥27/30 (90%) classifications correct.
Bucketed: 10 current, 10 historical, 10 ambiguous.

@spec(S13.5)
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "temporal_intent.py"


@pytest.fixture(scope="module")
def ti():
    spec = importlib.util.spec_from_file_location("temporal_intent_30q", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Gold dataset — each item is (question, expected_intent_name).
# 10 current — explicit 2026 OR "năm nay/hiện tại".
GOLD_CURRENT = [
    "Học phí năm 2026 ngành Công nghệ thông tin là bao nhiêu?",
    "Năm 2026 HUSC tuyển bao nhiêu chỉ tiêu?",
    "Điểm chuẩn năm nay ngành Toán là bao nhiêu?",
    "Hiện tại trường có bao nhiêu ngành đào tạo?",
    "Năm nay HUSC mở thêm ngành nào?",
    "Năm 2026 có những phương thức xét tuyển nào?",
    "Học phí hiện tại của khối Khoa học là bao nhiêu?",
    "Năm 2026 ngành Vật lý có tổ hợp gì?",
    "Hiện tại điểm chuẩn ngành CNTT là bao nhiêu?",
    "Năm hiện tại HUSC có bao nhiêu ngành?",
]

# 10 historical — explicit < 2026 OR "năm trước/ngoái".
GOLD_HISTORICAL = [
    "Điểm chuẩn năm 2025 ngành Toán là bao nhiêu?",
    "Năm 2024 chỉ tiêu ngành Sinh học là bao nhiêu?",
    "Năm trước HUSC tuyển bao nhiêu sinh viên?",
    "Năm ngoái điểm chuẩn ngành Hóa là bao nhiêu?",
    "Học phí năm 2025 là bao nhiêu?",
    "Năm 2023 HUSC có những ngành gì?",
    "Năm trước có những phương thức xét tuyển nào?",
    "Năm ngoái có ngành nào mới?",
    "Năm 2025 chỉ tiêu ngành Toán-Tin là bao nhiêu?",
    "Năm 2022 ngành Vật lý có tổ hợp gì?",
]

# 10 ambiguous — no temporal signal at all.
GOLD_AMBIGUOUS = [
    "Học phí ngành Công nghệ thông tin là bao nhiêu?",
    "Điểm chuẩn ngành Toán là bao nhiêu?",
    "HUSC có bao nhiêu ngành đào tạo?",
    "Ngành CNTT có tổ hợp xét tuyển nào?",
    "Có ký túc xá cho sinh viên không?",
    "Quy trình xét tuyển ra sao?",
    "Ngành Sinh học có khoa nào?",
    "Tỷ lệ sinh viên có việc làm sau tốt nghiệp?",
    "HUSC ở đâu?",
    "Ngành Triết học có tuyển không?",
]

CURRENT_YEAR = 2026


def _classify(ti, question):
    return ti.classify_temporal(question, CURRENT_YEAR)


def test_30q_classifier_meets_90pct_threshold(ti):
    correct = 0
    total = 30
    misses = []

    for q in GOLD_CURRENT:
        actual = _classify(ti, q).value
        if actual == "current":
            correct += 1
        else:
            misses.append((q, "current", actual))

    for q in GOLD_HISTORICAL:
        actual = _classify(ti, q).value
        if actual == "historical":
            correct += 1
        else:
            misses.append((q, "historical", actual))

    for q in GOLD_AMBIGUOUS:
        actual = _classify(ti, q).value
        if actual == "ambiguous":
            correct += 1
        else:
            misses.append((q, "ambiguous", actual))

    pct = correct / total
    msg = f"Accuracy {correct}/{total} = {pct:.1%}\nMisses:\n" + "\n".join(
        f"  '{q}' expected={exp} got={got}" for q, exp, got in misses
    )
    assert correct >= 27, msg


def test_current_bucket_at_least_8_of_10_correct(ti):
    correct = sum(
        1 for q in GOLD_CURRENT if _classify(ti, q) == ti.TemporalIntent.current
    )
    assert correct >= 8, f"Current bucket only {correct}/10"


def test_historical_bucket_at_least_8_of_10_correct(ti):
    correct = sum(
        1 for q in GOLD_HISTORICAL if _classify(ti, q) == ti.TemporalIntent.historical
    )
    assert correct >= 8, f"Historical bucket only {correct}/10"


def test_ambiguous_bucket_at_least_8_of_10_correct(ti):
    correct = sum(
        1 for q in GOLD_AMBIGUOUS if _classify(ti, q) == ti.TemporalIntent.ambiguous
    )
    assert correct >= 8, f"Ambiguous bucket only {correct}/10"


def test_no_ambiguous_question_misclassified_as_historical(ti):
    """Sanity: ambiguous questions should NOT silently get year-filtered."""
    bad = [q for q in GOLD_AMBIGUOUS if _classify(ti, q) == ti.TemporalIntent.historical]
    assert not bad, f"Ambiguous → historical leak: {bad}"


def test_buckets_are_disjoint_by_construction(ti):
    """Sanity: gold buckets don't overlap (no question shared)."""
    all_q = GOLD_CURRENT + GOLD_HISTORICAL + GOLD_AMBIGUOUS
    assert len(set(all_q)) == 30, "Gold buckets contain duplicates"
