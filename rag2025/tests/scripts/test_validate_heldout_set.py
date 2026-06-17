"""TDD tests for the HUSC held-out eval set validator (anti-overfit set).

The held-out JSON file (``data/eval/husc_heldout_v1.json``) is the canonical
artifact; the script-side validator is ``scripts/validate_heldout_set.py``.

Composition contract (enforced by ``validate_set()``):
  * ≥40 items, unique ``ho###`` ids
  * per-category counts (with tolerance ±1):
      paraphrase=10, multi_aspect=8, comparison=6, temporal=6,
      out_of_scope=5, edge_case=5
  * every item has id/question/gold_route/expected_behavior/category/notes
  * ``gold_route`` ∈ {"padded", "hybrid", "graph"}
  * ``expected_behavior`` ∈ {"answer", "abstain", "refuse"}
  * ``category`` ∈ {paraphrase, multi_aspect, comparison, temporal,
    out_of_scope, edge_case}
  * ``out_of_scope`` items must be ``refuse``
  * question non-empty string
  * no question meaningfully duplicates any 86Q or 60Q-adversarial
    (15-char bidirectional substring containment OR 5-gram content-shingle
    overlap)
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVAL_DATA = REPO_ROOT / "data" / "eval" / "husc_heldout_v1.json"
EXISTING_86Q = REPO_ROOT / "data" / "eval" / "husc_thi_sinh_thuc.json"
EXISTING_ADV = REPO_ROOT / "data" / "eval" / "husc_adversarial.json"
SCRIPT_PATH = SCRIPTS_DIR / "validate_heldout_set.py"


# ---------- Required constants ----------

MIN_ITEMS = 40
EXPECTED_COUNTS = {
    "paraphrase": 10,
    "multi_aspect": 8,
    "comparison": 6,
    "temporal": 6,
    "out_of_scope": 5,
    "edge_case": 5,
}
VALID_ROUTES = {"padded", "hybrid", "graph"}
VALID_BEHAVIORS = {"answer", "abstain", "refuse"}
VALID_CATEGORIES = set(EXPECTED_COUNTS)


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def items() -> List[Dict[str, Any]]:
    assert EVAL_DATA.is_file(), f"missing held-out set: {EVAL_DATA}"
    with EVAL_DATA.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "held-out set must be a JSON array"
    return data


@pytest.fixture(scope="module")
def corpus_86q() -> List[str]:
    with EXISTING_86Q.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [it["question"] for it in data if isinstance(it, dict)]


@pytest.fixture(scope="module")
def corpus_adv() -> List[str]:
    with EXISTING_ADV.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [it["question"] for it in data if isinstance(it, dict)]


@pytest.fixture(scope="module")
def validator_module():
    """Import scripts/validate_heldout_set.py."""
    if not SCRIPT_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("validate_heldout_set", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- 1. Schema + id sanity ----------

def test_total_is_at_least_40(items):
    assert len(items) >= MIN_ITEMS, f"need ≥{MIN_ITEMS} items, got {len(items)}"


def test_ids_unique(items):
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), f"duplicate ids: {[i for i, c in Counter(ids).items() if c > 1]}"


def test_ids_follow_ho_naming(items):
    """ids should look like ho001..hoNNN (lower-case, 3-digit suffix)."""
    import re
    for it in items:
        iid = it["id"]
        assert re.fullmatch(r"ho\d{3}", iid), f"id {iid!r} must match ho###"


# ---------- 2. Required fields on every item ----------

REQUIRED_FIELDS = ("id", "question", "gold_route", "expected_behavior", "category", "notes")


def test_required_fields_present(items):
    for it in items:
        for fld in REQUIRED_FIELDS:
            assert fld in it, f"item {it.get('id')!r}: missing field {fld!r}"


def test_question_is_nonempty_string(items):
    for it in items:
        q = it.get("question")
        assert isinstance(q, str), f"item {it.get('id')!r}: question must be str, got {type(q).__name__}"
        assert q.strip(), f"item {it.get('id')!r}: question must be non-empty"


# ---------- 3. gold_route, expected_behavior, category validity ----------

def test_gold_route_valid_for_every_item(items):
    for it in items:
        r = it["gold_route"]
        assert r in VALID_ROUTES, (
            f"item {it['id']}: bad gold_route {r!r} (want one of {sorted(VALID_ROUTES)})"
        )


def test_expected_behavior_valid_for_every_item(items):
    for it in items:
        eb = it["expected_behavior"]
        assert eb in VALID_BEHAVIORS, (
            f"item {it['id']}: bad expected_behavior {eb!r} "
            f"(want one of {sorted(VALID_BEHAVIORS)})"
        )


def test_category_valid_for_every_item(items):
    for it in items:
        c = it["category"]
        assert c in VALID_CATEGORIES, (
            f"item {it['id']}: bad category {c!r} "
            f"(want one of {sorted(VALID_CATEGORIES)})"
        )


# ---------- 4. Category distribution roughly matches plan ----------

def test_category_counts_within_tolerance(items):
    counts = Counter(it["category"] for it in items)
    TOLERANCE = 1
    for cat, want in EXPECTED_COUNTS.items():
        got = counts.get(cat, 0)
        assert abs(got - want) <= TOLERANCE, (
            f"category {cat}: want {want} (±{TOLERANCE}), got {got}"
        )


def test_no_unexpected_categories(items):
    counts = Counter(it["category"] for it in items)
    extras = set(counts) - VALID_CATEGORIES
    assert not extras, f"unexpected categories: {extras}"


# ---------- 5. out_of_scope MUST be refuse ----------

def test_out_of_scope_is_always_refuse(items):
    for it in items:
        if it["category"] == "out_of_scope":
            assert it["expected_behavior"] == "refuse", (
                f"item {it['id']} (out_of_scope) must have expected_behavior='refuse', "
                f"got {it['expected_behavior']!r}"
            )


# ---------- 6. Bonus: optional validator script runs cleanly on real data ----------

def test_optional_validator_script_passes(validator_module, items):
    """If the script is present, its validate_set() must accept the data."""
    if validator_module is None:
        pytest.skip("validate_heldout_set.py not yet created")
    report = validator_module.validate_set(items)
    assert isinstance(report, dict), "validate_set() must return dict"
    assert report.get("ok") is True, f"validate_set() reported issues: {report}"
    # Stats block must be present
    assert "stats" in report
    assert report["stats"]["total"] == len(items)
    assert report["stats"]["corpus_86q_size"] == 86
    assert report["stats"]["corpus_adv_size"] == 60


# ---------- 7. Validator rejects crafted bad items ----------

@pytest.fixture
def base_good_item() -> Dict[str, Any]:
    return {
        "id": "hoXX1",
        "question": "Câu hỏi mới hoàn toàn không trùng với bất kỳ câu nào trong 86Q hoặc adversarial",
        "gold_route": "padded",
        "expected_behavior": "answer",
        "category": "paraphrase",
        "notes": "Test fixture for bad-item rules.",
    }


def _bad_variation(base: Dict[str, Any], **overrides) -> Dict[str, Any]:
    item = dict(base)
    item.update(overrides)
    return item


def test_validator_rejects_missing_field(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    bad = _bad_variation(base_good_item)
    del bad["gold_route"]
    report = validator_module.validate_set_with_corpus([bad], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("gold_route" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_bad_route(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    bad = _bad_variation(base_good_item, gold_route="bogus_route")
    report = validator_module.validate_set_with_corpus([bad], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("gold_route" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_bad_behavior(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    bad = _bad_variation(base_good_item, expected_behavior="unsure")
    report = validator_module.validate_set_with_corpus([bad], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("expected_behavior" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_bad_category(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    bad = _bad_variation(base_good_item, category="not_a_real_category")
    report = validator_module.validate_set_with_corpus([bad], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("category" in e or "unexpected" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_out_of_scope_without_refuse(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    bad = _bad_variation(
        base_good_item,
        category="out_of_scope",
        expected_behavior="answer",
        question="Một câu hỏi hoàn toàn không liên quan đến HUSC để test quy tắc refuse",
    )
    report = validator_module.validate_set_with_corpus([bad], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("out_of_scope" in e and "refuse" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_duplicate_id(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    # Two items with the same id → must be flagged
    item1 = _bad_variation(
        base_good_item,
        question="Câu hỏi thứ nhất có nội dung riêng biệt không trùng câu nào trong 86Q",
    )
    item2 = _bad_variation(
        base_good_item,
        question="Câu hỏi thứ hai có nội dung riêng biệt khác hoàn toàn câu thứ nhất",
    )
    report = validator_module.validate_set_with_corpus([item1, item2], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("duplicate ids" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_question_duplicate_of_86q(validator_module, corpus_86q, corpus_adv):
    """A verbatim copy of an 86Q question must be flagged as a duplicate."""
    if validator_module is None:
        pytest.skip("validator not loaded")
    # msg001 = "Em muốn tìm hiểu về các ngành đào tạo của trường mình"
    dup_item = {
        "id": "hoXX2",
        "question": "Em muốn tìm hiểu về các ngành đào tạo của trường mình",
        "gold_route": "padded",
        "expected_behavior": "answer",
        "category": "paraphrase",
        "notes": "Test fixture: this is an exact 86Q copy, must be flagged.",
    }
    report = validator_module.validate_set_with_corpus([dup_item], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("86Q" in e and ("substring" in e or "shingle" in e) for e in report["errors"]), (
        f"expected a 86Q duplicate error, got: {report['errors']}"
    )


def test_validator_rejects_question_duplicate_of_adversarial(validator_module, corpus_86q, corpus_adv):
    """A verbatim copy of an adversarial question must be flagged."""
    if validator_module is None:
        pytest.skip("validator not loaded")
    # adv001 = "Điểm chuẩn ngành CNTT HUSC năm 2025 là bao nhiêu?"
    dup_item = {
        "id": "hoXX3",
        "question": "Điểm chuẩn ngành CNTT HUSC năm 2025 là bao nhiêu?",
        "gold_route": "padded",
        "expected_behavior": "abstain",
        "category": "temporal",
        "notes": "Test fixture: exact adversarial copy, must be flagged.",
    }
    report = validator_module.validate_set_with_corpus([dup_item], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("adversarial" in e and ("substring" in e or "shingle" in e) for e in report["errors"]), (
        f"expected an adversarial duplicate error, got: {report['errors']}"
    )


def test_validator_rejects_empty_question(validator_module, base_good_item, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    bad = _bad_variation(base_good_item, question="   ")
    report = validator_module.validate_set_with_corpus([bad], corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("non-empty" in e for e in report["errors"]), report["errors"]


def test_validator_rejects_too_few_items(validator_module, corpus_86q, corpus_adv):
    if validator_module is None:
        pytest.skip("validator not loaded")
    # Just 1 item — below the 40-item floor.
    tiny = [{
        "id": "ho001",
        "question": "Một câu hỏi rất mới hoàn toàn không trùng với bất kỳ câu nào trong tập cũ",
        "gold_route": "padded",
        "expected_behavior": "answer",
        "category": "paraphrase",
        "notes": "intentionally tiny set",
    }]
    report = validator_module.validate_set_with_corpus(tiny, corpus_86q, corpus_adv)
    assert report["ok"] is False
    assert any("total" in e for e in report["errors"]), report["errors"]


# ---------- 8. Disjointness — direct check on real data ----------

def test_no_question_is_exact_duplicate_of_86q(items, corpus_86q):
    """No held-out question may equal an 86Q question (case/whitespace insensitive)."""
    norm_held = {q.strip().lower() for q in (it["question"] for it in items)}
    norm_86q = {q.strip().lower() for q in corpus_86q}
    overlap = norm_held & norm_86q
    assert not overlap, f"held-out has verbatim 86Q duplicates: {overlap}"


def test_no_question_is_exact_duplicate_of_adversarial(items, corpus_adv):
    """No held-out question may equal an adversarial question (case/whitespace insensitive)."""
    norm_held = {q.strip().lower() for q in (it["question"] for it in items)}
    norm_adv = {q.strip().lower() for q in corpus_adv}
    overlap = norm_held & norm_adv
    assert not overlap, f"held-out has verbatim adversarial duplicates: {overlap}"
