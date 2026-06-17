"""TDD tests for the HUSC adversarial eval set (60 questions across 5 categories).

Adversarial set is the hallucination-pressure test:
  15 temporal          — year-token minimal pairs
  15 fabricated_entity — fake major codes (must fail is_valid_2026_major)
  10 out_of_scope      — completely off-topic (weather, cooking, other unis)
  10 multi_hop         — chain ≥2 real facts from husc_2026_canonical.json
  10 numeric_precision — exact real number, with trap_value to assert

Each item is shaped:
  {"id", "question", "category", "expected_behavior", "notes",
   "trap_value": null, "trap_code": null}
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
RAG2025_ROOT = REPO_ROOT  # same here, tests/scripts/ → 2 parents = rag2025
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVAL_DATA = REPO_ROOT / "data" / "eval" / "husc_adversarial.json"
SCRIPT_PATH = SCRIPTS_DIR / "gen_adversarial_set.py"

# Validator import: rag2025/src/services/major_code_validator.py
SRC_DIR = REPO_ROOT / "src"


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def items() -> List[Dict[str, Any]]:
    assert EVAL_DATA.is_file(), f"missing adversarial set: {EVAL_DATA}"
    with EVAL_DATA.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "adversarial set must be a JSON array"
    return data


@pytest.fixture(scope="module")
def validator():
    """Import rag2025.src.services.major_code_validator via sys.path.insert."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from services.major_code_validator import is_valid_2026_major  # type: ignore
    return is_valid_2026_major


@pytest.fixture(scope="module")
def gen_module():
    """Import the optional gen_adversarial_set.py validator script."""
    if not SCRIPT_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("gen_adversarial_set", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Required constants (the test enforces these counts) ----------

EXPECTED_COUNTS = {
    "temporal": 15,
    "fabricated_entity": 15,
    "out_of_scope": 10,
    "multi_hop": 10,
    "numeric_precision": 10,
}
TOTAL = sum(EXPECTED_COUNTS.values())  # 60
VALID_BEHAVIORS = {"answer", "abstain", "refuse"}
YEAR_RX = re.compile(r"\b20\d{2}\b")


# ---------- 1. Total count + unique ids ----------

def test_total_is_60(items):
    assert len(items) == TOTAL, f"expected {TOTAL} items, got {len(items)}"


def test_ids_unique(items):
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), f"duplicate ids: {[i for i, c in Counter(ids).items() if c > 1]}"


def test_ids_follow_adv_naming(items):
    """ids should look like adv001..adv060 (lower-case, 3-digit suffix)."""
    for it in items:
        iid = it["id"]
        assert re.fullmatch(r"adv\d{3}", iid), f"id {iid!r} must match adv###"


# ---------- 2. Category counts EXACTLY ----------

def test_category_counts_exact(items):
    counts = Counter(it["category"] for it in items)
    for cat, want in EXPECTED_COUNTS.items():
        assert counts.get(cat, 0) == want, f"category {cat}: want {want}, got {counts.get(cat, 0)}"
    # No extras
    extras = set(counts) - set(EXPECTED_COUNTS)
    assert not extras, f"unexpected categories: {extras}"


# ---------- 3. expected_behavior validity ----------

def test_expected_behavior_valid_for_every_item(items):
    for it in items:
        assert it["expected_behavior"] in VALID_BEHAVIORS, (
            f"item {it['id']}: bad expected_behavior {it['expected_behavior']!r}"
        )


# ---------- 4. out_of_scope MUST be refuse ----------

def test_out_of_scope_is_always_refuse(items):
    for it in items:
        if it["category"] == "out_of_scope":
            assert it["expected_behavior"] == "refuse", (
                f"item {it['id']} (out_of_scope) must have expected_behavior='refuse', "
                f"got {it['expected_behavior']!r}"
            )


# ---------- 5. fabricated_entity trap_code must FAIL is_valid_2026_major ----------

def test_fabricated_entity_trap_codes_invalid(items, validator):
    fabs = [it for it in items if it["category"] == "fabricated_entity"]
    assert fabs, "no fabricated_entity items found"
    for it in fabs:
        tc = it.get("trap_code")
        assert tc is not None, f"item {it['id']} (fabricated_entity) missing trap_code"
        assert not validator(tc), (
            f"item {it['id']}: trap_code {tc!r} IS valid 2026 — should be a fake/non-existent code"
        )


# ---------- 6. temporal: year token + ≥5 minimal pairs ----------

def test_temporal_items_contain_4digit_year(items):
    temps = [it for it in items if it["category"] == "temporal"]
    assert len(temps) == EXPECTED_COUNTS["temporal"]
    for it in temps:
        q = it["question"]
        assert YEAR_RX.search(q), (
            f"item {it['id']} (temporal) has no 4-digit year in question: {q!r}"
        )


def test_temporal_has_at_least_5_minimal_pairs(items):
    """Normalize by replacing any 4-digit year with 'YYYY'; ≥5 normalized strings
    must appear ≥2 times each (i.e. form minimal pairs differing only by year)."""
    temps = [it for it in items if it["category"] == "temporal"]
    normalized = [YEAR_RX.sub("YYYY", it["question"]) for it in temps]
    c = Counter(normalized)
    pairs = [q for q, n in c.items() if n >= 2]
    assert len(pairs) >= 5, (
        f"need ≥5 minimal year-pairs in temporal, got {len(pairs)}: {pairs}"
    )


# ---------- 7. numeric_precision: trap_value must be set ----------

def test_numeric_precision_has_trap_value(items):
    nums = [it for it in items if it["category"] == "numeric_precision"]
    assert len(nums) == EXPECTED_COUNTS["numeric_precision"]
    for it in nums:
        tv = it.get("trap_value")
        assert tv is not None, f"item {it['id']} (numeric_precision) missing trap_value"
        # also reject empty strings
        assert isinstance(tv, str) and tv.strip(), (
            f"item {it['id']}: trap_value must be non-empty string, got {tv!r}"
        )


# ---------- 8. Bonus: optional gen_adversarial_set.py validator runs cleanly ----------

def test_optional_gen_script_validator_passes(gen_module):
    """If the script is present, its validate_set() must accept the data."""
    if gen_module is None:
        pytest.skip("gen_adversarial_set.py not yet created")
    with EVAL_DATA.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result = gen_module.validate_set(data)
    assert isinstance(result, dict)
    assert result.get("ok") is True, f"validate_set() reported issues: {result}"
