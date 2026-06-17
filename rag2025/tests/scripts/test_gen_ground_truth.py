"""TDD tests for the HUSC ground-truth dataset (86Q) generator.

This file is part of Slice S3: ground-truth dataset (fact-level, traceable).

Schema under test (per entry in husc_thi_sinh_thuc_gt.json):
    {
        "id": "msg001".."msg086",
        "question": "...",
        "reference_answer": "...",
        "expected_behavior": "answer" | "abstain" | "refuse",
        "source_doc_refs": ["canonical:..." | "opqa:..."],
        "critical_facts": [
            {"value": "<string>",
             "source_doc_ref": "<ref that exists in build_source_map()>",
             "fact_kind": "<short tag, e.g. hoc_phi / chi_tieu / phuong_thuc>"},
            ...
        ],
        "verified": False  # human/Opus verification flag (always False at slice time)
    }

The traceability validator checks:
  * expected_behavior ∈ {answer, abstain, refuse}
  * for every critical_fact: source_doc_ref ∈ source_map AND value (with numeric
    normalization) is a substring of source_map[source_doc_ref]
  * no fabricated major code leaks into reference_answer (C11)
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
RAG2025_ROOT = REPO_ROOT
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVAL_DATA = REPO_ROOT / "data" / "eval" / "husc_thi_sinh_thuc.json"
GT_DATA = REPO_ROOT / "data" / "eval" / "husc_thi_sinh_thuc_gt.json"
SCRIPT_PATH = SCRIPTS_DIR / "gen_ground_truth.py"
SRC_DIR = REPO_ROOT / "src"

CANONICAL_PATH = REPO_ROOT / "data" / "raw" / "husc_2026_canonical.json"
OPQA_PATH = REPO_ROOT / "data" / "raw" / "husc_2026_operational_qa.json"

VALID_BEHAVIORS = {"answer", "abstain", "refuse"}


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def gen_module():
    """Import the gen_ground_truth.py script as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("gen_ground_truth", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_map(gen_module):
    return gen_module.build_source_map()


@pytest.fixture(scope="module")
def gt_list():
    assert GT_DATA.is_file(), f"missing GT file: {GT_DATA}"
    with GT_DATA.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "GT file must be a JSON array"
    return data


@pytest.fixture(scope="module")
def questions():
    with EVAL_DATA.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validator(gen_module, source_map, gt_list):
    """Run the validator over the full shipped GT file."""
    return gen_module.validate_ground_truth(gt_list, source_map)


# ---------- 1. File presence + structural integrity ----------

def test_gt_file_exists_and_has_86_entries(gt_list):
    assert len(gt_list) == 86, f"expected 86 GT entries, got {len(gt_list)}"


def test_gt_ids_msg001_to_msg086_no_dupes(gt_list):
    ids = [it["id"] for it in gt_list]
    assert len(ids) == len(set(ids)), "duplicate ids"
    expected = {f"msg{n:03d}" for n in range(1, 87)}
    assert set(ids) == expected, f"missing or extra ids: got {len(ids)}"


def test_gt_every_entry_has_valid_behavior(gt_list):
    bad = [it["id"] for it in gt_list
           if it.get("expected_behavior") not in VALID_BEHAVIORS]
    assert not bad, f"entries with invalid expected_behavior: {bad}"


def test_gt_every_entry_has_verified_flag(gt_list):
    for it in gt_list:
        assert "verified" in it, f"missing verified field on {it.get('id')}"
        assert isinstance(it["verified"], bool), \
            f"verified must be bool on {it.get('id')} (got {type(it['verified']).__name__})"


# ---------- 2. source_map sanity ----------

def test_source_map_contains_tuition_khoi_v(source_map):
    assert "canonical:tuition:khoi_v" in source_map, \
        "source_map must contain canonical:tuition:khoi_v"
    text = source_map["canonical:tuition:khoi_v"]
    # Must contain the amount "600000" (possibly with . or , or spaces).
    norm = text.replace(".", "").replace(",", "").replace(" ", "")
    assert "600000" in norm, \
        f"khoi_v source must contain 600000; got: {text[:200]!r}"


def test_source_map_contains_a_program(source_map):
    """At least one canonical:programs:<code> key must exist with a real program text."""
    prog_keys = [k for k in source_map if k.startswith("canonical:programs:")]
    assert prog_keys, "source_map must have canonical:programs:<code> entries"
    # Pick first, check it has the program's code somewhere in its text.
    key = prog_keys[0]
    code = key.split(":", 2)[2]
    assert code in source_map[key], \
        f"program source for {key} should contain its own code {code}"


def test_source_map_has_phuong_thuc_and_diem_cong(source_map):
    assert "canonical:phuong_thuc" in source_map
    assert "canonical:diem_cong" in source_map


def test_source_map_has_opqa_entries(source_map):
    opqa_keys = [k for k in source_map if k.startswith("opqa:")]
    assert len(opqa_keys) >= 1, "source_map must flatten operational_qa into opqa:* keys"


# ---------- 3. Traceability: positive (a real fact MUST pass) ----------

def test_traceability_positive_tuition(gen_module, source_map):
    """A hand-crafted GT entry with a real khoi_v tuition fact must validate OK."""
    entry = {
        "id": "test-ok-001",
        "question": "test?",
        "reference_answer": "test answer",
        "expected_behavior": "answer",
        "source_doc_refs": ["canonical:tuition:khoi_v"],
        "critical_facts": [
            {"value": "600000", "source_doc_ref": "canonical:tuition:khoi_v",
             "fact_kind": "hoc_phi"},
        ],
        "verified": False,
    }
    res = gen_module.validate_ground_truth([entry], source_map)
    assert res["ok"] is True, f"expected ok=True, got errors: {res['errors']}"
    assert res["n_verified"] == 1
    assert res["n_unverified"] == 0


# ---------- 4. Traceability: negative (a fake fact MUST be flagged) ----------

def test_traceability_negative_fake_value(gen_module, source_map):
    """A critical_fact with a value NOT present in the source must be flagged."""
    entry = {
        "id": "test-bad-001",
        "question": "test?",
        "reference_answer": "test answer",
        "expected_behavior": "answer",
        "source_doc_refs": ["canonical:tuition:khoi_v"],
        "critical_facts": [
            {"value": "999999", "source_doc_ref": "canonical:tuition:khoi_v",
             "fact_kind": "hoc_phi"},
        ],
        "verified": False,
    }
    res = gen_module.validate_ground_truth([entry], source_map)
    assert res["ok"] is False, "validator must flag the untraceable fact"
    assert res["n_unverified"] >= 1
    assert any("999999" in e for e in res["errors"]), \
        f"errors should mention the bad value 999999: {res['errors']}"


def test_traceability_negative_missing_source_ref(gen_module, source_map):
    """A critical_fact pointing to a non-existent source must be flagged."""
    entry = {
        "id": "test-bad-002",
        "question": "test?",
        "reference_answer": "test answer",
        "expected_behavior": "answer",
        "source_doc_refs": [],
        "critical_facts": [
            {"value": "600000", "source_doc_ref": "canonical:does_not_exist",
             "fact_kind": "hoc_phi"},
        ],
        "verified": False,
    }
    res = gen_module.validate_ground_truth([entry], source_map)
    assert res["ok"] is False
    assert any("does_not_exist" in e for e in res["errors"]), \
        f"errors should mention missing ref: {res['errors']}"


# ---------- 5. Fabricated major code guard ----------

def test_fabricated_code_in_reference_answer_is_flagged(gen_module, source_map):
    """A reference_answer mentioning a fake 7-digit code must be flagged (C11)."""
    entry = {
        "id": "test-fab-001",
        "question": "test?",
        "reference_answer": "Ngành XYZ có mã 7999999 là ngành mới.",  # 7999999 not in WHITELIST_2026
        "expected_behavior": "answer",
        "source_doc_refs": [],
        "critical_facts": [],
        "verified": False,
    }
    res = gen_module.validate_ground_truth([entry], source_map)
    assert res["ok"] is False
    assert any("7999999" in e for e in res["errors"]), \
        f"fabricated code 7999999 must be flagged: {res['errors']}"


# ---------- 6. Validator over the ACTUAL shipped file ----------

def test_shipped_file_passes_traceability_on_facts(validator, gt_list):
    """Every entry in the shipped file that HAS critical_facts must pass traceability."""
    assert validator["ok"] is True, \
        f"shipped GT file has untraceable critical_facts: {validator['errors']}"
    # Sanity: there must be at least one entry with critical_facts (the seeded
    # ≥10) and the verified count must equal that many.
    entries_with_facts = [e for e in gt_list if e.get("critical_facts")]
    assert len(entries_with_facts) >= 10, \
        f"shipped file must have ≥10 entries with critical_facts, got {len(entries_with_facts)}"
    assert validator["n_verified"] == len(entries_with_facts), (
        f"verified count {validator['n_verified']} != entries-with-facts "
        f"{len(entries_with_facts)}"
    )


# ---------- 7. Verified flag helper ----------

def test_load_verified_only_helper(gen_module, gt_list):
    """A `load_verified_only(path)` helper must return only verified==True entries."""
    helper = getattr(gen_module, "load_verified_only", None)
    assert callable(helper), "gen_ground_truth.py must expose load_verified_only(path)"
    result = helper(GT_DATA)
    # The shipped file contains a seeded verified-true subset (13 entries) and
    # remaining verified=false entries. The helper returns exactly the true ones.
    assert isinstance(result, list)
    assert all(e.get("verified") is True for e in result), (
        f"load_verified_only returned non-verified entries: {result!r}"
    )
    assert len(result) == sum(1 for e in gt_list if e.get("verified") is True), (
        f"helper returned {len(result)} verified entries, expected "
        f"{sum(1 for e in gt_list if e.get('verified') is True)}"
    )
