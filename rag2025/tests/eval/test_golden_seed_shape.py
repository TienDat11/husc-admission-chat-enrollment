"""U1 acceptance: shape of the golden seed (plan §PL-6.6, §PL-6.7).

Thresholds are not lowered: >=40 in-scope, >=10 out-of-scope in five groups.
The 26 supersession cases are not yet available (see xfail reason).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.build_golden_seed import build  # noqa: E402

FIXTURES = ROOT / "tests" / "eval" / "fixtures"
SEED = FIXTURES / "golden_seed.jsonl"
MAPPING = FIXTURES / "legacy_id_mapping.json"
SOURCE = ROOT / "data" / "eval" / "husc86q_golden_records.json"

# §PL-6.6: {record_type}:{scope}:{subject}:{admission_year}:{admission_round}:{methods_hash}
COORDINATE_KEY = re.compile(r"^[A-Z_]+:[a-z_]+:[^:]+:\d{4}:[^:]+:(all|[0-9a-f]{12})$")
OUT_OF_SCOPE_GROUPS = {
    "truong_khac",
    "tin_tuc_thoi_tiet",
    "tu_van_ngoai_tuyen_sinh",
    "nam_chua_cong_bo",
    "chao_hoi",
}


def _rows() -> list[dict]:
    return [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line]


def test_seed_file_exists_and_is_nonempty():
    assert SEED.exists(), "run: python eval/build_golden_seed.py"
    assert _rows()


def test_in_scope_has_at_least_40_cases():
    in_scope = [r for r in _rows() if r["group"] == "in_scope"]
    assert len(in_scope) >= 40


def test_out_of_scope_has_at_least_10_cases_in_five_groups():
    out = [r for r in _rows() if r["group"] == "out_of_scope"]
    assert len(out) >= 10
    assert {r["out_of_scope_group"] for r in out} == OUT_OF_SCOPE_GROUPS
    assert all(r["expect_abstain"] is True for r in out)
    assert all(r["expected_coordinate_keys"] == [] for r in out)


def test_in_scope_coordinate_keys_follow_plan_format():
    for row in _rows():
        if row["group"] != "in_scope":
            continue
        assert row["expected_coordinate_keys"], row["id"]
        for key in row["expected_coordinate_keys"]:
            assert COORDINATE_KEY.match(key), (row["id"], key)


def test_question_ids_are_unique():
    ids = [r["id"] for r in _rows()]
    assert len(ids) == len(set(ids))
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert not duplicates


def _mapping() -> dict:
    return json.loads(MAPPING.read_text(encoding="utf-8"))["mapping"]


def _fact_of_legacy(legacy_id: str) -> tuple[str, str, str, str]:
    record_type, scope, subject, year = legacy_id.split(":")[:4]
    return record_type, scope, subject, year


def test_one_key_identifies_one_fact_identity():
    # Source copies of the same fact (different source tags) may share a key.
    # Two different fact identities (e.g. two programs of one major code) must not.
    key_to_facts: dict[str, set] = {}
    for legacy, entry in _mapping().items():
        key_to_facts.setdefault(entry["coordinate_key"], set()).add(_fact_of_legacy(legacy))
    assert all(len(facts) == 1 for facts in key_to_facts.values())


def test_admission_round_is_the_none_token_everywhere():
    # admission_round is null in the source; the round slot is the fixed token "none".
    for entry in _mapping().values():
        assert entry["coordinate_key"].split(":")[4] == "none", entry


def test_legacy_source_tag_is_kept_in_mapping_not_in_key():
    for legacy, entry in _mapping().items():
        assert entry["legacy_source_tag"] == ":".join(legacy.split(":")[4:]), legacy
        assert entry["legacy_source_tag"] not in entry["coordinate_key"].split(":"), legacy


def test_subject_starts_with_seven_digit_major_code():
    for entry in _mapping().values():
        subject = entry["coordinate_key"].split(":")[2]
        assert re.match(r"^\d{7}", subject), entry


def test_program_suffix_only_when_major_code_has_several_programs():
    subjects_by_code: dict[str, set[str]] = {}
    for legacy in _mapping():
        subject = legacy.split(":")[2]
        subjects_by_code.setdefault(subject[:7], set()).add(subject)
    for entry in _mapping().values():
        subject = entry["coordinate_key"].split(":")[2]
        if subject != subject[:7]:
            assert len(subjects_by_code[subject[:7]]) > 1, entry


@pytest.mark.skipif(not SOURCE.exists(), reason="legacy golden file is gitignored; attach data/ to run")
def test_build_is_deterministic():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    first = build(source)
    second = build(source)
    assert first == second


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BLOCKER U1: 26 supersession cases (§PL-6.7 matrix) need old/new record "
        "pairs with ground-truth outcomes; tests/fixtures/legal_vn/golden_40.jsonl "
        "holds single sentences, not pairs. Awaiting user decision."
    ),
)
def test_supersession_has_26_matrix_cases():
    supersession = [r for r in _rows() if r.get("matrix_row")]
    assert len(supersession) >= 26
