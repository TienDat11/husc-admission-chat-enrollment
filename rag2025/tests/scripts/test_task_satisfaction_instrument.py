"""Ticket 30 — task-level satisfaction instrument.

Proves observable behavior, not wiring:

  * the rubric's three conditions (plan:786-790) and the exact admission-intent
    list the plan's "7-8 intent tuyen sinh chinh" (plan:794) refers to;
  * the design floors (>=200 baseline / >=400 release, plan:794-795);
  * the gate boundaries: exactly at the floor, one below, point exactly 99.0%,
    lower 95% CI exactly 97.5%, and one critical error;
  * a stratum regression that must survive an otherwise-passing aggregate;
  * a provider-dead log yields ``insufficient_evidence`` with a named reason and
    NO satisfaction rate -- the environment cannot manufacture a pass;
  * byte-identical re-runs on the same log.

Fakes live ONLY in this test file. No network, no real pipeline.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts"
SATISFACTION_PATH = SCRIPTS_DIR / "eval_task_satisfaction.py"

# The real ticket-08 artifact for the provider-dead environment.
PROVIDER_DEAD_RECORDS = (
    SCRIPTS_DIR.parent
    / "results"
    / "eval_harness"
    / "86q_records_bf01_86q_20260912T152147Z.jsonl"
)


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
def sat():
    return _load_via_spec("eval_task_satisfaction", SATISFACTION_PATH)


# =============================================================================
# Fixtures
# =============================================================================

def task(
    task_id: str,
    *,
    role: str = "student",
    intent: Optional[str] = None,
    question: str = "Học phí ngành CNTT 2026?",
    completed: Any = True,
    trusted: Any = True,
    critical: Any = False,
    repeated: bool = False,
    answer: Optional[str] = None,
) -> Dict[str, Any]:
    """One already-collected task row (the questionnaire's serialized form)."""
    row: Dict[str, Any] = {
        "task_id": task_id,
        "role": role,
        "question": question,
        "task_completed": completed,
        "citation_trusted": trusted,
        "critical_misinformation": critical,
    }
    if intent is not None:
        row["admission_intent"] = intent
    if repeated:
        row["repeated_session"] = True
    if answer is not None:
        row["answer"] = answer
    return row


def uniform_log(
    n: int,
    *,
    role: str = "student",
    intent: str = "học phí",
    satisfactory: Optional[int] = None,
    critical_errors: int = 0,
) -> List[Dict[str, Any]]:
    """``n`` tasks in ONE stratum; ``satisfactory`` of them satisfactory."""
    if satisfactory is None:
        satisfactory = n
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        # The last `critical_errors` tasks are non-satisfactory *because* of a
        # critical error, which is the case the ceiling must catch.
        if i < satisfactory:
            rows.append(task(f"t{i}", role=role, intent=intent))
        elif i < satisfactory + critical_errors:
            rows.append(task(f"t{i}", role=role, intent=intent, critical=True))
        else:
            rows.append(task(f"t{i}", role=role, intent=intent, completed=False))
    return rows


def verdict_of(sat, rows, phase: str = "release") -> Dict[str, Any]:
    return sat.build_report(rows, phase=phase)


# =============================================================================
# Rubric + intent list (the plan's definition)
# =============================================================================

def test_admission_intents_are_the_seven_from_the_ux_requirements(sat):
    """The plan says "7-8 intent tuyen sinh chinh" (line 794) without listing
    them; the instrument must use the inherited list, not invent one."""
    assert sat.ADMISSION_INTENTS == (
        "ngành",
        "phương thức",
        "chỉ tiêu",
        "học phí",
        "mốc thời gian",
        "hồ sơ",
        "điểm chuẩn",
    )
    assert len(sat.ADMISSION_INTENTS) == 7
    assert sat.INTENT_SOURCE["file"] == "docs/chatgpt-ux-admin-requirements.md"
    assert sat.INTENT_SOURCE["line"] == 10

    doc = pathlib.Path(sat.INTENT_SOURCE["file"])
    if not doc.exists():
        doc = SCRIPTS_DIR.parent.parent / sat.INTENT_SOURCE["file"]
    line = doc.read_text(encoding="utf-8").splitlines()[sat.INTENT_SOURCE["line"] - 1]
    for intent in sat.ADMISSION_INTENTS:
        assert intent in line, f"{intent!r} is not on the cited source line"


def test_rubric_encodes_the_three_plan_conditions(sat):
    conditions = sat.RUBRIC["conditions"]
    assert [c["id"] for c in conditions] == [
        "a_task_completed",
        "b_citation_trusted",
        "c_no_critical_misinformation",
    ]
    assert [(c["field"], c["pass_when"]) for c in conditions] == [
        ("task_completed", True),
        ("citation_trusted", True),
        ("critical_misinformation", False),
    ]
    assert "AND" in sat.RUBRIC["satisfactory"]
    assert sat.QUESTIONNAIRE["fields"]["critical_misinformation"].startswith(
        "condition C"
    )


@pytest.mark.parametrize(
    "completed,trusted,critical,expected",
    [
        (True, True, False, True),   # all three conditions hold
        (False, True, False, False),  # (a) user did not complete the task
        (True, False, False, False),  # (b) citation judged untrustworthy
        (True, True, True, False),    # (c) critical misinformation found
        (False, False, True, False),
    ],
)
def test_satisfactory_requires_all_three_conditions(sat, completed, trusted, critical, expected):
    row = task("t0", completed=completed, trusted=trusted, critical=critical)
    built = sat.build_task(row)
    assert built["satisfactory"] is expected


def test_unadjudicated_task_is_excluded_not_counted_as_satisfied(sat):
    """A missing rubric field must never be silently read as satisfaction."""
    rows = [task("t0", trusted=None)]
    built = sat.build_task(rows[0])
    assert built["satisfactory"] is None
    report = verdict_of(sat, rows)
    assert report["sample"]["scored"] == 0
    assert report["sample"]["unadjudicated_by_condition"]["citation_trusted"] == 1
    assert report["aggregate"]["rate"] is None
    assert report["verdict"] == "insufficient_evidence"


def test_questionnaire_answers_accept_serialized_booleans(sat):
    assert sat.as_bool_or_none(True) is True
    assert sat.as_bool_or_none("no") is False
    assert sat.as_bool_or_none("Không") is False
    assert sat.as_bool_or_none(0) is False
    assert sat.as_bool_or_none("maybe") is None
    assert sat.as_bool_or_none(None) is None


# =============================================================================
# Intent classification + stratification
# =============================================================================

@pytest.mark.parametrize(
    "question,expected",
    [
        ("Học phí tín chỉ ngành CNTT 2026 là bao nhiêu?", "học phí"),
        ("Điểm chuẩn ngành Toán học năm 2026?", "điểm chuẩn"),
        ("Chỉ tiêu tuyển sinh ngành Sinh học?", "chỉ tiêu"),
        ("Các phương thức xét tuyển năm 2026?", "phương thức"),
        ("Hạn nộp hồ sơ xét tuyển là khi nào?", "hồ sơ"),
        ("Thời gian công bố kết quả trúng tuyển?", "mốc thời gian"),
        ("Trường có những ngành đào tạo nào?", "ngành"),
        ("Xin chào", "khác"),
    ],
)
def test_intent_derived_from_question_text(sat, question, expected):
    intent, source = sat.classify_admission_intent(question)
    assert intent == expected
    assert source == "derived_rule"


def test_explicit_intent_label_wins_over_derived_rule(sat):
    intent, source = sat.classify_admission_intent(
        "Học phí ngành CNTT?", explicit="điểm chuẩn"
    )
    assert (intent, source) == ("điểm chuẩn", "explicit_label")


def test_router_vocabulary_is_not_mistaken_for_an_admission_intent(sat):
    """Ticket-08 records carry intent in the ROUTER's vocabulary."""
    row = task("t0", intent="general", question="Học phí ngành CNTT 2026?")
    row["intent"] = "general"
    row.pop("admission_intent")
    built = sat.build_task(row)
    assert built["intent"] == "học phí"
    assert built["intent_source"] == "derived_rule"


def test_stratification_reports_every_role_and_intent_separately(sat):
    rows = [
        task("s1", role="student", intent="học phí"),
        task("s2", role="student", intent="học phí", completed=False),
        task("t1", role="teacher", intent="điểm chuẩn"),
        task("t2", role="teacher", intent="điểm chuẩn"),
    ]
    report = verdict_of(sat, rows)
    assert report["strata"]["by_role"]["student"]["scored"] == 2
    assert report["strata"]["by_role"]["student"]["satisfactory"] == 1
    assert report["strata"]["by_role"]["teacher"]["scored"] == 2
    assert report["strata"]["by_role"]["teacher"]["rate"] == 1.0
    assert report["strata"]["by_intent"]["học phí"]["rate"] == 0.5
    assert report["strata"]["by_intent"]["điểm chuẩn"]["rate"] == 1.0
    # every stratum key is present even when empty, so nothing is hidden
    assert set(report["strata"]["by_intent"]) == set(sat.INTENT_STRATA)
    assert set(report["strata"]["by_role"]) == set(sat.ROLES) | {sat.UNKNOWN_ROLE}


def test_repeated_spam_session_is_not_an_independent_sample(sat):
    rows = uniform_log(399) + [
        task("spam1", repeated=True),
        task("spam2", repeated=True),
    ]
    report = verdict_of(sat, rows)
    assert report["sample"]["records"] == 401
    assert report["sample"]["excluded_repeated_session"] == 2
    assert report["sample"]["independent"] == 399
    assert report["sample"]["scored"] == 399
    assert report["verdict"] == "insufficient_evidence"
    assert "below_design_floor" in report["verdict_reasons"]


# =============================================================================
# Gate boundaries
# =============================================================================

def test_exactly_at_the_release_floor_passes(sat):
    report = verdict_of(sat, uniform_log(400), phase="release")
    assert report["sample"]["scored"] == 400
    assert report["sample"]["floor"] == 400
    assert report["aggregate"]["rate"] == 1.0
    assert report["aggregate"]["meets_point"] is True
    assert report["aggregate"]["meets_lower_ci"] is True
    assert report["critical_misinformation"]["observed"] == 0
    assert report["verdict"] == "pass"


def test_one_below_the_release_floor_is_insufficient_evidence(sat):
    report = verdict_of(sat, uniform_log(399), phase="release")
    assert report["sample"]["scored"] == 399
    assert report["verdict"] == "insufficient_evidence"
    assert report["verdict_reasons"] == ["below_design_floor"]
    assert "399" in report["reason_detail"]
    assert "400" in report["reason_detail"]


def test_baseline_floor_is_200(sat):
    assert verdict_of(sat, uniform_log(200), phase="baseline")["verdict"] == "pass"
    below = verdict_of(sat, uniform_log(199), phase="baseline")
    assert below["verdict"] == "insufficient_evidence"
    assert below["sample"]["floor"] == 200


def test_point_estimate_exactly_99_percent(sat):
    """396/400 == 0.99 exactly: the point criterion is met at the boundary,
    and the Wilson lower bound (0.97457) then decides the verdict."""
    rows = uniform_log(400, satisfactory=396)
    report = verdict_of(sat, rows)
    assert report["aggregate"]["rate"] == 0.99
    assert report["aggregate"]["meets_point"] is True
    assert report["aggregate"]["wilson_95"][0] == 0.9746
    assert report["aggregate"]["meets_lower_ci"] is False
    assert report["verdict"] == "fail"
    assert report["verdict_reasons"] == ["lower_ci_below_target"]


def test_point_below_99_percent_does_not_pass_by_display_rounding(sat):
    """1084/1095 == 0.98995 prints as "99.00%" at 4dp but is truly below target."""
    rows = uniform_log(1095, satisfactory=1084)
    report = verdict_of(sat, rows)
    assert report["aggregate"]["rate"] == 0.99  # what the report prints
    assert 1084 / 1095 < 0.99  # what the gate reads
    assert report["aggregate"]["meets_point"] is False
    assert report["verdict"] == "fail"
    assert report["verdict_reasons"] == ["point_below_target"]


def test_lower_ci_exactly_97_5_percent_passes(sat):
    """403/407: point 0.99017 and Wilson lower bound 0.9750 exactly."""
    rows = uniform_log(407, satisfactory=403)
    report = verdict_of(sat, rows)
    assert report["aggregate"]["rate"] == 0.9902
    assert report["aggregate"]["wilson_95"][0] == 0.975
    assert report["aggregate"]["meets_point"] is True
    assert report["aggregate"]["meets_lower_ci"] is True
    assert report["verdict"] == "pass"


def test_one_point_below_the_lower_ci_boundary_fails(sat):
    """402/407: point 0.98772 falls under 99% before the CI even matters."""
    report = verdict_of(sat, uniform_log(407, satisfactory=402))
    assert report["aggregate"]["rate"] == 0.9877
    assert report["aggregate"]["meets_point"] is False
    assert report["verdict"] == "fail"
    assert report["verdict_reasons"] == ["point_below_target"]


def test_a_single_critical_error_blocks_the_release_regardless_of_point(sat):
    rows = uniform_log(400, satisfactory=399, critical_errors=1)
    report = verdict_of(sat, rows)
    # The satisfaction numbers alone would clear both thresholds...
    assert report["aggregate"]["rate"] == 0.9975
    assert report["aggregate"]["meets_point"] is True
    assert report["aggregate"]["meets_lower_ci"] is True
    assert report["sample"]["scored"] == 400
    # ...but one critical error fails the release.
    assert report["critical_misinformation"]["observed"] == 1
    assert report["critical_misinformation"]["within_ceiling"] is False
    assert report["verdict"] == "fail"
    assert report["verdict_reasons"] == ["critical_misinformation"]


def test_a_stratum_regression_blocks_an_unqualified_pass(sat):
    """The aggregate clears every threshold; the small stratum does not."""
    rows = uniform_log(380, role="student", intent="học phí") + [
        task(f"x{i}", role="teacher", intent="hồ sơ", completed=i < 18)
        for i in range(20)
    ]
    report = verdict_of(sat, rows)

    # Aggregate alone: 398/400, point 99.5%, lower 98.20% -> would pass.
    assert report["aggregate"]["rate"] == 0.995
    assert report["aggregate"]["meets_point"] is True
    assert report["aggregate"]["meets_lower_ci"] is True
    assert report["sample"]["scored"] >= report["sample"]["floor"]
    assert report["critical_misinformation"]["observed"] == 0

    # The 20-task stratum cannot support 99%, so no unqualified PASS.
    teacher = report["strata"]["by_role"]["teacher"]
    assert teacher["scored"] == 20
    assert teacher["rate"] == 0.9
    assert teacher["wilson_95"][1] < 0.99
    assert teacher["regressed"] is True
    assert report["verdict"] != "pass"
    assert report["verdict"] == "fail"
    assert report["verdict_reasons"] == ["stratum_regression"]
    assert report["masking_guard"]["triggered"] is True
    names = {s["stratum"] for s in report["masking_guard"]["regressed_strata"]}
    assert names == {"teacher", "hồ sơ"}


def test_small_stratum_above_target_is_not_a_regression(sat):
    """A small stratum that is perfect cannot be evidence of a regression."""
    rows = uniform_log(380, role="student", intent="học phí") + [
        task(f"x{i}", role="teacher", intent="hồ sơ") for i in range(20)
    ]
    report = verdict_of(sat, rows)
    assert report["strata"]["by_role"]["teacher"]["regressed"] is False
    assert report["masking_guard"]["triggered"] is False
    assert report["verdict"] == "pass"


# =============================================================================
# Critical misinformation: one-sided upper bound
# =============================================================================

def test_critical_misinformation_reports_one_sided_95_upper_bound(sat):
    """Plan:798 — with ~300+ zero-error observations the bound sits near 1%."""
    report = verdict_of(sat, uniform_log(400), phase="release")
    critical = report["critical_misinformation"]
    assert critical["observed"] == 0
    assert critical["adjudicated"] == 400
    assert critical["ci95_method"] == "clopper_pearson_one_sided_upper_alpha0.05"
    assert 0.0074 <= critical["one_sided_95_upper"] <= 0.0075

    at_300 = sat.critical_block(0, 300)
    assert 0.0099 <= at_300["one_sided_95_upper"] <= 0.0100
    assert at_300["within_ceiling"] is True


def test_critical_upper_bound_widens_with_an_error_and_is_absent_when_unadjudicated(sat):
    one_error = sat.critical_block(1, 400)
    zero_error = sat.critical_block(0, 400)
    assert one_error["one_sided_95_upper"] > zero_error["one_sided_95_upper"]
    assert one_error["within_ceiling"] is False

    none = sat.critical_block(0, 0)
    assert none["one_sided_95_upper"] is None
    assert none["rate"] is None
    assert none["verdict"] == "insufficient_evidence"


# =============================================================================
# The provider-dead environment must not produce a PASS
# =============================================================================

BLOCKED_ANSWER = (
    "Hệ thống đã tìm thấy tài liệu liên quan nhưng dịch vụ tạo câu trả lời "
    "đang tạm thời không khả dụng. Vui lòng thử lại sau."
)


def test_provider_dead_log_yields_insufficient_evidence_with_a_named_reason(sat):
    """The instrument cannot be fed nothing and report 99%."""
    rows = [
        {
            "id": f"q{i}",
            "question": "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?",
            "answer": BLOCKED_ANSWER,
        }
        for i in range(298)
    ]
    report = verdict_of(sat, rows, phase="release")

    assert report["verdict"] == "insufficient_evidence"
    assert report["verdict_reasons"] == ["no_usable_adjudication"]
    assert report["aggregate"]["rate"] is None
    assert report["aggregate"]["wilson_95"] is None
    assert report["sample"]["scored"] == 0
    assert report["sample"]["unadjudicated"] == 298
    assert report["generation"]["status"] == "blocked"
    assert report["generation"]["blocked"] == 298
    assert report["generation"]["ok"] == 0
    # Ticket 08's detector reports the verbatim fallback MARKER as the evidence.
    assert report["generation"]["evidence"] in BLOCKED_ANSWER
    assert report["generation"]["evidence"] == (
        "dịch vụ tạo câu trả lời đang tạm thời không khả dụng"
    )
    assert "generation blocked by the provider" in report["reason_detail"]


def test_empty_log_never_passes(sat):
    report = verdict_of(sat, [])
    assert report["verdict"] == "insufficient_evidence"
    assert report["verdict_reasons"] == ["no_records"]
    assert report["aggregate"]["rate"] is None


@pytest.mark.skipif(
    not PROVIDER_DEAD_RECORDS.exists(),
    reason="ticket-08 303q baseline records are not present",
)
def test_real_303q_artifact_is_insufficient_evidence(sat):
    """Ground truth: the frozen 303q run has generation_ok=5, so no answer was
    ever adjudicated and the release verdict cannot be a pass."""
    rows = sat.load_rows(PROVIDER_DEAD_RECORDS)
    report = verdict_of(sat, rows, phase="release")

    assert report["sample"]["records"] == 303
    assert report["verdict"] == "insufficient_evidence"
    assert report["verdict_reasons"] == ["no_usable_adjudication"]
    assert report["aggregate"]["rate"] is None
    assert report["sample"]["scored"] == 0
    assert report["generation"]["blocked"] == 298
    assert report["generation"]["status"] == "blocked"
    # The stratum table is still emitted, so the missing data is visible.
    assert report["strata"]["by_intent"]["học phí"]["scored"] == 0


# =============================================================================
# Determinism
# =============================================================================

def test_same_log_produces_byte_identical_reports(sat):
    rows = uniform_log(380, role="student", intent="học phí") + [
        task(f"x{i}", role="teacher", intent="hồ sơ", completed=i < 18)
        for i in range(20)
    ]
    first = sat.canonical_json(sat.build_report(rows, phase="release"))
    second = sat.canonical_json(sat.build_report(rows, phase="release"))
    assert first == second
    assert len(first) > 0


def test_reports_are_byte_identical_across_processes(sat, tmp_path):
    """Run the CLI twice and diff the written artifacts."""
    log = tmp_path / "log.jsonl"
    rows = uniform_log(407, satisfactory=403)
    log.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert sat.main(["--log", str(log), "--phase", "release", "--out", str(out_a), "--format", "json"]) == 0
    assert sat.main(["--log", str(log), "--phase", "release", "--out", str(out_b), "--format", "json"]) == 0

    json_a = (out_a / "task-satisfaction.release.json").read_bytes()
    json_b = (out_b / "task-satisfaction.release.json").read_bytes()
    assert json_a == json_b
    md_a = (out_a / "task-satisfaction.release.md").read_bytes()
    md_b = (out_b / "task-satisfaction.release.md").read_bytes()
    assert md_a == md_b

    parsed = json.loads(json_a.decode("utf-8"))
    assert parsed["verdict"] == "pass"
    assert parsed["content_hash"]


def test_cli_exit_code_reflects_the_verdict(sat, tmp_path, capsys):
    log = tmp_path / "dead.jsonl"
    log.write_text(
        json.dumps({"id": "q0", "question": "Học phí?", "answer": BLOCKED_ANSWER}) + "\n",
        encoding="utf-8",
    )
    assert sat.main(["--log", str(log), "--phase", "release"]) == 2
    assert "insufficient_evidence" in capsys.readouterr().out

    log2 = tmp_path / "good.jsonl"
    log2.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in uniform_log(400)) + "\n",
        encoding="utf-8",
    )
    assert sat.main(["--log", str(log2), "--phase", "release"]) == 0


def test_report_renders_the_gate_table_and_strata(sat):
    report = verdict_of(sat, uniform_log(400), phase="release")
    md = sat.render_markdown(report)
    assert "**Verdict: `pass`**" in md
    assert "| point estimate |" in md
    assert "| lower 95% CI (Wilson) |" in md
    assert "| critical misinformation |" in md
    assert "critical 1-sided 95% upper bound" in md
    assert "| by_role | student |" in md
