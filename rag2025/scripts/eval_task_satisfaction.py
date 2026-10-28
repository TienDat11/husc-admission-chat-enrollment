"""Ticket 30 — task-level satisfaction instrument (deterministic, offline).

Metric (``.scratch/chatgpt-plan-raw.md:786-803``)
-------------------------------------------------
A completed task is *satisfactory* only when ALL THREE conditions hold:

  A. the user completed the admissions task;
  B. the answer/citation was judged trustworthy;
  C. expert adjudication found NO critical misinformation.

Design (plan lines 792-799)
---------------------------
* baseline floor  : >= 200 completed, adjudicated tasks
* release floor   : >= 400 independent completed, adjudicated tasks
* target          : point estimate >= 99.0% satisfactory tasks
* interval        : two-sided 95% Wilson score interval
* gate            : point >= 99% AND lower 95% CI >= 97.5%
* critical errors : 0 observed, plus the one-sided 95% upper bound
* below the floor : ``insufficient_evidence`` -- NEVER ``pass``

Stratification (plan line 800)
------------------------------
Every rate is reported separately by role (student/teacher) and by the 7
admissions intents named in ``docs/chatgpt-ux-admin-requirements.md:10`` — the
list the plan's "7-8 intent tuyen sinh chinh" refers to. A stratum that is
statistically below target blocks an unqualified pass, so a large stratum can
never mask a small one's regression.

Why this instrument cannot pass on empty input
----------------------------------------------
A task that does not carry all three rubric fields is *unadjudicated* and is
excluded from the denominator -- never counted as satisfied. With no expert
adjudication the scored sample is 0 and the verdict is ``insufficient_evidence``
with an explicit reason. There is no code path that turns missing data into a
99% rate.

Reuse (ticket 08, ``eval_per_intent_report``): the Wilson interval, the
generation-block detector, the provider-error sidecar loader and canonical JSON
-- one definition, not a second style.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent  # rag2025/
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_intent import ATTRIBUTE_PATTERNS  # noqa: E402  (single lexicon source)
from eval_per_intent_report import (  # noqa: E402  (ticket 08 conventions)
    _r,
    canonical_json,
    generation_block_evidence,
    load_provider_errors,
    run_meta_path,
    wilson_interval,
)

TICKET = "30"
REPORT_SCHEMA = "task-satisfaction-report"
REPORT_SCHEMA_VERSION = 1

PLAN_REFERENCE = {
    "file": ".scratch/chatgpt-plan-raw.md",
    "lines": "786-803",
    "metric_lines": "786-790",
    "design_lines": "792-799",
    "stratification_line": "800",
}

# --- Design floors + gate thresholds (plan:794-798) --------------------------
BASELINE_FLOOR = 200
RELEASE_FLOOR = 400
PHASE_FLOORS: Dict[str, int] = {"baseline": BASELINE_FLOOR, "release": RELEASE_FLOOR}

TARGET_POINT = 0.99
TARGET_LOWER_CI = 0.975
MAX_CRITICAL_ERRORS = 0
ALPHA = 0.05  # one-sided upper bound for the critical-misinformation rate
Z_95 = 1.96

CI_METHOD = "wilson_score_interval_z1.96"
CRITICAL_CI_METHOD = "clopper_pearson_one_sided_upper_alpha0.05"

# --- Stratification keys -----------------------------------------------------
ROLES: Tuple[str, ...] = ("student", "teacher")
UNKNOWN_ROLE = "unknown"

# Source of the intent list. The plan (line 794) says "7-8 intent tuyen sinh
# chinh" without enumerating them; the UX requirements doc it inherits names
# exactly these seven, in this order.
INTENT_SOURCE = {
    "file": "docs/chatgpt-ux-admin-requirements.md",
    "line": 10,
    "text": (
        "One prominent search/chat entry with suggested intents: ngành, "
        "phương thức, chỉ tiêu, học phí, mốc thời gian, hồ sơ, điểm chuẩn."
    ),
}
ADMISSION_INTENTS: Tuple[str, ...] = (
    "ngành",
    "phương thức",
    "chỉ tiêu",
    "học phí",
    "mốc thời gian",
    "hồ sơ",
    "điểm chuẩn",
)
OTHER_INTENT = "khác"
INTENT_STRATA: Tuple[str, ...] = ADMISSION_INTENTS + (OTHER_INTENT,)

# Classification priority: most specific attribute first; the bare "ngành"
# program pattern is LAST because "ngành" is usually the entity a question is
# *about*, not the information need. Attribute keys reference the shared
# ATTRIBUTE_PATTERNS lexicon in ``eval_intent``.
INTENT_MATCH_PRIORITY: Tuple[str, ...] = (
    "học phí",
    "điểm chuẩn",
    "chỉ tiêu",
    "phương thức",
    "mốc thời gian",
    "hồ sơ",
    "ngành",
)
INTENT_ATTRIBUTE_KEYS: Dict[str, Tuple[str, ...]] = {
    "học phí": ("hoc_phi", "le_phi"),
    "điểm chuẩn": ("diem_chuan",),
    "chỉ tiêu": ("chi_tieu",),
    "phương thức": ("phuong_thuc", "to_hop", "hoc_ba", "uu_tien"),
    "mốc thời gian": ("thoi_gian",),
    "hồ sơ": ("ho_so", "dang_ky"),
    "ngành": ("ma_truong",),
}
_PROGRAM_PATTERN = re.compile(
    r"ngành|chuyên\s+ngành|chương\s+trình\s+đào\s+tạo|đào\s+tạo", re.IGNORECASE
)

# --- Questionnaire + rubric --------------------------------------------------
# The three questions HUSC asks of every completed task. ``pass_when`` is the
# value the field must take for the condition to hold.
QUESTIONNAIRE: Dict[str, Any] = {
    "instrument": "HUSC admissions task-satisfaction questionnaire",
    "administered_to": "the person who completed the task (student/teacher)",
    "adjudication": (
        "condition C is adjudicated by an admissions-domain expert, not the user"
    ),
    "fields": {
        "task_id": "stable id of the completed task",
        "session_id": "session the task belongs to (independence bookkeeping)",
        "role": "student | teacher",
        "admission_intent": (
            "optional explicit intent; one of: " + ", ".join(ADMISSION_INTENTS)
        ),
        "question": "the question asked (used to derive the intent when absent)",
        "task_completed": "condition A: did the user complete the admissions task?",
        "citation_trusted": "condition B: is the answer/citation trustworthy?",
        "critical_misinformation": (
            "condition C: did expert adjudication find critical misinformation? "
            "(true = an error was found)"
        ),
        "repeated_session": "true marks a repeated spam session (excluded sample)",
    },
}

RUBRIC: Dict[str, Any] = {
    "satisfactory": (
        "task_completed AND citation_trusted AND NOT critical_misinformation"
    ),
    "conditions": (
        {
            "id": "a_task_completed",
            "field": "task_completed",
            "pass_when": True,
            "plan_line": 788,
            "prompt": "Người dùng có hoàn thành được task tuyển sinh không?",
        },
        {
            "id": "b_citation_trusted",
            "field": "citation_trusted",
            "pass_when": True,
            "plan_line": 789,
            "prompt": "Đánh giá answer/citation có đáng tin không?",
        },
        {
            "id": "c_no_critical_misinformation",
            "field": "critical_misinformation",
            "pass_when": False,
            "plan_line": 790,
            "prompt": "Expert adjudication có tìm thấy critical misinformation không?",
        },
    ),
    "unadjudicated": (
        "a task missing ANY of the three fields is excluded from the denominator; "
        "it is never counted as satisfied"
    ),
}

_ATTR_RES: Dict[str, re.Pattern] = {
    key: re.compile(pattern, re.IGNORECASE) for key, pattern in ATTRIBUTE_PATTERNS
}

_TRUE_TOKENS = {"true", "yes", "y", "1", "có", "co", "đồng ý", "dong y"}
_FALSE_TOKENS = {"false", "no", "n", "0", "không", "khong", "không có", "khong co"}


# =============================================================================
# Input helpers
# =============================================================================

def load_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load a task log: a JSON array or newline-delimited JSON objects."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def file_ref(path: Optional[pathlib.Path]) -> Optional[Dict[str, Any]]:
    """Stable identity of the input (name + sha256), never an absolute path."""
    if path is None or not path.exists():
        return None
    return {
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def as_bool_or_none(value: Any) -> Optional[bool]:
    """Coerce a questionnaire answer to bool, or None when absent/ambiguous."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    return None


def normalize_role(value: Any) -> str:
    if not isinstance(value, str):
        return UNKNOWN_ROLE
    token = value.strip().lower()
    return token if token in ROLES else UNKNOWN_ROLE


def normalize_intent_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if token in ADMISSION_INTENTS else None


def classify_admission_intent(
    question: str, explicit: Any = None
) -> Tuple[str, str]:
    """Return ``(intent, source)`` for one task.

    ``source`` is ``"explicit_label"`` when the log named a canonical intent,
    else ``"derived_rule"``. A router-vocabulary ``intent`` field
    (``general``/``so_sanh``/``liet_ke``, as ticket-08 records carry) is NOT a
    canonical intent and falls through to the derived rule.
    """
    label = normalize_intent_label(explicit)
    if label is not None:
        return label, "explicit_label"

    text = question or ""
    for intent in INTENT_MATCH_PRIORITY:
        for key in INTENT_ATTRIBUTE_KEYS[intent]:
            pattern = _ATTR_RES.get(key)
            if pattern is not None and pattern.search(text):
                return intent, "derived_rule"
    if _PROGRAM_PATTERN.search(text):
        return "ngành", "derived_rule"
    return OTHER_INTENT, "derived_rule"


def is_questionnaire_row(row: Dict[str, Any]) -> bool:
    """A row is questionnaire data iff it carries any of the three rubric fields."""
    return any(
        field in row
        for field in ("task_completed", "citation_trusted", "critical_misinformation")
    )


def build_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one log row into the scored-task shape (adjudication may be None)."""
    question = row.get("question") or row.get("query") or ""
    explicit = row.get("admission_intent")
    if explicit is None and "task_completed" in row:
        # Only questionnaire rows may carry an admissions intent under `intent`;
        # ticket-08 records use that key for the router's vocabulary.
        explicit = row.get("intent")
    intent, intent_source = classify_admission_intent(question, explicit)

    a = as_bool_or_none(row.get("task_completed"))
    b = as_bool_or_none(row.get("citation_trusted"))
    c = as_bool_or_none(row.get("critical_misinformation"))

    answer = row.get("answer")
    block_evidence = (
        generation_block_evidence(str(answer))
        if isinstance(answer, str) and answer
        else None
    )
    blocked = None if block_evidence is None else block_evidence[0] is not None

    satisfactory: Optional[bool]
    if a is None or b is None or c is None:
        satisfactory = None
    else:
        satisfactory = bool(a) and bool(b) and (c is False)

    repeated = bool(row.get("repeated_session")) if row.get("repeated_session") is not None else False

    return {
        "task_id": row.get("task_id") or row.get("id"),
        "role": normalize_role(row.get("role") or row.get("user_role") or row.get("persona")),
        "intent": intent,
        "intent_source": intent_source,
        "questionnaire": is_questionnaire_row(row),
        "task_completed": a,
        "citation_trusted": b,
        "critical_misinformation": c,
        "satisfactory": satisfactory,
        "generation_blocked": blocked,
        "generation_evidence": block_evidence[0] if block_evidence else None,
        "repeated_session": repeated,
    }


# =============================================================================
# Statistics
# =============================================================================

def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p); deterministic, log-gamma terms."""
    if n <= 0:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(0, min(k, n) + 1):
        total += math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
    return min(1.0, total)


def clopper_pearson_upper(
    successes: int, n: int, alpha: float = ALPHA, iterations: int = 200
) -> Optional[float]:
    """One-sided ``1 - alpha`` Clopper-Pearson upper bound for a binomial rate.

    With zero observed errors and n >= 300 this is ~1%, which is the ceiling the
    plan (line 798) cites for the 400-task release sample.
    """
    if n <= 0:
        return None
    if successes >= n:
        return 1.0
    if successes == 0:
        return _r(1.0 - alpha ** (1.0 / n))
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if binom_cdf(successes, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return _r((lo + hi) / 2.0)


def meets_point(rate: Optional[float], target: float = TARGET_POINT) -> bool:
    """Boundary-true: a rate of exactly ``target`` meets the point criterion."""
    if rate is None:
        return False
    return rate >= target - 1e-12


def meets_lower_ci(lower: Optional[float], target: float = TARGET_LOWER_CI) -> bool:
    """Boundary-true: a lower bound of exactly ``target`` meets the criterion."""
    if lower is None:
        return False
    return lower >= target - 1e-12


def rate_block(satisfactory: int, scored: int) -> Dict[str, Any]:
    """Point estimate + Wilson 95% CI + the two gate comparisons."""
    if scored <= 0:
        return {
            "satisfactory": satisfactory,
            "scored": scored,
            "rate": None,
            "wilson_95": None,
            "ci95_method": CI_METHOD,
            "meets_point": False,
            "meets_lower_ci": False,
            "below_target_point": False,
            "regressed": False,
            "verdict": "insufficient_evidence",
            "note": "no scored tasks",
        }
    rate = satisfactory / scored
    interval = wilson_interval(satisfactory, scored, Z_95)
    lower = interval[0] if interval else None
    upper = interval[1] if interval else None
    # The gate reads the UNROUNDED point estimate (exact division) so display
    # rounding can never turn a sub-target rate into a pass; the interval is
    # compared exactly as ticket 08 serializes it, which the report prints.
    point_ok = meets_point(rate)
    lower_ok = meets_lower_ci(lower)
    # A stratum regresses only when its UPPER bound is below target: that is
    # 95%-one-sided evidence the true rate misses 99%, so a small noisy stratum
    # is surfaced (`below_target_point`) without being failed on noise.
    regressed = upper is not None and upper < TARGET_POINT
    if regressed:
        verdict = "fail"
    elif point_ok:
        verdict = "pass"
    else:
        verdict = "watch"
    return {
        "satisfactory": satisfactory,
        "scored": scored,
        "rate": _r(rate),
        "wilson_95": interval,
        "ci95_method": CI_METHOD,
        "meets_point": point_ok,
        "meets_lower_ci": lower_ok,
        "below_target_point": not point_ok,
        "regressed": regressed,
        "verdict": verdict,
        "note": None,
    }


def critical_block(errors: int, adjudicated: int) -> Dict[str, Any]:
    """Observed critical errors + the one-sided 95% upper bound."""
    if adjudicated <= 0:
        return {
            "observed": errors,
            "adjudicated": adjudicated,
            "rate": None,
            "one_sided_95_upper": None,
            "ci95_method": CRITICAL_CI_METHOD,
            "ceiling": MAX_CRITICAL_ERRORS,
            "within_ceiling": False,
            "verdict": "insufficient_evidence",
            "note": "no task carries an expert critical-misinformation adjudication",
        }
    return {
        "observed": errors,
        "adjudicated": adjudicated,
        "rate": _r(errors / adjudicated),
        "one_sided_95_upper": clopper_pearson_upper(errors, adjudicated),
        "ci95_method": CRITICAL_CI_METHOD,
        "ceiling": MAX_CRITICAL_ERRORS,
        "within_ceiling": errors <= MAX_CRITICAL_ERRORS,
        "verdict": "pass" if errors <= MAX_CRITICAL_ERRORS else "fail",
        "note": None,
    }


# =============================================================================
# Ingestion + stratification
# =============================================================================

def _tally(
    tasks: Sequence[Dict[str, Any]],
    key_fn,
    keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        members = [t for t in tasks if key_fn(t) == key]
        scored = [t for t in members if t["satisfactory"] is not None]
        block = rate_block(
            sum(1 for t in scored if t["satisfactory"]), len(scored)
        )
        block["tasks"] = len(members)
        out[key] = block
    return out


def ingest(
    rows: Sequence[Dict[str, Any]],
    phase: str = "release",
    provider_errors: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score a task log into the report body (no file I/O beyond the rows)."""
    floor = PHASE_FLOORS.get(phase)
    if floor is None:
        raise ValueError(f"unknown phase: {phase!r}")

    tasks = [build_task(row) for row in rows]

    repeated = [t for t in tasks if t["repeated_session"]]
    independent = [t for t in tasks if not t["repeated_session"]]
    scored = [t for t in independent if t["satisfactory"] is not None]

    unadjudicated = [t for t in independent if t["satisfactory"] is None]
    missing = {
        field: sum(1 for t in unadjudicated if t[field] is None)
        for field in ("task_completed", "citation_trusted", "critical_misinformation")
    }

    with_answer = [t for t in tasks if t["generation_blocked"] is not None]
    blocked = [t for t in with_answer if t["generation_blocked"]]
    if not with_answer:
        generation_status = "not_applicable"
    elif len(blocked) == len(with_answer):
        generation_status = "blocked"
    elif blocked:
        generation_status = "partial"
    else:
        generation_status = "ok"

    generation: Dict[str, Any] = {
        "observed": len(with_answer),
        "blocked": len(blocked),
        "ok": len(with_answer) - len(blocked),
        "status": generation_status,
        "provider_errors": list(provider_errors or []),
    }
    if blocked:
        generation["evidence"] = blocked[0].get("generation_evidence")

    sample = {
        "records": len(tasks),
        "independent": len(independent),
        "excluded_repeated_session": len(repeated),
        "scored": len(scored),
        "unadjudicated": len(unadjudicated),
        "unadjudicated_by_condition": missing,
        "floor": floor,
        "phase": phase,
        "intent_source": {
            "explicit_label": sum(
                1 for t in tasks if t["intent_source"] == "explicit_label"
            ),
            "derived_rule": sum(
                1 for t in tasks if t["intent_source"] == "derived_rule"
            ),
        },
    }

    aggregate = rate_block(
        sum(1 for t in scored if t["satisfactory"]), len(scored)
    )
    critical_adjudicated = [t for t in independent if t["critical_misinformation"] is not None]
    critical = critical_block(
        sum(1 for t in critical_adjudicated if t["critical_misinformation"]),
        len(critical_adjudicated),
    )

    strata = {
        "by_role": _tally(independent, lambda t: t["role"], ROLES + (UNKNOWN_ROLE,)),
        "by_intent": _tally(independent, lambda t: t["intent"], INTENT_STRATA),
    }
    regressed = [
        {"dimension": dim, "stratum": key, **block}
        for dim, blocks in strata.items()
        for key, block in blocks.items()
        if block.get("regressed")
    ]
    watched = [
        {"dimension": dim, "stratum": key, "rate": block["rate"], "scored": block["scored"]}
        for dim, blocks in strata.items()
        for key, block in blocks.items()
        if block.get("below_target_point") and not block.get("regressed")
    ]

    reasons: List[str] = []
    if len(tasks) == 0:
        verdict = "insufficient_evidence"
        reasons.append("no_records")
    elif len(scored) == 0:
        verdict = "insufficient_evidence"
        reasons.append("no_usable_adjudication")
    elif len(scored) < floor:
        verdict = "insufficient_evidence"
        reasons.append("below_design_floor")
    elif critical["observed"] > MAX_CRITICAL_ERRORS:
        verdict = "fail"
        reasons.append("critical_misinformation")
    elif not aggregate["meets_point"]:
        verdict = "fail"
        reasons.append("point_below_target")
    elif not aggregate["meets_lower_ci"]:
        verdict = "fail"
        reasons.append("lower_ci_below_target")
    elif regressed:
        verdict = "fail"
        reasons.append("stratum_regression")
    else:
        verdict = "pass"

    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "ticket": TICKET,
        "phase": phase,
        "plan_reference": PLAN_REFERENCE,
        "intent_source": INTENT_SOURCE,
        "admission_intents": list(ADMISSION_INTENTS),
        "thresholds": {
            "target_point": TARGET_POINT,
            "target_lower_ci": TARGET_LOWER_CI,
            "max_critical_errors": MAX_CRITICAL_ERRORS,
            "alpha": ALPHA,
            "z": Z_95,
        },
        "questionnaire": QUESTIONNAIRE,
        "rubric": RUBRIC,
        "sample": sample,
        "generation": generation,
        "aggregate": aggregate,
        "critical_misinformation": critical,
        "strata": strata,
        "masking_guard": {
            "rule": (
                "a stratum whose 95% Wilson UPPER bound is below 99% blocks an "
                "unqualified pass"
            ),
            "triggered": bool(regressed),
            "regressed_strata": regressed,
            "watched_strata": watched,
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
        "reason_detail": describe_reason(
            verdict, reasons, sample, generation, aggregate, critical, regressed
        ),
    }


def describe_reason(
    verdict: str,
    reasons: Sequence[str],
    sample: Dict[str, Any],
    generation: Dict[str, Any],
    aggregate: Dict[str, Any],
    critical: Dict[str, Any],
    regressed: Sequence[Dict[str, Any]],
) -> str:
    """One deterministic sentence naming exactly why the verdict was reached."""
    if verdict == "insufficient_evidence" and "no_records" in reasons:
        return (
            "the log contained 0 rows, so no task can be scored and no "
            "satisfaction rate exists."
        )
    if verdict == "insufficient_evidence" and "no_usable_adjudication" in reasons:
        base = (
            f"0 of {sample['records']} rows carry all three rubric fields "
            "(task_completed, citation_trusted, critical_misinformation), so 0 "
            "tasks are scored and no satisfaction rate exists."
        )
        if generation.get("blocked"):
            base += (
                f" {generation['blocked']} of {generation['observed']} rows had "
                "generation blocked by the provider, so there is no answer for an "
                "expert to adjudicate."
            )
        return base
    if verdict == "insufficient_evidence":
        return (
            f"{sample['scored']} scored tasks is below the {sample['floor']}-task "
            f"{sample['phase']} design floor; a point estimate of "
            f"{aggregate['rate']} is not trustworthy at this sample size."
        )
    if "critical_misinformation" in reasons:
        return (
            f"{critical['observed']} critical misinformation error(s) observed; the "
            "release ceiling is 0 regardless of the point estimate."
        )
    if "stratum_regression" in reasons:
        names = ", ".join(
            f"{s['dimension']}={s['stratum']} ({s['rate']})" for s in regressed
        )
        return (
            f"aggregate passes but stratum regression blocks an unqualified pass: "
            f"{names}."
        )
    if "point_below_target" in reasons:
        return (
            f"aggregate point estimate {aggregate['rate']} is below the "
            f"{TARGET_POINT} target."
        )
    if "lower_ci_below_target" in reasons:
        return (
            f"aggregate lower 95% CI {aggregate['wilson_95'][0]} is below the "
            f"{TARGET_LOWER_CI} target."
        )
    return "all gate criteria met at or above the design floor."


def build_report(
    rows: Sequence[Dict[str, Any]],
    phase: str = "release",
    source: Optional[Dict[str, Any]] = None,
    provider_errors: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Full report + content hash. Pure function of its inputs."""
    report = ingest(rows, phase=phase, provider_errors=provider_errors)
    report["source"] = source
    payload = {k: v for k, v in report.items() if k != "content_hash"}
    report["content_hash"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return report


# =============================================================================
# Rendering
# =============================================================================

def _fmt_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _fmt_ci(interval: Optional[Sequence[float]]) -> str:
    if not interval:
        return "n/a"
    return f"[{interval[0] * 100:.2f}%, {interval[1] * 100:.2f}%]"


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"# Task-satisfaction report ({report['phase']})",
        "",
        f"**Verdict: `{report['verdict']}`** — {report['reason_detail']}",
        "",
        f"- ticket: {report['ticket']}  ·  schema: {report['schema']} v{report['schema_version']}",
        f"- metric: {report['plan_reference']['file']}:{report['plan_reference']['lines']}",
        f"- intents: {report['intent_source']['file']}:{report['intent_source']['line']}",
        f"- content_hash: `{report['content_hash']}`",
        "",
        "## Sample",
        "",
        f"- records: {report['sample']['records']}"
        f"  ·  independent: {report['sample']['independent']}"
        f"  ·  excluded (repeated session): {report['sample']['excluded_repeated_session']}",
        f"- scored (all 3 rubric fields): {report['sample']['scored']}"
        f"  ·  floor: {report['sample']['floor']}",
        f"- unadjudicated: {report['sample']['unadjudicated']}"
        f"  ·  by condition: {report['sample']['unadjudicated_by_condition']}",
        f"- generation: {report['generation']['status']}"
        f" (blocked {report['generation']['blocked']}/{report['generation']['observed']})",
        "",
        "## Gate",
        "",
        "| check | value | target | ok |",
        "|---|---|---|---|",
        f"| point estimate | {_fmt_rate(report['aggregate']['rate'])} "
        f"| >= {report['thresholds']['target_point']:.1%} "
        f"| {report['aggregate']['meets_point']} |",
        f"| lower 95% CI (Wilson) | "
        f"{_fmt_rate(report['aggregate']['wilson_95'][0]) if report['aggregate']['wilson_95'] else 'n/a'} "
        f"| >= {report['thresholds']['target_lower_ci']:.1%} "
        f"| {report['aggregate']['meets_lower_ci']} |",
        f"| critical misinformation | {report['critical_misinformation']['observed']} "
        f"| = {report['thresholds']['max_critical_errors']} "
        f"| {report['critical_misinformation']['within_ceiling']} |",
        f"| critical 1-sided 95% upper bound | "
        f"{_fmt_rate(report['critical_misinformation']['one_sided_95_upper'])} | report-only | — |",
        f"| sample floor ({report['phase']}) | {report['sample']['scored']} "
        f"| >= {report['sample']['floor']} | "
        f"{report['sample']['scored'] >= report['sample']['floor']} |",
        "",
        "## Strata",
        "",
        "| dimension | stratum | scored | satisfactory | rate | wilson 95% | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for dimension, blocks in report["strata"].items():
        for key, block in blocks.items():
            lines.append(
                f"| {dimension} | {key} | {block['scored']} | {block['satisfactory']} "
                f"| {_fmt_rate(block['rate'])} | {_fmt_ci(block['wilson_95'])} "
                f"| {block['verdict']} |"
            )
    if report["masking_guard"]["triggered"]:
        lines += [
            "",
            "**Masking guard triggered** — the aggregate alone would not be a "
            "trustworthy pass:",
            "",
        ]
        for stratum in report["masking_guard"]["regressed_strata"]:
            lines.append(
                f"- {stratum['dimension']}={stratum['stratum']}: "
                f"{_fmt_rate(stratum['rate'])} ({stratum['scored']} tasks, "
                f"upper bound {_fmt_rate(stratum['wilson_95'][1])})"
            )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ticket 30 — task-level satisfaction instrument (offline)."
    )
    parser.add_argument("--log", required=True, help="task log (JSON array or JSONL)")
    parser.add_argument(
        "--phase",
        choices=sorted(PHASE_FLOORS),
        default="release",
        help="design floor: baseline=200, release=400",
    )
    parser.add_argument("--out", help="directory for the .json/.md report")
    parser.add_argument(
        "--format", choices=("md", "json"), default="md", help="stdout format"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    path = pathlib.Path(args.log)
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        return 2

    rows = load_rows(path)
    sidecar = run_meta_path(path)
    report = build_report(
        rows,
        phase=args.phase,
        source=file_ref(path),
        provider_errors=load_provider_errors(sidecar if sidecar.exists() else None),
    )

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"task-satisfaction.{args.phase}"
        (out / f"{stem}.json").write_text(canonical_json(report), encoding="utf-8")
        (out / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")

    if args.format == "json":
        print(canonical_json(report), end="")
    else:
        print(render_markdown(report), end="")

    return {"pass": 0, "fail": 1, "insufficient_evidence": 2}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
