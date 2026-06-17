"""Slice S6 — Deterministic hallucination-type classifier (Type-1/4/5/6).

DETERMINISTIC detectors only — regex + numeric matching + major-code validator.
NO LLM in the tested path. Reports (does not enforce) the four types.

Type definitions (from `temporal_reingest_plan_v2.md` acceptance gate):
  * Type-1 (year-mix): answer asserts a 2025 fact as if it's the 2026 answer.
  * Type-4 (fabricated major code): answer mentions a code not in WHITELIST_2026.
  * Type-5 (fabricated numbers): number in answer not in any context / GT fact.
  * Type-6 (anti-fallback fabrication): specifics asserted with no current-year
    context supporting them.

Public API:
  * detect_type4(answer) -> list[str]
  * detect_type5(answer, contexts, critical_facts) -> list[str]
  * detect_type1(answer, contexts_meta, question, current_year=2026) -> bool
  * detect_type6(answer, retrieved_chunks, current_year=2026) -> bool
  * numeric_normalize(s) -> str  (helper exposed for downstream)
  * is_valid_2026_major(code) -> bool  (re-export from major_code_validator)
  * classify_record(record, gt_entry, current_year=2026) -> dict
  * build_confusion_matrix(classifications, gt_by_id) -> dict
  * main() — argparse: --run-id, --records, --gt, --out-dir
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------- Path bootstrapping (so we can import the validator) ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAG2025_SRC = RAG2025_ROOT / "src"
for _p in (str(RAG2025_ROOT), str(RAG2025_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the major-code validator. This is a PURE module (no I/O on import),
# safe to call from any environment.
from services.major_code_validator import (  # noqa: E402  (sys.path tweak above)
    is_valid_2026_major,
    validate_answer_majors,
    WHITELIST_2026,
)


# Re-export the public predicate the tests/spec want reachable.
__all__ = [
    "numeric_normalize",
    "is_valid_2026_major",
    "detect_type4",
    "detect_type5",
    "detect_type1",
    "detect_type6",
    "classify_record",
    "build_confusion_matrix",
    "main",
]


# ---------- Constants ----------

# Default output location: rag2025/results/eval_harness/judge_halltype/<run_id>.json
RESULTS_ROOT = RAG2025_ROOT / "results" / "eval_harness" / "judge_halltype"

# Acceptance gate (per temporal_reingest_plan_v2.md).
GATE: Dict[str, int] = {
    "type1_max": 3,
    "type4_max": 0,
    "type5_max": 3,
    "type6_max": 0,
}

# Numeric patterns in the answer.
# Money: 2-3 digits + (. or ,) + 3 digits, optional second group. e.g. 600.000, 18.000.000
_NUMERIC_RX = re.compile(r"\b\d{2,3}[\.,]\d{3}(?:[\.,]\d{3})?\b")
# Standalone chỉ-tiêu/điểm numbers like "18.50", "24,75" — but to keep Type-5
# focused on real candidates we also accept 2-digit "decimal-group" numerics.
_DECIMAL_RX = re.compile(r"\b\d{1,3}[\.,]\d{1,3}\b")
# "X triệu" money phrasing.
_TRIEU_RX = re.compile(r"\b\d{1,3}\s*tr(?:iệu)?\b", re.IGNORECASE)

# Year token (used in Type-1).
_YEAR_RX = re.compile(r"\b(?:năm\s*)?(20\d{2})\b", re.IGNORECASE)

# Safe-fallback markers (mirror the spirit of test_llm_generator_no_fabrication.py
# but deterministic — no async, no LLM).
_FALLBACK_MARKERS = (
    "chưa có thông tin",
    "chưa có thông tin chính thức",
    "chưa có",
    "chưa công bố",
    "chưa được công bố",
    "không tìm thấy thông tin",
    "không tìm thấy",
    "ngoài phạm vi",
)


# ---------- Helpers ----------

def numeric_normalize(s: Any) -> str:
    """Strip separators (., space) from a numeric string for comparison.

    Examples:
      numeric_normalize("600.000")  -> "600000"
      numeric_normalize("600,000")  -> "600000"
      numeric_normalize("600 000")  -> "600000"
      numeric_normalize("999.999")  -> "999999"
      numeric_normalize(600000)     -> "600000"
    """
    if s is None:
        return ""
    if isinstance(s, (int, float)):
        # int/float → str, then strip ".0" float artefact.
        s = str(s)
        if s.endswith(".0"):
            s = s[:-2]
    s = str(s)
    return re.sub(r"[\s\.,]", "", s)


def _coerce_chunk_year(c: Any) -> Optional[int]:
    """Return the int data_year from a chunk, robust to str/int/None."""
    if c is None:
        return None
    if isinstance(c, dict):
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else c
        y = meta.get("data_year") if isinstance(meta, dict) else None
    else:
        meta = getattr(c, "metadata", None)
        y = getattr(meta, "data_year", None) if meta is not None else None
    if y is None:
        return None
    try:
        return int(y)
    except (TypeError, ValueError):
        return None


def _chunk_text(c: Any) -> str:
    """Return the chunk's text field (str)."""
    if isinstance(c, dict):
        t = c.get("text", "")
    else:
        t = getattr(c, "text", "")
    return t if isinstance(t, str) else ""


def _is_fallback_answer(answer: str) -> bool:
    """True if the answer reads as a graceful fallback / abstain / out-of-scope."""
    a = (answer or "").lower()
    if not a.strip():
        return True
    return any(m in a for m in _FALLBACK_MARKERS)


def _extract_answer_numerics(answer: str) -> List[str]:
    """Return all candidate numeric tokens (raw + normalized) in the answer.

    The list preserves first-seen order and is de-duplicated by normalized form.
    """
    if not isinstance(answer, str) or not answer:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for rx in (_NUMERIC_RX, _DECIMAL_RX, _TRIEU_RX):
        for m in rx.finditer(answer):
            raw = m.group(0)
            norm = numeric_normalize(raw)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(raw)
    return out


# =============================================================================
# Detectors
# =============================================================================

def detect_type4(answer: str) -> List[str]:
    """Type-4: list of major codes in `answer` NOT in WHITELIST_2026.

    Delegates to `validate_answer_majors` from major_code_validator.py — the
    regex there already enforces the 7-digit + optional 2-letter shape.
    """
    if not isinstance(answer, str) or not answer:
        return []
    return list(validate_answer_majors(answer))


def detect_type5(
    answer: str,
    contexts: Iterable[str],
    critical_facts: Iterable[Dict[str, Any]] | None,
) -> List[str]:
    """Type-5: list of answer numeric tokens not supported by context OR GT.

    A number is "supported" if its numeric-normalized form appears as a
    numeric-normalized substring of any context text OR any GT critical_fact
    value. Tokens that are not numeric-stable (e.g. empty after normalize,
    pure year, or with no digit pair) are skipped.
    """
    if not isinstance(answer, str) or not answer:
        return []
    ctx_list = [c for c in (contexts or []) if isinstance(c, str)]
    facts_list = list(critical_facts or [])

    # Build the bag of supported normalized numerics.
    supported: set[str] = set()
    for c in ctx_list:
        # Take every numeric-looking substring in the context and normalize.
        for rx in (_NUMERIC_RX, _DECIMAL_RX, _TRIEU_RX):
            for m in rx.finditer(c):
                supported.add(numeric_normalize(m.group(0)))
        # Also store the entire context's normalized form to handle cases where
        # the answer embeds a smaller chunk (e.g. "600000" inside "6000000").
        supported.add(numeric_normalize(c))
    for f in facts_list:
        if not isinstance(f, dict):
            continue
        v = f.get("value")
        if v is None:
            continue
        supported.add(numeric_normalize(v))

    # Walk the answer's numerics. Return BOTH the raw token and the normalized
    # form so downstream consumers can match either way (e.g. "999.999" or
    # "999999").
    unsupported: List[str] = []
    seen_norm: set[str] = set()
    for token in _extract_answer_numerics(answer):
        norm = numeric_normalize(token)
        if not norm or len(norm) < 2:
            continue
        # Skip pure 4-digit years (e.g. "2026", "2025") — those are not
        # fabrication candidates for Type-5 (Type-1 handles year-mix).
        if len(norm) == 4 and norm.isdigit() and norm.startswith("20"):
            continue
        # Check membership in supported set OR substring match (so 999.999 is
        # detected even if contexts store 9999990 etc.).
        supported_match = (
            norm in supported
            or any(norm in s for s in supported if s)
        )
        if not supported_match and norm not in seen_norm:
            seen_norm.add(norm)
            unsupported.append(token)
            unsupported.append(norm)
    return unsupported


def detect_type1(
    answer: str,
    contexts_meta: Iterable[Dict[str, Any]],
    question: str,
    current_year: int = 2026,
) -> bool:
    """Type-1: answer asserts a previous-year fact as if it's the current answer.

    Heuristic — true iff ALL of:
      1. The answer mentions a prior-year token (e.g. "2025") AND that year
         appears alongside an assertive numeric in the answer.
      2. The question implies a current-year ask (e.g. "năm nay", "2026", or
         no explicit year at all — we treat the absence of an explicit year
         in the question as "current-year implied").
      3. At least one retrieved context has data_year == prior-year.
      4. The answer does NOT contain a safe disambiguation that clearly tags
         the prior-year number as historical (e.g. "năm 2025 là X, năm nay
         chưa công bố").
    """
    if not isinstance(answer, str) or not answer:
        return False
    answer_lower = answer.lower()
    question_lower = (question or "").lower()

    # Safe-disambiguation check: explicit "năm 2025 ... năm nay chưa ..." or
    # "(năm trước)" tag attached to the year mention.
    safe_disambig = (
        "năm trước" in answer_lower
        or "(năm trước)" in answer_lower
        or "năm 2025" in answer_lower and (
            "năm nay chưa" in answer_lower
            or "chưa công bố" in answer_lower
            or "chưa có thông tin" in answer_lower
            or "chưa được công bố" in answer_lower
        )
    )
    if safe_disambig:
        return False

    # Check question implication: if question explicitly mentions current_year,
    # it implies "current". If it mentions only a PRIOR year, treat as a
    # historical ask → not a year-mix.
    q_years = {int(y) for y in _YEAR_RX.findall(question_lower)}
    if q_years and current_year not in q_years and all(y < current_year for y in q_years):
        return False

    # The answer must assert a non-year numeric (the year-mix "stake").
    has_numeric_non_year = False
    for tok in _extract_answer_numerics(answer):
        norm = numeric_normalize(tok)
        if norm and not (len(norm) == 4 and norm.isdigit() and norm.startswith("20")):
            has_numeric_non_year = True
            break
    if not has_numeric_non_year:
        return False

    # At least one context's data_year is a prior year (data_year < current_year).
    ctx_years: set[int] = set()
    for c in contexts_meta or []:
        y = _coerce_chunk_year(c)
        if y is not None:
            ctx_years.add(int(y))
    prior_ctx_years = {y for y in ctx_years if y < current_year}
    if not prior_ctx_years:
        return False

    # Year-mix fires when EITHER:
    #   (a) the answer explicitly mentions a prior-year token AND the cited
    #       context for that number has a matching data_year, OR
    #   (b) the answer asserts a number that only appears in prior-year
    #       contexts (no current-year context carries it), while the question
    #       implies a current-year ask — the LLM is silently presenting a
    #       prior-year fact as the current answer.
    prior_years_in_answer = {
        int(y) for y in _YEAR_RX.findall(answer_lower)
        if y.isdigit() and int(y) < current_year
    }
    if prior_years_in_answer and (prior_years_in_answer & prior_ctx_years):
        return True

    # Case (b): check whether any asserted numeric in the answer is supported
    # ONLY by a prior-year context (or not supported at all in any context).
    answer_norms = {
        numeric_normalize(t)
        for t in _extract_answer_numerics(answer)
    }
    current_year_norms: set[str] = set()
    prior_year_norms: set[str] = set()
    for c in contexts_meta or []:
        y = _coerce_chunk_year(c)
        if y is None or y == current_year:
            continue
        text = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
        if not isinstance(text, str):
            continue
        for rx in (_NUMERIC_RX, _DECIMAL_RX, _TRIEU_RX):
            for m in rx.finditer(text):
                n = numeric_normalize(m.group(0))
                if y < current_year:
                    prior_year_norms.add(n)
                else:
                    current_year_norms.add(n)
    # If any asserted number is in prior-year contexts AND NOT in any
    # current-year context, that's a year-mix (current_year_norms is empty
    # iff no current-year context exists or none carries the number).
    for n in answer_norms:
        if n in prior_year_norms and n not in current_year_norms:
            return True

    return False


def detect_type6(
    answer: str,
    retrieved_chunks: Iterable[Dict[str, Any]],
    current_year: int = 2026,
) -> bool:
    """Type-6: specifics asserted while no current-year context supports them.

    True iff ALL of:
      1. There is no retrieved chunk whose data_year == current_year
         (empty list, OR every chunk has data_year != current_year).
      2. The answer contains assertive numerics (money / điểm / chỉ-tiêu).
      3. The answer does NOT use a graceful-fallback wording.
    """
    if not isinstance(answer, str) or not answer:
        return False

    has_current_year_ctx = False
    has_any_ctx = False
    for c in retrieved_chunks or []:
        has_any_ctx = True
        y = _coerce_chunk_year(c)
        if y is not None and int(y) == int(current_year):
            has_current_year_ctx = True
            break
    if has_current_year_ctx:
        return False

    if _is_fallback_answer(answer):
        return False

    # Need an assertive numeric (or a major code) in the answer.
    has_numeric = False
    for tok in _extract_answer_numerics(answer):
        norm = numeric_normalize(tok)
        if norm and not (len(norm) == 4 and norm.isdigit() and norm.startswith("20")):
            has_numeric = True
            break
    if not has_numeric:
        # Major code also counts as a specific fact.
        has_numeric = bool(_CODE_IN_TEXT_RX.search(answer))
    if not has_numeric:
        return False

    return True


# Major-code regex (mirror of major_code_validator._CODE_IN_TEXT_RX, inlined
# so this module remains import-safe even if validator's privates move).
_CODE_IN_TEXT_RX = re.compile(r"\b7\d{6}(?:[A-Z]{2})?\b")


# =============================================================================
# Per-record classifier
# =============================================================================

def _coerce_record(rec: Any) -> Dict[str, Any]:
    """Accept either an EvalRecord (Pydantic) or a dict-like record."""
    if hasattr(rec, "model_dump"):
        return rec.model_dump()
    if isinstance(rec, dict):
        return rec
    raise TypeError(f"unsupported record type: {type(rec).__name__}")


def _gt_payload(gt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the GT-derived fields for a single record."""
    if not gt:
        return {"critical_facts": [], "expected_behavior": None}
    return {
        "critical_facts": list(gt.get("critical_facts") or []),
        "expected_behavior": gt.get("expected_behavior"),
    }


def classify_record(
    record: Any,
    gt_entry: Optional[Dict[str, Any]],
    current_year: int = 2026,
) -> Dict[str, Any]:
    """Classify one record into {id, types, type4_codes, type5_values}."""
    d = _coerce_record(record)
    rid = str(d.get("id", ""))
    answer = d.get("answer", "") or ""
    question = d.get("question", "") or ""
    chunks = d.get("retrieved_chunks") or []
    gt = _gt_payload(gt_entry)

    # Build the inputs each detector needs.
    contexts_text: List[str] = [_chunk_text(c) for c in chunks if _chunk_text(c)]
    contexts_meta: List[Dict[str, Any]] = [
        c if isinstance(c, dict) else {
            "text": getattr(c, "text", ""),
            "data_year": _coerce_chunk_year(c),
        }
        for c in chunks
    ]

    type4_codes = detect_type4(answer)
    type5_values = detect_type5(answer, contexts_text, gt["critical_facts"])
    type1 = detect_type1(answer, contexts_meta, question, current_year=current_year)
    type6 = detect_type6(answer, chunks, current_year=current_year)

    return {
        "id": rid,
        "answer_text": answer,
        "types": {
            "type1": bool(type1),
            "type4": bool(type4_codes),
            "type5": bool(type5_values),
            "type6": bool(type6),
        },
        "type4_codes": list(type4_codes),
        "type5_values": list(type5_values),
        "expected_behavior": gt.get("expected_behavior"),
    }


# =============================================================================
# Confusion matrix
# =============================================================================

# Observed categories (cols). Kept as constants so the matrix stays ordered.
_COLS: List[str] = [
    "answered_clean",
    "answered_with_hallucination",
    "abstained",
    "refused",
]
_ROWS: List[str] = ["answer", "abstain", "refuse"]


def _infer_observed(cls: Dict[str, Any]) -> str:
    """Map a classification → observed column.

    Priority:
      1. If the answer is empty / pure fallback wording AND it does NOT carry
         a hallucination → "abstained".
      2. If any Type-X fired → "answered_with_hallucination".
      3. If expected_behavior == "refuse" OR the answer is a refusal pattern
         (e.g. "Tôi không thể tư vấn...") → "refused".
      4. Otherwise → "answered_clean".
    """
    answer_text = (cls.get("answer_text") or "").lower()
    types = cls.get("types", {}) or {}
    any_fired = any(types.get(k) for k in ("type1", "type4", "type5", "type6"))

    if any_fired:
        return "answered_with_hallucination"

    if not answer_text.strip():
        return "abstained"
    if _is_fallback_answer(answer_text):
        return "abstained"
    if cls.get("expected_behavior") == "refuse":
        return "refused"
    # Plain refusal wording (no numeric, no code, declarative "không thể").
    refusal_markers = ("tôi không thể", "không thể tư vấn", "xin lỗi, tôi không")
    if any(m in answer_text for m in refusal_markers):
        return "refused"
    return "answered_clean"


def build_confusion_matrix(
    classifications: Sequence[Dict[str, Any]],
    gt_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a confusion matrix + per-type counts vs the gate.

    Input `classifications` items should at minimum contain:
      {id, types:{type1,type4,type5,type6}, type4_codes, type5_values,
       expected_behavior, answer_text?}

    If `answer_text` is missing, it is looked up from gt_by_id (if present) —
    otherwise the row falls back to "answered_clean" for non-empty `expected`.
    """
    grid: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {r: {c: 0 for c in _COLS} for r in _ROWS}
    per_type: Dict[str, int] = {"type1": 0, "type4": 0, "type5": 0, "type6": 0}

    # Build the enriched list with answer_text when missing.
    enriched: List[Dict[str, Any]] = []
    for c in classifications:
        item = dict(c)
        if "answer_text" not in item or item.get("answer_text") is None:
            gt = gt_by_id.get(item.get("id")) if gt_by_id else None
            if isinstance(gt, dict):
                item["answer_text"] = gt.get("answer") or gt.get("reference_answer") or ""
            else:
                item["answer_text"] = ""
        enriched.append(item)

    for c in enriched:
        # Resolve row (expected_behavior) — default to "answer" if unspecified.
        row = c.get("expected_behavior") or "answer"
        if row not in _ROWS:
            row = "answer"
        col = _infer_observed(c)
        if col not in _COLS:
            col = "answered_clean"
        counts[row][col] += 1
        for k in per_type:
            if c.get("types", {}).get(k):
                per_type[k] += 1

    for r in _ROWS:
        grid.append({"row": r, "counts": dict(counts[r])})

    rows_total = sum(sum(v.values()) for v in counts.values())
    cols_total = rows_total  # by construction.

    # Gate evaluation (counts vs acceptance gate).
    gate_eval: Dict[str, Dict[str, Any]] = {}
    for k, cap in GATE.items():
        actual = per_type.get(k.replace("_max", ""), 0)
        gate_eval[k] = {
            "actual": actual,
            "max": cap,
            "pass": actual <= cap,
        }

    return {
        "rows": list(_ROWS),
        "cols": list(_COLS),
        "grid": grid,
        "totals": {"rows_total": rows_total, "cols_total": cols_total},
        "per_type_counts": per_type,
        "gate": gate_eval,
    }


# =============================================================================
# I/O helpers + main
# =============================================================================

def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_gt_by_id(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Read a GT JSON (array of entries) into {id: entry}."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"GT file must be a JSON array, got {type(data).__name__}")
    return {e["id"]: e for e in data if isinstance(e, dict) and "id" in e}


def classify_records(
    records: Sequence[Any],
    gt_by_id: Dict[str, Dict[str, Any]],
    current_year: int = 2026,
) -> List[Dict[str, Any]]:
    """Classify a sequence of records + return enriched list (with answer_text)."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        d = _coerce_record(rec)
        rid = str(d.get("id", ""))
        gt = gt_by_id.get(rid) if gt_by_id else None
        cls = classify_record(rec, gt, current_year=current_year)
        cls["answer_text"] = d.get("answer", "") or ""
        out.append(cls)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic hallucination-type classifier (S6)."
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument(
        "--records",
        required=True,
        type=pathlib.Path,
        help="Path to S2 EvalRecord JSONL.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        type=pathlib.Path,
        help="Path to GT JSON (array of {id, ...}).",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Override output dir (default: rag2025/results/eval_harness/judge_halltype/<run_id>/).",
    )
    parser.add_argument(
        "--current-year",
        type=int,
        default=2026,
        help="Current admission year (default 2026).",
    )
    args = parser.parse_args(argv)

    records = _read_jsonl(args.records)
    gt_by_id = _read_gt_by_id(args.gt)
    classifications = classify_records(
        records, gt_by_id, current_year=args.current_year
    )
    matrix = build_confusion_matrix(classifications, gt_by_id)

    out_dir = args.out_dir or (RESULTS_ROOT / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_halltype_{args.run_id}.json"
    payload = {
        "run_id": args.run_id,
        "current_year": args.current_year,
        "n_records": len(records),
        "n_classified": len(classifications),
        "confusion_matrix": matrix,
        "classifications": classifications,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[eval_judge_halltype] run_id={args.run_id} "
        f"records={len(records)} out={out_path}"
    )
    print(
        f"  per_type={matrix['per_type_counts']} "
        f"gate_pass={all(v['pass'] for v in matrix['gate'].values())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
