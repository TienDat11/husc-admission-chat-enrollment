"""Slice S3 — Ground-truth dataset generator + traceability validator.

Builds a fact-level ground-truth file (husc_thi_sinh_thuc_gt.json) for the 86
real student questions in husc_thi_sinh_thuc.json. Every critical NUMBER/CODE in
a GT entry is traceable to a RAW source document (NOT a lancedb chunk id) — this
breaks the circularity where retrieval candidates become the eval truth.

Public API:
  * build_source_map() -> dict[str, str]
        Flatten canonical + operational_qa into {source_doc_ref: text}. The refs
        point to RAW documents, not indexed chunks.

  * validate_ground_truth(gt_list, source_map) -> dict
        Traceability + major-code validator. Returns
        {"ok": bool, "errors": [..], "n_verified": int, "n_unverified": int}.

  * load_verified_only(path) -> list
        Helper for downstream precision/recall gating: returns only the entries
        whose `verified` flag is True.

  * main()
        Loads the 86 questions, builds source_map, and writes a DRAFT GT file
        (many entries are `verified:false` with empty critical_facts; the human
        / Opus verification step is a later slice). ≥10 entries are seeded with
        real, traceable critical_facts so the validator test has positive
        signal.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# --- Constants ---

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = RAG2025_ROOT / "src"
EVAL_DIR = RAG2025_ROOT / "data" / "eval"
RAW_DIR = RAG2025_ROOT / "data" / "raw"

CANONICAL_PATH = RAW_DIR / "husc_2026_canonical.json"
OPQA_PATH = RAW_DIR / "husc_2026_operational_qa.json"
QUESTIONS_PATH = EVAL_DIR / "husc_thi_sinh_thuc.json"
GT_OUT_PATH = EVAL_DIR / "husc_thi_sinh_thuc_gt.json"

VALID_BEHAVIORS = {"answer", "abstain", "refuse"}

# Vietnamese numeric separators we treat as equivalent when comparing a critical
# fact's value to a source document. Used only when the value is "numeric-like".
_NUMERIC_RX = re.compile(r"^[\d\.,\s]+$")
_STRIP_NUM_SEP_RX = re.compile(r"[\.,\s]")


# --- Path resolution helpers ---

def _resolve(path: pathlib.Path) -> pathlib.Path:
    """Return path if it exists, else raise FileNotFoundError with context."""
    if not path.is_file():
        raise FileNotFoundError(f"required RAW/EVAL file missing: {path}")
    return path


# --- (A) source map ---

def build_source_map() -> Dict[str, str]:
    """Flatten canonical + operational_qa into {source_doc_ref: text}.

    Keys (no chunk ids, only RAW doc refs):
      canonical:tuition:<khoi_key>          e.g. canonical:tuition:khoi_v
      canonical:programs:<code>             one per program (26 total)
      canonical:phuong_thuc                 all 5 phuong_thuc methods
      canonical:diem_cong                   bonus-point items
      opqa:<opqa_key>                       one per operational_qa entry
    """
    source_map: Dict[str, str] = {}

    # --- canonical ---
    with _resolve(CANONICAL_PATH).open("r", encoding="utf-8") as f:
        canonical = json.load(f)

    # Tuition
    for khoi_key, payload in canonical.get("tuition", {}).items():
        codes = ", ".join(payload.get("nganh_codes", []))
        text = (
            f"Học phí {khoi_key}: {payload.get('amount')} "
            f"{payload.get('unit', 'VNĐ/tín chỉ')}. "
            f"Mã ngành áp dụng: {codes}."
        )
        source_map[f"canonical:tuition:{khoi_key}"] = text

    # Programs
    for prog in canonical.get("programs", []):
        code = prog["code"]
        to_hop = ", ".join(prog.get("to_hop", []))
        text = (
            f"Mã ngành {code} - {prog.get('name', '')} "
            f"(tên ngắn: {prog.get('name_short', '')}). "
            f"Chỉ tiêu: {prog.get('chi_tieu')}. "
            f"Tổ hợp xét tuyển: {to_hop}. "
            f"Khối học phí: {prog.get('khoi_hoc_phi')}. "
            f"Thời gian đào tạo: {prog.get('duration_years')} năm."
        )
        source_map[f"canonical:programs:{code}"] = text

    # Phuong thuc
    pt_items = canonical.get("phuong_thuc", [])
    pt_lines = [f"{pt['id']} - {pt.get('name', '')}: {pt.get('mo_ta', '')}"
                for pt in pt_items]
    source_map["canonical:phuong_thuc"] = "Các phương thức xét tuyển HUSC 2026: " + " | ".join(pt_lines) + "."

    # Diem cong
    dc = canonical.get("diem_cong", {})
    dc_items = dc.get("items", [])
    dc_lines = [f"{it.get('type', '')}: {it.get('value', '')}" for it in dc_items]
    note = dc.get("note", "")
    source_map["canonical:diem_cong"] = (
        f"Điểm cộng tối đa {dc.get('max_pct', '10% thang điểm')}. "
        f"Các mục: " + " | ".join(dc_lines) + f". Ghi chú: {note}"
    )

    # --- operational_qa ---
    with _resolve(OPQA_PATH).open("r", encoding="utf-8") as f:
        opqa = json.load(f)
    for key, payload in opqa.items():
        source_map[f"opqa:{key}"] = payload.get("answer_canonical", "")

    return source_map


# --- (B) traceability validator ---

def _value_in_text(value: str, text: str) -> bool:
    """True if value appears in text. Numeric values are compared with . , space
    treated as equivalent so "600000" matches "600.000"."""
    if not value or not text:
        return False
    if value in text:
        return True
    # Numeric normalization (Vietnamese thousands separator)
    if _NUMERIC_RX.match(value):
        norm_value = _STRIP_NUM_SEP_RX.sub("", value)
        norm_text = _STRIP_NUM_SEP_RX.sub("", text)
        if norm_value and norm_value in norm_text:
            return True
    return False


def validate_ground_truth(
    gt_list: List[Dict[str, Any]],
    source_map: Dict[str, str],
) -> Dict[str, Any]:
    """Validate GT entries for traceability + fabricated-code guard.

    Returns {"ok": bool, "errors": [str], "n_verified": int, "n_unverified": int}.
    """
    errors: List[str] = []
    n_verified = 0
    n_unverified = 0

    # Lazy-import major_code_validator to keep this module import-cheap.
    try:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from services.major_code_validator import validate_answer_majors  # type: ignore
    except Exception as exc:  # pragma: no cover
        errors.append(f"could not import major_code_validator: {exc}")
        validate_answer_majors = None  # type: ignore

    for entry in gt_list:
        eid = entry.get("id", "<no-id>")
        # behavior check
        beh = entry.get("expected_behavior")
        if beh not in VALID_BEHAVIORS:
            errors.append(f"{eid}: invalid expected_behavior={beh!r}")

        # major code check on reference_answer
        ref_ans = entry.get("reference_answer") or ""
        if ref_ans and validate_answer_majors is not None:
            bad_codes = validate_answer_majors(ref_ans)
            for code in bad_codes:
                errors.append(f"{eid}: fabricated major code {code!r} in reference_answer")

        # critical_facts traceability
        facts = entry.get("critical_facts") or []
        if not facts:
            n_unverified += 1
            continue

        all_facts_ok = True
        for fact in facts:
            value = (fact.get("value") or "").strip()
            ref = (fact.get("source_doc_ref") or "").strip()
            if not value:
                errors.append(f"{eid}: critical_fact with empty value")
                all_facts_ok = False
                continue
            if not ref:
                errors.append(f"{eid}: critical_fact with empty source_doc_ref (value={value!r})")
                all_facts_ok = False
                continue
            if ref not in source_map:
                errors.append(
                    f"{eid}: critical_fact value={value!r} ref={ref!r} — "
                    f"ref NOT in source_map"
                )
                all_facts_ok = False
                continue
            text = source_map[ref]
            if not _value_in_text(value, text):
                errors.append(
                    f"{eid}: critical_fact value={value!r} not found in {ref}"
                )
                all_facts_ok = False
                continue

        if all_facts_ok:
            n_verified += 1
        else:
            n_unverified += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_verified": n_verified,
        "n_unverified": n_unverified,
    }


# --- (C) load_verified_only helper ---

def load_verified_only(path: pathlib.Path | str) -> List[Dict[str, Any]]:
    """Read a GT JSON file and return only entries where `verified` is True.

    Used by downstream precision/recall gating to compare predictions only
    against the human-verified subset.
    """
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [e for e in data if e.get("verified") is True]


# --- (D) seed facts + draft builder ---

def _seed_critical_facts(qid: str) -> List[Dict[str, str]]:
    """Return (possibly empty) list of real, traceable critical_facts for qid.

    Only seeds questions where a confidently-mappable fact exists. Everything
    else gets an empty list and is left to the human-verification step.
    """
    seeds: Dict[str, List[Dict[str, str]]] = {
        # msg016 — phương thức tuyển sinh: list 5 methods, include the well-known
        # "XTT" + "THPT" + "HB" + "NL" + "KH" ids.
        "msg016": [
            {"value": "XTT", "source_doc_ref": "canonical:phuong_thuc",
             "fact_kind": "phuong_thuc"},
            {"value": "THPT", "source_doc_ref": "canonical:phuong_thuc",
             "fact_kind": "phuong_thuc"},
            {"value": "HB", "source_doc_ref": "canonical:phuong_thuc",
             "fact_kind": "phuong_thuc"},
            {"value": "NL", "source_doc_ref": "canonical:phuong_thuc",
             "fact_kind": "phuong_thuc"},
            {"value": "KH", "source_doc_ref": "canonical:phuong_thuc",
             "fact_kind": "phuong_thuc"},
        ],
        # msg017 — trường có ngành gì: real program codes.
        "msg017": [
            {"value": "7220104", "source_doc_ref": "canonical:programs:7220104",
             "fact_kind": "ma_nganh"},
            {"value": "7480201", "source_doc_ref": "canonical:programs:7480201",
             "fact_kind": "ma_nganh"},
        ],
        # msg001 — tìm hiểu ngành: confirm a real program exists.
        "msg001": [
            {"value": "7229030", "source_doc_ref": "canonical:programs:7229030",
             "fact_kind": "ma_nganh"},
        ],
        # msg015 — xem thông tin ngành ở đâu: confirm a real program.
        "msg015": [
            {"value": "7480201", "source_doc_ref": "canonical:programs:7480201",
             "fact_kind": "ma_nganh"},
        ],
        # msg018 — ngành văn học: real code + name.
        "msg018": [
            {"value": "7229030", "source_doc_ref": "canonical:programs:7229030",
             "fact_kind": "ma_nganh"},
        ],
        # msg024 — học phí bao nhiêu: 3 khoi amounts.
        "msg024": [
            {"value": "600000", "source_doc_ref": "canonical:tuition:khoi_v",
             "fact_kind": "hoc_phi"},
            {"value": "540000", "source_doc_ref": "canonical:tuition:khoi_iv",
             "fact_kind": "hoc_phi"},
            {"value": "525000", "source_doc_ref": "canonical:tuition:khoi_vii",
             "fact_kind": "hoc_phi"},
        ],
        # msg006 — xét học bạ ngành CNTT
        "msg006": [
            {"value": "7480201", "source_doc_ref": "canonical:programs:7480201",
             "fact_kind": "ma_nganh"},
        ],
        # msg035 — so sánh CNTT và CNNT Việt-Nhật: 2 real codes
        "msg035": [
            {"value": "7480201", "source_doc_ref": "canonical:programs:7480201",
             "fact_kind": "ma_nganh"},
            {"value": "7480201VJ", "source_doc_ref": "canonical:programs:7480201VJ",
             "fact_kind": "ma_nganh"},
        ],
        # msg048 — chỉ tiêu 2025: canonical is 2026, so use a real 2026 chi_tieu
        # as a reference ("chỉ tiêu 2025" → answer uses 2026 plan reference).
        "msg048": [
            {"value": "400", "source_doc_ref": "canonical:programs:7480201",
             "fact_kind": "chi_tieu"},
        ],
        # msg031 — ngành Kiến trúc tổ hợp: real code 7580101 + one of its to_hop.
        "msg031": [
            {"value": "7580101", "source_doc_ref": "canonical:programs:7580101",
             "fact_kind": "ma_nganh"},
            {"value": "V00", "source_doc_ref": "canonical:programs:7580101",
             "fact_kind": "to_hop"},
        ],
        # msg079 — ngành Lịch sử: real code 7229010.
        "msg079": [
            {"value": "7229010", "source_doc_ref": "canonical:programs:7229010",
             "fact_kind": "ma_nganh"},
        ],
        # msg081 — ngành Văn học (job question): same code as msg018
        "msg081": [
            {"value": "7229030", "source_doc_ref": "canonical:programs:7229030",
             "fact_kind": "ma_nganh"},
        ],
        # msg068 — ngành Truyền thông số + số tín chỉ / học phí (we don't have TC
        # count in canonical, but we have chi_tieu + khoi)
        "msg068": [
            {"value": "7320111", "source_doc_ref": "canonical:programs:7320111",
             "fact_kind": "ma_nganh"},
            {"value": "525000", "source_doc_ref": "canonical:tuition:khoi_vii",
             "fact_kind": "hoc_phi"},
        ],
    }
    return seeds.get(qid, [])


# --- (E) main: build the draft GT file ---

def _classify_expected_behavior(question: str, qid: str) -> str:
    """Conservative default classifier. Returns one of VALID_BEHAVIORS.

    Heuristic:
      * "điểm chuẩn" / "điểm sàn năm ngoái" / "dự kiến năm nay điểm có tăng"
        → abstain (we don't have year-over-year cutoff data in canonical)
      * "group / zalo tư vấn" → abstain (opqa marks this as missing)
      * everything else → answer
    """
    q = question.lower()
    if any(kw in q for kw in ("điểm chuẩn", "điểm sàn", "dự kiến điểm",
                              "điểm 3 năm", "điểm xét học bạ năm trước",
                              "điểm vẽ từ uah", "có link đăng ký",
                              "mã trường", "group hay zalo")):
        return "abstain"
    return "answer"


def _draft_entry(question_obj: Dict[str, Any]) -> Dict[str, Any]:
    qid = question_obj["id"]
    question = question_obj["question"]
    facts = _seed_critical_facts(qid)
    behavior = _classify_expected_behavior(question, qid)
    source_refs = sorted({f["source_doc_ref"] for f in facts}) if facts else []
    return {
        "id": qid,
        "question": question,
        "reference_answer": "",  # left blank for human/Opus verification pass
        "expected_behavior": behavior,
        "source_doc_refs": source_refs,
        "critical_facts": facts,
        "verified": False,
    }


def main() -> int:
    """Build source_map, load 86 questions, write the draft GT file. Returns 0."""
    with _resolve(QUESTIONS_PATH).open("r", encoding="utf-8") as f:
        questions = json.load(f)
    assert isinstance(questions, list), "questions file must be a JSON array"

    gt_list: List[Dict[str, Any]] = []
    for q in questions:
        gt_list.append(_draft_entry(q))

    # Make sure all 86 ids are present, no dupes
    ids = [e["id"] for e in gt_list]
    assert len(ids) == 86, f"expected 86 entries, got {len(ids)}"
    assert len(ids) == len(set(ids)), "duplicate ids in draft GT"

    GT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GT_OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(gt_list, f, ensure_ascii=False, indent=2)

    # Self-validate before writing? We wrote already; but re-validate now.
    source_map = build_source_map()
    res = validate_ground_truth(gt_list, source_map)
    n_with_facts = sum(1 for e in gt_list if e.get("critical_facts"))
    print(f"[gen_ground_truth] wrote {len(gt_list)} entries → {GT_OUT_PATH}")
    print(f"[gen_ground_truth] source_map size: {len(source_map)}")
    print(f"[gen_ground_truth] entries with critical_facts: {n_with_facts}")
    print(f"[gen_ground_truth] validator ok={res['ok']} "
          f"verified={res['n_verified']} unverified={res['n_unverified']}")
    if not res["ok"]:
        print(f"[gen_ground_truth] VALIDATION ERRORS:")
        for e in res["errors"]:
            print(f"  - {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
