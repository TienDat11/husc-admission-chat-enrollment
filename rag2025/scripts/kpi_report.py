#!/usr/bin/env python3
"""kpi_report.py — đo KPI vận hành (T16) từ artifact baseline_T00.

Quy tắc
-------
- Mọi số đều kèm (a) đường dẫn artifact nguồn, (b) ngày đo.
- KPI chưa đo được -> in đúng nhãn "chưa đo được: <lý do>", KHÔNG crash,
  KHÔNG bịa số.
- Đầu vào thiếu/khỏi -> lỗi rõ ràng (exit 1), không im lặng trả 0.

Mỗi KPI là một hàm đo riêng; `main()` chọn lệnh và in bảng KPI tổng hợp ở cuối.

Usage
-----
  kpi_report.py [k1|k2|k5|k3|k4|all] [--baseline-dir PATH] [--data-dir PATH] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import dataclasses
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR_HELP = "thư mục data (chứa kpi/probe_log.jsonl và unanswered/queue.jsonl)"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASELINE_DIR_DEFAULT = REPO_ROOT / "results" / "baseline_T00"

MEASUREMENT_DATE = date.today().isoformat()


class KpiError(RuntimeError):
    """Lỗi cấu hình / missing artifact -> main() in lỗi và exit 1."""


def find_baseline_dir(path: Optional[Path]) -> Path:
    if path is not None:
        if not path.is_dir():
            raise KpiError(f"không tìm thấy artifact baseline_T00: {path}")
        return path
    for cand in (BASELINE_DIR_DEFAULT,):
        if cand.is_dir():
            return cand
    raise KpiError(
        f"không tìm thấy artifact baseline_T00 tại {BASELINE_DIR_DEFAULT}\n"
        "Dùng --baseline-dir để trỏ tới thư mục baseline_T00 (chứa "
        "baseline_T00_detail.json, heldout_metrics.json, *.jsonl)."
    )


def _path_to_baseline(p: Path) -> Path:
    """Chuẩn hóa đường dẫn artifact dưới baseline dir thành chuỗi báo cáo."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise KpiError(f"file không tồn tại: {_path_to_baseline(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise KpiError(f"file không tồn tại: {_path_to_baseline(path)}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "0/0"
    return f"{num}/{den} = {num / den:.1%}"


# ---------------------------------------------------------------------------
# KPI sections
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print("=" * 72)
    print(f"### {title}")
    print("=" * 72)


def print_row(label: str, value: str) -> None:
    print(f"{label:<44} {value}")


# Chuỗi marker của câu fallback khi provider LLM chết (thấy trong T00).
FALLBACK_MARKER = "tạm thời không khả dụng"



def _coarse_route_match(gold: object, route: object) -> bool:
    """Coarse match: padded/hybrid <-> hybrid; graph <-> graph_rag (06 spec §2)."""
    g = gold.lower() if gold else None
    r = route.lower() if route else None
    if g in ("padded", "hybrid"):
        return r == "hybrid"
    if g == "graph":
        return r == "graph_rag"
    return False

def _is_fallback_answer(answer: object) -> bool:
    """True nếu answer rỗng hoặc là chuỗi fallback (không phải câu trả lời thật)."""
    a = (answer or "").strip() if isinstance(answer, str) else ""
    return (not a) or (FALLBACK_MARKER in a)


def _has_measurable_answers(recs: List[Dict[str, Any]]) -> bool:
    """True nếu ≥1 record có câu trả lời thật KÈM tín hiệu phân loại
    (expected_behavior hoặc judge_scores) để áp bộ lọc 'không abstain'.
    Thiếu một trong hai → không đo được K2/K5, không đoán."""
    for r in recs:
        if r.get("expected_behavior") is None and r.get("judge_scores") is None:
            continue
        if _is_fallback_answer(r.get("answer")):
            continue
        return True
    return False


def _is_judged_correct(judge: object) -> bool:
    """Đúng/sai theo judge_scores. Chỉ hiểu 2 dạng rõ ràng; dạng lạ → lỗi
    rõ ràng (không đoán đúng/sai thay judge)."""
    if isinstance(judge, dict):
        if isinstance(judge.get("correct"), bool):
            return judge["correct"]
        if isinstance(judge.get("is_correct"), bool):
            return judge["is_correct"]
        score = judge.get("score")
        if isinstance(score, (int, float)) and 0 <= score <= 1:
            return score >= 0.5
    raise KpiError(f"schema judge_scores chưa biết ({judge!r:.80}) — không đoán đúng/sai")


def measure_k1() -> int:
    """K1 — quy trình tay 2 tuần (con người, không phải script)."""
    print_section("K1 — Câu lặp lại thầy cô phải tự trả lời")
    print("Đây là quy trình người, KHÔNG phải số do script đo. Thầy cô ghi tay 2 "
          "tuần trước khi bật chatbot, theo bảng template bên dưới.")
    print()
    print("1. Thời gian: 2 tuần liên tiếp (tuần 1 + tuần 2).")
    print("2. Người ghi: bộ phận tuyển sinh / trực hotline.")
    print("3. Kênh ghi nhận: Fanpage / Zalo OA / Hotline.")
    print("4. Mỗi câu hỏi ghi 1 dòng vào bảng; nhóm nội dung theo 5 loại bên dưới.")
    print()
    print("Bảng ghi (Excel / Google Sheets):")
    print("  +-----+----------+-----------+-------------------------------------------------+-------+")
    print("  | STT | Ngày     | Kênh      | Nội dung câu hỏi (chép lại)                   | Nhóm  |")
    print("  +-----+----------+-----------+-------------------------------------------------+-------+")
    print("  |  1  | dd/mm/yy | Fanpage   | 'Điểm chuẩn ngành A năm nay bao nhiêu?'       | điểm  |")
    print("  |  2  | dd/mm/yy | Hotline   | 'Hồ sơ xét học bạ gồm những giấy tờ gì?'       | hồ sơ |")
    print("  +-----+----------+-----------+-------------------------------------------------+-------+")
    print()
    print("5. Nhóm nội dung: 'chiến_lược' | 'hoc_phi' | 'nganh_diem_chuan' | "
          "'ho_so_nhap_hoc' | 'khac'.")
    print("6. Chỉ đếm dòng thuộc nhóm 'đã có trong tài liệu'.")
    print()
    print("Kết quả cuối 2 tuần:  K1 = (số câu/tuần) = tổng dòng đã có trong "
          "tài liệu / 2.")
    print("Đơn vị: câu/tuần.")
    print()
    print("Vị trí tài liệu: rag2025/docs/kpi-van-hanh.md")
    return 0


def measure_k2() -> int:
    """K2 — tỷ lệ bot trả lời có trích dẫn = (không abstain, >=1 nguồn) / tổng."""
    baseline = find_baseline_dir(args.baseline_dir)
    path86 = baseline / "husc86q_records_t00.jsonl"
    recs = load_jsonl(path86)
    n = len(recs)
    if n == 0:
        raise KpiError(
            f"artifact rỗng: {_path_to_baseline(path86)} (0 dòng) — "
            "không có dữ liệu để đo, không trả về 0")
    print_section("K2 — Tỷ lệ bot trả lời có trích dẫn")
    print_row("định nghĩa", "(câu không abstain) có >=1 nguồn / tổng câu")
    print_row("nguồn", _path_to_baseline(path86))

    if not _has_measurable_answers(recs):
        fallbacks = sum(1 for r in recs if _is_fallback_answer(r.get("answer")))
        print()
        print("K2 (86q eval): chưa đo được: baseline degraded (provider LLM chết) — "
              f"{fallbacks}/{n} câu là fallback 'dịch vụ tạo câu trả lời tạm thời không khả dụng'; "
              "không có câu trả lời thật để chấm")
    else:
        non_abst = [r for r in recs if r.get("expected_behavior") != "abstain"]
        with_src = [r for r in non_abst if r.get("retrieved_chunks")]
        print_row("86q (eval):", fmt_pct(len(with_src), n))

    # heldout
    hm = load_json(baseline / "heldout_metrics.json")
    per = hm["per_question"]
    hrecs = load_jsonl(baseline / "heldout40_records_t00.jsonl")
    nh = len(hrecs)
    if nh == 0:
        raise KpiError(
            f"artifact rỗng: {_path_to_baseline(baseline / 'heldout40_records_t00.jsonl')} (0 dòng) — "
            "không có dữ liệu để đo, không trả về 0")
    if not _has_measurable_answers(hrecs):
        hfallbacks = sum(1 for r in hrecs if _is_fallback_answer(r.get("answer")))
        print(f"K2 (heldout40): chưa đo được: baseline degraded — "
              f"{hfallbacks}/{nh} câu là fallback; không có câu trả lời thật để chấm")
    else:
        non_abst_h = [r for r in hrecs if r.get("expected_behavior") != "abstain"]
        with_src_h = [r for r in non_abst_h if r.get("retrieved_chunks")]
        print_row("heldout40:", fmt_pct(len(with_src_h), nh))

    # Proxy retrieval (thông tin riêng, KHÔNG phải K2)
    print()
    print_row("Proxy truy xuất (KHÔNG phải K2)", "(chỉ số chunk có ≥1 nguồn, không phản ánh câu trả lời thật)")
    with_chunks_86 = sum(1 for r in recs if r.get("retrieved_chunks"))
    print_row("      ", f"86q: {fmt_pct(with_chunks_86, n)} | {n} câu, nguồn: {_path_to_baseline(path86)}")
    with_chunks_h = sum(1 for r in hrecs if r.get("retrieved_chunks"))
    print_row("      ", f"heldout40: {fmt_pct(with_chunks_h, nh)} | {nh} câu, nguồn: {_path_to_baseline(baseline / 'heldout40_records_t00.jsonl')}")
    return 0


def measure_k5() -> int:
    """K5 — đúng trên bộ câu vàng = % câu ĐÚNG do người chấm (judge_scores /
    expected_behavior KỲ vọng đáp án). Baseline T00 degraded -> chưa đo được.
    Proxy truy xuất (route match / chunk có nguồn) chỉ là THAM KHẢO riêng."""
    baseline = find_baseline_dir(args.baseline_dir)
    hm = load_json(baseline / "heldout_metrics.json")
    per = hm["per_question"]
    hrecs = load_jsonl(baseline / "heldout40_records_t00.jsonl")
    recs = load_jsonl(baseline / "husc86q_records_t00.jsonl")
    if len(hrecs) == 0:
        raise KpiError(
            f"artifact rỗng: {_path_to_baseline(baseline / 'heldout40_records_t00.jsonl')} (0 dòng) — "
            "không có dữ liệu để đo, không trả về 0")
    if len(recs) == 0:
        raise KpiError(
            f"artifact rỗng: {_path_to_baseline(baseline / 'husc86q_records_t00.jsonl')} (0 dòng) — "
            "không có dữ liệu để đo, không trả về 0")

    print_section("K5 — Đúng trên bộ câu vàng")
    print_row("định nghĩa", "% câu ĐÚNG do chấm sự kiện (judge_scores / expected_behavior 'answer' THẬT)")
    print_row("nguồn", _path_to_baseline(baseline / "heldout40_records_t00.jsonl"))
    print()

    # K5 THẬT cần câu trả lời thật KÈM chấm đúng/sai — T00 degraded:
    if not _has_measurable_answers(hrecs):
        hfallback = sum(1 for r in hrecs if _is_fallback_answer(r.get("answer")))
        print(f"K5 (heldout40): chưa đo được: thiếu chấm sự kiện của chuyên gia/judge "
              f"({hfallback}/{len(hrecs)} câu là fallback, judge_scores = None x/{len(hrecs)}) "
              "— không có câu trả lời đúng/sai để chấm %")
        print()
        print("K5 (86q eval): chưa đo được: cùng lý do — 86q JSONL "
              "expected_behavior/gold_route = None x/303, judge_scores = None")
        print()
        # Proxy riêng (tham khảo, KHÔNG gán K5):
        has_gold = any(q.get("gold_route") for q in per)
        has_exp = any(q.get("expected_behavior") is not None for q in per)
        if has_gold and has_exp:
            route_match = sum(
                1 for q in per
                if _coarse_route_match(q.get("gold_route"), q.get("route"))
                and q.get("expected_behavior") == "answer"
            )
            print_row("Proxy route+expected_behavior (KHÔNG phải K5)",
                      f"(route coarse-match gold_route AND expected_behavior=='answer', "
                      "đếm theo heldout_metrics.json — KHÔNG chấm câu trả lời đúng sự kiện)")
            print_row("      ", f"heldout40: {fmt_pct(route_match, len(per))} | "
                               f"nguồn: {_path_to_baseline(baseline / 'heldout_metrics.json')}")
        with_c86 = sum(1 for r in recs if r.get("retrieved_chunks"))
        with_ch = sum(1 for r in hrecs if r.get("retrieved_chunks"))
        print_row("      Proxy truy xuất (KHÔNG phải K5)",
                  f"86q: {fmt_pct(with_c86, len(recs))} | "
                  f"heldout40: {fmt_pct(with_ch, len(hrecs))} (chunk có >=1 nguồn)")
        return 0

    # Có câu trả lời thật + judge: đo thật (vẽ đường cho sau)
    correct = sum(
        1 for r in hrecs
        if r.get("expected_behavior") == "answer"
        and not _is_fallback_answer(r.get("answer"))
        and _is_judged_correct(r.get("judge_scores"))
    )
    print_row("heldout40:", fmt_pct(correct, len(hrecs)))
    return 0



def _resolve_data_dir() -> Path:
    """Data dir cho K3/K4: --data-dir > CANONICAL_STORE_ROOT env > mặc định repo."""
    arg_dir = getattr(globals().get("args", None), "data_dir", None)
    if arg_dir is not None:
        return Path(arg_dir)
    env_root = os.environ.get("CANONICAL_STORE_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    return REPO_ROOT / "data" / "canonical_store"


def _parse_ts(raw: object) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _load_probe_durations(probe_path: Path) -> List[float]:
    """Phút (served_at − upload_received_at) cho mỗi entry đủ 2 mốc.

    Entry thiếu 1 trong 2 mốc (upload chưa rõ, release chưa từng serve)
    bị BỎ QUA — không điền 0, không bịa số.
    """
    if not probe_path.is_file():
        return []
    merged: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(probe_path):
        rid = row.get("release_id")
        if not rid:
            continue
        e = merged.setdefault(str(rid), {})
        if row.get("event") == "published" or ("published_at" in row and "published_at" not in e):
            for key in ("upload_received_at", "published_at"):
                if e.get(key) is None and row.get(key) is not None:
                    e[key] = row[key]
        if row.get("served_at"):
            e["served_at"] = row["served_at"]
    minutes: List[float] = []
    for e in merged.values():
        up = _parse_ts(e.get("upload_received_at"))
        sv = _parse_ts(e.get("served_at"))
        if up is None or sv is None:
            continue
        minutes.append((sv - up).total_seconds() / 60)
    return minutes


def _load_k4_counts(queue_path: Path) -> tuple[int, int]:
    """(tử số, mẫu số) K4: mục tạo trong 7 ngày qua, trừ mục out_of_scope.

    Tử số = mục có resolved_at − asked_at ≤ 48h. Mẫu số = mục tạo trong
    tuần trừ mục bị đánh spam/ngoài phạm vi (02 §9).
    """
    now = datetime.now(timezone.utc)
    num = den = 0
    for row in load_jsonl(queue_path):
        if row.get("out_of_scope"):
            continue
        asked = _parse_ts(row.get("asked_at"))
        if asked is None or (now - asked) > timedelta(days=7):
            continue
        den += 1
        resolved = _parse_ts(row.get("resolved_at"))
        if resolved is not None and (resolved - asked) <= timedelta(hours=48):
            num += 1
    return num, den


def measure_k3() -> int:
    """K3 — phút từ upload_received_at tới served_at (median trên probe log)."""
    data_dir = _resolve_data_dir()
    probe_path = data_dir / "kpi" / "probe_log.jsonl"
    print_section("K3 — Thời gian từ thông báo mới -> bot trả lời đúng")
    print_row("nguồn", _path_to_baseline(probe_path))
    minutes = _load_probe_durations(probe_path)
    if not minutes:
        print("chưa đo được: chưa có dữ liệu (probe log trống hoặc thiếu mốc thời gian)")
        return 0
    print_row("số release đo được", str(len(minutes)))
    print_row("K3 (median phút)", f"{_median(minutes):.1f}")
    print_row("K3 (trung bình phút)", f"{sum(minutes) / len(minutes):.1f}")
    return 0


def measure_k4() -> int:
    """K4 — tỷ lệ hàng đợi unanswered được xử lý trong 48h."""
    data_dir = _resolve_data_dir()
    queue_path = data_dir / "unanswered" / "queue.jsonl"
    print_section("K4 — Hàng đợi câu chưa trả lời được xử lý trong 48h")
    print_row("nguồn", _path_to_baseline(queue_path))
    print("Công thức: tử số = mục có resolved_at - asked_at <= 48h;")
    print("  mẫu số = mục tạo trong 7 ngày qua (trừ mục out_of_scope)")
    if not queue_path.is_file():
        print("chưa đo được: chưa có hàng đợi unanswered (thiếu queue.jsonl)")
        return 0
    num, den = _load_k4_counts(queue_path)
    print_row("K4 (trong 48h / tạo trong tuần)", fmt_pct(num, den))
    return 0


# ---------------------------------------------------------------------------
# Unified table + main
# ---------------------------------------------------------------------------

@dataclass
class KpiRow:
    kpi: str
    value: str
    source: str
    date: str


def build_table() -> List[KpiRow]:
    rows: List[KpiRow] = []
    rows.append(KpiRow(
        "K1 Câu lặp lại thầy cô phải tự trả lời",
        "quy trình tay 2 tuần (template: rag2025/docs/kpi-van-hanh.md)",
        "rag2025/docs/kpi-van-hanh.md",
        MEASUREMENT_DATE))

    baseline = BASELINE_DIR_DEFAULT
    if baseline.is_dir():
        recs = load_jsonl(baseline / "husc86q_records_t00.jsonl")
        n = len(recs)
        if n == 0:
            raise KpiError(
                f"artifact rỗng: {_path_to_baseline(baseline / 'husc86q_records_t00.jsonl')} (0 dòng)")
        if not _has_measurable_answers(recs):
            fallbacks = sum(1 for r in recs if _is_fallback_answer(r.get("answer")))
            k2_value = (f"chưa đo được: baseline degraded — {fallbacks}/{n} câu là "
                       "fallback; không có câu trả lời thật để chấm")
        else:
            non_abst = [r for r in recs if r.get("expected_behavior") != "abstain"]
            with_src = [r for r in non_abst if r.get("retrieved_chunks")]
            k2_value = fmt_pct(len(with_src), n)
        rows.append(KpiRow(
            "K2 Tỷ lệ bot trả lời có trích dẫn",
            k2_value,
            "baseline_T00/husc86q_records_t00.jsonl",
            "baseline 2026-10-06 -> đo " + MEASUREMENT_DATE))
    else:
        rows.append(KpiRow(
            "K2 Tỷ lệ bot trả lời có trích dẫn",
            "chưa đo được: thiếu artifact baseline_T00",
            "-", "-"))

    data_dir = _resolve_data_dir()
    probe_path = data_dir / "kpi" / "probe_log.jsonl"
    minutes = _load_probe_durations(probe_path)
    if minutes:
        k3_value = (f"median {_median(minutes):.1f} phút "
                    f"(trung bình {sum(minutes) / len(minutes):.1f}; "
                    f"{len(minutes)} release)")
        k3_src, k3_date = str(_path_to_baseline(probe_path)), MEASUREMENT_DATE
    else:
        k3_value = "chưa đo được: chưa có dữ liệu (probe log trống hoặc thiếu mốc thời gian)"
        k3_src, k3_date = str(_path_to_baseline(probe_path)), "-"
    rows.append(KpiRow(
        "K3 Thời gian từ thông báo mới -> trả lời đúng",
        k3_value, k3_src, k3_date))
    queue_path = data_dir / "unanswered" / "queue.jsonl"
    if queue_path.is_file():
        num, den = _load_k4_counts(queue_path)
        k4_value = f"K4 (trong 48h / tạo trong tuần) {fmt_pct(num, den)}"
        k4_src, k4_date = str(_path_to_baseline(queue_path)), MEASUREMENT_DATE
    else:
        k4_value = "chưa đo được: chưa có hàng đợi unanswered (thiếu queue.jsonl)"
        k4_src, k4_date = str(_path_to_baseline(queue_path)), "-"
    rows.append(KpiRow(
        "K4 Hàng đợi unanswered xử lý trong 48h",
        k4_value, k4_src, k4_date))

    if baseline.is_dir():
        hrecs = load_jsonl(baseline / "heldout40_records_t00.jsonl")
        if len(hrecs) == 0:
            raise KpiError(
                f"artifact rỗng: {_path_to_baseline(baseline / 'heldout40_records_t00.jsonl')} (0 dòng)")
        if not _has_measurable_answers(hrecs):
            hfallback = sum(1 for r in hrecs if _is_fallback_answer(r.get("answer")))
            k5_value = (f"chưa đo được: thiếu chấm sự kiện — {hfallback}/{len(hrecs)} câu là "
                       "fallback, judge_scores = None (baseline T00 degraded)")
        else:
            correct = sum(1 for r in hrecs if r.get("expected_behavior") == "answer"
                          and not _is_fallback_answer(r.get("answer"))
                          and _is_judged_correct(r.get("judge_scores")))
            k5_value = fmt_pct(correct, len(hrecs))
        rows.append(KpiRow(
            "K5 Đúng trên bộ câu vàng",
            k5_value,
            "baseline_T00/heldout40_records_t00.jsonl",
            "baseline 2026-10-06 -> đo " + MEASUREMENT_DATE))
    else:
        rows.append(KpiRow(
            "K5 Đúng trên bộ câu vàng",
            "chưa đo được: thiếu artifact baseline_T00",
            "-", "-"))
    return rows


def print_table(rows: List[KpiRow]) -> None:
    print()
    print("=" * 72)
    print("BẢNG KPI TỔNG HỢP")
    print(f"Ngày đo: {MEASUREMENT_DATE}   |  Mọi số đều kèm nguồn (column 'Nguồn').")
    print("=" * 72)
    header = "KPI".ljust(38) + "Giá trị".ljust(42) + "Nguồn".ljust(36) + "Ngày đo"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = r.kpi.ljust(38) + r.value.ljust(42) + r.source.ljust(36) + r.date
        print(line)
    print("-" * len(header))
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Đo KPI vận hành (T16) từ artifact baseline_T00.",
        epilog="Mỗi KPI là một hàm đo riêng; 'all' in bảng tổng hợp.",
    )
    parser.add_argument("kpi", choices=["k1", "k2", "k5", "k3", "k4", "all"],
                        default="all", help="KPI cần đo")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="đường dẫn tới baseline_T00 (mặc định: "
                        "rag2025/results/baseline_T00)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="thư mục data cho K3/K4 (chứa kpi/probe_log.jsonl "
                        "và unanswered/queue.jsonl; mặc định CANONICAL_STORE_ROOT "
                        "hoặc rag2025/data/canonical_store)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="ghi bảng KPI dưới dạng JSON vào thư mục này")
    global args
    args = parser.parse_args(argv)

    missing_baseline = (args.baseline_dir is None) and (not BASELINE_DIR_DEFAULT.exists())

    def _run(fn):
        """Run a measure_<k> function; swallow KpiError as clean exit 1."""
        try:
            return fn()
        except KpiError as exc:
            print("LỖI:", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 1

    if args.kpi in ("k1", "k2", "k5"):
        # K1/K2/K5 cần artifact T00
        if missing_baseline:
            print("LỖI:", file=sys.stderr)
            print(find_baseline_dir(None), file=sys.stderr)
            return 1
        code = _run({"k1": measure_k1, "k2": measure_k2, "k5": measure_k5}[args.kpi])
    elif args.kpi in ("k3", "k4"):
        # K3/K4 luôn in interface, bất kể có T00 hay không
        code = _run({"k3": measure_k3, "k4": measure_k4}[args.kpi])
    else:
        code = 0
        for fn in (measure_k1, measure_k2, measure_k5, measure_k3, measure_k4):
            code = _run(fn)
            if code != 0:
                break
        print_table(build_table())

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        rows = build_table()
        out = args.out_dir / f"kpi_report_{MEASUREMENT_DATE}.json"
        out.write_text(
            json.dumps([dataclasses.asdict(r) for r in rows], ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"Đã ghi bảng JSON: {_path_to_baseline(out)}", file=sys.stderr)

    return code


if __name__ == "__main__":
    sys.exit(main())
