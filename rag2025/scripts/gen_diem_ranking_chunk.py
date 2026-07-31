# @spec(V5-R011) — Generate ranking chunks for 2025 diem_chuan + chi_tieu + diem_san.
#
# Deterministic (no LLM, no network). Reads:
#   data/major_codes/diem_chuan_history.json  (34 entries, 2025)
# Writes to:
#   data/chunked/husc_diem_ranking_2025.jsonl
#
# Chunks produced:
#   - diem_chuan_2025_full         — sorted ranking by diem_chuan (both methods)
#   - chi_tieu_2025_full           — chỉ tiêu 2025
#   - historical_diem_san_2024_2025_husc — điểm sàn 2024-2025
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY_JSON = ROOT / "data" / "major_codes" / "diem_chuan_history.json"
OUTPUT_FILE = ROOT / "data" / "chunked" / "husc_diem_ranking_2025.jsonl"

# ── 2025 chỉ tiêu (from scripts/ingest_husc_aggregation.py) ──────────────────
CHI_TIEU_2025 = {
    "7220104": 30, "7229001": 30, "7229010": 60, "7229030": 100,
    "7229042": 30, "7310205": 75, "7310301": 30, "7310608": 40,
    "7320101": 230, "7320111": 250, "7420201": 40, "7440102": 20,
    "7440112": 25, "7440301": 25, "7480103": 60, "7480107TD": 30,
    "7480201": 380, "7480201VJ": 30, "7510302": 95, "7510401": 35,
    "7520503": 20, "7580101": 100, "7580211": 20, "7760101": 45,
    "7850101": 25, "7850105": 25,
}


def _load_history() -> list[dict]:
    with HISTORY_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["entries"]


def build_diem_chuan_ranking(entries: list[dict]) -> dict:
    """Build ``diem_chuan_2025_full`` — sorted ranking of 29 majors by
    diem_chuan in 2025 (both methods), CAO→THẤP.

    Returns chunk dict ready for jsonl serialization.
    """
    # Group entries by method
    by_method: dict[str, list[dict]] = {"thi_thpt": [], "hoc_ba": []}
    seen_codes: set[str] = set()
    for e in entries:
        if e["year"] != 2025:
            continue
        m = e["method"]
        if m in by_method:
            by_method[m].append(e)
            seen_codes.add(e["major_code"])

    lines: list[str] = [
        "BẢNG XẾP HẠNG ĐIỂM CHUẨN HUSC NĂM 2025 (CAO → THẤP)",
        "",
        "Lưu ý: Đây là dữ liệu năm 2025 (tham khảo). Điểm chuẩn 2026 chưa công bố.",
        "",
    ]

    # thi_thpt ranking
    thpt_sorted = sorted(by_method["thi_thpt"], key=lambda x: -x["diem_chuan"])
    lines.append("─" * 60)
    lines.append("PHƯƠNG THỨC XÉT ĐIỂM THI THPT 2025:")
    lines.append("─" * 60)
    # Track top for summary
    top_thpt_name = thpt_sorted[0]["major_name"]
    top_thpt_score = thpt_sorted[0]["diem_chuan"]
    for i, e in enumerate(thpt_sorted, 1):
        to_hop_str = ", ".join(e["to_hop"])
        lines.append(
            f"  {i:2d}. {e['major_name']} (mã {e['major_code']}): "
            f"{e['diem_chuan']} điểm — tổ hợp {to_hop_str}"
        )

    lines.append("")
    lines.append(f"  → Ngành CAO NHẤT (THPT): {top_thpt_name} — {top_thpt_score} điểm")
    lines.append(f"  → Ngành THẤP NHẤT (THPT): {thpt_sorted[-1]['major_name']} — {thpt_sorted[-1]['diem_chuan']} điểm")
    lines.append("")

    # hoc_ba ranking
    hb_sorted = sorted(by_method["hoc_ba"], key=lambda x: -x["diem_chuan"])
    lines.append("─" * 60)
    lines.append("PHƯƠNG THỨC XÉT HỌC BẠ 2025:")
    lines.append("─" * 60)
    top_hb_name = hb_sorted[0]["major_name"]
    top_hb_score = hb_sorted[0]["diem_chuan"]
    for i, e in enumerate(hb_sorted, 1):
        to_hop_str = ", ".join(e["to_hop"])
        lines.append(
            f"  {i:2d}. {e['major_name']} (mã {e['major_code']}): "
            f"{e['diem_chuan']} điểm — tổ hợp {to_hop_str}"
        )

    lines.append("")
    lines.append(f"  → Ngành CAO NHẤT (học bạ): {top_hb_name} — {top_hb_score} điểm")
    lines.append(f"  → Ngành THẤP NHẤT (học bạ): {hb_sorted[-1]['major_name']} — {hb_sorted[-1]['diem_chuan']} điểm")
    lines.append("")
    lines.append(
        "Tổng số: 17 ngành có đủ dữ liệu điểm chuẩn 2025 "
        "(các ngành không trong danh sách 2026 hoặc không có dữ liệu bảng được bỏ qua). "
        "Nguồn: tuyensinh.husc.edu.vn (2025)."
    )

    return {
        "chunk_id": "diem_chuan_2025_full",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn (2025)",
        "faq_type": "diem_chuan_ranking_2025",
        "metadata": {
            "data_year": 2025,
            "info_type": "diem_chuan",
            "is_reference": True,
            "school": "HUSC",
            "summary": (
                f"Xếp hạng điểm chuẩn 2025. Cao nhất THPT: {top_thpt_name} "
                f"({top_thpt_score}đ). Cao nhất học bạ: {top_hb_name} "
                f"({top_hb_score}đ)."
            ),
        },
    }


def build_chi_tieu_2025(entries: list[dict]) -> dict:
    """Build ``chi_tieu_2025_full`` — chỉ tiêu 2025 for all known majors."""
    # Get unique major names from history entries
    major_names: dict[str, str] = {}
    for e in entries:
        if e["year"] == 2025:
            major_names[e["major_code"]] = e["major_name"]

    # Sort by chi_tieu DESC
    sorted_codes = sorted(CHI_TIEU_2025.items(), key=lambda x: -x[1])
    total = sum(CHI_TIEU_2025.values())

    lines = [
        "BẢNG CHỈ TIÊU TUYỂN SINH HUSC NĂM 2025 (CAO → THẤP)",
        "",
        "Lưu ý: Chỉ tiêu 2025 là dữ liệu lịch sử. Chỉ tiêu 2026 có thể khác.",
        "",
    ]
    lines.append(f"{'STT':>4} {'Ngành':40s} {'Mã':12s} {'Chỉ tiêu':>10s}")
    lines.append("-" * 68)
    for i, (code, ct) in enumerate(sorted_codes, 1):
        name = major_names.get(code, code)
        lines.append(f"{i:4d} {name:40s} {code:12s} {ct:10d}")
    lines.append("-" * 68)
    lines.append(f"{'':44s} {'Tổng':12s} {total:10d}")
    lines.append("")
    lines.append(f"Ngành có chỉ tiêu cao nhất 2025: Công nghệ thông tin (380).")
    lines.append("Nguồn: tuyensinh.husc.edu.vn (2025).")

    return {
        "chunk_id": "chi_tieu_2025_full",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn (2025)",
        "faq_type": "chi_tieu_2025",
        "metadata": {
            "data_year": 2025,
            "info_type": "chi_tieu",
            "is_reference": True,
            "school": "HUSC",
            "summary": f"Chỉ tiêu 2025: tổng {total}, cao nhất CNTT 380.",
        },
    }


def build_diem_san_2024_2025(entries: list[dict]) -> dict:
    """Build ``historical_diem_san_2024_2025_husc`` — điểm sàn / ngưỡng đầu
    vào 2024-2025.

    Derived from the minimum observed cutoff score per year+method.
    """
    # Compute min per year+method
    min_by: dict[tuple[int, str], float] = {}
    for e in entries:
        key = (e["year"], e["method"])
        if key not in min_by or e["diem_chuan"] < min_by[key]:
            min_by[key] = e["diem_chuan"]

    lines = [
        "NGƯỠNG ĐẦU VÀO / ĐIỂM SÀN HUSC (2024-2025)",
        "",
        "Dữ liệu dưới đây thể hiện mức điểm chuẩn THẤP NHẤT quan sát được "
        "(ngưỡng đầu vào thực tế thấp nhất theo từng phương thức).",
        "",
    ]

    # 2025
    thpt_2025 = min_by.get((2025, "thi_thpt"), "N/A")
    hb_2025 = min_by.get((2025, "hoc_ba"), "N/A")
    lines.append("Năm 2025:")
    lines.append(f"  - Phương thức THPT: {thpt_2025} điểm")
    lines.append(f"  - Phương thức học bạ: {hb_2025} điểm")
    lines.append("")

    # 2024 — no data
    lines.append("Năm 2024:")
    lines.append("  - Không có dữ liệu bảng điểm (file HTML chứa ảnh, không parse được).")
    lines.append("  - Tham khảo trực tiếp tại tuyensinh.husc.edu.vn hoặc liên hệ trường.")
    lines.append("")
    lines.append("Nguồn: tuyensinh.husc.edu.vn (dữ liệu lịch sử).")

    return {
        "chunk_id": "historical_diem_san_2024_2025_husc",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn (2024-2025)",
        "faq_type": "diem_san_lich_su",
        "metadata": {
            "data_year": 2025,
            "info_type": "diem_san",
            "is_reference": True,
            "school": "HUSC",
            "summary": (
                f"Điểm sàn 2025: THPT thấp nhất {thpt_2025}đ, "
                f"học bạ thấp nhất {hb_2025}đ. 2024: không có dữ liệu."
            ),
        },
    }


def build_diem_chuan_2026_stub(entries: list[dict]) -> dict:
    """Build ``diem_chuan_2026_full`` — stub chunk for current year.

    Since 2026 cutoff scores are not yet published, this chunk points
    users to the 2025 reference data.
    """
    # Get the top THPT score from 2025 for reference
    thpt_2025 = sorted(
        [e for e in entries if e["year"] == 2025 and e["method"] == "thi_thpt"],
        key=lambda x: -x["diem_chuan"],
    )
    top = thpt_2025[0] if thpt_2025 else None

    lines = [
        "ĐIỂM CHUẨN XÉT TUYỂN HUSC NĂM 2026",
        "",
        "Điểm chuẩn năm 2026 CHƯA được công bố tại thời điểm hiện tại.",
        "Thông tin điểm chuẩn sẽ được HUSC cập nhật trên "
        "tuyensinh.husc.edu.vn sau khi có quyết định chính thức.",
        "",
    ]
    if top:
        lines.append(
            "Tham khảo: Năm 2025, ngành có điểm chuẩn THPT cao nhất là "
            f"{top['major_name']} với {top['diem_chuan']} điểm."
        )
        lines.append(
            "Đây chỉ là dữ liệu lịch sử. Điểm chuẩn 2026 có thể thay đổi."
        )
    lines.append("")
    lines.append("Nguồn: tuyensinh.husc.edu.vn.")

    return {
        "chunk_id": "diem_chuan_2026_full",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn",
        "faq_type": "diem_chuan_2026_pending",
        "metadata": {
            "data_year": 2026,
            "info_type": "diem_chuan",
            "is_pending": True,
            "school": "HUSC",
            "summary": "Điểm chuẩn 2026 chưa công bố. Tham khảo 2025.",
        },
    }


def main() -> int:
    entries = _load_history()
    if not entries:
        print("ERROR: No entries in diem_chuan_history.json", file=sys.stderr)
        return 1

    chunks = [
        build_diem_chuan_ranking(entries),
        build_diem_chuan_2026_stub(entries),
        build_chi_tieu_2025(entries),
        build_diem_san_2024_2025(entries),
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done: {written} chunks written to {OUTPUT_FILE}")
    for c in chunks:
        print(f"  - {c['chunk_id']}: {c['metadata']['summary'][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
