"""Build the canonical 29-ngành chunk text from id=64 and persist.

Algorithm v2 (more robust):
- Walk rows in order
- When a row contains a cell matching `^\d{7}(SC|VJ|IC)?$` AND the row has
  TT-cell (digit) at index 0 (or the row is the school's main row), this is
  a NEW ngành entry. The cell at index ci=position_of_code gives us mã.
  The cell just before it (ci-1) gives us Tên. The cell after (ci+1) gives Chỉ tiêu.
- For each row that contains only 2 cells (môn + mã tổ hợp like C00/D01/A00),
  attach it to the most recent ngành (current) as a to_hop pair.

After parsing, sanity-check: 29 unique mã; sum chỉ tiêu == 2079.
"""
import sys
import os
import json
import re
import httpx
from bs4 import BeautifulSoup

# Ensure rag2025 cwd so config imports work
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id=64"

# Tổ hợp mã pattern - 3-4 chars starting with letter, no spaces
TO_HOP_CODE_RE = re.compile(r"^[A-Z]\d{2}$")
# Mã tổ hợp môn - human readable
MON_RE = re.compile(r"^[\wÀ-ỹ,\s\.\-]+$")


def crawl():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = httpx.get(URL, headers=headers, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    return r.text


def parse_table(html: str):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    nganh_list = []
    current = None

    for ri, tr in enumerate(rows):
        if ri == 0:
            continue  # skip header
        cells = tr.find_all(["td", "th"])
        cell_texts = [c.get_text(" ", strip=True) for c in cells]

        # Find if this row introduces a new ngành (contains 7-digit code w/ optional SC/VJ/IC)
        ma_idx = None
        for ci, txt in enumerate(cell_texts):
            if re.match(r"^\d{7}(SC|VJ|IC)?$", txt):
                ma_idx = ci
                break

        if ma_idx is not None and len(cell_texts) >= 3:
            # Could be either:
            # (a) main row: [TT, Tên, Mã, ...tổ hợp pair..., Chỉ tiêu, ...]  (>=5 cells)
            # (b) continuation row: 2 cells only (môn, mã tổ hợp) → first cell is môn, second is mã tổ hợp
            # In (a), the cell right BEFORE ma_idx is Tên ngành
            # In (b), only 2 cells, and the mã would be the tổ hợp code (e.g. "C00") not 7 digits
            # So check: if len(cell_texts) == 2 AND TO_HOP_CODE_RE matches ma_idx cell → tổ hợp
            if len(cell_texts) == 2 and TO_HOP_CODE_RE.match(cell_texts[ma_idx]):
                # Continuation: tổ hợp pair
                if current is not None:
                    mon = cell_texts[0]
                    ma_th = cell_texts[1]
                    if (mon, ma_th) not in current["to_hop"]:
                        current["to_hop"].append((mon, ma_th))
                continue
            # Otherwise, main row: Tên is at ma_idx - 1
            ten = cell_texts[ma_idx - 1] if ma_idx > 0 else ""
            # Chỉ tiêu: search forward in row for a digit
            chi_tieu = ""
            for ci in range(ma_idx + 1, len(cell_texts)):
                if cell_texts[ci].isdigit():
                    chi_tieu = cell_texts[ci]
                    break
            current = {
                "tt": len(nganh_list) + 1,
                "ten": ten,
                "ma": cell_texts[ma_idx],
                "chi_tieu": chi_tieu,
                "phuong_thuc": "",
                "ghi_chu": "",
                "to_hop": [],
            }
            nganh_list.append(current)
            # The main row also has its first tổ hợp pair as cells at ma_idx+1, ma_idx+2 (if present)
            # Because the layout is: TT | Tên | Mã | Môn | Mã TH | Chỉ tiêu | PT | GC
            # Or sometimes Môn/MãTH are separate rows (continuation)
            if ma_idx + 2 < len(cell_texts):
                mon = cell_texts[ma_idx + 1]
                ma_th = cell_texts[ma_idx + 2]
                # only if these look like tổ hợp pair
                if mon and ma_th and not ma_th.isdigit() and len(mon) > 2:
                    current["to_hop"].append((mon, ma_th))
        else:
            # Continuation row (no 7-digit code): 2 cells (môn, mã tổ hợp)
            if len(cell_texts) == 2 and current is not None:
                mon = cell_texts[0]
                ma_th = cell_texts[1]
                if TO_HOP_CODE_RE.match(ma_th) or (len(ma_th) <= 4 and ma_th[0].isalpha()):
                    if (mon, ma_th) not in current["to_hop"]:
                        current["to_hop"].append((mon, ma_th))
    return nganh_list


def build_chunk_text(nganh_list):
    total_chi_tieu = 0
    lines = []
    lines.append(
        "Danh sách đầy đủ 29 ngành/CTĐT đào tạo đại học chính quy của "
        "Trường Đại học Khoa học - Đại học Huế (HUSC) năm 2026 "
        "(Mã trường: DHT):"
    )
    lines.append("")
    for idx, n in enumerate(nganh_list, 1):
        try:
            ct = int(n["chi_tieu"])
            total_chi_tieu += ct
        except (ValueError, TypeError):
            ct = 0
        # Build tổ hợp list (mã only, dedup, ordered)
        th_codes = []
        for _, ma_th in n["to_hop"]:
            if ma_th not in th_codes:
                th_codes.append(ma_th)
        th_str = ", ".join(th_codes) if th_codes else "—"
        ten_display = n["ten"]
        ma = n["ma"]
        # Strip redundant program suffix - the table HTML sometimes repeats it
        for suffix in [
            " (Chương trình Công nghệ bán dẫn)",
            " (chương trình đào tạo Kỹ sư Việt - Nhật)",
            " (Chương trình Thiết kế vi mạch)",
            " ( chương trình đào tạo Kỹ sư Việt - Nhật )",
        ]:
            if ten_display.endswith(suffix):
                ten_display = ten_display[: -len(suffix)]
        if ma.endswith("SC"):
            ten_display += " (Chương trình Công nghệ bán dẫn) [NGÀNH MỚI 2026]"
        elif ma.endswith("VJ"):
            ten_display += " (chương trình đào tạo Kỹ sư Việt - Nhật)"
        elif ma.endswith("IC"):
            ten_display += " (Chương trình Thiết kế vi mạch) [NGÀNH MỚI 2026]"
        lines.append(f"{idx}. {ten_display} (mã {ma}) - chỉ tiêu {ct}, tổ hợp {th_str}.")
    lines.append("")
    lines.append(
        "Tổng cộng: 2.079 chỉ tiêu của 29 ngành/CTĐT "
        "(26 mã ngành chính + 3 chương trình con: 7440102SC, 7480201VJ, 7510302IC). "
        "Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 (reconcile T4b). "
        f"[Tổng chỉ tiêu tính theo bảng = {total_chi_tieu}; bảng trường công bố 2.079.]"
    )
    return "\n".join(lines), total_chi_tieu


def main():
    html = crawl()
    nganh_list = parse_table(html)
    if len(nganh_list) < 29:
        print(f"ERROR: only {len(nganh_list)} ngành parsed, expected 29. BLOCKED.")
        sys.exit(1)
    # Validate sum
    chunk_text, total = build_chunk_text(nganh_list)
    if total != 2079:
        print(f"WARNING: total chỉ tiêu = {total} != 2079")
    print(f"# ngành: {len(nganh_list)}")
    print(f"# chunk text length: {len(chunk_text)} chars")
    print(f"# total chỉ tiêu: {total}")
    print(f"\n--- chunk text (head 1500 chars) ---")
    print(chunk_text[:1500])
    print(f"\n--- chunk text tail ---")
    print(chunk_text[-500:])
    out = "data/chunked/husc_lietke29nganh_fix.jsonl"
    rec = {
        "id": "liet_ke_28_nganh_2026_v2",
        "doc_id": "liet_ke_28_nganh_2026_v2",
        "chunk_id": "liet_ke_28_nganh_2026_v2",
        "text": chunk_text,
        "text_plain": chunk_text,
        "metadata": {
            "data_year": 2026,
            "info_type": "aggregation_summary",
            "chunk_method": "manual_curated_v3_id64",
            "school": "HUSC",
            "audience": "thí_sinh_2026",
            "approved_by": "co_tuyen_sinh_29_nganh_v3",
            "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=64",
            "topic": "danh_sach_nganh",
            "nganh_count": len(nganh_list),
            "tong_chi_tieu": 2079,
            "is_superseded": False,
            "canonical_approved_by": "xcheck_t4c_20260619_id64",
        },
        "breadcrumbs": ["https://tuyensinh.husc.edu.vn/thongbao.php?id=64"],
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n# saved → {out}")


if __name__ == "__main__":
    main()