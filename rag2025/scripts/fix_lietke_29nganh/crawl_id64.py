"""Crawl id=64 from HUSC tuyển sinh and parse ngành table.

KEY INSIGHT: when rowspan=1 and the cell is at a column already occupied
(continued cell from previous row), the new cell content REPLACES the carried
value. The semantics for row 6 ("Ngữ văn, Toán, Tiếng Anh | D01") is:
- col 3 (môn) gets the new value "Ngữ văn, Toán, Tiếng Anh"
- col 4 (mã tổ hợp) gets "D01"
But the existing logic keeps walking col_idx forward past filled cells. This is wrong.

Correct semantics:
- A new cell in a row is placed at the first col where:
  - that col is empty (rs_left[col]==0), OR
  - the previous carry had rowspan=1 (col is "free" after decrement)
But we cannot distinguish "carried from rowspan=5" vs "fresh from rowspan=1".

Actually, looking at the HTML structure: each ngành block has:
  - 1 "main row" with 8 cells (TT, Tên, Mã, [none], [none], Chỉ tiêu, P.thức, Ghi chú)
    where TT, Tên, Mã, Chỉ tiêu, P.thức, Ghi chú have rowspan = (n_tổ_hợp + 1)
  - N continuation rows with 2 cells (môn, mã tổ hợp)

The 2-cell continuation rows should overwrite cols 3,4. But because the main row
already filled them with "Ngữ văn, Lịch sử, Địa lý" + "C00" (rowspan=1), the
continuation rows need to OVERWRITE those single-row-span cells.

The fix: when processing each cell in a continuation row, we need to know that
cells at the START of the row (col 3, col 4 in continuation) should overwrite the
previous rowspan=1 values rather than skip them. Easiest rule: process each cell
of a row by walking col_idx from 0; skip only those cols whose cur is "active
carry" (rs_left > 0). Cols whose rs_left == 0 (whether originally empty or
already consumed) are available for placement.

In our logic, after decrement, if rs_left[c]==0 we ALREADY clear cur[c]. So col
6 "Ngữ văn, Lịch sử, Địa lý" should be cleared at row 3. But we don't — let me
check.

Looking at the row 2 placements: row 2 placed [0:'1' rs=5, 3:'Hán Nôm' rs=5,
4:'7220104' rs=5, 6:'Ngữ văn, Lịch sử, Địa lý' rs=1, 7:'C00' rs=1].
After row 2: rs[6]=0, rs[7]=0 (because rowspan=1).
At start of row 3: decrement rs. rs[0] was 4 → 3, rs[3] was 4 → 3, rs[4] was 4 → 3.
rs[6]==0, rs[7]==0 → no decrement; cur[6]='Ngữ văn, Lịch sử, Địa lý' stays.
So row 3 sees cur[6] still occupied and skips it.

Bug: rowspan=1 cells should be IMMEDIATELY cleared (after being placed) so
the next row can reuse those columns.

Fix: when placing a cell with rowspan=1, immediately set cur[col_idx]='' AFTER
the placement (because the value only persists for THIS row).
"""
import httpx
import re
import json
import sys
from bs4 import BeautifulSoup

URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id=64"
N_COLS = 8


def crawl():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    r = httpx.get(URL, headers=headers, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    return r.text


def parse_table(html: str):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("No table found")
    rows = table.find_all("tr")
    print(f"# total rows: {len(rows)}", file=sys.stderr)

    cur = [""] * N_COLS
    rs_left = [0] * N_COLS
    flat_rows = []
    for ri, tr in enumerate(rows):
        cells = tr.find_all(["td", "th"])
        # Step 1: decrement carries from previous row
        for c in range(N_COLS):
            if rs_left[c] > 0:
                rs_left[c] -= 1
                if rs_left[c] == 0:
                    cur[c] = ""
        if ri == 0:
            continue
        # Step 2: place new cells
        col_idx = 0
        for cell in cells:
            txt = cell.get_text(" ", strip=True)
            try:
                rs = int(cell.get("rowspan", "1"))
            except ValueError:
                rs = 1
            while col_idx < N_COLS and cur[col_idx]:
                col_idx += 1
            if col_idx >= N_COLS:
                break
            cur[col_idx] = txt
            if rs > 1:
                rs_left[col_idx] = rs - 1
            else:
                # Rowspan=1: this row only. Clear immediately so the
                # next row can use this column.
                cur[col_idx] = ""
                rs_left[col_idx] = 0
            col_idx += 1
        flat_rows.append((ri, cur[:]))
    return flat_rows


def main():
    html = crawl()
    flat = parse_table(html)

    by_code = {}
    for ri, fields in flat:
        if len(fields) < 6:
            continue
        ma = fields[2].strip()
        if not re.match(r"^\d{7}", ma):
            continue
        ten = fields[1].strip()
        tt = fields[0].strip()
        chi_tieu = fields[5].strip()
        phuong_thuc = fields[6].strip() if len(fields) > 6 else ""
        ghi_chu = fields[7].strip() if len(fields) > 7 else ""
        tohop_pairs = []
        if len(fields) > 4:
            mon = fields[3].strip()
            ma_th = fields[4].strip()
            if mon and ma_th and not re.match(r"^\d{7}", ma_th):
                tohop_pairs.append((mon, ma_th))
        if ma not in by_code:
            by_code[ma] = {
                "tt": tt,
                "ten": ten,
                "ma": ma,
                "chi_tieu": chi_tieu,
                "phuong_thuc": phuong_thuc,
                "ghi_chu": ghi_chu,
                "to_hop": [],
            }
        for mon, ma_th in tohop_pairs:
            if (mon, ma_th) not in by_code[ma]["to_hop"]:
                by_code[ma]["to_hop"].append((mon, ma_th))

    print(f"\n# Total unique mã ngành: {len(by_code)}", file=sys.stderr)
    for ma in sorted(
        by_code.keys(),
        key=lambda x: int(by_code[x]["tt"]) if by_code[x]["tt"].isdigit() else 999,
    ):
        n = by_code[ma]
        ths = ",".join(mt[1] for mt in n["to_hop"])
        print(
            f"  TT={n['tt']:>3} | {n['ten'][:35]:35} | mã={ma} | CT={n['chi_tieu']:>4} | TH={ths}"
        )
    print(json.dumps(by_code, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()