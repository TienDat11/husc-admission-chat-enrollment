"""Audit 2025/2026 ngành leakage — verify 2026 answers don't include 2025-only ngành.

Checks:
  1. Whitelist 2026 ngành (28 codes from nganh_detail_2026.jsonl)
  2. Forbidden patterns (2025-only ngành/text)
  3. Scan blue table husc_v2026_clean for leaks
  4. Scan 86Q answers for leaks
"""
import json
from pathlib import Path
import lancedb

ROOT = Path(__file__).parent.parent

# Step 1: Build whitelist
codes_2026 = set()
for line in (ROOT / "data" / "chunked" / "nganh_detail_2026.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    cid = obj["chunk_id"].replace("nganh_detail_", "").replace("_2026", "")
    codes_2026.add(cid)

print(f"=== 28 ngành 2026 whitelist ===")
for c in sorted(codes_2026):
    print(f"  - {c}")
print()

# Forbidden = 2025-only ngành no longer offered in 2026
# Known: Quản trị và phân tích dữ liệu (7340405), Toán ứng dụng (7460112?), Hóa phân tích?
FORBIDDEN_CODES = ["7340405", "7480107TD"]
FORBIDDEN_PHRASES = [
    "Quản trị và phân tích dữ liệu",
    "Quản trị, phân tích dữ liệu",
]

# Step 2: Scan blue table
print(f"=== Blue table husc_v2026_clean leak audit ===")
db = lancedb.connect(str(ROOT / "data" / "lancedb"))
t = db.open_table("husc_v2026_clean")
df = t.to_pandas()
print(f"Total rows: {len(df)}")

leaks = []
for idx, row in df.iterrows():
    cid = row["id"]
    text = str(row.get("text") or "")
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    year = meta.get("data_year") if meta else None
    info_type = meta.get("info_type") if meta else None

    # Allowed exceptions: 2025-reference docs (thongbao 40,49,50,53 + baiviet)
    is_2025_ref = (year == 2025) or (info_type in ("baiviet", "vision_ocr"))

    for pat in FORBIDDEN_CODES + FORBIDDEN_PHRASES:
        if pat in text:
            if is_2025_ref:
                continue  # OK - allowed in 2025 reference docs
            leaks.append({
                "chunk_id": cid,
                "year": year,
                "info_type": info_type,
                "pattern": pat,
                "text_snippet": text[max(0, text.find(pat)-40):text.find(pat)+80],
            })

if leaks:
    print(f"!! FOUND {len(leaks)} 2025-leak in 2026 chunks !!")
    for L in leaks[:10]:
        print(f"  - id={L['chunk_id']} year={L['year']} info_type={L['info_type']}")
        print(f"    pat={L['pattern']!r} | ctx: ...{L['text_snippet']!r}...")
else:
    print("OK No 2025-only ngành leaked into 2026 chunks.")
print()

# Step 3: Scan 86Q answers
print(f"=== 86Q answer leak audit ===")
results = json.loads((ROOT / "results" / "86q_thi_sinh_results.json").read_text(encoding="utf-8"))["results"]

# Questions about 2025 historical data — leak là legitimate (đang trả lời lịch sử 2025)
HISTORY_2025_QIDS = {"msg027", "msg028", "msg048"}

# Disclaimer phrases that indicate the leak is intentional ("đã ngừng" / "đã ngừng tuyển" / "không còn")
DISCLAIMER_PHRASES = [
    "đã ngừng",
    "đã ngừng tuyển",
    "không còn tuyển",
    "không còn nằm",
    "ngừng tuyển từ",
    "thay bằng ngành",
    "năm 2025",  # 'năm 2025' adjacent to mention => historical reference
]

ans_leaks = []
for r in results:
    qid = r["id"]
    if qid in HISTORY_2025_QIDS:
        continue  # Question about 2025 — historical leak is legitimate
    ans = r.get("answer", "")
    for pat in FORBIDDEN_CODES + FORBIDDEN_PHRASES:
        idx = ans.find(pat)
        if idx == -1:
            continue
        # Check if a disclaimer phrase appears within ±200 chars of the leak
        ctx_window = ans[max(0, idx-200):idx+200].lower()
        if any(d in ctx_window for d in DISCLAIMER_PHRASES):
            continue  # Intentional disclaimer leak — OK
        ans_leaks.append({
            "id": qid,
            "question": r["question"][:60],
            "pattern": pat,
            "snippet": ans[max(0, idx-40):idx+100],
        })

if ans_leaks:
    print(f"!! FOUND {len(ans_leaks)} 2025-leak in 86Q answers !!")
    for L in ans_leaks:
        print(f"  - {L['id']}: {L['question']}")
        print(f"    pat={L['pattern']!r} | ctx: ...{L['snippet']!r}")
else:
    print("OK No 2025-only ngành leaked into 86Q answers.")
print()

# Step 4: Validate 28-ngành count in liet_ke chunk
print(f"=== 'liet_ke_26_nganh_2026' content audit ===")
liet_ke = df[df["id"] == "liet_ke_26_nganh_2026"]
if len(liet_ke):
    text = liet_ke.iloc[0]["text"]
    found_codes = []
    for c in codes_2026:
        if c in text:
            found_codes.append(c)
    missing = codes_2026 - set(found_codes)
    print(f"  Codes found in liet_ke chunk: {len(found_codes)}/{len(codes_2026)}")
    if missing:
        print(f"  MISSING: {sorted(missing)}")
    else:
        print(f"  OK All 28 codes present.")
else:
    print(f"  WARNING liet_ke_26_nganh_2026 chunk not found in blue table")

# Step 5: Validate the chunk text doesn't claim "26 ngành" instead of 28
print()
print(f"=== '26 ngành' vs '28 ngành' wording audit ===")
for kw in ["26 ngành", "28 ngành"]:
    hits = df[df["text"].str.contains(kw, na=False, regex=False)]
    print(f"  '{kw}': {len(hits)} chunks contain this phrase")
    for _, row in hits.head(5).iterrows():
        print(f"    - {row['id']}")
