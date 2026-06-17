"""Find which chunks in blue table contain 2025-only ngành (root cause of leakage)."""
import json
from pathlib import Path
import lancedb

ROOT = Path(__file__).parent.parent
db = lancedb.connect(str(ROOT / "data" / "lancedb"))
t = db.open_table("husc_v2026_clean")
df = t.to_pandas()

PATTERNS = ["Quản trị và phân tích dữ liệu", "7480107TD", "7340405", "26 ngành"]

print(f"Total rows: {len(df)}")
for pat in PATTERNS:
    print(f"\n=== Pattern: {pat!r} ===")
    hits = df[df["text"].str.contains(pat, na=False, regex=False)]
    print(f"  {len(hits)} hits")
    for _, row in hits.iterrows():
        cid = row["id"]
        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
        year = meta.get("data_year")
        info_type = meta.get("info_type")
        text = row["text"]
        idx = text.find(pat)
        snippet = text[max(0, idx-30):idx+80].replace("\n", " ")
        print(f"  - {cid} | year={year} | info_type={info_type}")
        print(f"    ctx: ...{snippet}...")
