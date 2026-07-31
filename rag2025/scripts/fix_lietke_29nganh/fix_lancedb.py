"""Idempotent fix for chunk liet_ke_28_nganh_2026_v2 in LanceDB.

Steps:
1. Load canonical chunk text from data/chunked/husc_lietke29nganh_fix.jsonl
2. For each target chunk_id:
   - Check existing row's mã ngành count
   - If < 29: delete + embed new + insert
   - If >= 29: idempotent skip (no-op)
3. Targets: liet_ke_28_nganh_2026_v2 AND bang_xep_hang_chi_tieu_2026_v2
"""
import sys
import os
import json
import re
import lancedb

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService


def load_fix_chunk():
    p = "data/chunked/husc_lietke29nganh_fix.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    return rec


def count_codes(text: str) -> int:
    codes = re.findall(r"\b\d{7}(?:SC|VJ|IC)?\b", text)
    return len(set(codes))


def delete_chunk(tbl, chunk_id: str) -> int:
    """Return number of rows deleted (best-effort)."""
    try:
        before = tbl.count_rows()
        tbl.delete(f"chunk_id = '{chunk_id}'")
        after = tbl.count_rows()
        return before - after
    except Exception as e:
        print(f"  delete via filter failed: {e}")
        return 0


def get_table():
    db = lancedb.connect("data/lancedb")
    return db.open_table("rag2025")


def build_bang_xep_hang_text(chunk_text: str) -> str:
    line_re = re.compile(
        r"^\s*(\d+)\.\s+(.+?)\s+\(mã\s+(\d{7}(?:SC|VJ|IC)?)\)\s*-\s*chỉ tiêu\s+(\d+)",
        re.UNICODE | re.MULTILINE,
    )
    entries = []
    for m in line_re.finditer(chunk_text):
        idx, ten, ma, ct = (
            int(m.group(1)),
            m.group(2).strip(),
            m.group(3),
            int(m.group(4)),
        )
        entries.append((idx, ten, ma, ct))
    entries.sort(key=lambda x: -x[3])
    lines = ["Bảng xếp hạng 29 ngành/CTĐT HUSC 2026 theo chỉ tiêu giảm dần:"]
    for i, (_, ten, ma, ct) in enumerate(entries, 1):
        lines.append(f"{i}. {ten} (mã {ma}) - chỉ tiêu {ct}.")
    lines.append(
        "Tổng: 2.079 chỉ tiêu, 29 ngành/CTĐT (26 mã chính + 3 CT con: 7440102SC, 7480201VJ, 7510302IC). "
        "Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 (reconcile T4b)."
    )
    return "\n".join(lines), len(entries)


def fix_chunk(tbl, e, chunk_id: str, text: str, faq_type: str, metadata: dict) -> bool:
    """Returns True if a change was applied."""
    df = tbl.to_pandas()
    rows = df[df["chunk_id"] == chunk_id]
    if len(rows) > 0:
        existing_codes = count_codes(rows.iloc[0]["text"])
        if existing_codes >= 29:
            print(f"[fix]   {chunk_id}: idempotent SKIP (existing {existing_codes} mã)")
            return False
    # Need rebuild
    print(f"[fix]   {chunk_id}: rebuilding (existing had <29 mã or missing)")
    delete_chunk(tbl, chunk_id)
    vec = e.encode_query(text)
    assert len(vec) == 1024, f"vec len {len(vec)} != 1024"
    new_row = {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": "https://tuyensinh.husc.edu.vn/thongbao.php?id=64",
        "faq_type": faq_type,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "vector": vec,
    }
    tbl.add([new_row])
    print(f"[fix]   {chunk_id}: inserted ({len(text)} chars, {count_codes(text)} mã)")
    return True


def main():
    rec = load_fix_chunk()
    chunk_id_lk = rec["chunk_id"]
    chunk_text_lk = rec["text"]
    metadata_lk = rec["metadata"]
    new_code_count = count_codes(chunk_text_lk)
    print(f"[fix] new liet_ke chunk has {new_code_count} unique mã ngành")
    if new_code_count < 29:
        print(f"ERROR: BLOCKED. new chunk only has {new_code_count} mã.")
        sys.exit(1)

    tbl = get_table()
    print(f"[fix] table rows before: {tbl.count_rows()}")

    s = RAGSettings()
    e = EmbeddingService(s)

    # 1) fix liet_ke
    fix_chunk(tbl, e, chunk_id_lk, chunk_text_lk, "danh_sach_nganh", metadata_lk)

    # 2) build + fix bang_xep_hang
    bang_text, bang_count = build_bang_xep_hang_text(chunk_text_lk)
    bang_meta = {
        "data_year": 2026,
        "info_type": "ranking_by_chi_tieu",
        "chunk_method": "manual_curated_v3_id64",
        "school": "HUSC",
        "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=64",
        "nganh_count": bang_count,
        "tong_chi_tieu": 2079,
        "canonical_approved_by": "xcheck_t4c_20260619_id64",
    }
    fix_chunk(tbl, e, "bang_xep_hang_chi_tieu_2026_v2", bang_text, "xep_hang_nganh", bang_meta)

    print(f"[fix] table rows after: {tbl.count_rows()}")

    # Final verification
    final = tbl.to_pandas()
    for cid in [chunk_id_lk, "bang_xep_hang_chi_tieu_2026_v2"]:
        rows = final[final["chunk_id"] == cid]
        if len(rows) > 0:
            t = rows.iloc[0]["text"]
            print(f"[fix]   {cid}: {len(t)} chars, {count_codes(t)} mã unique, vec_len={len(rows.iloc[0]['vector'])}")


if __name__ == "__main__":
    main()