"""@spec(V5-R013/FIX-3) Add canonical chunks for 5 câu-note đáp án
that the user explained. Idempotent upsert to rag2025 + jsonl.

Why this exists (per user note rag2025/note_5_question_chua_tra_loi_duoc.txt):
  - C1/C3 (msg011, msg080): thí sinh tự do (TN năm nào/2004 đều được) xét
    học bạ NHƯNG phải tạo TÀI KHOẢN BỘ. Existing chunks (chunked_13_chunk_2
    title-only + chunked_23_chunk_41 mốc 01/5-20/5) only had TITLE in
    context → generator fallback. Add canonical explicit chunk.
  - C5 (msg082): "Hết hạn đăng ký mà trường chưa duyệt minh chứng" — phân
    biệt "thời gian đăng ký" KHÁC "thời gian minh chứng".
  - C2/C4: handled by FIX 1 (guardrail) — no new chunks.

This script is idempotent: every add is gated on
``delete WHERE chunk_id = ?`` followed by ``table.add([row])``. Running
twice is a no-op.

Two new canonical chunks:
  - husc_tstd_xethocba_taikhoan_2026
  - husc_phanbiet_dangky_minhchung_2026

Both carry:
  data_year=2026
  info_type=thu_tuc_tsts (procedure / thi sinh tu do)
  vector dim 1024 (asserted at embed time)

Writes to data/chunked/husc_thongbao_fix_2026.jsonl for graph parity.
Commander rebuilds graph after this script runs (out of scope here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# TDD-locked canonical chunks (decision approved by user 2026-06-19)
# ---------------------------------------------------------------------------
SOURCE_URL_TSTD = "tuyensinh.husc.edu.vn/thongbao.php?id=71"
SOURCE_URL_MINHCHUNG = "tuyensinh.husc.edu.vn/thongbao.php?id=73"
APPROVAL = "v5r013_fix3_note5_20260619"

CHUNK_TSTD: str = (
    "Thí sinh tự do (đã tốt nghiệp THPT các năm trước, không phân biệt năm "
    "tốt nghiệp, bao gồm cả thí sinh tốt nghiệp năm 2004 hay các năm cũ hơn) "
    "ĐƯỢC xét tuyển bằng học bạ tại HUSC. Điều kiện BẮT BUỘC: thí sinh "
    "phải đăng ký cấp tài khoản trên Hệ thống của Bộ GD&ĐT tại địa chỉ "
    "https://thisinh.thitotnghiepthpt.edu.vn trong thời gian từ 01/5/2026 "
    "đến 20/5/2026 (mốc đăng ký HSDT và tài khoản). Quá hạn mốc trên sẽ "
    "KHÔNG tạo được tài khoản và KHÔNG thể đăng ký xét tuyển — kể cả xét "
    "học bạ. Sau khi có tài khoản trên Cổng Bộ, thí sinh dùng chính tài "
    "khoản đó để đăng nhập và tiến hành đăng ký xét tuyển (cả THPT lẫn "
    "học bạ). Lưu ý quan trọng: đăng ký tài khoản là thủ tục hành chính "
    "trên Hệ thống Bộ, hoàn toàn ĐỘC LẬP với việc đăng ký xét tuyển tại "
    "HUSC hay tại trường ĐH khác. (Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=71 "
    "— V5-R013/FIX-3 note-C1/C3.)"
)

META_TSTD: Dict[str, Any] = {
    "source": SOURCE_URL_TSTD,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "thu_tuc_tsts",
    "school": "HUSC",
    "audience": "thi_sinh_tu_do",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "tstd_xethocba_taikhoan_v1",
    "registration_window": "2026-05-01_to_2026-05-20",
    "portal_url": "https://thisinh.thitotnghiepthpt.edu.vn",
    "canonical_approved_by": APPROVAL,
    "is_superseded": False,
}


CHUNK_PHANBIET: str = (
    "Phân biệt hai mốc thời gian độc lập trong quy trình xét tuyển ĐH "
    "chính quy HUSC 2026: (1) THỜI GIAN ĐĂNG KÝ (ĐK) nguyện vọng — là "
    "khoảng thời gian thí sinh đăng ký NV trên cổng ĐKXT (thường từ "
    "02/7/2026 đến 17h ngày 14/7/2026); (2) THỜI GIAN MINH CHỨNG — là "
    "giai đoạn nhà trường xét/duyệt hồ sơ minh chứng (ưu tiên, giải "
    "thưởng, chứng chỉ ngoại ngữ, đối tượng XTT...) xem có hợp lệ không, "
    "diễn ra SAU hạn đăng ký NV. Hai quy trình này ĐỘC LẬP: việc trường "
    "chưa duyệt xong minh chứng khi đã hết hạn đăng ký NV là bình thường; "
    "thí sinh chờ trường thông báo kết quả xét minh chứng sau đó. Hết hạn "
    "ĐK NV không đồng nghĩa với việc hết cơ hội xét minh chứng. (Nguồn: "
    "tuyensinh.husc.edu.vn/thongbao.php?id=73 — V5-R013/FIX-3 note-C5.)"
)

META_PHANBIET: Dict[str, Any] = {
    "source": SOURCE_URL_MINHCHUNG,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "thu_tuc",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "phanbiet_dangky_minhchung_v1",
    "canonical_approved_by": APPROVAL,
    "is_superseded": False,
}


# Output JSONL (for graph parity). Append-only (commander rebuilds graph).
CHUNKED_JSONL_OUT: Path = REPO_ROOT / "data" / "chunked" / "husc_thongbao_fix_2026.jsonl"


# ---------------------------------------------------------------------------
# Helpers (mirrors scripts/fix_canonical_64_73.py shape)
# ---------------------------------------------------------------------------
def _delete_by_chunk_id(table, chunk_id: str) -> None:
    table.delete(f"chunk_id = '{chunk_id}'")


def _has_chunk_id(table, chunk_id: str) -> bool:
    if hasattr(table, "rows"):
        return any((r.get("chunk_id") == chunk_id) for r in table.rows)
    try:
        df = table.to_pandas()
        return chunk_id in df["chunk_id"].tolist()
    except Exception:
        return False


def _embed_text(embedder, text: str) -> List[float]:
    vec = embedder.encode_documents([text])[0]
    vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    assert len(vec_list) == 1024, f"embedding dim mismatch: got {len(vec_list)}"
    return vec_list


def _build_row(
    chunk_id: str, text: str, meta: Dict[str, Any], vector: List[float],
    source: str, faq_type: str = "",
) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "faq_type": faq_type,
        "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
        "vector": vector,
    }


def _append_jsonl(rows: List[Dict[str, Any]]) -> int:
    CHUNKED_JSONL_OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with CHUNKED_JSONL_OUT.open("a", encoding="utf-8") as fh:
        for row in rows:
            payload = {
                "id": row["chunk_id"],
                "doc_id": row["chunk_id"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "text_plain": row["text"],
                "metadata": json.loads(row["metadata_json"]),
                "breadcrumbs": [row["source"]],
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Pure logic (testable without DB)
# ---------------------------------------------------------------------------
def plan_changes(embedder=None, table=None, dry_run: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "added": [],
        "skipped": [],
        "rows_added": 0,
        "rows_deleted": 0,
        "dry_run": dry_run,
        "jsonl_written": 0,
    }

    plan = [
        ("husc_tstd_xethocba_taikhoan_2026", CHUNK_TSTD, META_TSTD,
         SOURCE_URL_TSTD, "thu_tuc_tsts"),
        ("husc_phanbiet_dangky_minhchung_2026", CHUNK_PHANBIET, META_PHANBIET,
         SOURCE_URL_MINHCHUNG, "thu_tuc"),
    ]

    if dry_run:
        summary["added"] = [cid for cid, *_ in plan]
        return summary

    assert embedder is not None and table is not None, "live mode needs embedder + table"

    jsonl_rows: List[Dict[str, Any]] = []
    for cid, text, meta, source, faq in plan:
        # Idempotent: if already present (and the data is correct), no-op.
        if _has_chunk_id(table, cid):
            summary["skipped"].append(cid)
            continue
        # delete-by-id (defensive: 0 or 1 affected)
        _delete_by_chunk_id(table, cid)
        summary["rows_deleted"] += 1
        vec = _embed_text(embedder, text)
        row = _build_row(cid, text, meta, vec, source, faq_type=faq)
        table.add([row])
        summary["rows_added"] += 1
        summary["added"].append(cid)
        jsonl_rows.append(row)

    summary["jsonl_written"] = _append_jsonl(jsonl_rows)
    return summary


# ---------------------------------------------------------------------------
# Live entry point
# ---------------------------------------------------------------------------
def run_against_live_db(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return plan_changes(dry_run=True)

    import lancedb

    from config.settings import RAGSettings
    from src.services.embedding import EmbeddingService

    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)

    return plan_changes(embedder=embedder, table=table, dry_run=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="V5-R013 FIX 3: idempotent upsert of 2 canonical chunks "
                    "(tstd xet hoc ba tai khoan + phan biet dang ky minh chung)"
    )
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    args = p.parse_args()

    summary = run_against_live_db(dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))