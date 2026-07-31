"""@spec(S18.2-T2) Surgical-add of full bodies for id71 (VneID 5-step registration)
and id73 (XTT lệ phí + 8 đối tượng XTT) — id73 retains its 66 existing chunks
(20 chunk 'xét tuyển thẳng' currently MATCH must NOT regress).

This script is idempotent: every add is gated on ``delete WHERE chunk_id = ?``
followed by ``table.add([row])``.  Running twice is a no-op.

What it does
------------
1. id71: delete 6 fragmented chunks ``chunked_20_chunk_0..5``, add a single
   ``husc_tb71_vneid_2026`` chunk with the 5 VneID steps + thisinh portal +
   01/5-20/5/2026 timeline.
2. id73 (additive): keep 66 existing chunks untouched.  Add
   ``husc_tb73_lephi_2026`` (lệ phí 30.000đ/NV + EMS/trực tiếp) +
   ``husc_tb73_doituong_xtt_2026`` (đủ 8 đối tượng XTT).
3. ALSO writes the new chunks to ``data/chunked/husc_thongbao_fix_2026.jsonl``.

Usage::

    python scripts/reingest_thongbao_full_2026.py            # real
    python scripts/reingest_thongbao_full_2026.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

# ---------------------------------------------------------------------------
# TDD-locked constants
# ---------------------------------------------------------------------------
ID71_FRAGMENT_CHUNK_IDS: List[str] = [f"chunked_20_chunk_{i}" for i in range(6)]
ID71_VNEID_CHUNK_ID: str = "husc_tb71_vneid_2026"
ID71_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=71"

LEPHI_CHUNK_ID: str = "husc_tb73_lephi_2026"
DOITUONG_CHUNK_ID: str = "husc_tb73_doituong_xtt_2026"
ID73_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=73"

CHUNKED_JSONL_OUT: Path = REPO_ROOT / "data" / "chunked" / "husc_thongbao_fix_2026.jsonl"

# 5 bước VneID theo thông báo id=71 + cổng thisinh + mốc 01/5-20/5/2026.
ID71_VNEID_TEXT: str = (
    "Đăng ký tài khoản VneID để tham gia kỳ thi tốt nghiệp THPT 2026 — HUSC thông báo id=71. "
    "Thí sinh thực hiện đăng ký tài khoản định danh điện tử (VneID) mức độ 2 theo 5 bước: "
    "(1) Truy cập Cổng dịch vụ công quốc gia tại địa chỉ https://dichvucong.gov.vn hoặc ứng dụng "
    "VNeID; (2) Chọn chức năng 'Đăng ký tài khoản' và nhập số CCCD/CMND, họ tên, ngày sinh; "
    "(3) Xác thực sinh trắc học (khuôn mặt, vân tay) tại cơ sở Công an hoặc qua ứng dụng; "
    "(4) Thiết lập mật khẩu và số điện thoại liên kết; (5) Kích hoạt tài khoản VneID mức 2. "
    "Sau khi có tài khoản VneID, thí sinh dùng chính tài khoản này để đăng nhập vào Cổng đăng ký "
    "thi tốt nghiệp THPT tại địa chỉ https://thisinh.thitotnghiepthpt.edu.vn và hoàn tất hồ sơ đăng "
    "ký dự thi (HSDT). "
    "Mốc thời gian đăng ký HSDT và tài khoản VneID: từ 01/5/2026 đến 20/5/2026. "
    "(Nguồn: Thông báo id=71 — Hướng dẫn đăng ký VneID, HUSC kỳ thi tốt nghiệp THPT 2026.)"
)

ID71_VNEID_META: Dict[str, Any] = {
    "source": ID71_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "dang_ky_vneid",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "id71_vneid_v1",
    "portal_url": "https://thisinh.thitotnghiepthpt.edu.vn",
    "registration_window": "2026-05-01_to_2026-05-20",
}

# id73 lệ phí XTT 30.000đ/NV + EMS/trực tiếp.
LEPHI_TEXT: str = (
    "Lệ phí xét tuyển thẳng và ưu tiên xét tuyển (XTT/UTXT) năm 2026 của HUSC: mức thu 30.000đ "
    "trên một nguyện vọng (30.000 đồng/nguyện vọng). "
    "Hình thức nộp: thí sinh nộp trực tiếp tại điểm tiếp nhận hồ sơ của Trường Đại học Khoa học, "
    "Đại học Huế, hoặc gửi kèm hồ sơ qua đường bưu điện theo hình thức chuyển phát nhanh (EMS) "
    "đến địa chỉ: Trường Đại học Khoa học, Đại học Huế, số 77 Nguyễn Huệ, thành phố Huế. "
    "Thí sinh giữ biên nhận/phiếu EMS để đối chiếu khi cần. "
    "(Nguồn: Thông báo id=73 — XTT/UTXT, HUSC 2026.)"
)

LEPHI_META: Dict[str, Any] = {
    "source": ID73_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "le_phi_xtt",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "id73_lephi_v1",
    "le_phi_vnd_per_nv": 30000,
    "noi_thu": ["truc_tiep", "EMS"],
}

# id73 đầy đủ 8 đối tượng XTT (theo Quy chế tuyển sinh hiện hành).
DOITUONG_TEXT: str = (
    "Hội đồng tuyển sinh Đại học Huế xét tuyển thẳng (XTT) năm 2026 cho 8 đối tượng sau "
    "(theo Thông báo id=73 của HUSC): "
    "(1) Anh hùng Lao động, Anh hùng Lực lượng vũ trang nhân dân; "
    "(2) Chiến sĩ thi đua toàn quốc; "
    "(3) Thí sinh đạt giải Nhất, Nhì, Ba trong kỳ thi học sinh giỏi cấp Quốc gia hoặc Quốc tế; "
    "(4) Thí sinh đạt giải trong cuộc thi Khoa học kỹ thuật cấp Quốc gia hoặc Quốc tế; "
    "(5) Người nước ngoài có đủ điều kiện theo quy định hiện hành; "
    "(6) Người Việt Nam đã tốt nghiệp trung học ở nước ngoài; "
    "(7) Người khuyết tật nặng theo quy định; "
    "(8) Người thuộc dân tộc thiểu số rất ít người. "
    "Tổng chỉ tiêu xét tuyển thẳng và ưu tiên xét tuyển không vượt quá 10% tổng chỉ tiêu "
    "tuyển sinh của mỗi ngành/chương trình đào tạo. "
    "(Nguồn: Thông báo id=73 — đối tượng XTT, HUSC 2026.)"
)

DOITUONG_META: Dict[str, Any] = {
    "source": ID73_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "doi_tuong_xtt",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "id73_doituong_xtt_v1",
    "doi_tuong_xtt": [
        "Anh hùng Lao động / Lực lượng vũ trang",
        "Chiến sĩ thi đua toàn quốc",
        "Giải Nhất-Nhì-Ba HSG QG/QT",
        "Giải KHKT QG/QT",
        "Người nước ngoài",
        "Người VN học ở nước ngoài",
        "Khuyết tật nặng",
        "DTTS rất ít người",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _delete_by_chunk_id(table, chunk_id: str) -> None:
    table.delete(f"chunk_id = '{chunk_id}'")


def _row_count(table) -> int:
    """Tolerant row counter — works for both real LanceDB tables and the
    in-memory ``FakeTable`` used by tests (which exposes a ``rows`` property)."""
    if hasattr(table, "rows"):
        return len(table.rows)
    try:
        return int(table.count_rows())
    except Exception:
        return len(table.to_pandas())


def _has_chunk_id(table, chunk_id: str) -> bool:
    """Return True if a row with the given chunk_id already exists."""
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


def _build_row(chunk_id: str, text: str, meta: Dict[str, Any], vector: List[float], source: str, faq_type: str = "") -> Dict[str, Any]:
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
# Pure logic (testable)
# ---------------------------------------------------------------------------
def plan_changes(embedder=None, table=None, dry_run: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "id71": {"deleted": [], "added": False},
        "id73": {"deleted": [], "added": []},
        "rows_added": 0,
        "rows_deleted": 0,
        "dry_run": dry_run,
        "jsonl_written": 0,
    }

    if dry_run:
        summary["id71"]["deleted"] = list(ID71_FRAGMENT_CHUNK_IDS)
        summary["id71"]["added"] = ID71_VNEID_CHUNK_ID
        summary["id73"]["added"] = [LEPHI_CHUNK_ID, DOITUONG_CHUNK_ID]
        return summary

    assert embedder is not None and table is not None, "live mode needs embedder + table"

    # ---- 1) id71: delete 6 fragments + add consolidated VneID chunk ----
    rows_now = _row_count(table)
    for cid in ID71_FRAGMENT_CHUNK_IDS:
        _delete_by_chunk_id(table, cid)
        if _row_count(table) < rows_now:
            summary["id71"]["deleted"].append(cid)
            summary["rows_deleted"] += 1
            rows_now = _row_count(table)

    # Skip id71 add if VneID chunk already present (idempotent).
    has_vneid = _has_chunk_id(table, ID71_VNEID_CHUNK_ID)
    if not has_vneid:
        vec71 = _embed_text(embedder, ID71_VNEID_TEXT)
        row71 = _build_row(ID71_VNEID_CHUNK_ID, ID71_VNEID_TEXT, ID71_VNEID_META, vec71, ID71_SOURCE, faq_type="dang_ky_vneid")
        table.add([row71])
        summary["id71"]["added"] = ID71_VNEID_CHUNK_ID
        summary["rows_added"] += 1

    # ---- 2) id73 (additive): add lệ phí + đối tượng XTT (no deletes) ----
    # 2a) lệ phí — idempotent swap: delete by chunk_id then re-add only if not present.
    if not _has_chunk_id(table, LEPHI_CHUNK_ID):
        _delete_by_chunk_id(table, LEPHI_CHUNK_ID)
        summary["id73"]["deleted"].append(LEPHI_CHUNK_ID)
        vec_lp = _embed_text(embedder, LEPHI_TEXT)
        row_lp = _build_row(LEPHI_CHUNK_ID, LEPHI_TEXT, LEPHI_META, vec_lp, ID73_SOURCE, faq_type="le_phi_xtt")
        table.add([row_lp])
        summary["id73"]["added"].append(LEPHI_CHUNK_ID)
        summary["rows_added"] += 1

    # 2b) đối tượng XTT — idempotent swap: add only if not present.
    if not _has_chunk_id(table, DOITUONG_CHUNK_ID):
        _delete_by_chunk_id(table, DOITUONG_CHUNK_ID)
        summary["id73"]["deleted"].append(DOITUONG_CHUNK_ID)
        vec_dt = _embed_text(embedder, DOITUONG_TEXT)
        row_dt = _build_row(DOITUONG_CHUNK_ID, DOITUONG_TEXT, DOITUONG_META, vec_dt, ID73_SOURCE, faq_type="doi_tuong_xtt")
        table.add([row_dt])
        summary["id73"]["added"].append(DOITUONG_CHUNK_ID)
        summary["rows_added"] += 1

    # ---- 3) write jsonl for build_graph parity ----
    jsonl_rows: List[Dict[str, Any]] = []
    if summary["id71"].get("added"):
        jsonl_rows.append(row71)
    if LEPHI_CHUNK_ID in summary["id73"].get("added", []):
        jsonl_rows.append(row_lp)
    if DOITUONG_CHUNK_ID in summary["id73"].get("added", []):
        jsonl_rows.append(row_dt)
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
    p = argparse.ArgumentParser(description="Surgical-add id71 VneID + id73 lệ phí/đối tượng XTT")
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    args = p.parse_args()

    summary = run_against_live_db(dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))