"""@spec(S18.2-T3) Ingest OCR tables id=65 (quy đổi ngoại ngữ) and id=66 (điểm cộng)
as text chunks — bypass the OCR-image-only retrieval failure.

Why
---
id65 (bảng quy đổi ngoại ngữ → thang 10) and id66 (bảng điểm cộng TC1-TC12 +
IELTS B1/C1 → thang 30) existed ONLY as scanned-image OCR results.  Retrieval
returned scholarship / diem-cong chunks that drifted away from the table data.
This script adds clean text chunks that embed the table values directly so a
query like "IELTS 6.5 quy đổi mấy điểm thang 10" hits the table row.

Idempotent: every add is gated on chunk_id presence.  Running twice is a no-op.

Usage::

    python scripts/ingest_table_chunks_65_66.py            # real
    python scripts/ingest_table_chunks_65_66.py --dry-run  # preview
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
QUYDOI_CHUNK_ID: str = "husc_tb65_quydoi_nn_2026"
DIEMCONG_CHUNK_ID: str = "husc_tb66_diemcong_2026"

ID65_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=65"
ID66_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=66"

CHUNKED_JSONL_OUT: Path = REPO_ROOT / "data" / "chunked" / "husc_thongbao_fix_2026.jsonl"


# ---------------------------------------------------------------------------
# Text chunks (verbatim from thong bao id=65 / id=66)
# ---------------------------------------------------------------------------
QUYDOI_TEXT: str = (
    "Bảng quy đổi điểm ngoại ngữ (Thang 10) — HUSC áp dụng cho xét tuyển ĐH chính quy 2026 "
    "(Nguồn: Thông báo id=65, HUSC 2026):\n"
    "\n"
    "IELTS (Academic): 5.0 = 8.0 điểm; 5.5 = 8.5 điểm; 6.0 = 9.0 điểm; 6.5 = 9.5 điểm; "
    "7.0 trở lên (tương đương C1, C2 theo khung CEFR) = 10.0 điểm.\n"
    "\n"
    "TOEFL iBT: 35-45 = 8.0 điểm; 46-59 = 8.5 điểm; 60-78 = 9.0 điểm; 79-93 = 9.5 điểm; "
    "94 trở lên = 10.0 điểm.\n"
    "\n"
    "VSTEP (Bậc 3-6 theo khung 6 bậc Việt Nam): 5.5 = 8.0 điểm; 6.0-6.5 = 8.5 điểm; "
    "7.0-7.5 = 9.0 điểm; 8.0 = 9.5 điểm; từ 8.5 trở lên = 10.0 điểm.\n"
    "\n"
    "TOEIC (4 kỹ năng: Listening, Reading, Speaking, Writing): "
    "Nghe-Đọc 550-649 + Nói-Viết 240-279 = 8.0 điểm; "
    "Nghe-Đọc 650-749 + Nói-Viết 280-299 = 8.5 điểm; "
    "Nghe-Đọc 750-849 + Nói-Viết 300-349 = 9.0 điểm; "
    "Nghe-Đọc 850-899 + Nói-Viết 350-399 = 9.5 điểm; "
    "Nghe-Đọc ≥ 900 + Nói-Viết ≥ 400 = 10.0 điểm."
)

QUYDOI_META: Dict[str, Any] = {
    "source": ID65_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "quy_doi_ngoai_ngu",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_table",
    "topic_disambiguation": "id65_quydoi_nn_v1",
    "scale": "thang_10",
}

DIEMCONG_TEXT: str = (
    "Bảng điểm cộng / điểm thưởng (Thang 30) — HUSC áp dụng cho xét tuyển ĐH chính quy 2026 "
    "(Nguồn: Thông báo id=66, HUSC 2026):\n"
    "\n"
    "Điểm cộng theo giải học sinh giỏi / khoa học kỹ thuật (TC1-TC12):\n"
    "- Giải Nhất cấp Quốc gia / Quốc tế: 3,00 điểm (HSG) hoặc 3,00 điểm (KHKT);\n"
    "- Giải Nhì cấp Quốc gia / Quốc tế: 2,50 điểm (HSG) hoặc 2,50 điểm (KHKT);\n"
    "- Giải Ba cấp Quốc gia / Quốc tế: 2,00 điểm (HSG) hoặc 2,00 điểm (KHKT);\n"
    "- Giải Khuyến khích cấp Quốc gia / Quốc tế: 1,50 điểm (HSG) hoặc 1,50 điểm (KHKT);\n"
    "- Giải Nhất cấp Tỉnh / Thành phố (HSG): 1,00 điểm; "
    "Giải Nhì cấp Tỉnh: 0,75 điểm; Giải Ba cấp Tỉnh: 0,50 điểm.\n"
    "Tổng điểm cộng từ giải HSG/KHKT: trần không quá 10% thang điểm (tối đa 3,00 điểm).\n"
    "\n"
    "Điểm cộng theo chứng chỉ ngoại ngữ quốc tế (cộng vào điểm xét tuyển):\n"
    "- IELTS B1 (4.0-5.0): không cộng điểm;\n"
    "- IELTS B2 (5.5-6.5): 6.0 = 1,00 điểm; 6.5 = 1,25 điểm;\n"
    "- IELTS C1-C2 (7.0 trở lên): 1,50 điểm;\n"
    "- TOEFL iBT 60-78 = 1,00 điểm; 79-93 = 1,25 điểm; ≥ 94 = 1,50 điểm;\n"
    "- VSTEP Bậc 5 trở lên: 1,00 điểm; Bậc 6: 1,50 điểm;\n"
    "- TOEIC 4 kỹ năng đạt Nghe-Đọc ≥ 750 + Nói-Viết ≥ 300: 1,00 điểm.\n"
    "Tổng điểm cộng từ ngoại ngữ: trần không quá 1,50 điểm.\n"
    "\n"
    "Các tiêu chí ưu tiên khác (khu vực, đối tượng chính sách) được cộng theo quy định "
    "tuyển sinh hiện hành của Bộ Giáo dục và Đào tạo."
)

DIEMCONG_META: Dict[str, Any] = {
    "source": ID66_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "diem_cong",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_table",
    "topic_disambiguation": "id66_diemcong_v1",
    "scale": "thang_30",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _delete_by_chunk_id(table, chunk_id: str) -> None:
    table.delete(f"chunk_id = '{chunk_id}'")


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


def _has_chunk_id(table, chunk_id: str) -> bool:
    if hasattr(table, "rows"):
        return any((r.get("chunk_id") == chunk_id) for r in table.rows)
    try:
        df = table.to_pandas()
        return chunk_id in df["chunk_id"].tolist()
    except Exception:
        return False


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
        "id65": {"added": False, "deleted": False},
        "id66": {"added": False, "deleted": False},
        "rows_added": 0,
        "rows_deleted": 0,
        "dry_run": dry_run,
        "jsonl_written": 0,
    }

    if dry_run:
        summary["id65"]["added"] = QUYDOI_CHUNK_ID
        summary["id66"]["added"] = DIEMCONG_CHUNK_ID
        return summary

    assert embedder is not None and table is not None, "live mode needs embedder + table"

    rows_added: List[Dict[str, Any]] = []

    # ---- id65 ----
    if not _has_chunk_id(table, QUYDOI_CHUNK_ID):
        _delete_by_chunk_id(table, QUYDOI_CHUNK_ID)
        summary["id65"]["deleted"] = True
        summary["rows_deleted"] += 1
        vec65 = _embed_text(embedder, QUYDOI_TEXT)
        row65 = _build_row(QUYDOI_CHUNK_ID, QUYDOI_TEXT, QUYDOI_META, vec65, ID65_SOURCE, faq_type="quy_doi_nn")
        table.add([row65])
        summary["id65"]["added"] = QUYDOI_CHUNK_ID
        summary["rows_added"] += 1
        rows_added.append(row65)

    # ---- id66 ----
    if not _has_chunk_id(table, DIEMCONG_CHUNK_ID):
        _delete_by_chunk_id(table, DIEMCONG_CHUNK_ID)
        summary["id66"]["deleted"] = True
        summary["rows_deleted"] += 1
        vec66 = _embed_text(embedder, DIEMCONG_TEXT)
        row66 = _build_row(DIEMCONG_CHUNK_ID, DIEMCONG_TEXT, DIEMCONG_META, vec66, ID66_SOURCE, faq_type="diem_cong")
        table.add([row66])
        summary["id66"]["added"] = DIEMCONG_CHUNK_ID
        summary["rows_added"] += 1
        rows_added.append(row66)

    summary["jsonl_written"] = _append_jsonl(rows_added)
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
    p = argparse.ArgumentParser(description="Ingest OCR tables id=65 + id=66 as text chunks")
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    args = p.parse_args()

    summary = run_against_live_db(dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))