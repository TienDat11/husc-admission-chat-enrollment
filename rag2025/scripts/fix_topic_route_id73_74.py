"""@spec(S18.2-T1) Topic-disambiguation fix for id73 (XTT route=offline) and id74 (ĐK XT online).

Root cause (architect review, L2):
  chunked_22_chunk_54 (id73, year=2026) is the LEGITIMATE XTT-offline route
  describing how to nộp lệ phí (EMS/trực tiếp).  The leak ATTRIBUTION is
  topic: chunks about nộp-EMS leak into queries about ĐK-XT-online (id74).
  Fix = add a co-equal chunk for the id74 route=online so the offline chunk
  no longer "owns" the ĐK-XT topic.

This script is idempotent: every add is gated on ``delete WHERE chunk_id = ?``
followed by ``table.add([row])``.  Running twice is a no-op.

What it does
------------
1. ``chunked_22_chunk_54`` (id73, year=2026, route=offline):
   - delete-by-chunk_id then RE-ADD with ``info_type=xtt_nop_ho_so``,
     ``route=offline`` metadata.  Same text (the EMS sentence is correct).
2. NEW ``husc_tb74_dkxt_online_2026`` (id74, year=2026, route=online):
   - URL dkxt.hueuni.edu.vn + ĐK NV mốc 02/7→17h 14/7/2026.
3. ALSO writes the new chunk to ``data/chunked/husc_thongbao_fix_2026.jsonl``
   so ``build_graph.py`` can pick it up.

Usage::

    python scripts/fix_topic_route_id73_74.py            # real
    python scripts/fix_topic_route_id73_74.py --dry-run  # preview only
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
CHUNK_54_ID: str = "chunked_22_chunk_54"
CHUNK_54_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=73"

NEW_ONLINE_CHUNK_ID: str = "husc_tb74_dkxt_online_2026"
NEW_ONLINE_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=74"

CHUNKED_JSONL_OUT: Path = REPO_ROOT / "data" / "chunked" / "husc_thongbao_fix_2026.jsonl"

# Verbatim EMS sentence from baseline (id73, route=offline, year=2026).
CHUNK_54_TEXT: str = (
    "- Hình thức nộp lệ phí: Thí sinh nộp trực tiếp hoặc gửi kèm hồ sơ qua đường "
    "bưu điện theo hình thức chuyển phát nhanh (EMS)."
)

CHUNK_54_META_BASE: Dict[str, Any] = {
    "source": CHUNK_54_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "xtt_nop_ho_so",
    "route": "offline",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "id73_xtt_offline_v1",
}

NEW_ONLINE_TEXT: str = (
    "Đăng ký xét tuyển đại học hệ chính quy năm 2026 theo phương thức trực tuyến (online) "
    "tại HUSC: thí sinh truy cập cổng đăng ký xét tuyển của Đại học Huế tại địa chỉ "
    "https://dkxt.hueuni.edu.vn bằng tài khoản VneID (Định danh điện tử) hoặc Căn cước "
    "công dân gắn chip (CCCD). "
    "Mốc đăng ký nguyện vọng (ĐK NV) cho thí sinh: từ 02/7/2026 đến 17h ngày 14/7/2026. "
    "Thí sinh cần khai báo đầy đủ thông tin cá nhân, nguyện vọng ngành, tổ hợp xét tuyển "
    "và xác nhận trước thời hạn trên. (Nguồn: Thông báo id=74 — ĐK XT online, HUSC 2026.)"
)

NEW_ONLINE_META: Dict[str, Any] = {
    "source": NEW_ONLINE_SOURCE,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "dang_ky_online",
    "route": "online",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "id74_dkxt_online_v1",
    "url": "https://dkxt.hueuni.edu.vn",
    "dk_nv_window": "2026-07-02_to_2026-07-14T17:00",
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


def _build_row(chunk_id: str, text: str, meta: Dict[str, Any], vector: List[float], source: str) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "faq_type": "dang_ky_xet_tuyen" if "dkxt" in chunk_id or "online" in chunk_id else "xet_tuyen_thang",
        "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
        "vector": vector,
    }


def _write_to_jsonl(rows: List[Dict[str, Any]]) -> None:
    CHUNKED_JSONL_OUT.parent.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Pure logic (testable)
# ---------------------------------------------------------------------------
def plan_changes(embedder=None, table=None, dry_run: bool = False) -> Dict[str, Any]:
    """Plan + execute the topic-disambiguation fix.

    Returns a summary dict.
    """
    summary: Dict[str, Any] = {
        "chunk_54": {"action": "noop", "added": False, "deleted": False},
        "new_online": {"action": "noop", "added": False, "deleted": False},
        "rows_added": 0,
        "rows_deleted": 0,
        "dry_run": dry_run,
        "jsonl_written": 0,
    }

    if dry_run:
        # No DB / model work in dry-run.
        summary["chunk_54"]["action"] = "would_re_add_with_route_offline"
        summary["new_online"]["action"] = "would_add_husc_tb74_dkxt_online_2026"
        return summary

    assert embedder is not None and table is not None, "live mode needs embedder + table"

    # ---- 1) re-add chunked_22_chunk_54 with route=offline, info_type=xtt ----
    _delete_by_chunk_id(table, CHUNK_54_ID)
    summary["chunk_54"]["deleted"] = True
    summary["rows_deleted"] += 1

    vec54 = _embed_text(embedder, CHUNK_54_TEXT)
    row54 = _build_row(CHUNK_54_ID, CHUNK_54_TEXT, CHUNK_54_META_BASE, vec54, CHUNK_54_SOURCE)
    table.add([row54])
    summary["chunk_54"]["added"] = True
    summary["chunk_54"]["action"] = "re_added_with_route_offline"
    summary["rows_added"] += 1

    # ---- 2) add husc_tb74_dkxt_online_2026 ----
    _delete_by_chunk_id(table, NEW_ONLINE_CHUNK_ID)
    summary["new_online"]["deleted"] = True
    summary["rows_deleted"] += 1

    vec74 = _embed_text(embedder, NEW_ONLINE_TEXT)
    row74 = _build_row(NEW_ONLINE_CHUNK_ID, NEW_ONLINE_TEXT, NEW_ONLINE_META, vec74, NEW_ONLINE_SOURCE)
    table.add([row74])
    summary["new_online"]["added"] = True
    summary["new_online"]["action"] = "added"
    summary["rows_added"] += 1

    # ---- 3) write jsonl for build_graph parity ----
    _write_to_jsonl([row54, row74])
    summary["jsonl_written"] = 2

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
    p = argparse.ArgumentParser(description="Topic-disambiguation fix id73/74")
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    args = p.parse_args()

    summary = run_against_live_db(dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))