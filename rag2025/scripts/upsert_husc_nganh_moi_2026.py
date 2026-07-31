"""Build + upsert the curated chunk ``husc_nganh_moi_2026``.

Idempotent: deletes any existing row with id='husc_nganh_moi_2026' before
re-adding. Asserts the embedding dimension is 1024 (matches the
``rag2025`` table schema enforced by EmbeddingService(RAGSettings())).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lancedb
from dotenv import load_dotenv
from loguru import logger

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

load_dotenv()

CHUNK_ID = "husc_nganh_moi_2026"
TABLE_NAME = "rag2025"


def _build_text() -> str:
    lines = [
        "Ngành mới mở / chương trình mới mở tại HUSC năm 2026",
        "",
        "Năm tuyển sinh 2026, Trường Đại học Khoa học (Đại học Huế) mở 3 chương trình/ngành mới:",
        "",
        "1. Vật lý học (Chương trình Công nghệ bán dẫn)",
        "   - Mã ngành: 7440102SC",
        "   - Thời gian đào tạo: 4 năm",
        "   - Chỉ tiêu tuyển sinh: 30",
        "   - Tổ hợp xét tuyển: A00, A01, C01, D01",
        "",
        "2. Khoa học dữ liệu",
        "   - Mã ngành: 7460108",
        "   - Thời gian đào tạo: 4 năm",
        "   - Chỉ tiêu tuyển sinh: 40",
        "   - Tổ hợp xét tuyển: C01, A01, D01, X02",
        "",
        "3. Công nghệ Kỹ thuật điện tử - Viễn thông (Chương trình thiết kế vi mạch)",
        "   - Mã ngành: 7510302IC",
        "   - Thời gian đào tạo: 5 năm",
        "   - Chỉ tiêu tuyển sinh: 35",
        "   - Tổ hợp xét tuyển: A00, A01, C01, X06",
        "",
        "Lưu ý: Đây là các ngành/chương trình mới được bổ sung trong năm tuyển sinh 2026, "
        "đón đầu nhu cầu nhân lực chất lượng cao về công nghệ bán dẫn, khoa học dữ liệu "
        "và thiết kế vi mạch. Thí sinh có thể theo dõi thông tin chi tiết tại trang "
        "tuyển sinh HUSC (https://tuyensinh.husc.edu.vn/).",
    ]
    return "\n".join(lines)


def main() -> int:
    settings = RAGSettings()
    embedding_service = EmbeddingService(settings)
    text = _build_text()
    vector = embedding_service.encode_documents([text])[0]
    assert len(vector) == 1024, (
        f"Embedding dim mismatch: expected 1024, got {len(vector)}"
    )

    meta = {
        "data_year": 2026,
        "info_type": "nganh_moi",
        "school": "HUSC",
        "chunk_method": "manual_curated",
        "source_url": "https://tuyensinh.husc.edu.vn/",
        "source_chunk_ids": [
            "nganh_detail_7440102SC_2026",
            "nganh_detail_7460108_2026",
            "nganh_detail_7510302IC_2026",
        ],
        "summary": "3 ngành/CT mới mở 2026: 7440102SC bán dẫn, 7460108 KHDL, 7510302IC vi mạch",
    }

    record = {
        "id": CHUNK_ID,
        "chunk_id": CHUNK_ID,
        "text": text,
        "source": "tuyensinh.husc.edu.vn (2026) + nganh_detail_*_2026",
        "faq_type": "nganh_moi_2026",
        "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
        "vector": vector.tolist(),
    }

    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(TABLE_NAME)

    # Idempotent: delete existing row with this id, then re-add.
    table.delete(f"id = '{CHUNK_ID}'")
    table.add([record])
    new_count = table.count_rows()
    logger.info(f"Upserted {CHUNK_ID}. Total rows now: {new_count}")

    # Also persist the chunk as a JSONL line for traceability.
    out_jsonl = ROOT / "data" / "chunked" / "husc_nganh_moi_fix.jsonl"
    payload = {
        "chunk_id": CHUNK_ID,
        "text": text,
        "source": record["source"],
        "faq_type": record["faq_type"],
        "metadata": meta,
    }
    with out_jsonl.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, default=str)
        fh.write("\n")
    logger.info(f"Wrote {out_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
