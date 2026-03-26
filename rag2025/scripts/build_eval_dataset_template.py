from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out = {
        "dataset_name": "husc_eval_2026",
        "description": "Evaluation set template for Vietnamese admission QA (simple, multihop, comparative).",
        "schema": {
            "id": "string",
            "category": "simple | multihop | comparative",
            "query": "string",
            "expected_chunks": ["chunk_id"],
            "gold_answer": "string",
            "must_include_entities": ["string"],
        },
        "samples": [
            {
                "id": "q001",
                "category": "simple",
                "query": "Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?",
                "expected_chunks": [],
                "gold_answer": "",
                "must_include_entities": ["NGANH:Công nghệ thông tin", "DIEM_CHUAN"],
            },
            {
                "id": "q002",
                "category": "multihop",
                "query": "Ngành nào có tổ hợp A00 và học phí thấp hơn ngành Kỹ thuật phần mềm?",
                "expected_chunks": [],
                "gold_answer": "",
                "must_include_entities": ["TO_HOP:A00", "HOC_PHI"],
            },
            {
                "id": "q003",
                "category": "comparative",
                "query": "So sánh ngành CNTT và Khoa học dữ liệu về điểm chuẩn và cơ hội học bổng.",
                "expected_chunks": [],
                "gold_answer": "",
                "must_include_entities": ["NGANH:CNTT", "NGANH:Khoa học dữ liệu", "DIEM_CHUAN", "CHINH_SACH"],
            },
        ],
    }

    output = Path(__file__).parent.parent / "data" / "eval" / "husc_eval_2026_template.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote template: {output}")


if __name__ == "__main__":
    main()
