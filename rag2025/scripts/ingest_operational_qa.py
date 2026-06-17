"""Generate operational Q-A chunks from husc_2026_operational_qa.json into JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "raw" / "husc_2026_operational_qa.json"
OUTPUT = ROOT / "data" / "chunked" / "husc_2026_operational_qa.jsonl"


def main():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for chunk_id, entry in data.items():
        text = f"Câu hỏi: {entry['question']}\nTrả lời: {entry['answer_canonical']}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "source": entry.get("source", "tuyensinh.husc.edu.vn"),
            "faq_type": "operational_qa",
            "metadata": entry.get("metadata", {"data_year": 2026, "info_type": "operational_qa"}),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Written {len(chunks)} operational Q-A chunks to {OUTPUT}")
    for c in chunks:
        print(f"  - {c['chunk_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
