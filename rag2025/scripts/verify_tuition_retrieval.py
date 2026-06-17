"""Quick verification: query the live table for HUSC 2026 tuition retrieval.

Verifies the LIVE table after the corrector ran:
  * "học phí ngành Văn học HUSC 2026"  → top chunk says 490.000 + 2026-2027/NĐ-238
  * "học phí ngành Vật lý"             → 555.000
  * "Kỹ thuật phần mềm"                 → 600.000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService


def main() -> int:
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        print("MISSING TABLE")
        return 2
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)

    queries = [
        # query, [groups of OR-markers, single markers]
        (
            "học phí ngành Văn học HUSC 2026",
            [["490.000", "490,000"], "Văn học", "2026-2027", "Nghị định 238"],
        ),
        (
            "học phí ngành Vật lý",
            [["555.000", "555,000"], "Vật lý"],
        ),
        (
            "Kỹ thuật phần mềm",
            [["600.000", "600,000"], "Kỹ thuật phần mềm"],
        ),
    ]

    failures = 0
    for q, must_have in queries:
        print(f"\n=== QUERY: {q!r} ===")
        vec = embedder.encode_single(q)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        rows = (
            table.search(vec)
            .metric("cosine")
            .limit(3)
            .to_list()
        )
        if not rows:
            print("  NO RESULTS")
            failures += 1
            continue
        for i, r in enumerate(rows):
            text = r.get("text", "")
            chunk_id = r.get("chunk_id") or r.get("id") or "?"
            print(f"  [{i+1}] chunk_id={chunk_id!r}")
            snippet = text[:280].replace("\n", " ")
            print(f"      text: {snippet}...")
            for marker in must_have:
                if isinstance(marker, list):
                    # OR-group: at least one variant must match
                    if any(m in text for m in marker):
                        print(f"      OK marker: {marker!r}")
                    else:
                        print(f"      MISSING marker: {marker!r}")
                        failures += 1
                else:
                    if marker in text:
                        print(f"      OK marker: {marker!r}")
                    else:
                        print(f"      MISSING marker: {marker!r}")
                        failures += 1

    print(f"\n=== TOTAL FAILURES: {failures} ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
