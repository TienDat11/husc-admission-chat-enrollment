"""
Idempotent fix: reconcile booster rule chunk IDs in husc_chunks (S3).

Group A (50 rows): id has "rag2025:" prefix from migration artifact.
  Fix: set id = doc_id (doc_id holds the correct identifier).
Group B (2 rows): id = "all_bogddt_tt_08_header_2025" but
  doc_id is "qa_xet_truc_tuyen_vs_giay_2026" / "qa_xet_tuyen_thang_2026".
  Fix: set id = doc_id for those 2 rows.

Idempotent: running twice produces the same result (no-op on second run).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rag2025"))

import lancedb
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "rag2025" / ".env")

from config.settings import RAGSettings

settings = RAGSettings()
LANCEDB_URI = settings.LANCEDB_URI
TABLE_NAME = settings.LANCEDB_TABLE


def _esc(raw: str) -> str:
    """SQL-escape single quotes for IN-clause values."""
    return raw.replace("'", "''")


def main() -> None:
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(TABLE_NAME)

    # ------------------------------------------------------------------
    # BEFORE snapshot
    # ------------------------------------------------------------------
    all_rows = table.to_arrow()
    total_before = len(all_rows)
    print(f"BEFORE: total rows = {total_before}")

    # Count group A candidates
    group_a_mask = [
        str(r["id"]).startswith("rag2025:") for r in all_rows.to_pylist()
    ]
    group_a_before = sum(group_a_mask)
    print(f"  id LIKE 'rag2025:%'  = {group_a_before} (expect >0)")

    # Count group B candidates
    group_b_before = sum(
        1 for r in all_rows.to_pylist() if str(r["id"]) == "all_bogddt_tt_08_header_2025"
    )
    print(f"  id = 'all_bogddt_tt_08_header_2025' = {group_b_before} (expect 3)")

    # ------------------------------------------------------------------
    # Group A: fix rows where id starts with "rag2025:" → id = doc_id
    # ------------------------------------------------------------------
    if group_a_before > 0:
        pylist = all_rows.to_pylist()
        group_a_rows = [r for r in pylist if str(r["id"]).startswith("rag2025:")]
        old_ids_a = [str(r["id"]) for r in group_a_rows]

        # Build modified records: change id to doc_id, keep everything else
        modified_a = []
        for r in group_a_rows:
            rec = dict(r)
            new_id = str(rec["doc_id"])
            if new_id == rec["id"]:
                # Already correct — skip (idempotency safety)
                continue
            rec["id"] = new_id
            modified_a.append(rec)

        if modified_a:
            # Delete old rows by old id
            escaped_old = ", ".join(f"'{_esc(oid)}'" for oid in old_ids_a)
            table.delete(f"id IN ({escaped_old})")

            # Re-insert with corrected id
            table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                modified_a
            )
            print(f"  Group A fixed: {len(modified_a)} row(s) (id = 'rag2025:...' → id = doc_id)")
        else:
            print("  Group A: no rows needed fixing (idempotent)")
    else:
        print("  Group A: nothing to fix")

    # ------------------------------------------------------------------
    # Group B: fix 2 rows where id='all_bogddt_tt_08_header_2025' but
    #          doc_id is one of the two correct QA ids
    # ------------------------------------------------------------------
    GROUP_B_DOC_IDS = {"qa_xet_truc_tuyen_vs_giay_2026", "qa_xet_tuyen_thang_2026"}

    # Re-read to get fresh state after Group A fix
    all_rows2 = table.to_arrow()
    pylist2 = all_rows2.to_pylist()

    group_b_rows = [
        r for r in pylist2
        if str(r["id"]) == "all_bogddt_tt_08_header_2025"
        and str(r["doc_id"]) in GROUP_B_DOC_IDS
    ]

    if group_b_rows:
        modified_b = []
        for r in group_b_rows:
            rec = dict(r)
            rec["id"] = str(rec["doc_id"])
            modified_b.append(rec)

        # Delete each row by compound condition (id + doc_id).
        # We CANNOT use id IN (...) because all 3 rows share the same id,
        # and a bare `id = '...'` would delete the legitimate row too.
        for r in group_b_rows:
            old_id_esc = _esc(str(r["id"]))
            old_doc_esc = _esc(str(r["doc_id"]))
            table.delete(
                f"id = '{old_id_esc}' AND doc_id = '{old_doc_esc}'"
            )

        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            modified_b
        )
        print(f"  Group B fixed: {len(modified_b)} row(s) (id = 'all_bogddt_tt_08_header_2025' -> id = doc_id)")
    else:
        print("  Group B: no rows needed fixing (idempotent)")

    # ------------------------------------------------------------------
    # AFTER snapshot
    # ------------------------------------------------------------------
    all_rows3 = table.to_arrow()
    total_after = len(all_rows3)
    pylist3 = all_rows3.to_pylist()

    group_a_after = sum(1 for r in pylist3 if str(r["id"]).startswith("rag2025:"))
    group_b_after = sum(
        1 for r in pylist3 if str(r["id"]) == "all_bogddt_tt_08_header_2025"
    )

    print(f"\nAFTER:  total rows = {total_after} (expect 2138)")
    print(f"  id LIKE 'rag2025:%'  = {group_a_after} (expect 0)")
    print(f"  id = 'all_bogddt_tt_08_header_2025' = {group_b_after} (expect 1)")

    # Integrity checks
    assert total_after == total_before, (
        f"ROW LOSS: {total_before} → {total_after}"
    )
    assert group_a_after == 0, f"Group A NOT fully fixed: {group_a_after} remain"
    assert group_b_after == 1, (
        f"Group B count wrong: {group_b_after} (expect 1 — the legitimate row)"
    )

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
