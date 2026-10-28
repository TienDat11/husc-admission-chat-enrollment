"""Back-fill ``major_codes`` on canonical records from ``record_id`` (ticket T03).

Records whose ``record_id`` embeds a major code in the form
``<TYPE>:nganh:<CODE>:<YEAR>:<SOURCE>`` get ``major_codes = [CODE]`` appended
as a new version (append-only; the seed version is never rewritten).

Usage (from ``rag2025/``):

    # Preview only — prints how many records would change, writes nothing.
    PYTHONPATH=src python scripts/backfill_major_codes.py --dry-run

    # For real — append one version per affected record.
    PYTHONPATH=src python scripts/backfill_major_codes.py

    # Point at a non-default store (tests, CI).
    PYTHONPATH=src python scripts/backfill_major_codes.py \
        --store /tmp/my_store
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/backfill_major_codes.py).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402

# record_id format: TYPE:nganh:C7-LETTER_SUFFIX:YEAR:SOURCE
# The major-code segment is a 7-digit code optionally followed by a lowercase
# alphabetic suffix (e.g. 7510302ic, 7480201vj, 7480107td, 7440102sc).
_MAJOR_CODE_RE = re.compile(r"^[\w]+:nganh:([0-9]{7}[a-z]*):")


def extract_major_code(record_id: str) -> str | None:
    """Return the major code embedded in ``record_id``, or ``None``."""
    m = _MAJOR_CODE_RE.match(record_id)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back-fill major_codes on canonical records from record_id."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Path to the canonical store directory (default: data/canonical_store)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the number of records that would change; write nothing.",
    )
    args = parser.parse_args()

    store_path = args.store or (_ROOT / "data" / "canonical_store")
    store = CanonicalRecordStore(store_path)

    changed = 0
    skipped = 0

    for record_id in store.record_ids():
        code = extract_major_code(record_id)
        if code is None:
            skipped += 1
            continue

        latest = store.get_record(record_id).latest

        if code in (latest.major_codes or []):
            # Already carries this code — nothing to do.
            skipped += 1
            continue

        if args.dry_run:
            print(f"  would back-fill: {record_id}  ->  major_codes=[{code}, ...]")
            changed += 1
            continue

        store.edit_record(record_id, major_codes=[code, *latest.major_codes])
        changed += 1
        print(f"  back-filled: {record_id}  ->  major_codes=[{code}, ...]")

    if args.dry_run:
        print(f"\n[dry-run] {changed} record(s) would change; {skipped} skipped.")
    else:
        print(f"\n{changed} record(s) updated; {skipped} skipped.")


if __name__ == "__main__":
    main()
