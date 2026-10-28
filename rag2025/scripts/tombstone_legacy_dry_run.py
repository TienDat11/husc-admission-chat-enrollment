"""Tombstone legacy rows per eligible (doc_type, data_year) group — T15.

Ticket T15 ("Di chuyển: duyệt lô 487 + tombstone legacy"), last sub-item
"Script dry-run". Encodes spec 02-ingestion-and-knowledge-qa.md L110:
once a doc_type has enough canonical rows published for a year, the legacy
rows of that same doc_type/data_year get is_tombstone=true.

Read-only by default (report mode). Apply requires both --apply and an
explicit acknowledgement that the 3-route HTTP check passed first
(ticket gate: "Không tombstone legacy trước khi test HTTP qua cả 3 route").

Usage:
  python -m scripts.tombstone_legacy_dry_run                    # report
  python -m scripts.tombstone_legacy_dry_run --json out.json     # report+json
  python -m scripts.tombstone_legacy_dry_run --print-eval-protocol
  python -m scripts.tombstone_legacy_dry_run --apply --i-ran-the-3-route-http-check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Maximum allowed recall@5 drop (percentage points) vs the T00 baseline.
RECALL_DELTA_THRESHOLD_PP: float = 2.0

#: T00 baseline 86q records artifact.
T00_BASELINE_DIR = Path(__file__).resolve().parent.parent / "results" / "baseline_T00"

#: The 3 retrieval routes the HTTP check must cover (spec 02 L110 / T15 gate).
HTTP_CHECK_ROUTES = ("padded", "hybrid", "graph")

ACKNOWLEDGEMENT_FLAG = "--i-ran-the-3-route-http-check"


# ---------------------------------------------------------------------------
# Pure helpers (testable without a live LanceDB)
# ---------------------------------------------------------------------------
def classify_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group rows by (doc_type, data_year) and classify each group.

    A row is a LEGACY row iff its release_id is the literal string "legacy"
    (the convention in lancedb_retrieval._pick_promoted_row: legacy rows
    carry release_id=="legacy"; canonical rows carry a real release id).

    A group is ELIGIBLE for tombstoning iff it has at least one canonical
    row (canonical coverage exists for that doc_type/data_year) — the
    explicit rule from ticket T15 / spec 02 L110: "Khi một loại đã publish
    đủ cho năm X → đánh dấu is_tombstone các dòng legacy cùng
    doc_type/data_year." Eligibility is per-group, not per-row: a group
    with zero canonical rows is NOT eligible, no matter how many legacy
    rows it holds.
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        doc_type = str(row.get("doc_type") or "")
        data_year = int(row.get("data_year") or 0)
        key = (doc_type, data_year)
        g = groups.setdefault(
            key,
            {"legacy_ids": [], "canonical_ids": [], "legacy_count": 0, "canonical_count": 0},
        )
        if str(row.get("release_id") or "") == "legacy":
            g["legacy_ids"].append(str(row.get("id") or ""))
            g["legacy_count"] += 1
        else:
            g["canonical_ids"].append(str(row.get("id") or ""))
            g["canonical_count"] += 1

    for (doc_type, data_year), g in groups.items():
        g["eligible"] = g["canonical_count"] > 0
        g["would_tombstone_ids"] = g["legacy_ids"] if g["eligible"] else []

    return {
        "groups": [
            {"doc_type": k[0], "data_year": k[1], **v} for k, v in sorted(groups.items())
        ],
        "total_legacy_rows": sum(g["legacy_count"] for g in groups.values()),
        "total_would_tombstone": sum(len(g["would_tombstone_ids"]) for g in groups.values()),
    }


def check_recall_delta(t00_recall_pp: float, after_recall_pp: float) -> Dict[str, Any]:
    """Encode the T15 acceptance rule: recall@5 may drop ≤ 2.0 percentage
    points vs T00, else fail (ticket: "Eval 86q so với T00: recall@5 record
    mong đợi không giảm quá 2 điểm %").

    Boundary semantics: exactly 2.0pp drop passes; 2.001pp drops fails.
    A recall ABOVE the T00 value (improvement) always passes.
    """
    delta_pp = t00_recall_pp - after_recall_pp  # positive = drop
    passed = delta_pp <= RECALL_DELTA_THRESHOLD_PP
    return {
        "t00_recall_pp": t00_recall_pp,
        "after_recall_pp": after_recall_pp,
        "delta_pp": delta_pp,
        "threshold_pp": RECALL_DELTA_THRESHOLD_PP,
        "passed": passed,
    }


def eval_protocol_commands(run_id: str = "t15_after_tombstone") -> List[str]:
    """The exact 86q eval commands to run BEFORE and AFTER the tombstone.

    BEFORE: run the instrumented 86q eval, capture recall@5 vs expected
    records (ground truth: data/eval/husc_thi_sinh_thuc_gt.json).
    AFTER: run the same harness with the post-tombstone release; compare
    recall@5 with the T00 baseline artifact (results/baseline_T00/
    husc86q_records_t00.jsonl) — must not drop more than 2pp.
    """
    base = "cd rag2025"
    return [
        f"{base} && PYTHONPATH=src python -m scripts.eval_run_instrumented "
        f"--run-id t15_before_tombstone",
        f"{base} && PYTHONPATH=src python -m scripts.eval_run_instrumented "
        f"--run-id {run_id}",
        (
            f"then compare recall@5-on-expected-records of "
            f"results/eval_harness/86q_records_{run_id}.jsonl against the T00 baseline "
            f"({T00_BASELINE_DIR.name}/husc86q_records_t00.jsonl) — "
            f"rule: drop must be ≤ {RECALL_DELTA_THRESHOLD_PP}pp"
        ),
    ]


# ---------------------------------------------------------------------------
# Table I/O (thin wrappers around the LanceDB-shaped table)
# ---------------------------------------------------------------------------
def load_all_rows(table: Any) -> List[Dict[str, Any]]:
    """Read the whole table as row dicts (no vector column)."""
    # lancedb 0.30 to_pandas() takes no `columns` arg — select columns after.
    df = table.to_pandas()
    if df is None or len(df) == 0:
        return []
    keep = ["id", "doc_id", "release_id", "doc_type", "data_year", "status", "is_tombstone"]
    keep = [c for c in keep if c in df.columns]
    return [row.to_dict() for _, row in df[keep].iterrows()]


def apply_tombstones(table: Any, tombstone_ids: List[str]) -> int:
    """Write is_tombstone=true only for the named ids. Idempotent.

    Rows already tombstoned are skipped (a second run reports 0 changes).
    A targeted table.update per row touches nothing else in the table.
    Returns the number of rows actually changed.
    """
    changed = 0
    target = set(tombstone_ids)
    for row in load_all_rows(table):
        rid = str(row.get("id") or "")
        if rid not in target:
            continue
        if bool(row.get("is_tombstone")):
            continue  # already tombstoned — idempotent no-op
        table.update(where=f'id = "{rid}"', values={"is_tombstone": True})
        changed += 1
    return changed


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def print_report(report: Dict[str, Any]) -> None:
    print("Tombstone legacy dry-run report")
    print(f"{'doc_type':<28} {'year':>5}  {'legacy':>7} {'canonical':>9}  eligible?  tombstone ids")
    print("-" * 90)
    for g in report["groups"]:
        print(
            f"{g['doc_type']:<28} {g['data_year']:>5}  {g['legacy_count']:>7} "
            f"{g['canonical_count']:>9}  {str(g['eligible']):<11}  "
            f"{len(g['would_tombstone_ids'])} ids"
        )
        for rid in g["would_tombstone_ids"]:
            print(f"    - {rid}")
    print("-" * 90)
    print(f"TOTAL legacy rows: {report['total_legacy_rows']}")
    print(f"TOTAL would-tombstone: {report['total_would_tombstone']}")


def write_json_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON report written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tombstone legacy rows for eligible (doc_type, data_year) groups (T15). "
        "Read-only report by default; --apply requires the HTTP-check acknowledgement flag.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write is_tombstone=true (default: dry-run report only)")
    parser.add_argument(ACKNOWLEDGEMENT_FLAG, action="store_true", dest="ack_http_check",
                        help="Required WITH --apply: operator acknowledgement that the 3-route "
                             "(padded/hybrid/graph) HTTP check passed and tombstoned legacy rows "
                             "no longer appear in sources on any route (T15 gate).")
    parser.add_argument("--json", type=Path, default=None, metavar="PATH",
                        help="Also write the report to PATH as JSON.")
    parser.add_argument("--print-eval-protocol", action="store_true",
                        help="Print the 86q eval before/after commands and exit.")
    args = parser.parse_args(argv)

    if args.print_eval_protocol:
        for cmd in eval_protocol_commands():
            print(cmd)
        return 0

    settings = RAGSettings()
    import lancedb

    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        print(f"ERROR: LanceDB table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'", file=sys.stderr)
        return 1
    table = db.open_table(settings.LANCEDB_TABLE)

    rows = load_all_rows(table)
    report = classify_rows(rows)

    if args.apply:
        if not args.ack_http_check:
            print(
                "REFUSED: --apply requires the explicit acknowledgement flag "
                f"{ACKNOWLEDGEMENT_FLAG}. T15 gate: 'Không tombstone legacy trước khi "
                "test HTTP qua cả 3 route' (padded, hybrid, graph). Run the 3-route "
                "HTTP check (tombstoned legacy rows must be absent from `sources` on "
                "all three routes), verify the eval gate (86q recall@5 vs T00, "
                "max 2.0pp drop), then re-run with the flag.",
                file=sys.stderr,
            )
            return 2
        print(f"Applying tombstones to {report['total_would_tombstone']} rows...")
        target_ids = [rid for g in report["groups"] for rid in g["would_tombstone_ids"]]
        changed = apply_tombstones(table, target_ids)
        print(f"Rows changed: {changed} (0 on a second run = idempotent)")
        if args.json is not None:
            write_json_report(report, args.json)
        return 0

    print_report(report)
    if args.json is not None:
        write_json_report(report, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
