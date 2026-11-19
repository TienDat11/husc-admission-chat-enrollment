"""Golden Gate CLI - blocking release verification (U9).

Usage:
    python scripts/run_golden_gate.py --release <id> --baseline eval/baseline_A1.json
        [--golden-set tests/eval/fixtures/golden_seed.jsonl]
        [--database-url postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc]

Evaluates ``<release>`` against the golden gate, persists ``golden_report`` and
status (``ready`` / ``gate_failed``) to PG when the release exists, prints a
JSON summary, and exits ``0`` on pass or ``1`` on fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from services.golden_gate import (  # noqa: E402
    GOLDEN_SEED_PATH,
    GoldenGateReport,
    persist_to_pg,
    release_exists,
    run_golden_gate,
)

from src.services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
)


def _build_retriever() -> LanceDBRetriever:
    return LanceDBRetriever.from_env()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden gate for a release.")
    parser.add_argument("--release", required=True, help="Release id to gate.")
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "eval" / "baseline_A1.json"),
        help="Baseline JSON (A1 numbers as floor).",
    )
    parser.add_argument(
        "--golden-set",
        default=str(GOLDEN_SEED_PATH),
        help="Golden seed JSONL.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PG URL (defaults to DATABASE_URL env or the local husc DSN).",
    )
    args = parser.parse_args(argv)

    import os

    database_url = args.database_url or os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc"
    )

    retriever = _build_retriever()
    from src.services.golden_retrieval_check import (  # noqa: E402
        GoldenRetrievalReport,
        _top5_record_ids,
        load_golden_set,
    )  # noqa: F401  (re-exported seam; kept for the reuse contract)

    report: GoldenGateReport = run_golden_gate(
        release_id=args.release,
        retriever=retriever,
        embedder=retriever.embedder,
        seed_path=Path(args.golden_set),
        baseline_path=Path(args.baseline),
    )

    summary = {
        "release_id": report.release_id,
        "verdict": report.verdict,
        "reason": report.reason,
        "matrix_complete": report.matrix_complete,
        "matrix_passed": report.matrix_passed,
        "matrix_total": report.matrix_total,
        "recall_at_5": report.recall_at_5,
        "recall_at_10": report.recall_at_10,
        "citation_precision": report.citation_precision,
        "abstention_ok": report.abstention_ok,
    }
    print(json.dumps(summary, ensure_ascii=False))

    # Persist to PG when the release row exists.
    try:
        if release_exists(args.release, database_url):
            persist_to_pg(args.release, report, database_url)
    except Exception as exc:  # noqa: BLE001 - persistence must not mask the verdict
        print(json.dumps({"persist_error": str(exc)}), file=sys.stderr)

    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
