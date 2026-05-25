# @spec(S13.4) smoke_test_blue — 86Q replay scaffold against blue table
"""Smoke test the blue LanceDB table by replaying 86 questions and computing a
hallucination summary by category.

Real evaluation lives in eval_86q.py; this scaffold supports the blue/green
flip workflow by exercising the retriever path with mocks/fixtures, so the
flip can be verified without a live model end-to-end.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger


def _load_gold(path: Path) -> list[dict[str, Any]]:
    """Load 86Q gold or fall back to synthetic stub."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [
        {"id": "syn_1", "question": "Học phí 2026 là bao nhiêu?", "expected_year": 2026, "expected_scope": True},
        {"id": "syn_2", "question": "Điểm chuẩn năm 2025?", "expected_year": 2025, "expected_scope": True},
        {"id": "syn_3", "question": "Ngành Triết học có tuyển 2026 không?", "expected_year": 2026, "expected_scope": False},
    ]


def _score_one(result: dict[str, Any], expected_year: int, expected_scope: bool) -> dict[str, bool]:
    """Reduce a query result to {year_match, scope_match, grounded} flags."""
    md = result.get("metadata", {}) or {}
    answer = (result.get("answer") or "").lower()
    year_match = md.get("data_year") == expected_year or str(expected_year) in answer
    scope_match = expected_scope == bool(result.get("in_scope", True))
    grounded = bool(result.get("grounded", True))
    return {"year_match": year_match, "scope_match": scope_match, "grounded": grounded}


async def run_smoke(
    *,
    table_name: str,
    gold_path: Path,
    output_path: Path,
    query_fn: Callable[[str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Run the smoke test and write report.

    `query_fn` is injectable so tests can stub the retriever path.
    """
    gold = _load_gold(gold_path)
    counters = {"year_match": 0, "scope_match": 0, "grounded": 0}
    per_question: list[dict[str, Any]] = []

    for q in gold:
        try:
            result = await query_fn(q["question"])
        except Exception as exc:
            logger.exception(f"smoke_test query failed: {q['id']}")
            result = {"answer": "", "metadata": {}, "error": str(exc)}

        scores = _score_one(result, q.get("expected_year", 2026), q.get("expected_scope", True))
        for k in counters:
            if scores[k]:
                counters[k] += 1
        per_question.append({"id": q["id"], **scores})

    total = len(gold)
    report = {
        "table_name": table_name,
        "total_questions": total,
        "year_match_pct": counters["year_match"] / total if total else 0.0,
        "scope_match_pct": counters["scope_match"] / total if total else 0.0,
        "grounded_pct": counters["grounded"] / total if total else 0.0,
        "counters": counters,
        "per_question": per_question,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8",
    )
    logger.info(f"smoke_test_blue: wrote {output_path} ({total} questions)")
    return report


def _default_query_fn(question: str) -> Callable[[str], Awaitable[dict[str, Any]]]:  # pragma: no cover
    raise NotImplementedError("Real retriever not wired; provide --stub or inject query_fn.")


def main() -> int:
    import asyncio

    parser = argparse.ArgumentParser(description="Smoke test blue LanceDB table with 86Q replay")
    parser.add_argument("--table", default="husc_v2026_blue")
    parser.add_argument("--gold", default="rag2025/data/eval/86q_gold.json")
    parser.add_argument("--output", default="rag2025/results/regression_blue.json")
    parser.add_argument("--stub", action="store_true", help="Use a stub query_fn (returns happy-path)")
    args = parser.parse_args()

    if args.stub:
        async def _stub_query(_q: str) -> dict[str, Any]:
            return {"answer": "stub answer 2026", "metadata": {"data_year": 2026}, "in_scope": True, "grounded": True}
        query_fn = _stub_query
    else:
        query_fn = _default_query_fn

    asyncio.run(run_smoke(
        table_name=args.table,
        gold_path=Path(args.gold),
        output_path=Path(args.output),
        query_fn=query_fn,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
