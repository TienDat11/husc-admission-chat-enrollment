# @spec(S13.8) yearly rotation orchestrator + 2027 auto-trigger detector (C14)
"""Drive a year-namespaced reingest when a new admission cycle starts.

Two entry points:
  - detect_new_year_signal(text, max_known_id=None): pure pattern matcher.
    Returns the detected year as int when fresh, or None.
  - rotate(year, runners): wires crawl + 3-way chunker + reload-table in
    sequence with injectable runners. All side effects via callables so the
    test suite can mock the entire pipeline.

CLI: python yearly_rotation.py --year 2027 [--dry-run]
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Any, Awaitable, Callable, Optional

from loguru import logger


# C14 — admission notifications mention the upcoming year explicitly.
# Match years strictly in [2027, 2039] — aligned with rotate()'s validator.
# Avoids signals about 2040+ that rotate() would reject anyway.
_NEXT_YEAR_RX = re.compile(r"\b(202[7-9]|203\d)\b")

# Semantic-context guard: text must mention an admissions-cycle keyword
# near the year for the signal to count as a real "new cycle" event,
# not just a retrospective comparison.
_ADMISSION_CONTEXT_RX = re.compile(
    r"(?:tuyển\s+sinh|thông\s+báo|xét\s+tuyển|đề\s+án|chỉ\s+tiêu)",
    re.IGNORECASE,
)


def detect_new_year_signal(
    text: str,
    *,
    max_known_id: Optional[int] = None,
    seen_id: Optional[int] = None,
    current_year: Optional[int] = None,
) -> Optional[int]:
    """Detect signals that a new admission cycle has begun.

    Two independent triggers (per C14):
      1. Content match: the text mentions a year strictly newer than current_year.
      2. ID overflow: an unseen notification id appears (seen_id > max_known_id).

    Args:
        text: Crawler payload (HTML or excerpt).
        max_known_id: Highest notification id ever ingested.
        seen_id: Newly observed notification id.
        current_year: Anchor — defaults to env CURRENT_ADMISSION_YEAR or 2026.

    Returns:
        The detected year (int) when a new cycle is observed; None otherwise.
        When both content and id signals fire, content takes precedence.

    Caller contract:
        This is a *signal* function — it does NOT itself trigger rotation.
        Production callers MUST log the detected signal and require human or
        secondary confirmation (e.g., a Slack approval step) before invoking
        rotate(). The semantic-context guard reduces false positives but is
        not a substitute for human-in-the-loop confirmation.
    """
    anchor = current_year if current_year is not None else int(
        os.getenv("CURRENT_ADMISSION_YEAR", "2026")
    )

    if isinstance(text, str) and text:
        # Require admissions-context keyword to be present alongside the
        # future year — guards against retrospective comparisons producing
        # a false-positive rotation trigger.
        has_admissions_context = bool(_ADMISSION_CONTEXT_RX.search(text))
        if has_admissions_context:
            for match in _NEXT_YEAR_RX.finditer(text):
                try:
                    year = int(match.group(1))
                except ValueError:
                    continue
                if year > anchor:
                    return year

    if max_known_id is not None and seen_id is not None and seen_id > max_known_id:
        # ID overflow but no explicit year signal — we cannot know the year
        # yet, so return anchor + 1 so the operator can verify before rotation.
        return anchor + 1

    return None


# Runner protocol — all side effects injectable for testability.
CrawlRunner = Callable[[int], Awaitable[dict[str, Any]]]
ChunkerRunner = Callable[[int], Awaitable[dict[str, Any]]]
SupersedeRunner = Callable[[int], Awaitable[None]]
ReloadRunner = Callable[[str], Awaitable[dict[str, Any]]]


async def rotate(
    *,
    year: int,
    crawl: CrawlRunner,
    chunker: ChunkerRunner,
    supersede_prior_year: SupersedeRunner,
    reload_table: ReloadRunner,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the rotation pipeline for `year`.

    Order is fixed:
      1. crawl(year): fetch new HUSC notifications (idempotent).
      2. chunker(year): run 3-way chunker + arbiter; emit canonical chunks
         and load into LanceDB husc_v{year}_blue.
      3. supersede_prior_year(year - 1): mark prior-year chunks
         is_superseded=true, valid_to=admission close.
      4. reload_table("husc_v{year}_blue"): atomic flip via /admin/reload-table.

    `dry_run=True` skips supersede and reload_table — safe smoke test path.
    """
    if not isinstance(year, int):
        raise TypeError(f"year must be int, got {type(year).__name__}")
    if year < 2024 or year > 2039:
        raise ValueError(f"year out of supported range [2024,2039]: {year}")

    logger.info(f"yearly_rotation: starting year={year} dry_run={dry_run}")

    crawl_res = await crawl(year)
    logger.info(f"yearly_rotation: crawl done year={year}")

    chunker_res = await chunker(year)
    logger.info(f"yearly_rotation: chunker done year={year}")

    if dry_run:
        return {
            "year": year,
            "crawl": crawl_res,
            "chunker": chunker_res,
            "supersede": "skipped (dry_run)",
            "reload": "skipped (dry_run)",
        }

    await supersede_prior_year(year - 1)
    logger.info(f"yearly_rotation: supersede prior_year={year - 1} done")

    reload_res = await reload_table(f"husc_v{year}_blue")
    logger.info(f"yearly_rotation: reload done table=husc_v{year}_blue")

    return {
        "year": year,
        "crawl": crawl_res,
        "chunker": chunker_res,
        "supersede": {"prior_year": year - 1, "status": "marked_superseded"},
        "reload": reload_res,
    }


def main() -> int:
    import asyncio
    import json as _json

    parser = argparse.ArgumentParser(description="Yearly admissions rotation orchestrator")
    parser.add_argument("--year", type=int, default=None, help="Target admission year")
    parser.add_argument("--dry-run", action="store_true", help="Skip supersede + reload")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only run detect_new_year_signal on text supplied via stdin and exit",
    )
    parser.add_argument(
        "--current-year",
        type=int,
        default=None,
        help="Override anchor year for --detect-only (default: env CURRENT_ADMISSION_YEAR or 2026)",
    )
    args = parser.parse_args()

    if args.detect_only:
        text = sys.stdin.read()
        detected = detect_new_year_signal(text, current_year=args.current_year)
        result = {"detected_year": detected, "current_year_anchor": args.current_year}
        print(_json.dumps(result, ensure_ascii=False))
        return 0 if detected is None else 1  # nonzero so cron can branch

    if args.year is None:
        logger.error(
            "yearly_rotation: --year is required for rotation. Use --detect-only "
            "to scan text from stdin instead."
        )
        return 2

    logger.error(
        "yearly_rotation: real crawl/chunker/supersede/reload runners are not "
        "wired into the CLI. Invoke rotate() programmatically with injected "
        "runners (see rotate() docstring + YEARLY_REINGEST_PLAYBOOK.md)."
    )
    return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
