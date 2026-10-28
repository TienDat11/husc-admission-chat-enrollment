# @spec(S13.3) idempotent fetch of HUSC tuyen sinh notifications by ID list
"""Fetch HTML pages for HUSC tuyển sinh notifications.

Default ID set: 63..74 (2026 cohort) + 59 (2025 history).
Output: rag2025/data/raw/{notification_id}.html
Idempotent: SHA256 compare existing → skip if unchanged.
Bounded backoff: exponential 1s → 30s cap, 5 attempts max.

The fetch policy itself (retry/backoff, streamed byte cap, User-Agent) lives in
:mod:`src.services.web_fetch`, because ticket T21 needs the same policy for a URL
an operator pastes and a second HTTP client would be a second set of bugs. This
module keeps only what is specific to the ID-list crawl: the URL it builds, the
default IDs, and the CLI.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any

import requests
from loguru import logger

# ``python rag2025/scripts/crawl_urls.py`` puts rag2025/ on the path, so the
# shared core imports the same way it does for the service that uses it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.web_fetch import (  # noqa: E402
    MAX_ATTEMPTS,
    TIMEOUT_S,
    fetch_html,
    sha256_bytes,
)


BASE_URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id={nid}"
DEFAULT_IDS = list(range(63, 75)) + [59]  # 12 × 2026 + 1 × 2025


def fetch_url(
    nid: int,
    *,
    session: requests.Session | None = None,
    timeout: float = TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> bytes:
    """Fetch HTML body for one notification ID with bounded backoff.

    Raises RuntimeError on permanent failure after MAX_ATTEMPTS.

    ``validate=False``: the URL is built here from ``BASE_URL`` and an int, so
    there is no request field to guard — the SSRF check belongs to the caller
    that takes a URL from a human.
    """
    url = BASE_URL.format(nid=nid)
    try:
        return fetch_html(
            url, session=session, timeout=timeout, max_attempts=max_attempts, validate=False
        )
    except Exception as exc:
        # The CLI's contract (and its tests') is a RuntimeError naming the id,
        # not the web_fetch hierarchy.
        raise RuntimeError(f"Failed to fetch id={nid} after {max_attempts} attempts: {exc}") from exc


def crawl_one(
    nid: int,
    output_dir: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch + idempotent write. Returns metadata dict.

    Status codes:
      "fetched" — newly fetched and written.
      "unchanged" — existing file hash matches; skipped.
      "replaced" — existing file hash differs; overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{nid}.html"

    new_bytes = fetch_url(nid, session=session)
    new_hash = sha256_bytes(new_bytes)

    if target.exists() and not force:
        existing_hash = sha256_bytes(target.read_bytes())
        if existing_hash == new_hash:
            return {"id": nid, "path": str(target), "sha256": new_hash, "status": "unchanged"}
        target.write_bytes(new_bytes)
        return {"id": nid, "path": str(target), "sha256": new_hash, "status": "replaced"}

    existed_before = target.exists()
    target.write_bytes(new_bytes)
    return {
        "id": nid,
        "path": str(target),
        "sha256": new_hash,
        "status": "replaced" if existed_before else "fetched",
    }


def crawl_many(
    ids: list[int],
    output_dir: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Crawl all IDs sequentially. Continue on per-ID failure."""
    results: list[dict[str, Any]] = []
    sess = session or requests.Session()
    for nid in ids:
        try:
            r = crawl_one(nid, output_dir, force=force, session=sess)
            logger.info(f"id={nid} {r['status']} sha256={r['sha256'][:12]}")
            results.append(r)
        except Exception as exc:
            logger.error(f"id={nid} FAILED: {exc}")
            results.append({"id": nid, "status": "failed", "error": str(exc)})
    return results


def _parse_ids(raw: str) -> list[int]:
    if not raw:
        return list(DEFAULT_IDS)
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl HUSC tuyển sinh notifications by ID")
    parser.add_argument("--ids", default="", help="Comma-separated ID list (default: 63-74 + 59)")
    parser.add_argument("--output-dir", default="rag2025/data/raw", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Bypass hash skip")
    args = parser.parse_args()

    ids = _parse_ids(args.ids)
    out_dir = Path(args.output_dir)
    results = crawl_many(ids, out_dir, force=args.force)

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        logger.error(f"{len(failed)}/{len(results)} ids failed")
        return 1
    logger.info(f"{len(results)} ids OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
