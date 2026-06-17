# @spec(U2) idempotent fetch of HUSC ngành detail pages by major code
"""Fetch HTML pages for HUSC ngành đào tạo detail pages.

URL pattern: https://tuyensinh.husc.edu.vn/nganhdaotao-dh.php?manganh={code}
Output: rag2025/data/raw/nganh_{code}.html
Idempotent: SHA256 compare existing → skip if unchanged.
Bounded backoff: exponential 1s → 30s cap, 5 attempts max.
Sequential (not parallel) to avoid rate limiting.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


BASE_URL = "https://tuyensinh.husc.edu.vn/nganhdaotao-dh.php?manganh={code}"
DEFAULT_CODES = [
    "7460108", "7480201", "7480103", "7480201VJ", "7510302", "7440102",
    "7510302IC", "7440102SC", "7440112", "7510401", "7420201", "7850101",
    "7520503", "7580211", "7440301", "7850105", "7580101", "7220104",
    "7229030", "7229010", "7310608", "7229042", "7229001", "7310205",
    "7320101", "7320111", "7310301", "7760101",
]
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
MAX_ATTEMPTS = 5
TIMEOUT_S = 30
USER_AGENT = "HUSC-Admission-Chat/1.0 (+nganh-detail-crawl)"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _backoff_delay(attempt: int) -> float:
    return min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_CAP_S)


def fetch_url(
    code: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> bytes:
    """Fetch HTML body for one ngành code with bounded backoff.

    Raises RuntimeError on permanent failure after max_attempts.
    404 raises RuntimeError with 'HTTP 404' in message — caller treats as skip.
    """
    _client = client or httpx.Client()
    url = BASE_URL.format(code=code)
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = _client.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                # Not retryable — page simply doesn't exist yet
                raise RuntimeError(f"HTTP 404 on {url} (no retry)")
            if 500 <= resp.status_code < 600:
                last_exc = RuntimeError(f"HTTP {resp.status_code} on {url}")
                logger.warning(
                    f"code={code} attempt={attempt + 1} status={resp.status_code} retrying"
                )
            else:
                # Other 4xx — do not retry
                raise RuntimeError(f"HTTP {resp.status_code} on {url} (no retry)")
        except httpx.RequestError as exc:
            last_exc = exc
            logger.warning(f"code={code} attempt={attempt + 1} error={exc} retrying")
        if attempt + 1 < max_attempts:
            time.sleep(_backoff_delay(attempt))
    raise RuntimeError(
        f"Failed to fetch code={code} after {max_attempts} attempts: {last_exc}"
    )


def crawl_one(
    code: str,
    output_dir: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch + idempotent write for one ngành code. Returns metadata dict.

    Status values:
      "fetched"   — newly fetched and written.
      "unchanged" — existing file hash matches; skipped.
      "replaced"  — existing file hash differs; overwritten.
      "skipped_404" — server returned 404; file not written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"nganh_{code}.html"

    try:
        new_bytes = fetch_url(code, client=client)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            logger.warning(f"code={code} 404 — skipping (page not yet published)")
            return {"code": code, "status": "skipped_404", "error": str(exc)}
        raise

    new_hash = _sha256_bytes(new_bytes)

    if target.exists() and not force:
        existing_hash = _sha256_bytes(target.read_bytes())
        if existing_hash == new_hash:
            return {
                "code": code,
                "path": str(target),
                "sha256": new_hash,
                "status": "unchanged",
            }
        target.write_bytes(new_bytes)
        return {
            "code": code,
            "path": str(target),
            "sha256": new_hash,
            "status": "replaced",
        }

    existed_before = target.exists()
    target.write_bytes(new_bytes)
    return {
        "code": code,
        "path": str(target),
        "sha256": new_hash,
        "status": "replaced" if existed_before else "fetched",
    }


def crawl_many(
    codes: list[str],
    output_dir: Path,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Crawl all codes sequentially. Continue on per-code failure."""
    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for code in codes:
            try:
                r = crawl_one(code, output_dir, force=force, client=client)
                sha_preview = r.get("sha256", "")[:12]
                logger.info(f"code={code} {r['status']} sha256={sha_preview}")
                results.append(r)
            except Exception as exc:
                logger.error(f"code={code} FAILED: {exc}")
                results.append({"code": code, "status": "failed", "error": str(exc)})
    return results


def _parse_codes(raw: str) -> list[str]:
    if not raw:
        return list(DEFAULT_CODES)
    return [t.strip() for t in raw.split(",") if t.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crawl HUSC ngành đào tạo detail pages by major code"
    )
    parser.add_argument(
        "--codes",
        default="",
        help="Comma-separated major code list (default: all 28 codes)",
    )
    parser.add_argument(
        "--output-dir",
        default="rag2025/data/raw",
        help="Output directory for raw HTML files",
    )
    parser.add_argument("--force", action="store_true", help="Bypass hash skip")
    args = parser.parse_args()

    codes = _parse_codes(args.codes)
    out_dir = Path(args.output_dir)
    results = crawl_many(codes, out_dir, force=args.force)

    fetched = [r for r in results if r.get("status") in ("fetched", "replaced")]
    skipped = [r for r in results if r.get("status") in ("unchanged", "skipped_404")]
    failed = [r for r in results if r.get("status") == "failed"]

    summary = {
        "fetched_count": len(fetched),
        "skipped_count": len(skipped),
        "failed_codes": [r["code"] for r in failed],
    }
    logger.info(f"Summary: {summary}")

    if failed:
        logger.error(f"{len(failed)}/{len(results)} codes failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
