# @spec(S13.8) freshness_alert — daily check + Slack notify when stale
"""Daily freshness check.

Reads either /api/meta endpoint (preferred — picks up retriever snapshot) or
the audit JSON fallback directly. When freshness_lag_days exceeds the
threshold, posts a Slack warning. Designed to run via GitHub Actions cron.

Idempotent: when audit JSON missing entirely (early bootstrap), returns
status="no_data" and does NOT send a notification (avoid cron noise).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AUDIT_PATH = _REPO_ROOT / "rag2025" / "data" / "audit_pre_reingest.json"
DEFAULT_THRESHOLD_DAYS = 90


def _read_audit(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _compute_lag_days(latest_iso: str) -> Optional[int]:
    try:
        normalized = latest_iso.replace("Z", "+00:00")
        latest = datetime.fromisoformat(normalized)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = datetime.now(timezone.utc) - latest
    return max(0, delta.days)


def check_freshness(
    *,
    audit_path: Path | None = None,
    threshold_days: int | None = None,
    notifier=None,
) -> dict[str, Any]:
    """Run the freshness check.

    Args:
        audit_path: override default audit JSON path.
        threshold_days: override default 90-day threshold.
        notifier: callable(message, level=...) → bool. Defaults to
            slack_notify.send_slack.

    Returns:
        Dict with status (ok | stale | no_data), lag_days, threshold,
        notified (bool), and source.
    """
    if notifier is None:
        from observability.slack_notify import send_slack as notifier  # type: ignore[no-redef]

    threshold = threshold_days if threshold_days is not None else int(
        os.getenv("FRESHNESS_ALERT_DAYS", str(DEFAULT_THRESHOLD_DAYS))
    )
    path = audit_path or Path(os.getenv("AUDIT_FALLBACK_PATH", str(_DEFAULT_AUDIT_PATH)))

    audit = _read_audit(path)
    if audit is None:
        logger.info(f"freshness_alert: audit json not found at {path}; skipping (no-data)")
        return {
            "status": "no_data",
            "lag_days": None,
            "threshold": threshold,
            "notified": False,
            "source": str(path),
        }

    latest_iso = audit.get("audit_date") or audit.get("latest_crawl_date")
    if not latest_iso:
        return {
            "status": "no_data",
            "lag_days": None,
            "threshold": threshold,
            "notified": False,
            "source": str(path),
        }

    lag = _compute_lag_days(latest_iso)
    if lag is None:
        return {
            "status": "no_data",
            "lag_days": None,
            "threshold": threshold,
            "notified": False,
            "source": str(path),
        }

    if lag <= threshold:
        return {
            "status": "ok",
            "lag_days": lag,
            "threshold": threshold,
            "notified": False,
            "source": str(path),
        }

    # Stale — notify Slack but do not raise.
    message = (
        f"HUSC tuyển sinh data is stale: {lag} days since last crawl "
        f"(threshold {threshold}d). Run yearly_rotation if a new admissions "
        f"cycle has begun."
    )
    notified = False
    try:
        notified = bool(notifier(message, level="warning"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"freshness_alert: notifier raised {exc}; treating as not-notified")

    return {
        "status": "stale",
        "lag_days": lag,
        "threshold": threshold,
        "notified": notified,
        "source": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily freshness check + Slack notify")
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--threshold-days", type=int, default=None)
    args = parser.parse_args()

    try:
        res = check_freshness(
            audit_path=Path(args.audit_path) if args.audit_path else None,
            threshold_days=args.threshold_days,
        )
    except Exception as exc:  # noqa: BLE001 — observability script must never crash CI
        logger.error(f"freshness_alert: unhandled error {exc}")
        res = {"status": "error", "error": str(exc)}

    print(json.dumps(res, ensure_ascii=False, indent=2))
    # Always exit 0 — this is an observability script, not a test gate.
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
