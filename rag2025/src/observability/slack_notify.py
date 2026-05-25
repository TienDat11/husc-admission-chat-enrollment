# @spec(S13.8) Slack ops notification helper for yearly rotation + freshness alerts
"""Post a structured message to Slack (or no-op when webhook unset).

Used by yearly_rotation.py and freshness_alert.py. Non-blocking: any
network error is swallowed and logged so the calling script never fails
because of an ops notification.
"""
from __future__ import annotations
import os
from typing import Literal

import httpx
from loguru import logger


Level = Literal["info", "warning", "error"]


_LEVEL_COLORS: dict[Level, str] = {
    "info": "#FFD700",      # yellow
    "warning": "#FFA500",   # orange
    "error": "#FF0000",     # red
}


def send_slack(
    message: str,
    *,
    level: Level = "info",
    webhook_url: str | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """Post `message` to Slack with the given severity level.

    Args:
        message: Plain-text message body.
        level: One of "info", "warning", "error".
        webhook_url: Override SLACK_RAG_OPS_WEBHOOK env. None -> read env.
        timeout_s: HTTP timeout.

    Returns:
        True on success (HTTP 200), False on any failure (env unset, network
        error, non-2xx response). Never raises.
    """
    url = webhook_url or os.getenv("SLACK_RAG_OPS_WEBHOOK")
    if not url:
        logger.debug("send_slack: no webhook configured; skipping")
        return False

    color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
    payload = {
        "attachments": [
            {
                "color": color,
                "text": message,
                "fields": [
                    {"title": "level", "value": level, "short": True},
                    {"title": "source", "value": "rag-platform", "short": True},
                ],
            }
        ]
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=payload)
            if 200 <= resp.status_code < 300:
                return True
            # Truncate response body — Slack error bodies may echo request fragments.
            logger.warning(
                f"send_slack: non-2xx status={resp.status_code} "
                f"body={resp.text[:100]!r}"
            )
            return False
    except httpx.HTTPError as exc:
        logger.warning(f"send_slack: network error {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"send_slack: unexpected error {exc}")
        return False
