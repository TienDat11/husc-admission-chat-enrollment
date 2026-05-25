# @spec(S13.7) /api/meta — current admission year + data freshness for FE banner
"""Public meta endpoint exposing the admission year banner data.

Frontend uses this to show "Tư vấn tuyển sinh năm 2026 • Cập nhật ..."
plus a warning chip when freshness_lag_days exceeds the alert threshold.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel


# Anchor relative paths to the repo root so the endpoint works regardless of CWD.
# meta.py → routers → src → rag2025 → repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AUDIT_PATH = _REPO_ROOT / "rag2025" / "data" / "audit_pre_reingest.json"

router = APIRouter(prefix="/api", tags=["meta"])

# Default freshness threshold — surfaces a warning in the FE when crossed.
def _get_freshness_alert_days() -> int:
    """Read FRESHNESS_ALERT_DAYS env (defaults to 90). Always read fresh so tests can monkeypatch."""
    try:
        return int(os.getenv("FRESHNESS_ALERT_DAYS", "90"))
    except ValueError:
        return 90


class MetaResponse(BaseModel):
    """Public payload for /api/meta."""

    current_admission_year: int
    latest_crawl_date: Optional[str]
    total_notifications: Optional[int]
    freshness_lag_days: Optional[int]
    freshness_alert: bool


def _get_current_admission_year() -> int:
    """Read CURRENT_ADMISSION_YEAR env (defaults to 2026). Safe parse — invalid → 2026."""
    try:
        return int(os.getenv("CURRENT_ADMISSION_YEAR", "2026"))
    except ValueError:
        return 2026


def _read_audit_fallback(audit_path: Path) -> dict[str, Any] | None:
    """Read audit_pre_reingest.json as a freshness fallback when retriever
    state is unavailable.

    Returns None when the file is missing or malformed (caller falls back
    to env-only response).
    """
    if not audit_path.exists():
        return None
    try:
        return json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _compute_freshness_lag(latest_iso: Optional[str]) -> Optional[int]:
    """Days between now (UTC) and the latest crawl date. None if no date."""
    if not latest_iso:
        return None
    try:
        # Accept Z-suffix or +00:00 form.
        normalized = latest_iso.replace("Z", "+00:00")
        latest = datetime.fromisoformat(normalized)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = datetime.now(timezone.utc) - latest
    return max(0, delta.days)


@router.get("/meta", response_model=MetaResponse)
async def get_meta(request: Request) -> MetaResponse:
    """Return banner-ready meta data: current year + freshness status.

    Resolution order for freshness fields:
      1. app.state.retriever.metadata_snapshot() if present.
      2. rag2025/data/audit_pre_reingest.json fallback.
      3. None (FE renders with env year only).

    Contract: `retriever.metadata_snapshot()` MUST be a pure in-memory read.
    If the retriever needs I/O to compute snapshot data, it should pre-cache
    those values during startup. Otherwise wrap the call in asyncio.to_thread
    to avoid blocking the event loop.
    """
    current_year = _get_current_admission_year()
    state = request.app.state

    latest_crawl: Optional[str] = None
    total_notifications: Optional[int] = None

    # Path 1: ask the retriever directly when it can.
    retriever = getattr(state, "retriever", None)
    if retriever is not None:
        snap = getattr(retriever, "metadata_snapshot", None)
        if callable(snap):
            try:
                meta = snap()
                if isinstance(meta, dict):
                    if meta.get("latest_crawl_date") is not None:
                        latest_crawl = meta["latest_crawl_date"]
                    if meta.get("total_notifications") is not None:
                        total_notifications = meta["total_notifications"]
            except Exception:
                # Public endpoint must not fail on retriever quirks.
                pass

    # Path 2: audit JSON fallback.
    if latest_crawl is None or total_notifications is None:
        audit_path = Path(
            os.getenv("AUDIT_FALLBACK_PATH", str(_DEFAULT_AUDIT_PATH))
        )
        audit = _read_audit_fallback(audit_path)
        if audit:
            if latest_crawl is None and audit.get("audit_date") is not None:
                latest_crawl = audit["audit_date"]
            if total_notifications is None and audit.get("total_rows") is not None:
                total_notifications = audit["total_rows"]

    lag_days = _compute_freshness_lag(latest_crawl)
    return MetaResponse(
        current_admission_year=current_year,
        latest_crawl_date=latest_crawl,
        total_notifications=total_notifications,
        freshness_lag_days=lag_days,
        freshness_alert=(lag_days is not None and lag_days > _get_freshness_alert_days()),
    )
