# @spec(S13)
"""Single source of truth for current admission year.

Resolves the current admission year in this precedence:
1. CURRENT_ADMISSION_YEAR env (authoritative in prod).
2. APP_ENV=prod + env unset/invalid → raises RuntimeError (fail-loud).
3. Dev/test fallback: admissions-cycle calendar (month >= START_MONTH → year+1).

The retriever argument is EVIDENCE-ONLY and never changes the returned authority.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, Any


ADMISSION_YEAR_START_MONTH = int(os.getenv("ADMISSION_YEAR_START_MONTH", "8"))


def _now() -> datetime:
    """Testable clock — monkeypatch this in tests to control time."""
    return datetime.now()


def get_current_admission_year(retriever: Optional[Any] = None) -> int:
    """Return the current admission year.

    Args:
        retriever: EVIDENCE-ONLY (for API compat). Does NOT affect the returned year.

    Returns:
        int: Current admission year.

    Raises:
        RuntimeError: When APP_ENV=prod and CURRENT_ADMISSION_YEAR is unset/invalid.

    Precedence:
        (a) CURRENT_ADMISSION_YEAR env if set & valid int → return it (authoritative).
        (b) APP_ENV='prod' and env unset/invalid → raise RuntimeError (fail-loud).
        (c) Dev/test fallback: admissions-cycle calendar.
    """
    # (a) Env override — authoritative
    env_year = os.getenv("CURRENT_ADMISSION_YEAR")
    if env_year:
        try:
            year = int(env_year)
            # Sanity floor only — NO upper cap (system must survive 10+ years and beyond).
            if year >= 2020:
                return year
        except ValueError:
            pass
        # Invalid value in env
        app_env = os.getenv("APP_ENV", "").lower()
        if app_env == "prod":
            raise RuntimeError(
                f"CURRENT_ADMISSION_YEAR env is invalid: {env_year!r}. "
                "Production requires a valid integer >= 2020."
            )
        # Dev/test: fall through to calendar fallback

    # (b) Prod + env unset → fail loud
    app_env = os.getenv("APP_ENV", "").lower()
    if app_env == "prod":
        raise RuntimeError(
            "CURRENT_ADMISSION_YEAR env is required in production (APP_ENV=prod) but is unset or invalid."
        )

    # (c) Dev/test fallback: admissions-cycle calendar
    now = _now()
    clock_year = now.year
    if now.month >= ADMISSION_YEAR_START_MONTH:
        return clock_year + 1
    return clock_year
