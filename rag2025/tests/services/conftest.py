"""tests/services/conftest.py

Autouse fixture to reset year-state pollution between tests.

The root cause of cross-test failures is that `aggregation_rules._YEAR` and
`aggregation_booster._YEAR` are module-level constants derived at import time.
When `test_booster_year_durability.py` sets `CURRENT_ADMISSION_YEAR=2027` and
reloads the modules, the 2027 value leaks into subsequent tests that import
`from services.aggregation_booster import ...` because they reference the
reloaded module object (or a stale copy).

This conftest ensures every test in `tests/services/` starts and ends with a
deterministic year-state pinned to 2026 (data cycle hien tai) by:
1. Setting `CURRENT_ADMISSION_YEAR=2026` (and popping `APP_ENV` to avoid prod
   checks) before each test. Pinned to 2026 — data cycle hien tai; bump to
   2027 only when 2027 data is ingested.
2. Reloading the temporal-authority / aggregation-rules / aggregation-booster
   modules (both `src.services.*` and `services.*` namespaces) to re-derive
   `_YEAR` from the pinned env.
3. Restoring the original env and reloading again after the test.

The reload order is critical: temporal_authority -> aggregation_rules -> aggregation_booster.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def _reset_year_state() -> Generator[None, None, None]:
    """Reset year-state pollution for every test in tests/services/."""
    # Save original env
    original_year = os.environ.get("CURRENT_ADMISSION_YEAR")
    original_app_env = os.environ.get("APP_ENV")

    # Pin year to known state — 2026 = data cycle hien tai (max data_year=2026).
    # Calendar fallback would now resolve 2027 (month>=8), so we must SET the
    # env var explicitly, not pop it, to stay deterministic.
    os.environ["CURRENT_ADMISSION_YEAR"] = "2026"
    os.environ.pop("APP_ENV", None)

    # Reload modules to derive _YEAR from the pinned env (2026)
    _reload_year_modules()

    yield  # ── test runs ──

    # Restore original env
    if original_year is not None:
        os.environ["CURRENT_ADMISSION_YEAR"] = original_year
    else:
        os.environ.pop("CURRENT_ADMISSION_YEAR", None)
    if original_app_env is not None:
        os.environ["APP_ENV"] = original_app_env
    else:
        os.environ.pop("APP_ENV", None)

    # Reload again to restore original year-state
    _reload_year_modules()


def _reload_year_modules() -> None:
    """Reload temporal_authority -> aggregation_rules -> aggregation_booster
    in both `src.services.*` and `services.*` namespaces to re-derive `_YEAR`."""
    module_names = [
        "src.services.temporal_authority",
        "src.services.aggregation_rules",
        "src.services.aggregation_booster",
        "services.temporal_authority",
        "services.aggregation_rules",
        "services.aggregation_booster",
    ]
    for name in module_names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
