"""Shared fixtures for the document-upload integration tests.

The admin and upload limiters (tickets 27 / T04) are module-level
singletons keyed by ``Identity.actor_id``; resetting them per test is
the same seam ``tests/routers/conftest.py`` states for the admin
limiter.
"""
from __future__ import annotations

import pytest

from src.routers.admin_rate_limit import admin_rate_limiter, upload_rate_limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiters() -> None:
    admin_rate_limiter.reset()
    upload_rate_limiter.reset()
    yield
    admin_rate_limiter.reset()
    upload_rate_limiter.reset()
