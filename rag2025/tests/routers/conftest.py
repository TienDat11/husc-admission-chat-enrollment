"""Shared fixtures for the router tests.

One fixture, for one reason: the admin limiter (ticket 27) is a module-level
singleton keyed by ``Identity.actor_id``, and the admin tests reuse a handful of
actor ids across hundreds of requests in a single process. Without a reset, the
budget one test spends is charged to the next test that authenticates as the
same actor — so tests fail in ways that depend on execution order rather than on
the behaviour under test.

Resetting per test is the correct seam, not a workaround: each test is an
independent session against the API, and the production property being modelled
("one actor cannot exceed its budget in a minute") is asserted directly in
``test_admin_rate_limit.py`` rather than left to emerge from suite-wide
accumulation.
"""

from __future__ import annotations

import pytest

from src.routers.admin_rate_limit import admin_rate_limiter


@pytest.fixture(autouse=True)
def _reset_admin_rate_limit() -> None:
    """Start every test with empty rate-limit buckets."""
    admin_rate_limiter.reset()
