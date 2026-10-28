"""tests/conftest.py

Autouse reset for the process-global seams that app startup configures.

The problem this exists to remove
---------------------------------
``configure_canonical_record_store`` is the real startup wiring, and nine router
test files call it over a throwaway root — correctly, because that is the call
app startup makes. But ticket 11 added a step to that function:
``configure_conflict_source(conflict_queue_service.items)``, and
``abstention_policy._conflict_source`` is a **module-level** global, not app
state.

Production is one app per process, so a global is the right shape there — the
policy module is deliberately pure and cannot import the queue itself. Under
pytest, though, each test builds its own app, and the last ``configure_*`` call
wins: the conflict queue a router test pointed at its own ``tmp_path`` stays
installed for every test that runs afterwards. ``tests/services/
test_abstention_policy.py`` then sees that leftover queue, so queries that should
answer start returning ``conflict`` instead. Measured before this file existed:

    pytest tests/routers/test_admin_conflicts_api.py tests/services/test_abstention_policy.py
    -> 8 failed, 23 passed
    pytest tests/services/test_abstention_policy.py            (alone)
    -> 26 passed

Resetting per test is the correct seam, not a workaround — the same argument
``tests/routers/conftest.py`` makes for the admin rate limiter, which is a
module-level singleton for the same reason. Each test is an independent app; the
production property ("a queued contradiction is surfaced, not self-selected")
is asserted directly in ``test_abstention_policy.py`` rather than left to emerge
from whatever a previously-run test happened to install.

Note the reset is to ``None`` — the fail-soft default — not to a fake feed.
A test that wants conflicts registers its own source and restores it, as
``test_injected_conflict_source_is_used_at_the_seam`` already does.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_abstention_conflict_source() -> None:
    """Start every test with no conflict feed registered."""
    from services import abstention_policy

    abstention_policy.configure_conflict_source(None)
    yield
    abstention_policy.configure_conflict_source(None)
