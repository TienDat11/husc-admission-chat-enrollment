"""Admin rate limiting — ticket 27.

The ticket requires the anonymous query limit to be *separate* from the admin
one. These tests assert the two properties that make that separation real, over
HTTP, against the mounted router:

1. An actor that exceeds its budget gets 429 (and the budget is per actor, so a
   second actor is unaffected) — the admin surface has a limiter at all.
2. The admin budget is configured independently of ``RATE_LIMIT_PER_MINUTE``,
   so tightening public traffic does not throttle administrators.

Why HTTP rather than calling the limiter directly: the failure this ticket
guards against is a limiter that exists but is not wired to any route. Only a
request through the mounted router proves the dependency is attached.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from rag2025.src.app.lifecycle import (  # noqa: E402
    configure_canonical_record_store,
)
from src.routers.admin import router as admin_router  # noqa: E402
from src.routers.admin_rate_limit import (  # noqa: E402
    ADMIN_RATE_LIMIT_ENV,
    admin_rate_limiter,
)

TOKENS = {"admin": "tok-quan-tri", "editor": "tok-bien-tap"}
ACTORS = {"admin": "cb.quan-tri", "editor": "cb.bien-tap"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    """The real admin router, with two actors registered and a small budget."""
    entries = [
        {"token": TOKENS[role], "actor_id": ACTORS[role], "role": role}
        for role in TOKENS
    ]
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv(ADMIN_RATE_LIMIT_ENV, "3")

    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "store")
    )
    with TestClient(application) as test_client:
        yield test_client


def _identity(client: TestClient, role: str):
    """GET /admin/identity — needs a credential, no action, so it isolates the limiter."""
    return client.get("/admin/identity", headers={"X-Admin-Token": TOKENS[role]})


def test_actor_is_cut_off_after_its_budget(client: TestClient) -> None:
    """The 4th call from one actor in the window is refused with 429."""
    assert [_identity(client, "admin").status_code for _ in range(3)] == [200, 200, 200]

    refused = _identity(client, "admin")
    assert refused.status_code == 429
    # A limiter that returns 429 without telling the client when to come back
    # leaves well-behaved callers retrying immediately.
    assert refused.headers["Retry-After"] == "60"


def test_one_actor_cannot_spend_anothers_budget(client: TestClient) -> None:
    """Budget is per actor: exhausting admin leaves editor unaffected.

    This is the property an IP-keyed limiter cannot provide, and the reason the
    admin surface does not reuse the anonymous limiter.
    """
    for _ in range(4):
        _identity(client, "admin")
    assert _identity(client, "admin").status_code == 429

    assert _identity(client, "editor").status_code == 200


def test_admin_budget_is_independent_of_the_anonymous_limit(
    client: TestClient, monkeypatch
) -> None:
    """The anonymous limit does not govern the admin surface.

    ``RATE_LIMIT_PER_MINUTE=0`` disables the anonymous limiter; it must not
    silently disable the admin one, and vice versa — otherwise the two surfaces
    are one control wearing two names.
    """
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    assert [_identity(client, "admin").status_code for _ in range(4)] == [
        200,
        200,
        200,
        429,
    ]


def test_budget_is_read_per_request_so_retuning_needs_no_restart(
    client: TestClient, monkeypatch
) -> None:
    """Raising the env var takes effect without rebuilding the app."""
    for _ in range(4):
        _identity(client, "admin")
    assert _identity(client, "admin").status_code == 429

    monkeypatch.setenv(ADMIN_RATE_LIMIT_ENV, "10")
    assert _identity(client, "admin").status_code == 200


def test_zero_disables_the_admin_limiter(client: TestClient, monkeypatch) -> None:
    """An explicit 0 turns the admin limiter off (ops escape hatch)."""
    monkeypatch.setenv(ADMIN_RATE_LIMIT_ENV, "0")
    assert {_identity(client, "admin").status_code for _ in range(20)} == {200}


def test_unparsable_budget_keeps_the_default(client: TestClient, monkeypatch) -> None:
    """A typo'd budget keeps the limit rather than silently removing it.

    Same posture as ``four_eyes_enabled``: misconfiguration must not disable a
    control.
    """
    monkeypatch.setenv(ADMIN_RATE_LIMIT_ENV, "not-a-number")
    admin_rate_limiter.reset()
    codes = [_identity(client, "admin").status_code for _ in range(62)]
    assert codes[:60] == [200] * 60
    assert codes[60:] == [429, 429]


def test_budget_follows_the_actor_across_a_role_change(
    tmp_path: Path, monkeypatch
) -> None:
    """A role change must not hand the same actor a fresh budget.

    Ticket 27 keys the limiter on ``Identity.actor_id``. If the role were mixed
    into the key, an actor could exceed its budget by presenting a credential
    registered under a different role — the limit would follow the role instead
    of the actor.
    """
    entries = [
        {"token": "tok-before", "actor_id": "cb.mot-nguoi", "role": "editor"},
    ]
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv(ADMIN_RATE_LIMIT_ENV, "2")

    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "store")
    )

    with TestClient(application) as test_client:
        def call(token: str):
            return test_client.get(
                "/admin/identity", headers={"X-Admin-Token": token}
            ).status_code

        assert [call("tok-before"), call("tok-before")] == [200, 200]

        # Same actor, now registered with a different role and token.
        monkeypatch.setenv(
            "ADMIN_IDENTITIES",
            json.dumps(
                [{"token": "tok-after", "actor_id": "cb.mot-nguoi", "role": "publisher"}]
            ),
        )
        assert call("tok-after") == 429


def test_every_admin_route_is_covered_by_the_limiter() -> None:
    """No admin route can be added without inheriting the limit.

    Guards the wiring itself: the limiter is attached at router level, so a new
    route is covered by construction. This asserts that property instead of
    trusting it — a route that overrode ``dependencies`` would drop the limit
    while still passing the behavioural tests above.
    """
    spec = importlib.util.spec_from_file_location(
        "husc_admin_routes_probe", ROOT / "src" / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # T08: /admin/login and /admin/logout are on a separate router (session
    # lockout guards them, not the admin_rate_limit dependency).
    _T08_SESSION_ROUTES = frozenset({"/admin/login", "/admin/logout"})
    admin_paths = {
        route.path for route in module.app.routes
        if route.path.startswith("/admin") and route.path not in _T08_SESSION_ROUTES
    }
    assert admin_paths, "expected the admin surface to be mounted"

    for path in sorted(admin_paths):
        for route in module.app.routes:
            if getattr(route, "path", None) != path:
                continue
            dependency_names = {
                getattr(dep.call, "__name__", "") for dep in route.dependant.dependencies
            }
            assert "admin_rate_limit" in dependency_names, (
                f"{path} does not inherit the admin rate limiter"
            )
