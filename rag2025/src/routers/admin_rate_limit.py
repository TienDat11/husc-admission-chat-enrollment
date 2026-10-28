"""Admin-surface rate limiting, keyed by authenticated actor — ticket 27.

The anonymous query limiter keys on client IP. That key is wrong for the admin
surface for two reasons:

1. Admin callers share egress addresses (one office NAT, one reverse proxy), so
   an IP bucket lets one admin's burst throttle every other admin.
2. Conversely, one admin behind several IPs (laptop + phone + VPN) would get
   several budgets.

So this limiter keys on ``Identity.actor_id``, which is resolved from the
credential registry server-side and therefore cannot be spoofed by a caller.

The budget is read from the environment per request (not captured at import) so
an operator can retune it without a restart, and the env var is *separate* from
``RATE_LIMIT_PER_MINUTE``: ticket 27 requires the anonymous and admin paths to
be independently configurable, and sharing one variable would make tightening
public traffic also throttle administrators mid-incident.

Disabled only by an explicit ``<= 0``; a missing or unparsable value keeps the
default. That mirrors ``four_eyes_enabled`` in ``src.config.admin_identity``:
a deployment that forgets to configure this gets a limit, not silence.
"""

from __future__ import annotations

import os

from fastapi import Depends

from src.domain.admin_identity import Identity
from src.routers.admin_auth import current_identity
from src.services.rate_limit import FixedWindowLimiter

__all__ = [
    "ADMIN_RATE_LIMIT_ENV",
    "UPLOAD_RATE_LIMIT_ENV",
    "admin_rate_limit",
    "admin_rate_limiter",
    "upload_rate_limit",
    "upload_rate_limiter",
]
# 10 uploads/hour per actor. Separate from the 60/min admin budget: an
# upload is an expensive, rare operation (a subprocess parse, a store write,
# a draft record), so it gets its own tighter window.
UPLOAD_RATE_LIMIT_ENV = "UPLOAD_RATE_LIMIT_PER_HOUR"
_DEFAULT_UPLOAD_LIMIT = 10


def _upload_limit() -> int:
    """The configured per-actor upload budget, or the default when unset/invalid."""
    raw = (os.environ.get(UPLOAD_RATE_LIMIT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_UPLOAD_LIMIT
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_UPLOAD_LIMIT


ADMIN_RATE_LIMIT_ENV = "ADMIN_RATE_LIMIT_PER_MINUTE"

# 60/min per actor. Measured against the admin UI's real call pattern: the
# records/config screens issue one request per user action (no polling — only
# the chat readiness hook polls, and it hits /health, not /admin), so 60 leaves
# a full order of magnitude of headroom for a human while still bounding a
# runaway script or a leaked credential.
_DEFAULT_LIMIT = 60


def _admin_limit() -> int:
    """The configured per-actor budget, or the default when unset/invalid."""
    raw = (os.environ.get(ADMIN_RATE_LIMIT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_LIMIT
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_LIMIT


def _actor_key(identity: Identity) -> str:
    """Bucket key for one authenticated actor: its ``actor_id``, nothing else.

    Ticket 27 names ``Identity.actor_id`` as the key, and the role is
    deliberately *not* mixed in. An earlier revision used
    ``admin:<actor_id>:<role>``, which let a role change hand the same actor a
    fresh budget — the actor is the thing being limited, so anything else in
    the key is a way to escape the limit. Namespace separation from the
    anonymous limiter comes from this being a separate limiter instance, not
    from the key's shape.
    """
    return identity.actor_id


# One limiter per process, shared by every admin route. Keyed per actor, so
# sharing the instance is what makes the budget a per-actor budget.
admin_rate_limiter = FixedWindowLimiter(limit=_admin_limit)


async def admin_rate_limit(
    identity: Identity = Depends(current_identity),
) -> None:
    """FastAPI dependency: spend one unit of the calling actor's budget.

    Depends on ``current_identity`` (FastAPI caches it within a request), so a
    route can list this alongside ``require_action(...)`` without resolving the
    credential twice.
    """
    admin_rate_limiter.enforce(_actor_key(identity))


# One uploader limiter, separate from the admin limiter: a distinct
# instance so the budgets never bleed into each other — the same
# "namespace separation is the instance, not the key shape" rule the
# admin limiter states. The window is an hour: uploads are the rare,
# expensive admin operation this module exists to bound.
upload_rate_limiter = FixedWindowLimiter(
    limit=_upload_limit,
    window_s=3600.0,
)


async def upload_rate_limit(
    identity: Identity = Depends(current_identity),
) -> None:
    """FastAPI dependency: spend one unit of the actor's upload budget."""
    upload_rate_limiter.enforce(_actor_key(identity))
