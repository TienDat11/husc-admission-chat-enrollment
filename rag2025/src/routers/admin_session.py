"""Password login + session cookie — ticket T08.

The existing header-token path (``X-Admin-Token``) is kept intact for ops and
machine credentials. This module adds a second authentication path:

- ``POST /admin/login`` — an identity whose entry has a ``password_hash`` sends
  its ``actor_id`` + ``password``. On success an HttpOnly session cookie is
  set; on failure the attempt is counted and, after five consecutive failures
  for the same actor, login is locked for fifteen minutes.
- ``POST /admin/logout`` — clears the session cookie.

The session cookie carries a short-lived token that is stored in-process on
``app.state`` (no Redis/SQL). A multi-worker deployment with cookie-based
sessions would need a shared store; the MVP runs a single worker behind one
reverse proxy, so the in-process registry is correct and dependency-free.

The lockout registry is also in-process, same reasoning: the admin surface is
a human office UI, not a public endpoint, so the abuse it defends against is
one attacker at one office NAT — bounded by the per-actor rate limiter that
already guards every admin route.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from threading import Lock

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.infrastructure.identity_store import (
    IdentityEntry,
    IdentityStore,
    IdentityStoreError,
    verify_password,
)
from src.routers.admin_auth import identity_store_for

__all__ = [
    "LoginRequest",
    "LoginResponse",
    "LoginLockout",
    "SessionRegistry",
    "SessionRecord",
    "router",
]

# How many consecutive failures before an actor is locked.
_LOGIN_FAIL_LIMIT = 5
# How long a lockout lasts, in seconds.
_LOGIN_LOCK_SECONDS = 900  # 15 minutes
# How long a session cookie lives, in seconds. 8 hours (spec 05 §6).
_SESSION_TTL_SECONDS = 8 * 3600

# The cookie name; the value is an opaque token, not the identity itself, so a
# logged response does not reveal the actor.
_SESSION_COOKIE = "admin_session"


# ---------------------------------------------------------------------------
# In-process session + lockout registries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionRecord:
    """One issued session: which actor it belongs to, when it expires."""

    actor_id: str
    expires_at: float  # time.monotonic()


class SessionRegistry:
    """In-memory session token -> actor mapping.

    Tokens are 256-bit random hex strings. The registry is a plain dict
    guarded by a lock; eviction happens lazily on read (a session that is
    asked for but already expired is dropped).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, SessionRecord] = {}

    def issue(self, actor_id: str, ttl_seconds: int = _SESSION_TTL_SECONDS) -> str:
        """Create a session for ``actor_id`` and return the token."""
        token = secrets.token_hex(32)
        expires = time.monotonic() + ttl_seconds
        with self._lock:
            self._sessions[token] = SessionRecord(actor_id=actor_id, expires_at=expires)
        return token

    def resolve(self, token: str) -> SessionRecord | None:
        """The live session for ``token``, or ``None`` (unknown or expired)."""
        now = time.monotonic()
        with self._lock:
            record = self._sessions.get(token)
            if record is None or record.expires_at < now:
                self._sessions.pop(token, None)
                return None
            return record

    def revoke(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def reset(self) -> None:
        """Test seam — clear all sessions."""
        with self._lock:
            self._sessions.clear()


# One registry per process; the lifespan hook in main.py keeps a reference
# on ``app.state`` so a test can reach it without importing this module by
# name.
_session_registry = SessionRegistry()


@dataclass
class LoginLockout:
    """Tracks consecutive login failures for one actor.

    ``failure_count`` resets on a successful login. When it reaches
    ``_LOGIN_FAIL_LIMIT`` the actor is locked until ``locked_until``.
    """

    failure_count: int = 0
    locked_until: float = 0.0  # time.monotonic()

    def record_failure(self) -> bool:
        """Record one failure; return True when the actor is now locked."""
        self.failure_count += 1
        if self.failure_count >= _LOGIN_FAIL_LIMIT:
            self.locked_until = time.monotonic() + _LOGIN_LOCK_SECONDS
            self.failure_count = 0  # reset counter; the lock is what holds now
        return self.is_locked()

    def record_success(self) -> None:
        self.failure_count = 0
        self.locked_until = 0.0

    def is_locked(self) -> bool:
        if self.locked_until == 0.0:
            return False
        if time.monotonic() >= self.locked_until:
            self.locked_until = 0.0
            self.failure_count = 0
            return False
        return True

    def seconds_remaining(self) -> int:
        """How many more seconds the lockout lasts (0 if not locked)."""
        if self.locked_until == 0.0:
            return 0
        remaining = self.locked_until - time.monotonic()
        return max(0, int(remaining))


class LockoutRegistry:
    """In-memory per-actor lockout table."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[str, LoginLockout] = {}

    def get_or_create(self, actor_id: str) -> LoginLockout:
        with self._lock:
            entry = self._entries.get(actor_id)
            if entry is None:
                entry = LoginLockout()
                self._entries[actor_id] = entry
            return entry

    def reset(self) -> None:
        """Test seam — clear all lockouts."""
        with self._lock:
            self._entries.clear()


_lockout_registry = LockoutRegistry()


# ---------------------------------------------------------------------------
# HTTP models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    actor_id: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    actor_id: str
    role: str
    session_seconds: int = _SESSION_TTL_SECONDS


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/admin", tags=["admin-auth"])


def _registry_for(request: Request) -> tuple[SessionRegistry, LockoutRegistry]:
    """The process registries; tests can replace them on app.state."""
    sessions = getattr(request.app.state, "session_registry", None)
    lockouts = getattr(request.app.state, "lockout_registry", None)
    if sessions is None:
        sessions = _session_registry
        request.app.state.session_registry = sessions
    if lockouts is None:
        lockouts = _lockout_registry
        request.app.state.lockout_registry = lockouts
    return sessions, lockouts


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
) -> LoginResponse:
    """Authenticate by actor_id + password and set the session cookie.

    Returns 401 for a wrong password or no configured password; 423 (Locked)
    when the actor is in a lockout window.
    """
    sessions, lockouts = _registry_for(request)

    actor_lock = lockouts.get_or_create(payload.actor_id)
    if actor_lock.is_locked():
        remaining = actor_lock.seconds_remaining()
        raise HTTPException(
            status_code=423,
            detail=f"account is locked; try again in {remaining}s",
        )

    store = identity_store_for(request)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="identity store is not configured",
        )

    try:
        entry = store.find_by_actor(payload.actor_id)
    except IdentityStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"identity registry is unusable: {exc}",
        ) from exc

    if entry is None or entry.disabled:
        # Unknown or disabled actor: count the failure to prevent enumeration
        # of valid actor_ids (same cost as a wrong-password failure).
        actor_lock.record_failure()
        raise HTTPException(status_code=401, detail="invalid credentials")

    if not entry.password_hash:
        # Actor exists but has no password set — only the header-token path
        # is available for this identity.
        actor_lock.record_failure()
        raise HTTPException(status_code=401, detail="invalid credentials")

    if not verify_password(payload.password, entry.password_hash):
        actor_lock.record_failure()
        raise HTTPException(status_code=401, detail="invalid credentials")

    # Success.
    actor_lock.record_success()
    token = sessions.issue(entry.actor_id)
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="strict",
        secure=True,
        max_age=_SESSION_TTL_SECONDS,
        path="/",
    )
    return LoginResponse(
        actor_id=entry.actor_id,
        role=entry.role.value,
        session_seconds=_SESSION_TTL_SECONDS,
    )


@router.post("/logout")
async def logout(request: Request, response: Response) -> dict[str, str]:
    """Clear the session cookie. Idempotent — a request with no cookie is a no-op."""
    sessions, _ = _registry_for(request)
    token = request.cookies.get(_SESSION_COOKIE, "")
    if token:
        sessions.revoke(token)
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return {"status": "logged out"}
