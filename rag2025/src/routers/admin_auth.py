"""FastAPI dependencies for admin identity + RBAC — tickets 13 and 37.

The HTTP seam: a request carries a credential in one header
(:data:`~src.config.admin_identity.ADMIN_CREDENTIAL_HEADER`), the credential is
resolved to an :class:`~src.domain.admin_identity.Identity` server-side, and a
route states the *action* it performs. The decision itself lives in the matrix
(:mod:`src.domain.admin_identity`); this module only turns it into 403s.

Two entry points, both thin:

- :func:`require_action` - for a route whose action is fixed at definition time
  (``dependencies=[Depends(require_action(AdminAction.RECORDS_PUBLISH))]``).
- :func:`enforce` - for a route whose action comes from its payload, i.e. the
  lifecycle transition endpoint, where ``submit``/``approve``/``publish`` are
  three different grants arriving on one route.

Neither decides policy, and neither is optional: a route that names no action
and calls neither is a route with no authorization, so the transition endpoint
resolves its action through ``action_for_transition`` and :func:`enforce`
denies ``None`.

Ticket 37 moved the *source* of the registry, not the shape of the credential:

- :func:`identity_store_for` - the durable store, when the deployment has a
  canonical data root. With no data root there is no store and the behavior is
  exactly ticket 13's: the environment is the whole registry.
- :func:`seed_from_env_if_empty` - the bootstrap. An empty store plus a
  configured ``ADMIN_IDENTITIES`` copies the env into the store **once**; from
  then on the store is the source of truth and a changed environment does not
  silently overwrite grants the API made.
- :func:`current_identity` consults the store first and falls back to the env
  (which still carries the shared ops credential). No process-local cache: a
  disable must bite on the next request, not the next restart.

Status codes are deliberate:

- 403 - credential present but unknown (or disabled), or known with a role
  that does not grant the action. The caller is authenticated-or-not, but not
  permitted.
- 422 - no credential at all (the header is required), which is FastAPI's
  validation failure. This is the surviving "no credential -> no publish"
  guard from ticket 12: ticket 13 replaces how identity is proven, not the
  requirement that publish is refused without it.
- 503 - the registry itself is misconfigured. Nobody gets in, loudly.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from src.config.admin_identity import (
    ADMIN_CREDENTIAL_HEADER,
    OPS_ACTOR_ID,
    AdminIdentityConfigError,
    load_identities,
    resolve_identity,
)
from src.domain.admin_identity import AdminAction, Identity, role_allows
from src.infrastructure.identity_store import (
    IdentityEntry,
    IdentityStore,
    IdentityStoreError,
)

__all__ = [
    "AdminAction",
    "Identity",
    "current_identity",
    "enforce",
    "identity_store_for",
    "require_action",
    "seed_from_env_if_empty",
]

def identity_store_for(request: Request) -> IdentityStore | None:
    """The deployment's identity store, or ``None`` without a data root.

    Lazily built on first use and kept on ``app.state`` so one app has one
    store object (the object is stateless - every read hits the file - but a
    stable instance keeps the lock registry shared). An app with no canonical
    record store has nowhere durable to put identities, and falls back to the
    ticket-13 env-only behavior rather than inventing a path.
    """
    existing = getattr(request.app.state, "identity_store", None)
    if existing is not None:
        return existing
    record_store = getattr(request.app.state, "canonical_record_store", None)
    if record_store is None:
        return None
    store = IdentityStore(record_store.root)
    request.app.state.identity_store = store
    return store

def seed_from_env_if_empty(store: IdentityStore) -> bool:
    """Copy the env registry into ``store`` the first time it is used.

    Returns whether seeding happened. The "only when empty" guard is the whole
    point: without it, every process that starts with an older environment
    rewrites the store and silently loses the grants the API made - which is
    precisely the failure ticket 37 exists to close. The ops machine
    credential is deliberately *not* seeded: it is a shared deployment key,
    not a person, and it keeps resolving through the env fallback (ticket 13's
    rule that it cannot act as a human is enforced in config, not here).
    """
    if store.entries():
        return False
    humans = [
        IdentityEntry(
            subject_id=identity.actor_id,
            actor_id=identity.actor_id,
            role=identity.role,
            credential=token,
        )
        for token, identity in load_identities()
        if identity.actor_id != OPS_ACTOR_ID
    ]
    if not humans:
        return False
    store.seed(humans)
    return True

def current_identity(
    request: Request,
    credential: str | None = Header(None, alias=ADMIN_CREDENTIAL_HEADER),
) -> Identity:
    """Resolve the request's credential to an identity, or refuse.

    Two paths, tried in order:

    1. **Header** — the traditional ops/machine credential in
       ``X-Admin-Token``. Unchanged behaviour from ticket 13: store-first,
       env-second, 403 on unknown, 503 on a broken registry.

    2. **Cookie** — the session token issued by ``POST /admin/login``
       (ticket T08). The token resolves to an ``actor_id`` on
       ``app.state.session_registry``; the actor is then looked up in the
       store the same way as the header path, so a disabled entry is denied
       identically.

    A request that carries neither a header nor a session cookie is refused with
    422 (no credential at all - the validation-failure code the admin surface
    has used since ticket 12).
    """
    # Path 1: header token (ops + machine + human API callers).
    if credential is not None:
        store = identity_store_for(request)
        if store is not None:
            try:
                seed_from_env_if_empty(store)
                entry = store.find(credential)
            except (IdentityStoreError, AdminIdentityConfigError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"admin identity registry is unusable: {exc}",
                ) from exc
            if entry is not None:
                if entry.disabled:
                    raise HTTPException(
                        status_code=403,
                        detail="admin identity is disabled",
                    )
                return Identity(actor_id=entry.actor_id, role=entry.role)

        try:
            identity = resolve_identity(credential)
        except AdminIdentityConfigError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"admin identity registry is misconfigured: {exc}",
            ) from exc

        if identity is None:
            raise HTTPException(status_code=403, detail="unknown admin credential")
        return identity

    # Path 2: session cookie (ticket T08, password login).
    from src.routers.admin_session import _SESSION_COOKIE

    token = request.cookies.get(_SESSION_COOKIE, "")
    if not token:
        raise HTTPException(
            status_code=422,
            detail="no credential: X-Admin-Token header or session cookie required",
        )

    sessions = getattr(request.app.state, "session_registry", None)
    if sessions is None:
        raise HTTPException(
            status_code=503,
            detail="session registry is not configured",
        )

    record = sessions.resolve(token)
    if record is None:
        raise HTTPException(
            status_code=403,
            detail="invalid or expired session",
        )

    # Resolve the actor in the store the same way the header path does.
    store = identity_store_for(request)
    if store is not None:
        try:
            entry = store.find_by_actor(record.actor_id)
        except IdentityStoreError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"admin identity registry is unusable: {exc}",
            ) from exc
        if entry is not None:
            if entry.disabled:
                raise HTTPException(
                    status_code=403,
                    detail="admin identity is disabled",
                )
            return Identity(actor_id=entry.actor_id, role=entry.role)

    # The actor was in the store when the session was issued but has since
    # been removed (identity-management PATCH disabled then re-activated as
    # a different subject). The session token is stale; refuse.
    raise HTTPException(
        status_code=403,
        detail="session no longer maps to a stored identity",
    )

def enforce(identity: Identity, action: AdminAction | None) -> None:
    """Refuse unless ``identity``'s role grants ``action``.

    ``action=None`` means the caller could not map the request onto a known
    action. That denies: an unmapped action is one nobody has been granted.
    """
    if action is not None and role_allows(identity.role, action):
        return

    label = action.value if isinstance(action, AdminAction) else "an unmapped action"
    raise HTTPException(
        status_code=403,
        detail=f"role '{identity.role.value}' is not permitted to perform '{label}'",
    )

def require_action(action: AdminAction):
    """A dependency that admits only the roles granted ``action``.

    Returns the identity so a route can use the authenticated actor (its
    ``actor_id`` is what the audit event carries) without resolving it twice:
    FastAPI caches the sub-dependency within one request.
    """

    def _dependency(identity: Identity = Depends(current_identity)) -> Identity:
        enforce(identity, action)
        return identity

    return _dependency
