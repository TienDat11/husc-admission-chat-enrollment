"""The identity-management surface - ticket 37.

Three endpoints, mounted under the admin router so they share its rate limit
and its authentication dependency:

- ``GET   /admin/identities``              list every stored identity
- ``POST  /admin/identities``              create one
- ``PATCH /admin/identities/{subject_id}`` change its role or disable it

All three are gated by ``AdminAction.ROLES_MANAGE``, which the matrix grants
to ``admin`` alone. Before this ticket the surface did not exist: the only
way to change what an actor may do was to edit ``ADMIN_IDENTITIES`` and
restart the process, and nothing recorded that it had happened. That is the
hole ``AuditAction.PERMISSION_CHANGE`` was declared for and never filled -
this module is its first producer.

Two rules are load-bearing here and neither is decoration:

**The event carries the caller, never the body.** A request may *name* any
actor it likes (creating an identity for a colleague is the normal case), but
``actor_id``/``role`` on the emitted event come from the resolved credential.
Otherwise a body field would decide who the ledger says did it, which is the
same as having no audit.

**Four-eyes is evaluated once, against a flag read once.** The switch is
consulted at the top of the mutation and the result is both the value the
comparison uses and the value recorded in the event's ``review_id``. Reading
it twice would let a deployment that flips the variable mid-request produce
an event describing a different request than the one that ran.

The ``reason`` guard exists at two layers on purpose: the router refuses with
422 before anything is written, and the audit model refuses a blank reason on
a ``PERMISSION_CHANGE`` event regardless of who built it. Either layer alone
is removable without noticing; the domain test pins the second one.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.config.admin_identity import four_eyes_enabled
from src.domain.admin_identity import AdminAction, Identity, Role
from src.infrastructure.identity_store import (
    IdentityConflictError,
    IdentityEntry,
    IdentityNotFoundError,
    IdentityStoreError,
    hash_password,
)
from src.routers.admin_auth import identity_store_for, require_action

__all__ = [
    "IdentityCreateRequest",
    "IdentityPatchRequest",
    "IdentityView",
    "mask_credential",
    "router",
]

router = APIRouter(prefix="/identities", tags=["admin-identities"])

# How much of a credential a listing may reveal. A reader can tell two similar
# tokens apart without being handed either of them; the point is that no
# response body ever carries the plaintext a caller authenticated with, so a
# leaked screenshot or a logged response is not a leaked credential.
_REVEAL_CHARS = 2
_MIN_MASK_CHARS = 4


def mask_credential(credential: str) -> str:
    """``credential`` with everything but its first characters replaced.

    Short tokens are masked entirely: revealing two of a four-character secret
    would be a worse trade than revealing none.
    """
    if len(credential) <= 2 * _REVEAL_CHARS:
        return "*" * _MIN_MASK_CHARS
    reveal = credential[:_REVEAL_CHARS]
    return reveal + "*" * max(len(credential) - _REVEAL_CHARS, _MIN_MASK_CHARS)


class IdentityView(BaseModel):
    """One stored identity, with its credential masked.

    ``subject_id`` is exposed so the admin surface has a stable key to PATCH;
    ``actor_id`` can be renamed by an office and is not a URL segment.
    """

    subject_id: str
    actor_id: str
    role: Role
    disabled: bool
    credential: str


class IdentityCreateRequest(BaseModel):
    """Body for ``POST /admin/identities``.

    The credential is supplied by the caller rather than generated here: this
    surface hands an existing secret to a named person, and inventing one in a
    response body would put a live credential in a place logs read.
    """

    actor_id: str
    role: Role
    credential: str
    subject_id: str | None = None
    password: str | None = None
    reason: str


class IdentityPatchRequest(BaseModel):
    """Body for ``PATCH /admin/identities/{subject_id}``.

    ``role`` and ``disabled`` are both optional but at least one is required -
    a mutation that changes nothing would still append an event, and an audit
    trail with no-ops in it stops being evidence.
    """

    role: Role | None = None
    disabled: bool | None = None
    password: str | None = None
    reason: str


def _store_or_503(request: Request):
    store = identity_store_for(request)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="identity store is not configured",
        )
    return store


def _ledger_for(request: Request):
    ledger = getattr(request.app.state, "audit_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=503,
            detail="audit ledger is not configured",
        )
    return ledger


def _view(entry: IdentityEntry) -> IdentityView:
    return IdentityView(
        subject_id=entry.subject_id,
        actor_id=entry.actor_id,
        role=entry.role,
        disabled=entry.disabled,
        credential=mask_credential(entry.credential),
    )


def _require_reason(reason: str) -> str:
    """Refuse a blank justification before anything is written."""
    if not reason.strip():
        raise HTTPException(
            status_code=422,
            detail="reason is required for a permission change",
        )
    return reason.strip()


def _four_eyes_blocks(
    identity: Identity,
    target: IdentityEntry,
    four_eyes_on: bool,
) -> None:
    """Refuse an actor changing their own grants while the control is on.

    The comparison is on ``actor_id`` rather than on the credential: the
    caller proves who they are through ``current_identity``, and the target
    entry is the one read from the store, so a body cannot point the check at
    someone else by restating an id.
    """
    if not four_eyes_on or target.actor_id != identity.actor_id:
        return
    raise HTTPException(
        status_code=403,
        detail=(
            f"cannot change permissions for {target.subject_id}: it is your "
            "own identity (four-eyes: a permission change needs a second "
            "pair of eyes)"
        ),
    )


@router.get("", response_model=list[IdentityView])
def list_identities(
    request: Request,
    _identity: Identity = Depends(require_action(AdminAction.ROLES_MANAGE)),
) -> list[IdentityView]:
    """Every stored identity, credentials masked."""
    store = _store_or_503(request)
    try:
        entries = store.entries()
    except IdentityStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [_view(entry) for entry in entries]


@router.post("", response_model=IdentityView)
def create_identity(
    payload: IdentityCreateRequest,
    request: Request,
    identity: Identity = Depends(require_action(AdminAction.ROLES_MANAGE)),
) -> IdentityView:
    """Register a credential for an actor and record why.

    ``before_role`` is ``None`` - the event says "this subject had no grants
    before", which is what makes a creation distinguishable in the ledger
    from a change down to ``viewer``.
    """
    store = _store_or_503(request)
    reason = _require_reason(payload.reason)
    # Read the switch exactly once; the check and the event share this value.
    four_eyes_on = four_eyes_enabled()
    entry = IdentityEntry(
        subject_id=(payload.subject_id or payload.actor_id).strip(),
        actor_id=payload.actor_id.strip(),
        role=payload.role,
        credential=payload.credential,
        password_hash=hash_password(payload.password) if payload.password else None,
    )

    try:
        store.add(entry)
    except IdentityConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IdentityStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _ledger_for(request).record_permission_change(
        actor_id=identity.actor_id,
        actor_role=identity.role.value,
        subject_id=entry.subject_id,
        before_role=None,
        after_role=entry.role.value,
        reason=reason,
        four_eyes=four_eyes_on,
    )
    return _view(entry)


@router.patch("/{subject_id}", response_model=IdentityView)
def patch_identity(
    subject_id: str,
    payload: IdentityPatchRequest,
    request: Request,
    identity: Identity = Depends(require_action(AdminAction.ROLES_MANAGE)),
) -> IdentityView:
    """Change one identity's role, or disable it, and record the change.

    Disabling is a role change like any other: the ledger sees
    ``after_role=None`` and can answer "when did this person lose access?".
    The entry is never deleted, because a removed row would take the target
    of the event with it.
    """
    store = _store_or_503(request)
    if payload.role is None and payload.disabled is None and payload.password is None:
        raise HTTPException(
            status_code=422,
            detail="role, disabled or password is required",
        )
    reason = _require_reason(payload.reason)
    four_eyes_on = four_eyes_enabled()

    try:
        current = store.get(subject_id)
    except IdentityStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if current is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown subject_id: {subject_id}",
        )
    _four_eyes_blocks(identity, current, four_eyes_on)

    try:
        entry = store.update(
            subject_id,
            role=payload.role,
            disabled=payload.disabled,
            password=payload.password,
        )
    except IdentityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IdentityStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _ledger_for(request).record_permission_change(
        actor_id=identity.actor_id,
        actor_role=identity.role.value,
        subject_id=entry.subject_id,
        before_role=current.role.value,
        after_role=None if entry.disabled else entry.role.value,
        reason=reason,
        four_eyes=four_eyes_on,
    )
    return _view(entry)
