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

import uuid
from dataclasses import replace
from datetime import datetime

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
from src.infrastructure.sql_identity_repo import (
    IdentityRuleError,
    LastAdminError,
    PermissionAudit,
    SqlIdentityRepository,
    issue_header_token,
)

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
    credential: str | None = None
    last_login_at: datetime | None = None
    header_token: str | None = None


class IdentityCreateRequest(BaseModel):
    """Body for ``POST /admin/identities``.

    File store: the caller supplies the credential, as before, so an existing
    secret can be handed to a named person. PostgreSQL store: the server issues
    the header token (``issue_header_token``) and returns it exactly once.
    """

    actor_id: str
    role: Role
    credential: str | None = None
    issue_header_token: bool = False
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


def _view(entry: IdentityEntry, *, header_token: str | None = None) -> IdentityView:
    return IdentityView(
        subject_id=entry.subject_id,
        actor_id=entry.actor_id,
        role=entry.role,
        disabled=entry.disabled,
        credential=mask_credential(entry.credential) if entry.credential else None,
        last_login_at=entry.last_login_at,
        header_token=header_token,
    )


def _refuse_self_change(
    identity: Identity,
    target: IdentityEntry,
    payload: IdentityPatchRequest,
) -> None:
    """Refuse changing your own role or disabling your own account (SQL path).

    Applies whether or not ``ADMIN_REQUIRE_FOUR_EYES`` is on: a second admin
    cannot approve a change to one's own permissions, so it is refused outright.
    """
    if target.actor_id != identity.actor_id:
        return
    changes_role = payload.role is not None and payload.role != target.role
    disables = payload.disabled is True and not target.disabled
    if changes_role or disables:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "SELF_ROLE_CHANGE_FORBIDDEN",
                "message": "Không thể tự thay đổi vai trò hoặc tự vô hiệu hóa tài khoản của chính mình.",
            },
        )


def _sql_entry_for_create(payload: IdentityCreateRequest) -> tuple[IdentityEntry, str | None]:
    """The entry for the PostgreSQL store.

    The server issues the subject id unless a UUID is supplied, and issues a
    header token when asked. The plaintext token is returned once by the caller
    and never stored.
    """
    if payload.credential is not None:
        raise HTTPException(
            status_code=422,
            detail="credential is not accepted on the PostgreSQL store; set issue_header_token",
        )
    if not payload.password and not payload.issue_header_token:
        raise HTTPException(
            status_code=422,
            detail="a password or issue_header_token is required",
        )
    raw_subject = (payload.subject_id or "").strip() or str(uuid.uuid4())
    try:
        subject_id = str(uuid.UUID(raw_subject))
    except ValueError:
        raise HTTPException(status_code=422, detail="subject_id must be a UUID") from None
    token = issue_header_token() if payload.issue_header_token else None
    entry = IdentityEntry(
        subject_id=subject_id,
        actor_id=payload.actor_id.strip(),
        role=payload.role,
        credential=token,
        password_hash=hash_password(payload.password) if payload.password else None,
    )
    return entry, token


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
    sql = isinstance(store, SqlIdentityRepository)
    if sql:
        entry, header_token = _sql_entry_for_create(payload)
    else:
        if not payload.credential:
            raise HTTPException(status_code=422, detail="credential is required")
        entry = IdentityEntry(
            subject_id=(payload.subject_id or payload.actor_id).strip(),
            actor_id=payload.actor_id.strip(),
            role=payload.role,
            credential=payload.credential,
            password_hash=hash_password(payload.password) if payload.password else None,
        )
        header_token = None

    audit = PermissionAudit(
        actor_id=identity.actor_id,
        actor_role=identity.role.value,
        reason=reason,
        four_eyes=four_eyes_on,
    )
    try:
        if sql:
            store.add(entry, created_by=identity.actor_id, audit=audit)
        else:
            store.add(entry)
    except IdentityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "IDENTITY_EXISTS", "message": str(exc)},
        ) from exc
    except IdentityRuleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IdentityStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not sql:
        _ledger_for(request).record_permission_change(
            actor_id=identity.actor_id,
            actor_role=identity.role.value,
            subject_id=entry.subject_id,
            before_role=None,
            after_role=entry.role.value,
            reason=reason,
            four_eyes=four_eyes_on,
        )
    view_entry = replace(entry, credential=None) if sql else entry
    return _view(view_entry, header_token=header_token)


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
    if isinstance(store, SqlIdentityRepository):
        _refuse_self_change(identity, current, payload)
    _four_eyes_blocks(identity, current, four_eyes_on)
    audit = PermissionAudit(
        actor_id=identity.actor_id,
        actor_role=identity.role.value,
        reason=reason,
        four_eyes=four_eyes_on,
    )

    try:
        if isinstance(store, SqlIdentityRepository):
            entry = store.update(
                subject_id,
                role=payload.role,
                disabled=payload.disabled,
                password=payload.password,
                audit=audit,
            )
        else:
            entry = store.update(
                subject_id,
                role=payload.role,
                disabled=payload.disabled,
                password=payload.password,
            )
    except IdentityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LastAdminError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "LAST_ADMIN", "message": str(exc)},
        ) from exc
    except IdentityRuleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IdentityStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not isinstance(store, SqlIdentityRepository):
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
