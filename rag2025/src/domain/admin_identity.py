"""Admin identity and the role x action matrix — ticket 13.

Pure: no env, no HTTP, no store, no clock. This module owns two things and
nothing else:

1. The vocabulary a caller is described in — :class:`Role` (who the caller is)
   and :class:`AdminAction` (what a route does).
2. The matrix that maps one to the other, as **data**.

The spec's matrix, verbatim (``.scratch/husc-rag-admin-upgrade/spec.md``)::

    viewer    : read admin surface
    editor    : CRUD draft; submit review
    reviewer  : approve/reject; KHÔNG publish; KHÔNG tự approve bản mình sửa
    publisher : publish approved release; rollback retrieval
    admin     : role/config; data rollback; conflict override

Why the matrix is a table rather than per-endpoint checks: an endpoint that
checks "is this an editor" is a second place the policy lives, and the two
places drift. Here a route names the *action* it performs and this table
decides who may perform it — so revoking a permission is a one-line data change
and a newly added action is granted to nobody until someone grants it
deliberately (:func:`role_allows` denies anything absent from the table).

Roles are derived from a credential server-side (:mod:`src.config.admin_identity`).
Nothing in this module reads a request, so a client cannot assert its own role.

Notes on two entries the spec names but this ticket has no endpoint for:

- ``records.rollback`` (data rollback) and ``records.override`` (conflict
  override) are granted to ``admin`` only. They are in the table now because
  the matrix is the contract tickets 16/22/33 build against; a route that
  performs them will name the action rather than re-derive the policy.
- ``ops`` is not a spec role. It is the machine credential's role: the shared
  ``ADMIN_API_TOKEN`` keeps serving the ops surface (warmup, reload-table,
  graph update) and is granted **nothing** on the admissions lifecycle, which
  is what makes "the shared token cannot publish" a property of the table
  instead of a promise in a comment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.domain.canonical.lifecycle import LifecycleAction

__all__ = [
    "AdminAction",
    "Identity",
    "ROLE_MATRIX",
    "Role",
    "action_for_transition",
    "granted_actions",
    "role_allows",
]

class Role(str, Enum):
    """Who a caller is, as decided by the server from its credential."""

    VIEWER = "viewer"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    PUBLISHER = "publisher"
    REVIEWER_PUBLISHER = "reviewer_publisher"
    ADMIN = "admin"
    # Machine credential (the shared ops token). Never a human actor, and never
    # granted an admissions action.
    OPS = "ops"

class AdminAction(str, Enum):
    """What a route does, in RBAC terms.

    Names are ``<surface>.<verb>`` so the matrix reads as a permission list and
    a route's requirement is greppable. Lifecycle transitions have one action
    each rather than a single ``records.transition``: "may submit" and "may
    publish" are different grants, and collapsing them would hand an editor the
    publish button.
    """

    # Admissions lifecycle (spec: viewer/editor/reviewer/publisher/admin rows)
    RECORDS_READ = "records.read"
    RECORDS_CREATE = "records.create"
    RECORDS_EDIT = "records.edit"
    RECORDS_SUBMIT = "records.submit"
    RECORDS_APPROVE = "records.approve"
    RECORDS_REJECT = "records.reject"
    RECORDS_PUBLISH = "records.publish"
    RECORDS_SUPERSEDE = "records.supersede"
    # The two ways back for a published record (ticket 36). Separate grants from
    # supersede on purpose: replacing a record and retracting one are different
    # decisions an office delegates differently.
    RECORDS_WITHDRAW = "records.withdraw"
    RECORDS_REVOKE_SUPERSESSION = "records.revoke_supersession"
    RECORDS_ROLLBACK = "records.rollback"
    RECORDS_OVERRIDE = "records.override"
    # Ops surface: infrastructure, never admissions content.
    OPS_WARMUP = "ops.warmup"
    OPS_RELOAD_TABLE = "ops.reload_table"
    GRAPH_UPDATE = "graph.update"
    # Role management (ticket 37): the identity surface. Deliberately NOT in
    # ``_TRANSITION_ACTIONS`` - it is not a lifecycle action, and the
    # transitions endpoint must keep rejecting it.
    ROLES_MANAGE = "roles.manage"

# The policy, as data. A role absent from this mapping grants nothing; an action
# absent from a role's set is denied. Both directions fail closed.
#
# ``admin`` is an explicit superset rather than ``frozenset(AdminAction)`` on
# purpose: computed-as-everything would silently grant each future action to
# admin the moment it is declared, which is exactly the "added later" case this
# table exists to catch.
ROLE_MATRIX: dict[Role, frozenset[AdminAction]] = {
    Role.VIEWER: frozenset(
        {
            AdminAction.RECORDS_READ,
        }
    ),
    Role.EDITOR: frozenset(
        {
            AdminAction.RECORDS_READ,
            AdminAction.RECORDS_CREATE,
            AdminAction.RECORDS_EDIT,
            AdminAction.RECORDS_SUBMIT,
        }
    ),
    Role.REVIEWER: frozenset(
        {
            AdminAction.RECORDS_READ,
            AdminAction.RECORDS_APPROVE,
            AdminAction.RECORDS_REJECT,
        }
    ),
    Role.PUBLISHER: frozenset(
        {
            AdminAction.RECORDS_READ,
            AdminAction.RECORDS_PUBLISH,
            AdminAction.RECORDS_SUPERSEDE,
            # A publisher may retract what it published, but un-doing someone
            # else's supersession decision is the admin's (ticket 36).
            AdminAction.RECORDS_WITHDRAW,
        }
    ),
    Role.REVIEWER_PUBLISHER: frozenset(
        {
            AdminAction.RECORDS_READ,
            AdminAction.RECORDS_APPROVE,
            AdminAction.RECORDS_REJECT,
            AdminAction.RECORDS_PUBLISH,
            AdminAction.RECORDS_SUPERSEDE,
            AdminAction.RECORDS_WITHDRAW,
        }
    ),
    Role.ADMIN: frozenset(
        {
            AdminAction.RECORDS_READ,
            AdminAction.RECORDS_CREATE,
            AdminAction.RECORDS_EDIT,
            AdminAction.RECORDS_SUBMIT,
            AdminAction.RECORDS_APPROVE,
            AdminAction.RECORDS_REJECT,
            AdminAction.RECORDS_PUBLISH,
            AdminAction.RECORDS_SUPERSEDE,
            AdminAction.RECORDS_WITHDRAW,
            AdminAction.RECORDS_REVOKE_SUPERSESSION,
            AdminAction.RECORDS_ROLLBACK,
            AdminAction.RECORDS_OVERRIDE,
            AdminAction.OPS_WARMUP,
            AdminAction.OPS_RELOAD_TABLE,
            AdminAction.GRAPH_UPDATE,
            AdminAction.ROLES_MANAGE,
        }
    ),
    Role.OPS: frozenset(
        {
            AdminAction.OPS_WARMUP,
            AdminAction.OPS_RELOAD_TABLE,
            AdminAction.GRAPH_UPDATE,
        }
    ),
}

# Which grant a lifecycle transition needs. Kept beside the matrix so the two
# are read together; a lifecycle action added by a later ticket is absent here
# and therefore denies (see ``action_for_transition``).
_TRANSITION_ACTIONS: Mapping[LifecycleAction, AdminAction] = {
    LifecycleAction.SUBMIT: AdminAction.RECORDS_SUBMIT,
    LifecycleAction.APPROVE: AdminAction.RECORDS_APPROVE,
    LifecycleAction.REJECT: AdminAction.RECORDS_REJECT,
    LifecycleAction.PUBLISH: AdminAction.RECORDS_PUBLISH,
    LifecycleAction.SUPERSEDE: AdminAction.RECORDS_SUPERSEDE,
    LifecycleAction.WITHDRAW: AdminAction.RECORDS_WITHDRAW,
    LifecycleAction.REVOKE_SUPERSESSION: AdminAction.RECORDS_REVOKE_SUPERSESSION,
}

@dataclass(frozen=True)
class Identity:
    """An authenticated caller: who they are and what they may do.

    ``actor_id`` is the name the audit event carries; ``role`` selects the
    matrix row. Both come from the credential registry, never from the request
    body — that is the whole point of this module.
    """

    actor_id: str
    role: Role

def _as_role(value: Role | str) -> Role | None:
    """``value`` as a :class:`Role`, or ``None`` when it names no known role."""
    if isinstance(value, Role):
        return value
    try:
        return Role(value)
    except (ValueError, TypeError):
        return None

def _as_action(value: AdminAction | str) -> AdminAction | None:
    """``value`` as an :class:`AdminAction`, or ``None`` when unknown."""
    if isinstance(value, AdminAction):
        return value
    try:
        return AdminAction(value)
    except (ValueError, TypeError):
        return None

def role_allows(role: Role | str, action: AdminAction | str) -> bool:
    """Whether ``role`` grants ``action``.

    Fails closed in both dimensions: an unknown role grants nothing, and an
    unknown action is granted to nobody — so an action declared later (or a
    typo in a route's requirement) denies until it is added to the table.
    """
    resolved_role = _as_role(role)
    resolved_action = _as_action(action)
    if resolved_role is None or resolved_action is None:
        return False
    return resolved_action in ROLE_MATRIX.get(resolved_role, frozenset())

def granted_actions(role: Role | str) -> frozenset[AdminAction]:
    """Every action ``role`` may perform (empty for an unknown role)."""
    resolved = _as_role(role)
    if resolved is None:
        return frozenset()
    return ROLE_MATRIX.get(resolved, frozenset())

def action_for_transition(action: LifecycleAction | str) -> AdminAction | None:
    """The grant a lifecycle transition requires, or ``None`` when unmapped.

    ``None`` means "deny": the caller of this function must refuse rather than
    fall back to a default, so a lifecycle action added by a later ticket
    cannot become callable simply by existing.
    """
    if isinstance(action, LifecycleAction):
        resolved = action
    else:
        try:
            resolved = LifecycleAction(action)
        except (ValueError, TypeError):
            return None
    return _TRANSITION_ACTIONS.get(resolved)
