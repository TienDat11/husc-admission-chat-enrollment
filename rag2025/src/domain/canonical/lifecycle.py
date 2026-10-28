"""Lifecycle state machine for canonical admissions records — ticket 12.

Spec state machine (verbatim from the spec's decision block)::

    draft --submit--> review --approve--> approved --publish--> published --supersede--> archived
      ^                  |                     |                    ^
      |                  +--reject(reason bắt buộc)--> draft        |
      |                    |--withdraw(reason bắt buộc)--> archived |
      +-------------------------------------------------------------+
                                        revoke_supersession (bắt buộc lý do)

Why this lives in the domain: the state machine is the rule that makes "no
shortcut from draft to published" and "a published version cannot be edited in
place" true regardless of caller. It is pure data plus pure functions — no
store, no HTTP, no clock. ``src/services/admin_lifecycle.py`` orchestrates it
against the versioned store, and the admin router maps its errors to 4xx.

Design note — forbidden transitions are *absent* from the table rather than
listed in a deny-list. A deny-list drifts: adding a status silently makes every
unlisted (status, action) pair legal. Here an unlisted pair is illegal by
construction, so the three named prohibitions hold without being special-cased.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping

from src.domain.canonical.record import CanonicalRecord, RecordStatus

ENTITY_NAME = "canonical_record"


class LifecycleAction(str, Enum):
    """A requested move along the lifecycle. Not every action is legal in
    every state — see ``_TRANSITIONS``."""

    SUBMIT = "submit"
    APPROVE = "approve"
    REJECT = "reject"
    PUBLISH = "publish"
    SUPERSEDE = "supersede"
    WITHDRAW = "withdraw"
    REVOKE_SUPERSESSION = "revoke_supersession"


class LifecycleError(Exception):
    """Base class for lifecycle rule violations."""


class IllegalTransitionError(LifecycleError):
    """The requested action is not permitted from the record's current status.

    Carries both ends so the caller can explain the move (and offer only
    ``legal_actions``) without re-deriving the table.
    """

    def __init__(self, current: RecordStatus, action: LifecycleAction) -> None:
        self.current = current
        self.action = action
        allowed = sorted(a.value for a in legal_actions(current))
        super().__init__(
            f"cannot {action.value} a record in status '{current.value}'; "
            f"allowed from here: {allowed or ['(terminal)']}"
        )


class RejectReasonRequiredError(LifecycleError):
    """A decision-bearing action without a reason is not a decision, it is a
    silent discard. ``action`` names the offender so the message a caller sees
    explains the move it actually refused (ticket 36 widened the set of
    reason-bearing actions beyond ``reject``)."""

    def __init__(self, action: "LifecycleAction | str | None" = None) -> None:
        verb = getattr(action, "value", action) or LifecycleAction.REJECT.value
        self.action = action
        super().__init__(f"{verb} requires a reason")


# The whole state machine, as data. An absent key means the move is forbidden.
# Published -> ARCHIVED on supersede (spec: archived is the terminal state and
# 'supersede' is the verb; superseded-ness itself is derived, see ticket 03).
#
# Ticket 36 adds the two ways *back*: WITHDRAW is a published record retracted
# with no replacement (so it reaches the same ARCHIVED status supersede leads
# to, but by its own edge — the distinction matters on disk, where withdrawal
# leaves no supersession claim), and REVOKE_SUPERSESSION is the only way out of
# ARCHIVED, undoing a supersede that should not have happened.
_TRANSITIONS: Mapping[tuple[RecordStatus, LifecycleAction], RecordStatus] = {
    (RecordStatus.DRAFT, LifecycleAction.SUBMIT): RecordStatus.REVIEW,
    (RecordStatus.REVIEW, LifecycleAction.APPROVE): RecordStatus.APPROVED,
    (RecordStatus.REVIEW, LifecycleAction.REJECT): RecordStatus.DRAFT,
    (RecordStatus.APPROVED, LifecycleAction.PUBLISH): RecordStatus.PUBLISHED,
    (RecordStatus.PUBLISHED, LifecycleAction.SUPERSEDE): RecordStatus.ARCHIVED,
    (RecordStatus.PUBLISHED, LifecycleAction.WITHDRAW): RecordStatus.ARCHIVED,
    (RecordStatus.ARCHIVED, LifecycleAction.REVOKE_SUPERSESSION): RecordStatus.PUBLISHED,
}

# Actions whose legitimacy rests on a recorded justification.
_REASON_REQUIRED: frozenset[LifecycleAction] = frozenset(
    {
        LifecycleAction.REJECT,
        LifecycleAction.WITHDRAW,
        LifecycleAction.REVOKE_SUPERSESSION,
    }
)


def is_legal_transition(current: RecordStatus, action: LifecycleAction) -> bool:
    """Whether ``action`` may be applied to a record in ``current``."""
    return (current, action) in _TRANSITIONS


def legal_actions(current: RecordStatus) -> frozenset[LifecycleAction]:
    """Every action permitted from ``current`` (empty in a terminal status)."""
    return frozenset(action for (status, action) in _TRANSITIONS if status is current)


def requires_reason(action: LifecycleAction) -> bool:
    """Whether ``action`` must carry a non-blank justification."""
    return action in _REASON_REQUIRED


def transition_status(
    current: RecordStatus,
    action: LifecycleAction,
    *,
    reason: str | None = None,
) -> RecordStatus:
    """The status ``current`` moves to under ``action``.

    Raises:
        IllegalTransitionError: the move is not in the state machine.
        RejectReasonRequiredError: a reason-bearing action arrived without one.
    """
    if requires_reason(action) and not (reason or "").strip():
        raise RejectReasonRequiredError(action)
    try:
        return _TRANSITIONS[(current, action)]
    except KeyError:
        raise IllegalTransitionError(current, action) from None


def apply_transition(
    record: CanonicalRecord,
    action: LifecycleAction,
    *,
    reason: str | None = None,
) -> CanonicalRecord:
    """A copy of ``record`` in the status ``action`` leads to.

    Pure status mapping: no version bump, no clock, no I/O. The store owns
    version numbers and the service owns the publication timestamp, so this
    stays usable as a rule oracle that a test can call without a store.
    """
    target = transition_status(record.status, action, reason=reason)
    return record.model_copy(update={"status": target})


@dataclass(frozen=True)
class LifecycleEvent:
    """One transition, recorded for the audit ledger.

    ``action`` is a :class:`LifecycleAction` value for a status transition, or
    ``"create"`` / ``"edit"`` for the two operations that write a version
    without necessarily moving status.

    Field set follows the spec's audit event shape. ``role`` / ``source_ip`` /
    ``review_id`` / ``release_id`` are optional because ticket 12 has no
    identity layer yet — ticket 13 (RBAC) and ticket 14 (append-only ledger)
    populate and persist them; the shape is fixed here so those tickets extend
    the event rather than redefine it.
    """

    actor_id: str
    action: str
    entity: str
    entity_id: str
    before_version: int
    after_version: int
    before_status: RecordStatus
    after_status: RecordStatus
    timestamp: datetime
    reason: str | None = None
    role: str | None = None
    source_ip: str | None = None
    review_id: str | None = None
    release_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form (enums by value, datetime as ISO-8601)."""
        return {
            "actor_id": self.actor_id,
            "role": self.role,
            "action": self.action,
            "entity": self.entity,
            "entity_id": self.entity_id,
            "before_version": self.before_version,
            "after_version": self.after_version,
            "before_status": self.before_status.value,
            "after_status": self.after_status.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "review_id": self.review_id,
            "release_id": self.release_id,
        }
