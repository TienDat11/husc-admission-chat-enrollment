"""Audit event model — ticket 14.

The question this module exists to answer, in the admissions office's words:
*"vì sao dữ liệu đổi và ai đổi"* — why did this fact change, and who changed
it. An event is therefore not a log line about an HTTP call; it is a statement
about a change to canonical data, carrying who made it, what the version was
before and after, and why.

Two rules are enforced **here, at construction**, rather than at the API edge:

1. **A justification-bearing action must carry one.** ``reject`` throws work
   away, ``rollback`` retracts live data, ``conflict_override`` picks a winner
   between two official sources. Each is a decision somebody owns, so an event
   for one without a non-blank ``reason`` raises ``ValueError``. A route that
   forgot to pass the reason cannot produce a valid event at all — there is no
   "reason missing" state sitting in the ledger waiting to be discovered
   months later.
2. **A publish names the release it belongs to.** ``publish`` is only
   answerable later if the release is recorded (spec user story 26: *publish là
   một release ID cụ thể*), so a publish event without a ``release_id`` raises.
   See :data:`LEGACY_RELEASE_ID` for the one case where the publishing
   deployment has no release machinery to name.

The event shape follows the spec's decision block, verbatim::

    actor_id, role, action, entity, entity_id,
    before_version, after_version, reason,
    timestamp, source_ip/session, review_id, release_id

Two spellings differ from that list on purpose:

- ``entity_id`` is spelled ``target``, because the id may name a canonical
  record, a release, or a *subject* whose permissions changed — and the reader
  is asking "what did this touch?". :class:`TargetKind` says which of the three
  it is, so a release id and a record id never have to be told apart by
  squinting at the string.
- ``before_version`` / ``after_version`` are **record version keys**
  (``record_id@version``, ticket 02), not bare integers. A key resolves through
  ``CanonicalRecordStore.get_version`` / ``resolve_provenance``, which is what
  makes "trước/sau ra sao" answerable long after the fact. ``None`` means "no
  version on that side": a create has no before. For a
  ``permission_change`` — whose target is a subject, not a record — the two
  fields carry the subject's **role state** instead, encoded
  ``subject_id@role`` (``subject_id@none`` for a first grant,
  ``subject_id@disabled`` for a revocation), so "ai đổi quyền, từ gì sang gì"
  is answered by the same two fields the rest of the ledger uses.

The enumerated kinds are the ones the ticket requires every one of which is
recordable — :attr:`AuditAction.PUBLISH`, :attr:`~AuditAction.ROLLBACK`,
:attr:`~AuditAction.PERMISSION_CHANGE`,
:attr:`~AuditAction.SOURCE_DOCUMENT_CHANGE`,
:attr:`~AuditAction.CONFLICT_OVERRIDE` — plus the lifecycle writes ticket 12
already produces (``create``/``edit``/``submit``/``approve``/``reject``/
``supersede``). The lifecycle verbs are in the same enum because a ledger that
recorded publishes but not the approvals that made them publishable would not
answer "ai đổi" either. An action outside this enum cannot be constructed:
``AuditAction(value)`` raises, so a new kind is a deliberate change here rather
than a string a caller invented.

Which path emits what today, stated because the gap is the interesting part:

- ``create``, ``edit``, ``submit``, ``approve``, ``reject``, ``supersede``,
  ``publish`` and ``source_document_change`` — ``AdminLifecycleService``, one
  event per write (a publish transition carries the release it belongs to; an
  edit that moves the source document carries a second event).
- ``rollback``, ``permission_change`` and ``conflict_override`` — the model and
  the ledger record them, and there is a test that appends each, but the paths
  that *perform* those operations do not exist in this repo yet: data rollback
  is ticket 22, permission changes are deployment configuration (ticket 13's
  registry is environment, not an endpoint), and conflict override has no route.
  They go through the same :meth:`~src.infrastructure.audit_ledger.AuditLedger.append`
  those paths will call, so wiring them adds a caller rather than a schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping
from uuid import uuid4

from src.domain.canonical.lifecycle import ENTITY_NAME

__all__ = [
    "AuditAction",
    "AuditEvent",
    "LEGACY_RELEASE_ID",
    "REASON_REQUIRED",
    "SCHEMA_VERSION",
    "TargetKind",
]

SCHEMA_VERSION = "1.0"

# The release label a publish carries when the publishing deployment has no
# named release to attach it to. It is the same label ticket 34 already stamps
# on canonical chunks that belong to no release (`metadata_json.release_id`),
# so a reader sees one convention rather than two. It is a *label*, not a
# fabrication: the event says "this publish belongs to no named release" in a
# way that is queryable and that flips to the real release id the moment the
# publish path names one (tickets 16/22 create releases; ticket 19 passes it).
LEGACY_RELEASE_ID = "legacy"


class AuditAction(str, Enum):
    """What happened. An action outside this enum is not recordable."""

    # -- the kinds the ticket requires ------------------------------------
    PUBLISH = "publish"
    ROLLBACK = "rollback"
    PERMISSION_CHANGE = "permission_change"
    SOURCE_DOCUMENT_CHANGE = "source_document_change"
    CONFLICT_OVERRIDE = "conflict_override"
    # -- lifecycle writes (ticket 12) the ledger also records -------------
    CREATE = "create"
    EDIT = "edit"
    SUBMIT = "submit"
    APPROVE = "approve"
    REJECT = "reject"
    SUPERSEDE = "supersede"
    WITHDRAW = "withdraw"
    REVOKE_SUPERSESSION = "revoke_supersession"
    # -- T15: claiming an unclaimed draft (the seed's ``authored_by = null``
    #    is the version a claim attaches an author to)
    CLAIM = "claim"


class TargetKind(str, Enum):
    """Which id space ``AuditEvent.target`` names."""

    RECORD = "record"
    RELEASE = "release"
    SUBJECT = "subject"


# Actions whose legitimacy rests on a recorded justification. Reject discards a
# submission, rollback retracts live data, override resolves a conflict between
# two official sources, a permission change hands someone power they did not
# have, and withdraw / revoke_supersession retract a published statement (or a
# supersession decision) - each is a decision somebody owns, so an event for one
# without a non-blank ``reason`` raises ``ValueError``. A route that forgot to
# pass the reason cannot produce a valid event at all - there is no default to
# leak.
REASON_REQUIRED: frozenset[AuditAction] = frozenset(
    {
        AuditAction.REJECT,
        AuditAction.ROLLBACK,
        AuditAction.CONFLICT_OVERRIDE,
        AuditAction.PERMISSION_CHANGE,
        AuditAction.WITHDRAW,
        AuditAction.REVOKE_SUPERSESSION,
    }
)


@dataclass(frozen=True)
class AuditEvent:
    """One recorded change: who, what, before/after, and why.

    Frozen and validated in ``__post_init__``: an invalid event cannot be built,
    so no caller can hand the ledger something it would have to reject later.
    """

    action: AuditAction
    actor_id: str
    target: str
    timestamp: datetime
    before_version: str | None = None
    after_version: str | None = None
    reason: str | None = None
    role: str | None = None
    target_kind: TargetKind = TargetKind.RECORD
    entity: str = ENTITY_NAME
    source_ip: str | None = None
    review_id: str | None = None
    release_id: str | None = None
    event_id: str = field(default_factory=lambda: uuid4().hex)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Coercion first, so a caller may pass the string form of either enum
        # (a JSON body, an env-derived value) and an unknown value still fails
        # here rather than at the hash chain.
        object.__setattr__(self, "action", _as_action(self.action))
        object.__setattr__(self, "target_kind", _as_target_kind(self.target_kind))

        if not isinstance(self.timestamp, datetime):
            raise ValueError(
                f"audit event timestamp must be a datetime, got {type(self.timestamp).__name__}"
            )
        if not (self.actor_id or "").strip():
            raise ValueError(
                "an audit event requires actor_id — 'who' is the question this ledger answers"
            )
        if not (self.target or "").strip():
            raise ValueError(
                "an audit event requires a target (record id, release id or subject id)"
            )
        if not (self.event_id or "").strip():
            raise ValueError("an audit event requires an event_id")
        if self.action in REASON_REQUIRED and not (self.reason or "").strip():
            raise ValueError(
                f"'{self.action.value}' requires a reason — a decision without one "
                "is not accountable"
            )
        if self.action is AuditAction.PUBLISH and not (self.release_id or "").strip():
            raise ValueError(
                "publish requires release_id — a publish that cannot name its release "
                "cannot be rolled back"
            )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form: enums by value, datetime as ISO-8601, no datetime
        objects and no enum instances, so it survives a round trip through a
        ledger line unchanged."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "role": self.role,
            "action": self.action.value,
            "entity": self.entity,
            "target": self.target,
            "target_kind": self.target_kind.value,
            "before_version": self.before_version,
            "after_version": self.after_version,
            "reason": self.reason,
            "source_ip": self.source_ip,
            "review_id": self.review_id,
            "release_id": self.release_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AuditEvent":
        """Rebuild an event written by :meth:`to_dict`.

        Validates through the same constructor, so a ledger line whose event was
        edited into an invalid state (a reject with its reason removed) is
        rejected on read rather than silently trusted.
        """
        return cls(
            action=_as_action(data.get("action")),
            actor_id=str(data.get("actor_id") or ""),
            target=str(data.get("target") or ""),
            timestamp=_as_datetime(data.get("timestamp")),
            before_version=_optional_str(data.get("before_version")),
            after_version=_optional_str(data.get("after_version")),
            reason=_optional_str(data.get("reason")),
            role=_optional_str(data.get("role")),
            target_kind=_as_target_kind(data.get("target_kind") or TargetKind.RECORD),
            entity=str(data.get("entity") or ENTITY_NAME),
            source_ip=_optional_str(data.get("source_ip")),
            review_id=_optional_str(data.get("review_id")),
            release_id=_optional_str(data.get("release_id")),
            event_id=str(data.get("event_id") or ""),
            schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
        )


def _as_action(value: object) -> AuditAction:
    """``value`` as this module's :class:`AuditAction`.

    A member of the *other* module tree's ``AuditAction`` is accepted too: the
    repo loads ``src.…`` and ``rag2025.src.…`` in one process, and those members
    are ``==`` but not ``is`` — and ``str()`` of a ``(str, Enum)`` member is
    ``"AuditAction.PUBLISH"``, not ``"publish"``, so stringifying one would
    reject it with a message about an unknown action. Reading ``.value`` is what
    makes the two trees interchangeable.
    """
    if isinstance(value, AuditAction):
        return value
    try:
        return AuditAction(getattr(value, "value", value))
    except (ValueError, TypeError) as exc:
        known = sorted(action.value for action in AuditAction)
        raise ValueError(
            f"{value!r} is not an audit action; known actions: {known}"
        ) from exc


def _as_target_kind(value: object) -> TargetKind:
    """``value`` as this module's :class:`TargetKind`; see :func:`_as_action`."""
    if isinstance(value, TargetKind):
        return value
    try:
        return TargetKind(getattr(value, "value", value))
    except (ValueError, TypeError) as exc:
        known = sorted(kind.value for kind in TargetKind)
        raise ValueError(
            f"{value!r} is not a target kind; known kinds: {known}"
        ) from exc


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{value!r} is not an ISO-8601 timestamp") from exc


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
