"""Admin lifecycle orchestration for canonical admissions records — ticket 12.

Sits between the admin HTTP layer (``src/routers/admin.py``) and the versioned
record store. The state-machine rules live in ``src.domain.canonical.lifecycle``
(pure); this module is the part that needs I/O — reading the current version,
appending the next one, resolving the supersession relation, and emitting one
audit event per change.

Immutability and concurrency, concretely:

- **Published versions are never edited in place.** Any edit appends a new
  version. Because a content change invalidates whatever review the prior
  version carried, the appended version always returns to ``draft`` — so there
  is no path where edited content stays ``published`` or ``approved`` without
  being re-reviewed.
- **Concurrent edits are detected, not silently merged.** Callers pass the
  version they believe is current (``expected_version``). A mismatch raises
  :class:`VersionConflictError`. The store's own append-only check is the
  backstop: if a write races past the read, ``write_record`` refuses the
  version and its :class:`RecordVersionConflictError` is re-raised as the same
  conflict, so a lost update cannot happen silently.

Supersession is resolved through the store, never supplied by the caller. The
spec defines "superseded" as a *relation* — a record is superseded iff another
**published** record names it in ``supersedes`` — so ``supersede`` archives a
published record only when ``CanonicalRecordStore.superseding_record`` already
finds a published replacement for it. That is also what makes the action
impossible to self-inflict: there is no target parameter to set, and the store's
reverse index skips a version that names its own ``record_id``, so a record
cannot nominate itself as its own replacement.

Timestamps come from an injectable ``clock`` so tests are deterministic; the
default is UTC ``datetime.now``, matching the repo's existing convention.

Audit (ticket 14): every write this service performs also appends an
:class:`~src.domain.audit.AuditEvent` to the append-only ledger when one is
configured, carrying the **record version key** before and after — a key that
resolves through the store — and, for a publish, the release it belongs to. An
edit that moves the source document appends a ``source_document_change`` event
as well, so "which facts had their provenance moved, and by whom?" is a query
rather than a diff-reading exercise. Both events reference the same before and
after versions, because both describe the same version change from a different
question's angle.

The store is the source of truth, so the store write happens first and the audit
append follows it. That ordering has one honest failure mode: the version is
stored and the event is not. It is not hidden and not papered over — a failed
append raises :class:`AuditRecordingError`, which says exactly that, naming the
record, the version, the actor, the ledger path and the cause, because a caller
who cannot distinguish "nothing happened" from "it happened and is unaudited"
will retry a write that already landed and will report a publish that stands as
one that never occurred. Recording the event *before* the store write is not the
fix: it records a write that may never happen, which is the same lie pointing
the other way.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
import hashlib
from src.domain.audit import LEGACY_RELEASE_ID, AuditAction, AuditEvent
from src.domain.canonical.lifecycle import (
    ENTITY_NAME,
    IllegalTransitionError,
    LifecycleAction,
    LifecycleEvent,
    apply_transition,
    is_legal_transition,
)
from src.domain.canonical.record import CanonicalRecord, RecordStatus, RecordType
from src.infrastructure.audit_ledger import AuditLedger
from src.infrastructure.source_document_registry import (
    SOURCE_DOCUMENTS_DIRNAME,
    UPLOAD_ID_PREFIX,
    SourceDocumentRegistry,
)
from src.infrastructure.canonical_record_store import (
    CanonicalRecordStore,
    RecordNotFoundError,
    RecordVersionConflictError,
)
from src.services.reindex_outbox import ReindexOutbox

Clock = Callable[[], datetime]

# Which audit action each write this service performs is recorded as. Built from
# the lifecycle enum, so a lifecycle action added by a later ticket without an
# audit counterpart fails at import — loudly, at startup — instead of after a
# version has already been written and could no longer be recorded.
_AUDIT_ACTION: dict[str, AuditAction] = {
    action.value: AuditAction(action.value) for action in LifecycleAction
} | {
    "create": AuditAction.CREATE,
    "edit": AuditAction.EDIT,
    "claim": AuditAction.CLAIM,
}

# The provenance fields whose movement is a ``source_document_change``.
_PROVENANCE_FIELDS = ("source_document_id", "source_url", "source_page")


def _nfc(text: str) -> str:
    """Normalise one string to NFC (ticket 03, spec 02 §1).

    ``chunk_identity`` deliberately does *not* normalise Unicode, so the
    stored ``text`` must already be NFC by the time it reaches the store —
    this is that boundary. Legacy 1.0 records (seed, NFD-era) are never
    rewritten; only newly authored or edited text is normalised.
    """
    import unicodedata
    return unicodedata.normalize("NFC", text)


def _nfc_record_text(record: CanonicalRecord, *, fields: tuple[str, ...] = ("text",)) -> CanonicalRecord:
    """Return ``record`` with every string field in ``fields`` normalised to NFC."""
    changes = {f: _nfc(getattr(record, f)) for f in fields if isinstance(getattr(record, f), str)}
    return record.model_copy(update=changes) if changes else record



class AdminLifecycleError(Exception):
    """Base class for lifecycle orchestration failures."""


class AuditRecordingError(AdminLifecycleError):
    """A write reached the store but could not be recorded in the audit trail.

    Raised instead of letting the underlying I/O error escape, because the bare
    error reads as "the operation failed" while the truth is the opposite: the
    version is stored and live, and the audit trail does not know about it. A
    caller that cannot tell those apart will retry a write that already
    happened, or report a publish that stands as one that never occurred — and
    the audit ledger's whole job is to be the record nobody has to guess about.

    This is deliberately *not* fixed by recording the event before the store
    write: that records a write that may never happen, which is the same lie
    pointing the other way. The store is the source of truth; the honest
    response when the record cannot be written is to say so loudly and stop.
    """

    def __init__(
        self,
        *,
        action: str,
        record_id: str,
        version: int,
        actor_id: str,
        ledger_path: str | None,
        cause: BaseException,
    ) -> None:
        self.action = action
        self.record_id = record_id
        self.version = version
        self.actor_id = actor_id
        self.ledger_path = ledger_path
        self.cause = cause
        super().__init__(
            f"{action} of {record_id}@{version} by {actor_id} was stored but "
            f"NOT recorded in the audit trail"
            + (f" ({ledger_path})" if ledger_path else "")
            + f": {cause!r}. The version is live in the store; re-running the "
            "operation will fail with a version conflict. Reconcile the ledger "
            "before retrying, and do not treat this write as unaudited-but-fine."
        )


class RecordNotFound(AdminLifecycleError):
    """No stored version matches the requested ``record_id``."""


class VersionConflictError(AdminLifecycleError):
    """The caller's ``expected_version`` is not the current version.

    Raised instead of overwriting so two editors cannot silently lose each
    other's work.
    """

    def __init__(self, record_id: str, expected: int, actual: int) -> None:
        self.record_id = record_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"version conflict for {record_id}: expected version {expected}, "
            f"current version is {actual}"
        )


class NotSupersededError(AdminLifecycleError):
    """A ``supersede``/``revoke_supersession`` was requested but nothing
    supersedes this record.

    Without a replacement, archiving would retract live admissions data and
    leave nothing in its place — that is a revocation, not a supersession, and
    the spec keeps the two distinguishable. The same predicate answers both
    directions of ticket 36: supersede refuses to archive what nothing
    replaces, revoke refuses to un-supersede what was never superseded (a
    record archived by *withdrawal* has no claim to retract — its message
    says so, because "publish the replacement first" would be advice for an
    action that cannot fix this state).
    """

    def __init__(self, record_id: str, *, context: str = "supersede") -> None:
        self.record_id = record_id
        self.context = context
        if context == "revoke":
            super().__init__(
                f"cannot revoke supersession of {record_id}: it is not "
                "currently superseded (archived by withdrawal, or never "
                "superseded — there is no supersession claim to retract)"
            )
        else:
            super().__init__(
                f"cannot supersede {record_id}: no published record names it "
                "in 'supersedes' (publish the replacement first, or revoke "
                "explicitly)"
            )

class FourEyesViolation(AdminLifecycleError):
    """A reviewer tried to approve a version they authored (ticket 13), or a
    caller tried to revoke a supersession decision they themselves made
    (ticket 36).

    The spec's reviewer row ends with "KHÔNG tự approve bản mình sửa": approval
    is a second pair of eyes, so an approval by the author is not an approval.
    Unknown authorship denies too — "cannot prove it is someone else" is not
    the same as "it is someone else", and four-eyes is a control, not a
    best-effort check. The revoke case reuses the rule: un-superseding is the
    office cancelling *someone else's* supersession decision, so the same
    "second pair of eyes" bar applies with the verbs changed.
    """

    def __init__(
        self,
        record_id: str,
        version: int,
        author: str | None,
        *,
        operation: str = "approve",
    ) -> None:
        self.record_id = record_id
        self.version = version
        self.author = author
        self.operation = operation
        if operation == "approve":
            detail = (
                "its author is not recorded, so four-eyes cannot be verified"
                if author is None
                else f"it was authored by '{author}' (four-eyes: a reviewer "
                "must not approve their own version)"
            )
        else:
            detail = (
                "the supersession decision is not recorded, so four-eyes "
                "cannot be verified"
                if author is None
                else f"the supersession was decided by '{author}' "
                "(four-eyes: an operator must not revoke their own decision)"
            )
        super().__init__(f"cannot {operation} {record_id}@{version}: {detail}")


class ClaimRefusedError(AdminLifecycleError):
    """A ``claim`` was requested on a version it is not legal to claim (ticket T15).

    A claim is only a way to attach an author to a *draft* nobody has claimed
    yet (spec 02 §108: the seed drafts all carry ``authored_by = null``). The
    two refusals:

    - **a non-draft status** — a version that has been reviewed, approved,
      published or archived already carries a decision, and claiming it back
      into draft would silently retract that decision. The move is refused,
      never demoted: what claim refuses here is the same demotion an edit
      performs openly (and is never allowed to perform *without* an edit).
    - **a version already authored** — claiming your own draft is a no-op the
      store would record as a version with nothing to say, and claiming
      someone else's draft would steal their byline. Both refuse; the
      version is the version, and authorship is stamped when it is written.

    The refusal happens before any write, so a refused claim leaves the store
    byte-identical, like the other lifecycle refusals.
    """

    def __init__(self, record_id: str, version: int, reason: str) -> None:
        self.record_id = record_id
        self.version = version
        self.reason = reason
        super().__init__(f"cannot claim {record_id}@{version}: {reason}")


class AdminLifecycleService:
    """Create, edit, claim, and transition canonical records under lifecycle rules."""

    def __init__(
        self,
        store: CanonicalRecordStore,
        *,
        clock: Clock | None = None,
        events: list[LifecycleEvent] | None = None,
        four_eyes: bool | Callable[[], bool] = True,
        ledger: AuditLedger | None = None,
        outbox: ReindexOutbox | None = None,
        registry: SourceDocumentRegistry | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Four-eyes (ticket 13): when on, an approve by the version's author is
        # refused, and (ticket 36) so is a revoke of one's own supersession.
        # Configurable so a single-operator deployment can turn it off
        # deliberately; the default is on, so forgetting to configure it keeps
        # the control. A callable is consulted at *each check* rather than
        # snapshotted at construction, so the wired router (which passes
        # ``four_eyes_enabled`` itself) re-reads the environment per request.
        self._four_eyes: Callable[[], bool] = (
            four_eyes if callable(four_eyes) else lambda: bool(four_eyes)
        )
        # The in-process stream the admin router returns in its response
        # (ticket 12's shape), and the durable hash-chained ledger (ticket 14).
        # Two sinks on purpose: the first is one event per write for the caller
        # that just made the write, the second is the append-only record, which
        # may carry more than one event for a single write (an edit that also
        # moved the source document). The ledger is optional so a read-only or
        # test-built service needs no file.
        self._events: list[LifecycleEvent] = events if events is not None else []
        self._ledger = ledger
        # The re-index queue (ticket 15). Optional for the same reason the
        # ledger is: a service built only to read, or to exercise the state
        # machine, needs no file. When present it is fed from the publish
        # transition alone — a publication is the only event that changes what
        # students are served.
        self._outbox = outbox
        # T07: the source-document registry a HOI_DAP create writes its
        # generated document into. Optional so every non-HOI_DAP path and
        # every existing test-built service needs no registry file.
        self._registry = registry

    @property
    def store(self) -> CanonicalRecordStore:
        return self._store

    @property
    def ledger(self) -> AuditLedger | None:
        """The append-only ledger this service records to, or ``None``."""
        return self._ledger

    @property
    def outbox(self) -> ReindexOutbox | None:
        """The re-index outbox this service queues publications on, or ``None``."""
        return self._outbox

    @property
    def registry(self) -> SourceDocumentRegistry | None:
        """The source-document registry, or ``None`` for a test-built service."""
        return self._registry


    def emitted_events(self) -> tuple[LifecycleEvent, ...]:
        """Every event this service instance has emitted, in order."""
        return tuple(self._events)

    # -- reads ---------------------------------------------------------------

    def get(self, record_id: str) -> CanonicalRecord:
        """Current version of ``record_id``."""
        return self._latest(record_id)

    def history(self, record_id: str) -> tuple[CanonicalRecord, ...]:
        """Every stored version of ``record_id``, newest first."""
        chain = self._chain(record_id)
        return (chain.latest, *chain.prior)

    # -- writes --------------------------------------------------------------

    def create(
        self, record: CanonicalRecord, *, actor_id: str, role: str | None = None
    ) -> CanonicalRecord:
        """Store ``record`` as version 1 of its ``record_id``.

        A ``record_id`` that already has versions is a conflict, not a create —
        the store's append-only check refuses the write and we surface it as a
        version conflict naming the version that is actually next.

        Authorship is stamped from the authenticated ``actor_id``, overwriting
        whatever the caller sent: a caller must not be able to claim a version
        was written by someone else, or four-eyes would be a formality.

        T07 (spec 02 §1): a HOI_DAP record's source document is generated by
        the server at create time — a real text file under the registry's
        documents directory, holding the question–answer pair, the author and
        the date — and registered before the record is written, so version 1
        already carries a ``source_document_id`` the publish invariant accepts.
        A caller-supplied id on a HOI_DAP create is overwritten: the frontend
        used to fabricate one (``hoi-dap-${Date.now()}``) and the server
        trusted it, which let a hand-typed answer reach students with a
        citation chip pointing at nothing.
        """
        if record.version != 1:
            raise VersionConflictError(record.record_id, 1, record.version)
        authored = _nfc_record_text(record, fields=("text", "source_url")).model_copy(
            update={"authored_by": actor_id}
        )
        # T07 (spec 02 §1): generate the source document before the store
        # write, so version 1 already carries a real id. Non-HOI_DAP paths
        # skip this block entirely and keep today's behaviour.
        if record.record_type is RecordType.HOI_DAP:
            doc_id = self._generate_hoi_dap_source_document(authored, actor_id)
            authored = authored.model_copy(update={"source_document_id": doc_id})
        try:
            stored = self._store.write_record(authored)
        except RecordVersionConflictError as exc:
            raise VersionConflictError(
                record.record_id, 1, self._store.next_version(record.record_id)
            ) from exc
        self._emit(
            actor_id=actor_id,
            role=role,
            action="create",
            record=stored,
            before_version=0,
            before_status=stored.status,
            reason=None,
        )
        return stored

    def edit(
        self,
        record_id: str,
        *,
        actor_id: str,
        role: str | None = None,
        expected_version: int | None = None,
        reason: str | None = None,
        **changes: object,
    ) -> CanonicalRecord:
        """Append a new version of ``record_id`` with ``changes`` applied.

        The appended version is forced back to ``draft``: the previous version's
        review no longer describes this content, so carrying ``approved`` or
        ``published`` forward would let unreviewed text stay live.
        """
        current = self._require_current(record_id, expected_version)
        payload = current.model_dump()
        payload.update(changes)
        payload["status"] = RecordStatus.DRAFT
        payload["version"] = current.version + 1
        # A fresh draft carries no publication stamp.
        payload["published_at"] = None
        # …and its author is whoever is writing it now, not the previous
        # version's author — an edit is a new claim, so it needs new eyes.
        payload["authored_by"] = actor_id

        updated = _nfc_record_text(
            CanonicalRecord.model_validate(payload), fields=("text", "source_url")
        )
        stored = self._append(record_id, updated, expected_version=current.version)
        emitted_at = self._emit(
            actor_id=actor_id,
            role=role,
            action="edit",
            record=stored,
            before_version=current.version,
            before_status=current.status,
            reason=reason,
        )
        if _provenance_changed(current, stored):
            self._emit_source_document_change(
                actor_id=actor_id,
                role=role,
                record=stored,
                before_version=current.version,
                reason=reason,
                timestamp=emitted_at,
            )
        return stored

    def transition(
        self,
        record_id: str,
        action: LifecycleAction,
        *,
        actor_id: str,
        role: str | None = None,
        expected_version: int | None = None,
        reason: str | None = None,
        release_id: str | None = None,
    ) -> CanonicalRecord:
        """Move ``record_id`` along the state machine.

        Raises the domain's ``IllegalTransitionError`` for a move the state
        machine forbids and ``RejectReasonRequiredError`` for a reject with no
        justification. Both are raised before any write, so a rejected move
        leaves the store untouched.

        ``approve`` additionally enforces four-eyes: the approver must differ
        from the version's recorded author (see :class:`FourEyesViolation`).

        ``release_id`` names the release a ``publish`` belongs to and is what
        the audit event records; the publish path that knows its release (the
        swap of ticket 16, the rollback of ticket 22) passes it. See
        :meth:`_release_for_publish` for what a publish without one records.
        """
        # Coerced to *this* module's enum before anything compares it. The repo
        # loads two module trees in one process (``rag2025.src.…`` and
        # ``src.…``), so a caller can hand us ``LifecycleAction.PUBLISH`` from
        # the other tree: a different class object whose members are ``==`` but
        # not ``is``. Every ``is`` check below would then be False for a genuine
        # publish — and because the store write happens before the emit, the
        # version would be published with no audit event and a ValueError raised
        # from the event model. Coercing once, here, makes the rest of the
        # method independent of which tree the caller imported from.
        action = _as_action(action)
        current = self._require_current(record_id, expected_version)
        if action is LifecycleAction.APPROVE:
            self._require_independent_approver(current, actor_id)
        if action is LifecycleAction.SUPERSEDE:
            self._require_published_replacement(current)
        revoke_claims: tuple[CanonicalRecord, ...] | None = None
        if action is LifecycleAction.REVOKE_SUPERSESSION:
            revoke_claims = self._require_revocable_supersession(current, actor_id)
        # Rule check happens before any write, so an illegal move is a no-op.
        moved = apply_transition(current, action, reason=reason)

        payload = moved.model_dump()
        payload["version"] = current.version + 1
        if payload["status"] == RecordStatus.PUBLISHED.value and not payload.get(
            "published_at"
        ):
            payload["published_at"] = self._clock()
        updated = CanonicalRecord.model_validate(payload)

        # The retraction markers go down *before* the restored version is
        # appended (ticket 36): if the append then fails, the markers are
        # harmless — A is still archived, the index still excludes it, and a
        # retry passes the guard on the "already retracted" branch and
        # completes. The reverse order would strand the record: an appended
        # PUBLISHED version still excluded by live claims is a state revoke
        # can never re-enter (legality refuses PUBLISHED+revoke).
        stamp = self._clock() if revoke_claims else None
        if revoke_claims and stamp is not None:
            for claim in revoke_claims:
                self._store.retract_supersession(
                    record_id,
                    claim.record_version_key,
                    reason=reason or "",
                    retracted_by=actor_id,
                    at=stamp.isoformat(),
                )
        stored = self._append(record_id, updated, expected_version=current.version)
        # Hoisted: the release names both the audit event and the outbox entry,
        # and both must name the same one. ``release is not None`` is exactly
        # "this was a publish" — it is the value the PUBLISH branch produced.
        release = (
            self._release_for_publish(release_id)
            if action is LifecycleAction.PUBLISH
            else None
        )
        self._emit(
            actor_id=actor_id,
            role=role,
            action=action.value,
            record=stored,
            before_version=current.version,
            before_status=current.status,
            reason=reason,
            release_id=release,
            timestamp=stamp,
        )
        # The publish is stored and audited; this queues its re-index. Left to
        # raise: a publication that is served but not queued is a change nobody
        # re-indexes, and swallowing it here would make that divergence silent.
        # ``OutboxAppendError`` names the entry, the path and the affected ids,
        # and re-appending the same version is idempotent, so a caller can
        # retry. Same shape as ``AuditRecordingError``: stored, audited, not
        # queued, and told.
        if release is not None and self._outbox is not None:
            self._outbox.record_publication(stored, release_id=release, store=self._store)
        return stored

    def claim(
        self,
        record_id: str,
        *,
        actor_id: str,
        role: str | None = None,
        expected_version: int | None = None,
        reason: str | None = None,
    ) -> CanonicalRecord:
        """Append a new version of ``record_id`` claiming authorship (ticket T15).

        The seed drafts (spec 02 §108) all carry ``authored_by = null``, and an
        unclaimed version cannot be approved — four-eyes refuses a reviewer who
        cannot prove the author is someone else, and missing authorship fails
        that proof. A claim is the fix that does *not* destroy the seed
        version: it appends a new one that keeps every field the draft has and
        adds the caller as its ``authored_by``. The seed version and its audit
        history stay exactly where they were; the new version is the one that
        can now move through ``submit`` -> ``approve`` by a different person.

        Legal only from ``draft``, and only while ``authored_by`` is ``None``:
        a draft someone already claimed is not unclaimed, and a version past
        draft already carries a decision a claim would silently retract. Both
        refusals raise :class:`ClaimRefusedError` before any write.
        """
        current = self._require_current(record_id, expected_version)
        author = current.authored_by
        if current.status is not RecordStatus.DRAFT:
            raise ClaimRefusedError(
                record_id,
                current.version,
                f"its status is '{current.status.value}', not 'draft' — claiming "
                "re-claims a version a decision has already been made on",
            )
        if author is not None:
            detail = (
                "it is already yours — claiming your own draft appends a version "
                "that changes nothing"
                if author == actor_id
                else f"it was already claimed by '{author}' — claim only attaches "
                "an author to a draft that has none"
            )
            raise ClaimRefusedError(record_id, current.version, detail)
        payload = current.model_dump()
        payload["version"] = current.version + 1
        # A draft being claimed stays a draft; nothing is published, so no
        # publication stamp is created or cleared (drafts carry none).
        payload["authored_by"] = actor_id
        stored = self._append(
            record_id,
            CanonicalRecord.model_validate(payload),
            expected_version=current.version,
        )
        self._emit(
            actor_id=actor_id,
            role=role,
            action="claim",
            record=stored,
            before_version=current.version,
            before_status=current.status,
            reason=reason,
        )
        return stored

    def _release_for_publish(self, release_id: str | None) -> str:
        """The release a ``publish`` is recorded against.

        A publish the path names a release for (ticket 16's swap, ticket 22's
        rollback) records that id. A deployment with no release machinery yet
        records :data:`LEGACY_RELEASE_ID` — the label ticket 34 already stamps
        on canonical data that belongs to no release — because the event model
        refuses a publish that names no release at all: a publish nobody can
        point at is one nobody can roll back to.
        """
        return (release_id or "").strip() or LEGACY_RELEASE_ID

    # -- internals -----------------------------------------------------------

    def _require_independent_approver(
        self, record: CanonicalRecord, actor_id: str
    ) -> None:
        """Reject ``approve`` when the approver authored the version.

        Read from the *stored version under review* rather than from the event
        stream: the ledger is a log and (until ticket 14) lives in memory, while
        the rule has to hold after a restart. Unknown authorship is refused too
        — four-eyes that silently passes when the author is missing is not a
        control.
        """
        if not self._four_eyes():
            return
        author = record.authored_by
        if author is None or author == actor_id:
            raise FourEyesViolation(record.record_id, record.version, author)

    def _require_published_replacement(self, record: CanonicalRecord) -> None:
        """Reject ``supersede`` unless a published record replaces ``record``.

        The state machine already guarantees ``record`` is ``published``; this
        adds the spec's other half — a supersession exists only when another
        published record names it. Self-naming is structurally impossible: the
        candidate is looked up in the store's reverse index (which skips a
        version naming its own ``record_id``) rather than taken from the caller.
        """
        if self._store.superseding_record(record.record_id) is None:
            raise NotSupersededError(record.record_id)

    def _require_revocable_supersession(
        self, record: CanonicalRecord, actor_id: str
    ) -> tuple[CanonicalRecord, ...]:
        """Reject ``revoke_supersession`` unless this record is truly superseded.

        Three refusals, each with a message that names its own case (ticket 36
        criterion 5):

        - a status the state machine has no revoke edge from (``published``,
          ``draft``, …) raises the domain's ``IllegalTransitionError``, checked
          *before* the claims lookup so a plain published record answers "no
          such move" rather than "nothing supersedes you" — both are true of
          it, but only the first is the rule that was broken;
        - an ``archived`` record with no active claim naming it cannot be
          un-superseded: it was archived by *withdrawal* (or its supersession
          was already revoked and the operation never completed), and
          ``NotSupersededError(context="revoke")`` says so;
        - with four-eyes on, the caller must not be the one who decided the
          supersession — revoking your own decision is not a second pair of
          eyes. The decision actor is read from the ledger's most recent
          ``supersede`` event for this record; unknown (no ledger, no event)
          denies, the same posture as an unattributed author on ``approve``.

        Returns the active claims so the caller retracts exactly what this
        check saw. The check is not transactional with the append: a claim
        published *after* it lands while this revoke completes, and A's own
        version counter does not notice. That record is then excluded from
        the index as superseded by the new claim — a live state, not a lost
        one: ``withdraw`` then ``revoke_supersession`` on the same record is
        the documented way back (ticket 36's two-step retreat).
        """
        if not is_legal_transition(record.status, LifecycleAction.REVOKE_SUPERSESSION):
            raise IllegalTransitionError(record.status, LifecycleAction.REVOKE_SUPERSESSION)
        claims = self._store.superseding_versions(record.record_id)
        if not claims and not self._store.retracted_supersession_keys(
            record.record_id
        ):
            raise NotSupersededError(record.record_id, context="revoke")
        if self._four_eyes():
            decided_by = self._last_supersede_actor(record.record_id)
            if decided_by is None or decided_by == actor_id:
                raise FourEyesViolation(
                    record.record_id,
                    record.version,
                    decided_by,
                    operation="revoke_supersession",
                )
        return claims

    def _last_supersede_actor(self, record_id: str) -> str | None:
        """Who most recently superseded ``record_id``, per the durable ledger.

        ``None`` when there is no ledger or no supersede event in it: the
        in-process ``emitted_events`` stream would answer for the service that
        made the call but not across a restart, and four-eyes that forgets
        after a deploy is not a control (same reasoning as
        :meth:`_require_independent_approver`).
        """
        if self._ledger is None:
            return None
        entries = self._ledger.find(target=record_id, action=AuditAction.SUPERSEDE)
        return entries[-1].event.actor_id if entries else None

    def _chain(self, record_id: str):
        try:
            return self._store.get_record(record_id)
        except RecordNotFoundError as exc:
            raise RecordNotFound(record_id) from exc

    def _latest(self, record_id: str) -> CanonicalRecord:
        return self._chain(record_id).latest

    def _require_current(
        self, record_id: str, expected_version: int | None
    ) -> CanonicalRecord:
        current = self._latest(record_id)
        if expected_version is not None and current.version != expected_version:
            raise VersionConflictError(record_id, expected_version, current.version)
        return current

    def _append(
        self,
        record_id: str,
        updated: CanonicalRecord,
        *,
        expected_version: int,
    ) -> CanonicalRecord:
        try:
            return self._store.write_record(updated)
        except RecordVersionConflictError as exc:
            raise VersionConflictError(
                record_id, expected_version, self._store.next_version(record_id)
            ) from exc

    def _emit(
        self,
        *,
        actor_id: str,
        action: str,
        record: CanonicalRecord,
        before_version: int,
        before_status: RecordStatus,
        reason: str | None,
        role: str | None = None,
        release_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> datetime:
        """Record one write on both sinks, stamped with one clock reading.

        The timestamp is read once and returned, so every event describing this
        write — the edit and, when it applies, the source-document change — is
        stamped with the same instant rather than with two readings a few
        microseconds apart that would make one write look like two. A caller
        that already spent a clock reading for a companion write (the revoke's
        retraction markers, ticket 36) passes it in: the marker and the event
        for one decision then carry one timestamp.
        """
        if timestamp is None:
            timestamp = self._clock()
        # The durable write goes first. It is the one that can fail, and it must
        # fail *before* anything observable changes: an event sitting in
        # ``emitted_events`` for a write the ledger never accepted would be a
        # second, quieter version of the same divergence.
        self._record(
            AuditEvent(
                action=_AUDIT_ACTION[action],
                actor_id=actor_id,
                role=role,
                target=record.record_id,
                timestamp=timestamp,
                before_version=_version_key(record.record_id, before_version),
                after_version=record.record_version_key,
                reason=reason,
                release_id=release_id,
            ),
            operation=action,
            record=record,
            actor_id=actor_id,
        )
        self._events.append(
            LifecycleEvent(
                actor_id=actor_id,
                action=action,
                entity=ENTITY_NAME,
                entity_id=record.record_id,
                before_version=before_version,
                after_version=record.version,
                before_status=before_status,
                after_status=record.status,
                timestamp=timestamp,
                reason=reason,
                role=role,
                release_id=release_id,
            )
        )
        return timestamp

    def _record(
        self,
        event: AuditEvent,
        *,
        operation: str,
        record: CanonicalRecord,
        actor_id: str,
    ) -> None:
        """Append one event to the ledger, or raise :class:`AuditRecordingError`.

        The store write for this operation has already been committed by the
        time this runs, so a failure here is not "the operation failed" — it is
        "the operation happened and the audit trail does not know". Letting the
        underlying ``OSError``/``AuditLedgerError`` through would say the first
        thing while meaning the second, so it is re-raised as the error that
        says the second thing.
        """
        if self._ledger is None:
            return
        try:
            self._ledger.append(event)
        except Exception as exc:
            raise AuditRecordingError(
                action=operation,
                record_id=record.record_id,
                version=record.version,
                actor_id=actor_id,
                ledger_path=str(getattr(self._ledger, "path", None) or "") or None,
                cause=exc,
            ) from exc

    def _generate_hoi_dap_source_document(
        self, record: CanonicalRecord, actor_id: str
    ) -> str:
        """T07 (spec 02 §1): write the real source document for a HOI_DAP.

        The registry is optional on the service so every non-HOI_DAP path
        (and every test-built service) keeps working without one — but a
        HOI_DAP create without a registry is a configuration error, not
        a silent fallback: a record stored without provenance is exactly
        what this ticket forbids. Registration is idempotent: a retry of
        the same create reuses the id instead of raising, so a retry
        cannot produce a second document for the same record.

        Fails loudly if the registry write fails, so a HOI_DAP never
        reaches the store with a dangling or fabricated id.
        """
        if self._registry is None:
            raise AdminLifecycleError(
                f"cannot create HOI_DAP {record.record_id}: "
                "no source-document registry configured"
            )
        # T07 (spec 02 §1): the id is deterministic from the record
        # id, so a retry cannot mint a second document for the same
        # record. The ``uploads/`` prefix (``UPLOAD_ID_PREFIX``) is
        # the store-owned location the /content policy serves
        # unconditionally (``is_servable``), and ``resolve_path``
        # maps it under ``<root>/source_documents/`` — the file is
        # therefore openable from the citation chip, which a bare
        # ``hoi-dap-…`` id under ``documents/`` would not be.
        #
        # ``:`` is illegal in a Windows file name (NTFS reserves it
        # for ADS) and HUSC record ids carry it as their separator,
        # so ``:``, ``/`` and ``\`` are all replaced — the
        # sanitization is platform-agnostic and keeps the id valid
        # under the registry's path-traversal and kind checks.
        safe_id = (
            record.record_id.replace("\\", "-")
            .replace("/", "-")
            .replace(":", "-")
        )
        document_id = f"uploads/hoi-dap-{safe_id}.txt"
        now = self._clock()
        content = "\n".join(
            [
                "HOI_DAP",
                f"Câu hỏi: {record.text}",
                f"Người soạn: {actor_id}",
                f"Ngày: {now.date().isoformat()}",
                "",
            ]
        ).encode("utf-8")
        registry = self._registry
        # ``uploads/`` ids resolve under ``<root>/source_documents/``
        # (``resolve_path``), not ``documents/`` (the registry's entry
        # JSON dir) — write the blob where the /content policy will
        # actually open it.
        target = registry.root / SOURCE_DOCUMENTS_DIRNAME / document_id[
            len(UPLOAD_ID_PREFIX) :
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        if registry.is_registered(document_id):
            # A retry: reuse the id instead of raising FileExistsError.
            return document_id
        registry.create(
            document_id,
            sha256=digest,
            file_present=True,
            actor_id=actor_id,
            notes="Phòng tuyển sinh HUSC trả lời ngày "
            f"{now.date().isoformat()}",
            now=now,
        )
        return document_id

    def _emit_source_document_change(
        self,
        *,
        actor_id: str,
        role: str | None,
        record: CanonicalRecord,
        before_version: int,
        reason: str | None,
        timestamp: datetime,
    ) -> None:
        """Record that an edit moved which document a fact points at.

        A second event rather than a flag on the edit: "which facts had their
        provenance moved, and by whom?" is a question the admissions office asks
        on its own, and answering it must not mean reading every edit's diff.
        Only the ledger carries it — ``emitted_events`` stays one event per write,
        which is the contract the ticket-12 router's response shape depends on.

        Failing here means the edit is stored and its ``source_document_change``
        is not; the same :class:`AuditRecordingError` reports it, naming the
        provenance move as the unrecorded operation rather than the edit.
        """
        self._record(
            AuditEvent(
                action=AuditAction.SOURCE_DOCUMENT_CHANGE,
                actor_id=actor_id,
                role=role,
                target=record.record_id,
                timestamp=timestamp,
                before_version=_version_key(record.record_id, before_version),
                after_version=record.record_version_key,
                reason=reason,
            ),
            operation=AuditAction.SOURCE_DOCUMENT_CHANGE.value,
            record=record,
            actor_id=actor_id,
        )


def _as_action(action: LifecycleAction | str) -> LifecycleAction:
    """``action`` as *this* module's :class:`LifecycleAction`.

    The repo loads two module trees in one process (``rag2025.src.…`` and
    ``src.…``), so ``LifecycleAction`` can exist as two class objects whose
    members are ``==`` but not ``is``. Accepting either and returning ours is
    what lets :meth:`AdminLifecycleService.transition` compare with ``is``
    without depending on where its caller imported from.
    """
    if isinstance(action, LifecycleAction):
        return action
    try:
        return LifecycleAction(getattr(action, "value", action))
    except (ValueError, TypeError) as exc:
        raise AdminLifecycleError(f"unknown lifecycle action: {action!r}") from exc


def _version_key(record_id: str, version: int) -> str | None:
    """``record_id@version`` for a stored version, or ``None`` for "no version".

    ``None`` rather than a ``@0`` key: a create has no before-version, and a
    reference that resolves through the store must be visibly absent instead of
    pointing at a version that does not exist.
    """
    return f"{record_id}@{version}" if version >= 1 else None


def _provenance_changed(before: CanonicalRecord, after: CanonicalRecord) -> bool:
    """Whether an edit moved the document a fact points at (or its page)."""
    return any(
        getattr(before, field) != getattr(after, field) for field in _PROVENANCE_FIELDS
    )
