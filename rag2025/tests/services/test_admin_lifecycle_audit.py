"""Store/audit divergence — ticket 14.

The store is the source of truth and is written first, so there is one real
failure mode this file exists to pin down: **the version is stored and the audit
event is not**. The dangerous response to that is to let the raw ``OSError``
escape, because it reads as "the operation failed" while the truth is the
opposite — the caller then believes nothing happened while an unaudited publish
stands, which is ticket 14's first checkbox ("mọi publish... đều sinh event")
failing silently.

These tests assert the three things that make the failure safe to live with:

(a) the raised error is the distinct one, not a bare ``OSError``;
(b) its message says the store write succeeded and was not audited, and names
    what was not recorded;
(c) the divergence is *visible* — the version really is in the store, read back
    through a second store instance over the same root.

The failure is induced with real I/O (a ledger path whose parent is a file, so
the ledger's own ``mkdir`` fails) wherever the whole operation can be made to
fail, and with a subclass that fails on its Nth append where only part of a
multi-event write should fail.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction  # noqa: E402
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.admin_lifecycle import (  # noqa: E402
    AdminLifecycleService,
    AuditRecordingError,
    FourEyesViolation,
    VersionConflictError,
)
from src.services.reindex_outbox import ReindexOutbox  # noqa: E402

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
NOW = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)


class _FailingLedger(AuditLedger):
    """A ledger whose ``fail_from``-th append onward fails with a real OSError.

    Used to break the *second* event of a multi-event write while the first one
    lands — the shape a disk filling up mid-request produces.
    """

    def __init__(self, path: str | Path, *, fail_from: int) -> None:
        super().__init__(path)
        self._calls = 0
        self._fail_from = fail_from

    def append(self, event):  # type: ignore[override]
        self._calls += 1
        if self._calls >= self._fail_from:
            raise OSError(28, "No space left on device")
        return super().append(event)


def _record(**overrides) -> CanonicalRecord:
    fields = {
        "record_id": RECORD_ID,
        "record_type": RecordType.DIEM_CHUAN,
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-06-01",
        "status": RecordStatus.DRAFT,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": "de-an-tuyen-sinh-2026",
        "text": "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.",
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def blocked_ledger_path(tmp_path: Path) -> Path:
    """A ledger path that cannot be created: its parent is a file."""
    blocker = tmp_path / "audit"
    blocker.write_text("not a directory", encoding="utf-8")
    return blocker / "ledger.jsonl"


def test_a_create_that_cannot_be_audited_raises_loudly_and_stays_in_the_store(
    store_root: Path, blocked_ledger_path: Path
):
    """(a) distinct error, (b) it names the divergence, (c) the write is real."""
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(
        store, ledger=AuditLedger(blocked_ledger_path), clock=lambda: NOW
    )

    with pytest.raises(AuditRecordingError) as raised:
        service.create(_record(), actor_id="cb.bien-tap", role="editor")

    error = raised.value
    # (a) not a bare I/O error: the caller must be able to tell this apart from
    # "the write failed", which is a different situation entirely.
    assert not isinstance(error, OSError)
    assert isinstance(error.cause, OSError)

    # (b) the message says what happened, to what, and where the audit is missing.
    message = str(error)
    assert "create" in message
    assert f"{RECORD_ID}@1" in message
    assert "cb.bien-tap" in message
    assert "was stored" in message
    assert "NOT recorded in the audit trail" in message
    assert str(blocked_ledger_path) in message
    assert error.ledger_path == str(blocked_ledger_path)

    # Nothing observable claims the write was audited.
    assert service.emitted_events() == ()

    # (c) the divergence itself, read back from disk through a second store.
    persisted = CanonicalRecordStore(store_root).get_record(RECORD_ID)
    assert persisted.latest.version == 1
    assert persisted.latest.text == "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm."


def test_the_retry_of_a_write_that_was_stored_but_not_audited_is_a_version_conflict(
    store_root: Path, blocked_ledger_path: Path
):
    """Why the message says "do not retry": the retry cannot silently succeed.

    A caller that read the error as "nothing happened" would retry, and the
    store refuses the second version 1 — so the mistake surfaces rather than
    producing a duplicate. Documented here because it is the reason the loud
    error is enough instead of a rollback.
    """
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(
        store, ledger=AuditLedger(blocked_ledger_path), clock=lambda: NOW
    )
    with pytest.raises(AuditRecordingError):
        service.create(_record(), actor_id="cb.bien-tap", role="editor")

    with pytest.raises(VersionConflictError):
        service.create(_record(), actor_id="cb.bien-tap", role="editor")


def test_a_publish_that_cannot_be_audited_does_not_look_like_a_failed_publish(
    store_root: Path, tmp_path: Path
):
    """The case the ticket is about: an unaudited publish that stands.

    The ledger works for create/submit/approve and breaks before the publish, so
    the failure is exactly "the record is published and the audit trail does not
    say so" — the caller gets an error that names it instead of a 500 that looks
    like a publish which never happened.
    """
    store = CanonicalRecordStore(store_root)
    audit_dir = tmp_path / "audit"
    ledger = AuditLedger(audit_dir / "ledger.jsonl")
    service = AdminLifecycleService(store, ledger=ledger, clock=lambda: NOW)

    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")

    # The audit directory becomes unwritable mid-request.
    for entry in sorted(audit_dir.iterdir()):
        entry.unlink()
    audit_dir.rmdir()
    audit_dir.write_text("blocked", encoding="utf-8")

    with pytest.raises(AuditRecordingError) as raised:
        service.transition(
            RECORD_ID,
            LifecycleAction.PUBLISH,
            actor_id="cb.xuat-ban",
            release_id="R-2026-09-13",
        )

    assert "publish" in str(raised.value)
    assert "NOT recorded in the audit trail" in str(raised.value)

    # The publish is live: this is the divergence, stated rather than hidden.
    published = CanonicalRecordStore(store_root).get_record(RECORD_ID).latest
    assert published.version == 4
    assert published.status is RecordStatus.PUBLISHED


def test_a_provenance_move_that_cannot_be_audited_names_the_provenance_move(
    store_root: Path, tmp_path: Path
):
    """A multi-event write that fails partway says which event is missing.

    The edit is recorded and its ``source_document_change`` is not, so the error
    names the provenance move — "the audit trail is missing something" is not
    actionable, "it is missing the source-document change for this version" is.
    """
    store = CanonicalRecordStore(store_root)
    # create = append 1, edit = append 2, source_document_change = append 3.
    ledger = _FailingLedger(tmp_path / "audit" / "ledger.jsonl", fail_from=3)
    service = AdminLifecycleService(store, ledger=ledger, clock=lambda: NOW)
    service.create(_record(), actor_id="cb.bien-tap", role="editor")

    with pytest.raises(AuditRecordingError) as raised:
        service.edit(
            RECORD_ID,
            source_document_id="de-an-tuyen-sinh-2026-hieu-dinh",
            expected_version=1,
            actor_id="cb.bien-tap",
            reason="đối chiếu bản hiệu đính",
        )

    assert raised.value.action == AuditAction.SOURCE_DOCUMENT_CHANGE.value
    assert f"{RECORD_ID}@2" in str(raised.value)

    # The edit that *was* recorded is really in the ledger, and the provenance
    # move is really not: the error is a description of the file, not a guess.
    actions = [entry.event.action for entry in AuditLedger(ledger.path).entries()]
    assert actions == [AuditAction.CREATE, AuditAction.EDIT]
    assert AuditLedger(ledger.path).verify().ok is True

    # The edit itself is stored, which is what the message claims.
    assert CanonicalRecordStore(store_root).get_record(RECORD_ID).latest.version == 2


def test_a_page_edit_emits_exactly_one_source_document_change(store_root: Path):
    """Ticket 39 criterion 7, positive path: moving ``source_page`` is audited.

    ``source_page`` sits in ``_PROVENANCE_FIELDS`` alongside the document id and
    the url (``src/services/admin_lifecycle.py:90``), so an edit that moves only
    the page must emit exactly one ``SOURCE_DOCUMENT_CHANGE`` — not zero (the
    registry landing quietly changing what a page means) and not two (the field
    counted twice because it shares a prefix with ``source_document_id``).

    The failing-ledger test above proves the *missing* event is named; this is
    the complementary half — that the event is really there on the happy path,
    which is what makes the registry landing safe for the 188 cited records.
    """
    store = CanonicalRecordStore(store_root)
    ledger = AuditLedger(store_root / "audit" / "ledger.jsonl")
    service = AdminLifecycleService(store, ledger=ledger, clock=lambda: NOW)
    service.create(_record(source_page=12), actor_id="cb.bien-tap", role="editor")

    # Move the page only: same document, same url, new page.
    service.edit(
        RECORD_ID,
        source_page=30,
        expected_version=1,
        actor_id="cb.bien-tap",
        role="editor",
        reason="đối chiếu trang trong bản gốc",
    )

    actions = [entry.event.action for entry in ledger.entries()]
    assert actions == [
        AuditAction.CREATE,
        AuditAction.EDIT,
        AuditAction.SOURCE_DOCUMENT_CHANGE,
    ], f"expected exactly one provenance event, got {actions}"
    assert ledger.verify().ok is True

    # The page really moved, so the event describes a real change rather than
    # a field that was written and ignored.
    prov = CanonicalRecordStore(store_root).resolve_provenance(f"{RECORD_ID}@2")
    assert prov.source_page == 30
    assert prov.source_document_id == "de-an-tuyen-sinh-2026"


# ---------------------------------------------------------------------------
# the two module trees
# ---------------------------------------------------------------------------


def _foreign_lifecycle_action():
    """``LifecycleAction`` as imported through the *other* module tree.

    The repo genuinely loads both: ``rag2025/src/app/lifecycle.py`` imports
    ``rag2025.src.services.…`` while the routers import ``src.services.…``, so
    the same enum exists twice in one process — two class objects whose members
    compare ``==`` but are not ``is``. Skipped only if the duplication is ever
    removed, because then there is nothing to test.
    """
    try:
        import rag2025.src.domain.canonical.lifecycle as foreign  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover - layout dependent
        pytest.skip("the second module tree is not importable here")
    if foreign.LifecycleAction is LifecycleAction:
        pytest.skip("the two import paths resolve to one module; no duplication")
    assert foreign.LifecycleAction.PUBLISH is not LifecycleAction.PUBLISH
    assert foreign.LifecycleAction.PUBLISH == LifecycleAction.PUBLISH
    return foreign.LifecycleAction


def test_a_publish_using_the_other_tree_s_enum_still_publishes_and_is_audited(
    store_root: Path,
):
    """The landmine: an identity check that is False for a genuine publish.

    ``action is LifecycleAction.PUBLISH`` was False when the caller imported the
    enum from the other tree, so ``release_id`` was dropped, the event model
    refused the publish — and it refused *after* the store write had committed.
    The result was a published record with no publish event and an exception
    telling the caller nothing happened. Coercing the action at the top of
    ``transition`` is the fix; without it this test raises ValueError and the
    ledger has no publish event.
    """
    foreign = _foreign_lifecycle_action()
    store = CanonicalRecordStore(store_root)
    ledger = AuditLedger(store_root.parent / "audit" / "ledger.jsonl")
    service = AdminLifecycleService(store, ledger=ledger, clock=lambda: NOW)

    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, foreign.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, foreign.APPROVE, actor_id="cb.phan-bien")

    # (c) no exception.
    stored = service.transition(
        RECORD_ID, foreign.PUBLISH, actor_id="cb.xuat-ban", release_id="R-2026-09-13"
    )

    # (a) it publishes.
    assert stored.status is RecordStatus.PUBLISHED
    assert CanonicalRecordStore(store_root).get_record(RECORD_ID).latest.status is (
        RecordStatus.PUBLISHED
    )

    # (b) the release id reaches the event, which is what the dropped identity
    # check silently discarded.
    published = ledger.find(action=AuditAction.PUBLISH)
    assert len(published) == 1
    assert published[0].event.release_id == "R-2026-09-13"
    assert published[0].event.after_version == f"{RECORD_ID}@4"
    assert ledger.verify().ok is True


def test_four_eyes_still_applies_to_the_other_tree_s_enum(store_root: Path):
    """The same identity problem on the other ``is`` checks.

    ``approve`` and ``supersede`` are guarded by ``is`` comparisons too, so
    without the coercion a cross-tree approve would skip four-eyes entirely —
    the control silently not running is worse than the publish failing.
    """
    foreign = _foreign_lifecycle_action()
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(store, clock=lambda: NOW)
    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, foreign.SUBMIT, actor_id="cb.bien-tap")

    with pytest.raises(FourEyesViolation):
        service.transition(RECORD_ID, foreign.APPROVE, actor_id="cb.bien-tap")


# ---------------------------------------------------------------------------
# the publish transition feeds the re-index outbox (ticket 15's seam)
# ---------------------------------------------------------------------------


def _advance_to_publish(service: AdminLifecycleService) -> CanonicalRecord:
    """Drive a fresh record to published and return the published version."""
    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    return service.transition(
        RECORD_ID,
        LifecycleAction.PUBLISH,
        actor_id="cb.xuat-ban",
        release_id="R-2026-09-13",
    )


def test_a_publish_queues_exactly_one_outbox_entry_naming_the_stored_version(
    store_root: Path,
):
    """The only thing that re-indexes served data is a publication.

    Nothing else feeds the outbox, so a publish that is stored and audited but
    not queued is a change no job ever picks up — and nothing would report it.
    The entry must name the release and the version the store actually holds,
    because that pair is what the re-index job resolves.
    """
    store = CanonicalRecordStore(store_root)
    outbox = ReindexOutbox(store_root.parent / "outbox" / "reindex_outbox.jsonl")
    service = AdminLifecycleService(store, outbox=outbox, clock=lambda: NOW)

    stored = _advance_to_publish(service)

    entries = outbox.entries()
    assert len(entries) == 1, f"a publish must queue exactly one entry: {entries}"
    entry = entries[0]
    assert entry.release_id == "R-2026-09-13"
    assert entry.record_version_key == stored.record_version_key == f"{RECORD_ID}@4"
    assert entry.affected_record_ids == (RECORD_ID,)


def test_only_a_publish_queues_an_outbox_entry(store_root: Path):
    """create/submit/approve change no served data, so they queue nothing.

    Asserted because "queue on every write" would be a plausible way to satisfy
    the publish test above while re-indexing the corpus on every draft edit.
    """
    store = CanonicalRecordStore(store_root)
    outbox = ReindexOutbox(store_root.parent / "outbox" / "reindex_outbox.jsonl")
    service = AdminLifecycleService(store, outbox=outbox, clock=lambda: NOW)

    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    assert outbox.entries() == ()

    service.transition(
        RECORD_ID,
        LifecycleAction.PUBLISH,
        actor_id="cb.xuat-ban",
        release_id="R-2026-09-13",
    )
    assert len(outbox.entries()) == 1


def test_a_service_with_no_outbox_still_publishes(store_root: Path):
    """The outbox is optional, like the ledger: a read-only or test-built
    service must not need a file to publish."""
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(store, clock=lambda: NOW)

    stored = _advance_to_publish(service)

    assert stored.status is RecordStatus.PUBLISHED
