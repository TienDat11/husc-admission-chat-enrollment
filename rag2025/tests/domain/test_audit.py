"""Audit event model — ticket 14.

What is asserted here is the model's own contract, because that is where the
ticket put it: the kinds it must be able to record, and the two rules that must
hold *before* an event exists (a justification for reject/rollback/override; a
release id for a publish). A route that forgets either cannot produce a valid
event at all, so there is nothing for a later reader to discover.

The service-level consequences — that the references resolve through the store,
that a publish through the HTTP seam lands in the ledger — are asserted in
``tests/routers/test_admin_audit.py``, at the seam that produces them.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import (  # noqa: E402
    LEGACY_RELEASE_ID,
    REASON_REQUIRED,
    AuditAction,
    AuditEvent,
    TargetKind,
)

NOW = datetime(2026, 9, 13, 8, 30, tzinfo=timezone.utc)

REQUIRED_KINDS = (
    AuditAction.PUBLISH,
    AuditAction.ROLLBACK,
    AuditAction.PERMISSION_CHANGE,
    AuditAction.SOURCE_DOCUMENT_CHANGE,
    AuditAction.CONFLICT_OVERRIDE,
)


def _event(**overrides) -> AuditEvent:
    """A minimal valid event; each test overrides only what it is about."""
    fields: dict = {
        "action": AuditAction.EDIT,
        "actor_id": "cb.bien-tap",
        "target": "DIEM_CHUAN:NGANH:cntt:2026",
        "timestamp": NOW,
        "before_version": "DIEM_CHUAN:NGANH:cntt:2026@1",
        "after_version": "DIEM_CHUAN:NGANH:cntt:2026@2",
        "event_id": "evt-1",
    }
    fields.update(overrides)
    return AuditEvent(**fields)


def test_every_required_kind_is_recordable():
    """The five kinds the ticket names all exist and can be constructed."""
    for kind in REQUIRED_KINDS:
        event = _event(
            action=kind,
            reason="lý do",
            release_id="R-2026-09-13" if kind is AuditAction.PUBLISH else None,
        )
        assert event.action is kind


def test_the_reason_required_set_is_exactly_the_decision_actions():
    assert REASON_REQUIRED == frozenset(
        {
            AuditAction.REJECT,
            AuditAction.ROLLBACK,
            AuditAction.CONFLICT_OVERRIDE,
            AuditAction.PERMISSION_CHANGE,
            AuditAction.WITHDRAW,
            AuditAction.REVOKE_SUPERSESSION,
        }
    )


@pytest.mark.parametrize(
    "action",
    [
        AuditAction.REJECT,
        AuditAction.ROLLBACK,
        AuditAction.CONFLICT_OVERRIDE,
        AuditAction.PERMISSION_CHANGE,
        AuditAction.WITHDRAW,
        AuditAction.REVOKE_SUPERSESSION,
    ],
)
@pytest.mark.parametrize("reason", [None, "", "   "])
def test_a_justification_bearing_action_without_a_reason_raises(action, reason):
    """Absent, empty and whitespace-only are all "no reason"."""
    with pytest.raises(ValueError):
        _event(action=action, reason=reason)


@pytest.mark.parametrize(
    "action",
    [
        AuditAction.REJECT,
        AuditAction.ROLLBACK,
        AuditAction.CONFLICT_OVERRIDE,
        AuditAction.PERMISSION_CHANGE,
        AuditAction.WITHDRAW,
        AuditAction.REVOKE_SUPERSESSION,
    ],
)
def test_a_justification_bearing_action_with_a_reason_is_accepted(action):
    assert _event(action=action, reason="đã rà soát với phòng tuyển sinh").reason


def test_a_publish_without_a_release_id_raises():
    with pytest.raises(ValueError):
        _event(action=AuditAction.PUBLISH)
    with pytest.raises(ValueError):
        _event(action=AuditAction.PUBLISH, release_id="  ")


def test_a_publish_with_a_release_id_is_accepted():
    event = _event(action=AuditAction.PUBLISH, release_id="R-2026-09-13")
    assert event.release_id == "R-2026-09-13"


def test_the_legacy_release_label_is_the_blank_release_stand_in():
    """The label a release-less publish uses is a real value, not blank."""
    assert LEGACY_RELEASE_ID.strip()
    assert _event(action=AuditAction.PUBLISH, release_id=LEGACY_RELEASE_ID).release_id


def test_an_action_outside_the_enum_is_refused():
    """A kind is added to the enum deliberately, never spelled by a caller."""
    with pytest.raises(ValueError):
        _event(action="delete_everything")


def test_actor_and_target_must_be_present():
    with pytest.raises(ValueError):
        _event(actor_id="  ")
    with pytest.raises(ValueError):
        _event(target="")


def test_the_action_and_target_kind_accept_their_string_forms():
    """A JSON body or an env-derived value arrives as a string."""
    event = _event(action="publish", target_kind="release", release_id="R-1")
    assert event.action is AuditAction.PUBLISH
    assert event.target_kind is TargetKind.RELEASE


def test_a_dict_round_trip_preserves_every_field():
    original = _event(
        action=AuditAction.ROLLBACK,
        reason="hoàn nguyên vì điểm chuẩn sai",
        role="admin",
        source_ip="10.0.0.7",
        review_id="rev-9",
        target_kind=TargetKind.RECORD,
    )
    assert AuditEvent.from_dict(original.to_dict()) == original


def test_a_round_trip_keeps_vietnamese_text_as_utf8_not_escapes():
    reason = "Hoàn nguyên vì điểm chuẩn ngành CNTT bị sai"
    payload = _event(action=AuditAction.ROLLBACK, reason=reason).to_dict()
    assert payload["reason"] == reason
    assert "\\u" not in repr(payload["reason"])


def test_reading_back_an_event_whose_reason_was_stripped_is_refused():
    """The model rule holds on read too: an edited reject cannot be trusted."""
    payload = _event(action=AuditAction.REJECT, reason="sai số liệu").to_dict()
    payload["reason"] = None
    with pytest.raises(ValueError):
        AuditEvent.from_dict(payload)


def test_the_other_module_tree_s_action_enum_is_accepted():
    """``rag2025.src.domain.audit`` and ``src.domain.audit`` are two classes.

    The repo imports both paths, so an ``AuditAction`` from the other tree is a
    different class object whose members are ``==`` but not ``is`` — and whose
    ``str()`` is ``"AuditAction.PUBLISH"``, not ``"publish"``. Coercing by
    ``.value`` is what lets a caller holding either tree's enum build an event.
    """
    try:
        import rag2025.src.domain.audit as foreign  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover - layout dependent
        pytest.skip("the second module tree is not importable here")
    if foreign.AuditAction is AuditAction:
        pytest.skip("the two import paths resolve to one module; no duplication")

    event = _event(action=foreign.AuditAction.PUBLISH, release_id="R-1")
    assert event.action is AuditAction.PUBLISH

    # ...and the model's own rules still bind, rather than being skipped by an
    # action that never matched the enum.
    with pytest.raises(ValueError):
        _event(action=foreign.AuditAction.PUBLISH)
    with pytest.raises(ValueError):
        _event(action=foreign.AuditAction.REJECT)
