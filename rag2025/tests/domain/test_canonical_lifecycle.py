"""TDD: rag2025/src/domain/canonical/lifecycle.py — lifecycle state machine.

Ticket 12 acceptance (the rule half — the HTTP half lives in
``tests/routers/test_admin_lifecycle_api.py``):
- every legal transition reaches the status the spec's diagram names;
- the three named prohibitions hold: draft -> published, approved -> archived,
  publish without approved;
- reject requires a reason;
- every transition carries an event whose payload is JSON-ready.

The point of asserting the forbidden moves *as a set* is that the table is
data: a future edit that adds a status cannot silently legalize a pair nobody
listed, because the test enumerates the illegal pairs explicitly.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.lifecycle import (  # noqa: E402
    ENTITY_NAME,
    IllegalTransitionError,
    LifecycleAction,
    LifecycleEvent,
    RejectReasonRequiredError,
    apply_transition,
    is_legal_transition,
    legal_actions,
    requires_reason,
    transition_status,
)
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)


def _record(**overrides) -> CanonicalRecord:
    """A minimal valid record; each test overrides the status it probes."""
    fields = {
        "record_id": "DIEM_CHUAN:NGANH:cntt:2026",
        "record_type": RecordType.DIEM_CHUAN,
        "version": 1,
        "admission_year": 2026,
        "effective_from": date(2026, 6, 1),
        "effective_to": date(2026, 12, 31),
        "status": RecordStatus.DRAFT,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 12,
        "published_at": None,
        "supersedes": None,
        "text": "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.",
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


# The spec's diagram, verbatim, as (from, action) -> to, plus ticket 36's two
# retreat edges: WITHDRAW (published -> archived, no replacement) and
# REVOKE_SUPERSESSION (archived -> published, the only way out).
LEGAL_TRANSITIONS = [
    (RecordStatus.DRAFT, LifecycleAction.SUBMIT, RecordStatus.REVIEW),
    (RecordStatus.REVIEW, LifecycleAction.APPROVE, RecordStatus.APPROVED),
    (RecordStatus.REVIEW, LifecycleAction.REJECT, RecordStatus.DRAFT),
    (RecordStatus.APPROVED, LifecycleAction.PUBLISH, RecordStatus.PUBLISHED),
    (RecordStatus.PUBLISHED, LifecycleAction.SUPERSEDE, RecordStatus.ARCHIVED),
    (RecordStatus.PUBLISHED, LifecycleAction.WITHDRAW, RecordStatus.ARCHIVED),
    (
        RecordStatus.ARCHIVED,
        LifecycleAction.REVOKE_SUPERSESSION,
        RecordStatus.PUBLISHED,
    ),
]


@pytest.mark.parametrize("current,action,expected", LEGAL_TRANSITIONS)
def test_legal_transition_reaches_expected_status(current, action, expected):
    assert transition_status(current, action, reason="vì lý do") is expected
    assert is_legal_transition(current, action)


@pytest.mark.parametrize(
    "current,action",
    [
        # "Cấm: draft -> published" — the shortcut the whole ticket exists to stop.
        (RecordStatus.DRAFT, LifecycleAction.PUBLISH),
        # "publish khi chưa approved" — review is not approval.
        (RecordStatus.REVIEW, LifecycleAction.PUBLISH),
        # "approved -> archived trực tiếp"
        (RecordStatus.APPROVED, LifecycleAction.SUPERSEDE),
        # Nothing but a published version can be superseded or re-submitted.
        (RecordStatus.DRAFT, LifecycleAction.SUPERSEDE),
        (RecordStatus.REVIEW, LifecycleAction.SUPERSEDE),
        (RecordStatus.DRAFT, LifecycleAction.APPROVE),
        (RecordStatus.APPROVED, LifecycleAction.APPROVE),
        (RecordStatus.PUBLISHED, LifecycleAction.SUBMIT),
        (RecordStatus.PUBLISHED, LifecycleAction.APPROVE),
        (RecordStatus.PUBLISHED, LifecycleAction.PUBLISH),
        (RecordStatus.PUBLISHED, LifecycleAction.REJECT),
        # archived keeps exactly one exit (ticket 36); every other verb,
        # including a second withdraw or a re-publish, is still refused.
        (RecordStatus.ARCHIVED, LifecycleAction.SUBMIT),
        (RecordStatus.ARCHIVED, LifecycleAction.APPROVE),
        (RecordStatus.ARCHIVED, LifecycleAction.PUBLISH),
        (RecordStatus.ARCHIVED, LifecycleAction.SUPERSEDE),
        (RecordStatus.ARCHIVED, LifecycleAction.REJECT),
        (RecordStatus.ARCHIVED, LifecycleAction.WITHDRAW),
    ],
)
def test_forbidden_transition_is_rejected(current, action):
    assert not is_legal_transition(current, action)
    with pytest.raises(IllegalTransitionError) as excinfo:
        transition_status(current, action, reason="vì lý do")
    # The error names both ends so a caller can explain the refusal.
    assert excinfo.value.current is current
    assert excinfo.value.action is action


@pytest.mark.parametrize("reason", [None, "", "   ", "\n\t "])
def test_reject_without_a_reason_is_refused(reason):
    """A reject with no justification is a silent discard, not a decision."""
    assert requires_reason(LifecycleAction.REJECT)
    with pytest.raises(RejectReasonRequiredError):
        transition_status(RecordStatus.REVIEW, LifecycleAction.REJECT, reason=reason)


@pytest.mark.parametrize(
    "current,action",
    [
        (RecordStatus.PUBLISHED, LifecycleAction.WITHDRAW),
        (RecordStatus.ARCHIVED, LifecycleAction.REVOKE_SUPERSESSION),
    ],
)
@pytest.mark.parametrize("reason", [None, "", "   ", "\n\t "])
def test_the_retreat_actions_also_refuse_a_blank_reason(current, action, reason):
    # Ticket 36: a withdrawal or an un-supersede without a stated why is as
    # unaccountable as a silent reject, so they join the reason-required set.
    assert requires_reason(action)
    with pytest.raises(RejectReasonRequiredError):
        transition_status(current, action, reason=reason)


def test_reject_with_a_reason_returns_to_draft():
    assert (
        transition_status(
            RecordStatus.REVIEW, LifecycleAction.REJECT, reason="thiếu nguồn chính thức"
        )
        is RecordStatus.DRAFT
    )


def test_reason_is_not_required_for_other_actions():
    for action in (LifecycleAction.SUBMIT, LifecycleAction.APPROVE, LifecycleAction.PUBLISH):
        assert not requires_reason(action)


@pytest.mark.parametrize(
    "current,expected",
    [
        (RecordStatus.DRAFT, {LifecycleAction.SUBMIT}),
        (RecordStatus.REVIEW, {LifecycleAction.APPROVE, LifecycleAction.REJECT}),
        (RecordStatus.APPROVED, {LifecycleAction.PUBLISH}),
        (RecordStatus.PUBLISHED, {LifecycleAction.SUPERSEDE, LifecycleAction.WITHDRAW}),
        (RecordStatus.ARCHIVED, {LifecycleAction.REVOKE_SUPERSESSION}),
    ],
)
def test_legal_actions_match_the_state_machine(current, expected):
    assert legal_actions(current) == frozenset(expected)


def test_archived_has_only_the_un_supersede_exit():
    # Ticket 36: archived is no longer a dead end, but the one way out is
    # un-superseding a wrong replacement — never a re-submit or a re-publish.
    assert legal_actions(RecordStatus.ARCHIVED) == frozenset(
        {LifecycleAction.REVOKE_SUPERSESSION}
    )


def test_apply_transition_returns_a_new_record_and_leaves_the_original_alone():
    original = _record(status=RecordStatus.DRAFT)
    moved = apply_transition(original, LifecycleAction.SUBMIT)

    assert moved is not original
    assert moved.status is RecordStatus.REVIEW
    # The input record is untouched: the rule is a function, not a mutation.
    assert original.status is RecordStatus.DRAFT


def test_apply_transition_propagates_the_rule_violation():
    with pytest.raises(IllegalTransitionError):
        apply_transition(_record(status=RecordStatus.DRAFT), LifecycleAction.PUBLISH)


def test_event_payload_is_json_ready():
    """The audit ledger (ticket 14) persists this payload as JSON."""
    event = LifecycleEvent(
        actor_id="cb-tuyensinh-01",
        action=LifecycleAction.PUBLISH.value,
        entity=ENTITY_NAME,
        entity_id="DIEM_CHUAN:NGANH:cntt:2026",
        before_version=4,
        after_version=5,
        before_status=RecordStatus.APPROVED,
        after_status=RecordStatus.PUBLISHED,
        timestamp=datetime(2026, 6, 1, 8, 30, tzinfo=timezone.utc),
        reason=None,
    )
    payload = event.to_dict()

    assert payload["before_status"] == "approved"
    assert payload["after_status"] == "published"
    assert payload["action"] == "publish"
    # Round-trips as JSON without a custom encoder: enums are stored by value.
    assert json.loads(json.dumps(payload)) == payload
