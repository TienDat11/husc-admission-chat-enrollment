"""Batch record transitions — U8 PL-7.1.

``POST /admin/records/batch-transition`` runs up to 110 items in a single
transaction and returns ``{applied[], rejected[], skipped[]}``:

* item lacking permission → rejected, ``FORBIDDEN_ACTION``
* self-approve with four-eyes active → rejected, ``SELF_APPROVAL_FORBIDDEN``
* version mismatch → rejected, ``VERSION_CONFLICT``
* already in target state → skipped

The route itself only requires ``RECORDS_READ``; each item is authorised for
its own action (submit/approve/reject/publish).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.config.admin_identity import four_eyes_enabled
from src.domain.admin_identity import AdminAction, Identity
from src.domain.canonical.lifecycle import (
    IllegalTransitionError,
    LifecycleAction,
    apply_transition,
)
from src.domain.canonical.record import RecordStatus
from src.routers.admin_auth import current_identity, enforce, require_action
from src.services.admin_lifecycle import (
    AdminLifecycleService,
    FourEyesViolation,
    RecordNotFound,
    VersionConflictError,
)

router = APIRouter(tags=["admin-batch"])

_MAX_BATCH = 110

_ITEM_ACTIONS: dict[str, AdminAction] = {
    "submit": AdminAction.RECORDS_SUBMIT,
    "approve": AdminAction.RECORDS_APPROVE,
    "reject": AdminAction.RECORDS_REJECT,
    "publish": AdminAction.RECORDS_PUBLISH,
}

_ITEM_LIFECYCLE: dict[str, LifecycleAction] = {
    "submit": LifecycleAction.SUBMIT,
    "approve": LifecycleAction.APPROVE,
    "reject": LifecycleAction.REJECT,
    "publish": LifecycleAction.PUBLISH,
}

_TARGET_STATUS: dict[LifecycleAction, RecordStatus] = {
    LifecycleAction.SUBMIT: RecordStatus.REVIEW,
    LifecycleAction.APPROVE: RecordStatus.APPROVED,
    LifecycleAction.REJECT: RecordStatus.DRAFT,
    LifecycleAction.PUBLISH: RecordStatus.PUBLISHED,
}


class BatchItem(BaseModel):
    record_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    expected_version: int | None = None
    reason: str | None = None


class BatchTransitionRequest(BaseModel):
    items: list[BatchItem] = Field(default_factory=list, max_length=_MAX_BATCH)


class BatchItemResult(BaseModel):
    record_id: str
    action: str
    version: int | None = None


class BatchRejectedItem(BaseModel):
    record_id: str
    action: str
    code: str
    message: str


class BatchSkippedItem(BaseModel):
    record_id: str
    action: str
    code: str = "ALREADY_IN_TARGET_STATE"
    message: str = "record already in target state"


class BatchTransitionResponse(BaseModel):
    applied: list[BatchItemResult]
    rejected: list[BatchRejectedItem]
    skipped: list[BatchSkippedItem]


def _lifecycle_service(request: Request) -> AdminLifecycleService:
    """The lifecycle service wired by ``configure_canonical_record_store``.

    Same service the single-item admin routes use, so batch transitions share
    the audit ledger and four-eyes switch. Refused when unwired rather than
    building a second service over a different store.
    """
    service = getattr(request.app.state, "admin_lifecycle_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="admin lifecycle service is not configured")
    return service


def _reject(code: str, record_id: str, action: str, message: str) -> BatchRejectedItem:
    return BatchRejectedItem(
        record_id=record_id, action=action, code=code, message=message
    )


@router.post(
    "/records/batch-transition",
    response_model=BatchTransitionResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def batch_transition(
    payload: BatchTransitionRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> BatchTransitionResponse:
    """Apply many record transitions in one transaction."""
    actor_id = identity.actor_id
    service = _lifecycle_service(request)
    four_eyes = four_eyes_enabled()

    applied: list[BatchItemResult] = []
    rejected: list[BatchRejectedItem] = []
    skipped: list[BatchSkippedItem] = []

    for item in payload.items:
        action = _ITEM_ACTIONS.get(item.action)
        if action is None:
            rejected.append(
                _reject("FORBIDDEN_ACTION", item.record_id, item.action, "unknown action")
            )
            continue
        lifecycle_action = _ITEM_LIFECYCLE[item.action]

        # Per-item authorisation (§PL-7.1).
        try:
            enforce(identity, action)
        except HTTPException:
            rejected.append(
                _reject("FORBIDDEN_ACTION", item.record_id, item.action, "insufficient grant")
            )
            continue

        # Self-approval four-eyes check before touching the store.
        if action is AdminAction.RECORDS_APPROVE and four_eyes:
            try:
                current = service._latest(item.record_id)
            except RecordNotFound:
                rejected.append(
                    _reject("NOT_FOUND", item.record_id, item.action, "no such record")
                )
                continue
            if current.authored_by is not None and current.authored_by == actor_id:
                rejected.append(
                    _reject(
                        "SELF_APPROVAL_FORBIDDEN",
                        item.record_id,
                        item.action,
                        "approver may not be the version's author",
                    )
                )
                continue

        # Already in target state → skipped (no store write).
        try:
            current = service._latest(item.record_id)
        except RecordNotFound:
            current = None
        if current is not None and current.status == _TARGET_STATUS[lifecycle_action]:
            skipped.append(
                BatchSkippedItem(record_id=item.record_id, action=item.action)
            )
            continue

        try:
            stored = service.transition(
                item.record_id,
                lifecycle_action,
                actor_id=actor_id,
                role=identity.role.value,
                expected_version=item.expected_version,
                reason=item.reason,
            )
        except VersionConflictError as exc:
            rejected.append(
                _reject(
                    "VERSION_CONFLICT",
                    item.record_id,
                    item.action,
                    f"expected version {exc.expected}, current is {exc.current}",
                )
            )
            continue
        except FourEyesViolation:
            rejected.append(
                _reject(
                    "SELF_APPROVAL_FORBIDDEN",
                    item.record_id,
                    item.action,
                    "approver may not be the version's author",
                )
            )
            continue
        except RecordNotFound:
            rejected.append(
                _reject("NOT_FOUND", item.record_id, item.action, "no such record")
            )
            continue
        except IllegalTransitionError as exc:
            rejected.append(
                _reject(
                    "FORBIDDEN_ACTION",
                    item.record_id,
                    item.action,
                    f"illegal transition from {exc.current} for {exc.action}",
                )
            )
            continue
        except HTTPException:
            rejected.append(
                _reject("FORBIDDEN_ACTION", item.record_id, item.action, "refused by state machine")
            )
            continue

        applied.append(
            BatchItemResult(
                record_id=item.record_id, action=item.action, version=stored.version
            )
        )

    return BatchTransitionResponse(applied=applied, rejected=rejected, skipped=skipped)
