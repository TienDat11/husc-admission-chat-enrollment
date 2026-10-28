"""Admin endpoints: warmup post-deploy + atomic reload-table for blue-green swap.

@spec(S10.B7) /admin/warmup
@spec(S13.4) /admin/reload-table — atomic retriever swap with drain (RequestTracker).
SLA: <30s total = drain ≤25s + swap ≤1s + healthcheck ≤4s (per plan F4).
"""
import asyncio
import time
import unicodedata
from datetime import date
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

from config.settings import settings
from src.config.admin_identity import four_eyes_enabled
from src.domain.admin_identity import (
    AdminAction,
    Identity,
    action_for_transition,
    granted_actions,
)
from src.domain.canonical.lifecycle import (
    IllegalTransitionError,
    LifecycleAction,
    RejectReasonRequiredError,
)
from src.domain.canonical.record import AuthorityLevel, CanonicalRecord, RecordStatus
from src.domain.canonical.semantic_units import structured_payload
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from src.routers.admin_auth import current_identity, enforce, require_action
from src.routers.admin_rate_limit import admin_rate_limit
from src.routers.idempotency_guard import IdempotencyGuard
from src.services.admin_lifecycle import (
    AdminLifecycleService,
    ClaimRefusedError,
    FourEyesViolation,
    NotSupersededError,
    RecordNotFound,
    VersionConflictError,
)
from src.services.admin_rollback import (
    AdminRollbackService,
    InvalidReleaseIdError,
    NoPreviousReleaseError,
    ReleaseManifestMismatchError,
    ReleaseNotFoundError,
    RetrievalUnavailableError,
    RollbackError,
    RollbackValidationError,
    TargetIsActiveError,
    TargetNotTaggedError,
)
from src.services.release_status import (
    IndexJobsService,
    ReleaseStatusService,
    ReleaseView,
)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    # Ticket 27: every admin route spends one unit of the *calling actor's*
    # budget. Applied at router level rather than per route so a route added
    # later cannot silently ship without a limit — the same reason RBAC is
    # enforced by a named action instead of per-endpoint checks.
    dependencies=[Depends(admin_rate_limit)],
)

# F4 budget — drain bounded so total stays under 30s SLA.
DEFAULT_DRAIN_TIMEOUT_S = 25.0
ABS_MAX_DRAIN_TIMEOUT_S = 30.0

_reload_lock = asyncio.Lock()


@router.post("/warmup", dependencies=[Depends(require_action(AdminAction.OPS_WARMUP))])
async def warmup() -> dict:
    """Pre-warm heavy services (embedding model, LLM connection).
    Call after deploy before serving production traffic.
    @spec(S10.B7)
    """
    try:
        # Project layout puts src/ on sys.path; prefer relative-style import.
        from app.lifecycle import AppServices
    except ModuleNotFoundError:
        try:
            from rag2025.src.app.lifecycle import AppServices  # legacy path
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"AppServices module unavailable: {exc}",
            ) from exc

    services = AppServices()
    results: dict = {}

    # Warm embedding (~5s for Qwen3 model load)
    t0 = time.perf_counter()
    embedding = services.get_embedding()
    embedding.encode_query("warmup ping")
    results["embedding_ms"] = round((time.perf_counter() - t0) * 1000)

    # LLM warmup (connection check) — skip if no ns24 key.
    if settings.NS24_API_KEY:
        t1 = time.perf_counter()
        llm = services.get_llm_client()
        try:
            await llm.chat(
                user_message="ping",
                max_tokens=5,
            )
        except Exception:
            pass  # warmup is best-effort

        results["llm_ms"] = round((time.perf_counter() - t1) * 1000)

    return {"status": "warmed", "timings_ms": results}


# ============================================================================
# /admin/reload-table — @spec(S13.4) blue-green LanceDB atomic swap
# ============================================================================


class ReloadTableRequest(BaseModel):
    """Body for POST /admin/reload-table.

    ``force_while_in_flight`` is the emergency override for the drain gate, and
    is named for exactly what it does: swap while requests are still in flight,
    accepting that a request already reading the old retriever fails. It is a
    second, explicit decision by the caller — never a meaning read into the
    drain timeout. A plain reload whose drain times out is refused with 409.
    """

    table_name: str = Field(..., min_length=1, max_length=128)
    drain_timeout_s: float = Field(
        default=DEFAULT_DRAIN_TIMEOUT_S,
        gt=0.0,
        le=ABS_MAX_DRAIN_TIMEOUT_S,
    )
    force_while_in_flight: bool = Field(
        default=False,
        description=(
            "Emergency override: swap even though the drain timed out. A "
            "request already reading the old retriever may then fail; the "
            "response reports ``forced_while_in_flight``."
        ),
    )


class ReloadTableResponse(BaseModel):
    """Response for POST /admin/reload-table.

    ``drained`` is always True on a 200: a drain that times out is a 409 unless
    the caller forced it, in which case ``forced_while_in_flight`` is True.
    """

    active_table: str
    drained: bool
    drain_ms: float
    total_ms: float
    in_flight_at_swap: int
    forced_while_in_flight: bool


RetrieverBuilder = Callable[[str], Awaitable[Any]]
# The runtime swap seam. main.py owns it (it owns the module globals a query
# reads) and exposes it on the app object the same way it exposes ``_health_*``:
# this module cannot import main.py by name, because uvicorn runs it as ``main``
# while tests load it under their own module names.
RuntimeSwap = Callable[[Any], dict]


async def _safe_close(retriever: Any) -> None:
    """Close a retriever, swallowing exceptions so cleanup never raises."""
    close = getattr(retriever, "close", None)
    if close is None:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("reload-table: error closing old retriever (ignored)")


@router.post(
    "/reload-table",
    response_model=ReloadTableResponse,
    dependencies=[Depends(require_action(AdminAction.OPS_RELOAD_TABLE))],
)
async def reload_table(payload: ReloadTableRequest, request: Request) -> ReloadTableResponse:
    """Drain → build new retriever → healthcheck → atomic swap → close old.

    @spec(S13.4) atomic blue-green swap (M1+F1+F4 from plan v2.1).

    The swap itself is the ``app._swap_runtime_services`` seam (``src/main.py``),
    which rebinds every place a query can read a retriever. Assigning
    ``state.retriever`` alone left every query path — /query, /v2, /v3 — on the
    old table while this route reported a successful reload.

    A drain that does not finish inside ``drain_timeout_s`` is a refusal, not a
    warning: nothing is swapped, the old retriever is not closed, and the answer
    is 409. Swapping anyway would close a retriever a request is still reading,
    which is the AttributeError ``src/app/request_tracker.py`` documents. Only an
    explicit ``force_while_in_flight`` swaps past a timeout.
    """
    state = request.app.state
    tracker = getattr(state, "request_tracker", None)
    builder: RetrieverBuilder | None = getattr(state, "build_retriever", None)
    swap: RuntimeSwap | None = getattr(request.app, "_swap_runtime_services", None)
    if tracker is None or builder is None:
        raise HTTPException(
            status_code=503,
            detail="App state missing request_tracker or build_retriever",
        )
    if swap is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Runtime swap seam not wired: the app exposes no "
                "_swap_runtime_services. Swapping app.state alone would report a "
                "successful reload while every query kept the old retriever."
            ),
        )

    start = time.monotonic()
    async with _reload_lock:
        # 1. Build the replacement while the old retriever keeps serving. The
        #    build reads nothing a query reads, so it stays outside the gate: a
        #    slow build delays the swap, not the traffic. Closing admission
        #    here instead would block every query for the length of the build
        #    and buy nothing.
        try:
            new_retriever = await builder(payload.table_name)
        except Exception as exc:
            logger.exception(f"reload-table: builder failed for {payload.table_name}")
            raise HTTPException(status_code=503, detail=f"Builder error: {exc}") from exc

        # 2. Healthcheck.
        try:
            healthy = await new_retriever.healthcheck()
        except Exception as exc:
            await _safe_close(new_retriever)
            logger.exception("reload-table: healthcheck raised")
            raise HTTPException(status_code=503, detail=f"Healthcheck raised: {exc}") from exc
        if not healthy:
            await _safe_close(new_retriever)
            raise HTTPException(status_code=503, detail="New retriever failed healthcheck")

        # 3. Close admission, drain, and swap as one indivisible step (bounded
        #    timeout per F4). The gate — not the drain — is what makes the swap
        #    safe: ``drain()`` returning True only proves the counter *reached*
        #    zero, so with admission still open a request arriving in the window
        #    after it would increment, read the old retriever, and still be
        #    reading it when step 5 closed it. The gate reopens admission only
        #    once the swap is done, so every request it releases reads the
        #    runtime the swap already rebound.
        #
        #    A drain that times out is refused (FIX-01): the swap does not run,
        #    the old retriever is not closed, and the 409 tells the operator to
        #    retry. The gate's ``finally`` reopens admission on the way out of
        #    *every* path, the refusal included, so a failed reload can never
        #    leave the process refusing queries. The only way past a timeout is
        #    ``force_while_in_flight``, which the caller has to ask for by name.
        drain_start = time.monotonic()
        forced = False
        async with tracker.gate(timeout=payload.drain_timeout_s) as drained:
            drain_ms = (time.monotonic() - drain_start) * 1000.0
            if not drained:
                # FAIL CLOSED (FIX-01). Proceeding here would close a retriever
                # a request is still reading, and `request_tracker`'s own
                # docstring records what that costs: `close()` nulls
                # `LanceDBRetriever._adapter` and the reader then dies with
                # AttributeError. A reload has no emergency that justifies it —
                # the running release keeps serving — so the default is to
                # refuse and let the operator retry once the load drains.
                in_flight_at_timeout = tracker.in_flight
                if not payload.force_while_in_flight:
                    # The candidate was built and never swapped in, so nothing
                    # else holds it; closing it here leaks nothing. `close()` on
                    # the retriever this route builds only drops adapter
                    # references (LanceDBRetriever.close), so it cannot
                    # meaningfully extend the window admission stays closed.
                    await _safe_close(new_retriever)
                    logger.warning(
                        f"reload-table: drain timeout after "
                        f"{payload.drain_timeout_s}s with "
                        f"in_flight={in_flight_at_timeout}; refusing the swap "
                        "(the running release keeps serving)"
                    )
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Refusing to swap: drain timed out after "
                            f"{payload.drain_timeout_s}s with "
                            f"{in_flight_at_timeout} request(s) still in flight. "
                            "The running release is unchanged and admission has "
                            "been reopened. Retry once the load has drained, or "
                            "resend with force_while_in_flight=true to swap "
                            "anyway (a request already reading the old "
                            "retriever may then fail)."
                        ),
                        headers={"Retry-After": "1"},
                    )
                # The override: the caller asked for the swap knowing that a
                # request already holding the old retriever may fail. Admission
                # is still closed for the swap itself, so nothing *new* is
                # admitted into the half-swapped window.
                forced = True
                logger.warning(
                    f"reload-table: drain timeout after {payload.drain_timeout_s}s "
                    f"with in_flight={in_flight_at_timeout}; "
                    "force_while_in_flight=true — swapping anyway"
                )

            # 4. Atomic swap — one seam, so the retriever every query path reads
            #    is the new one (not merely the object app.state happens to
            #    hold). Admission has been closed since step 3, so nothing can
            #    have been admitted since the drain: this is the count the swap
            #    is actually safe at.
            in_flight_at_swap = tracker.in_flight
            try:
                swap_result = swap(new_retriever)
            except Exception as exc:
                await _safe_close(new_retriever)
                logger.exception("reload-table: runtime swap failed")
                raise HTTPException(status_code=503, detail=f"Swap error: {exc}") from exc
            if not isinstance(swap_result, dict) or "previous_retriever" not in swap_result:
                await _safe_close(new_retriever)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Swap seam returned no 'previous_retriever' mapping — the old "
                        "retriever would leak unclosed"
                    ),
                )
            old_retriever = swap_result["previous_retriever"]

            # 5. Async cleanup of the old retriever — do not block the response.
            #    Safe precisely because of the gate: on the drained path no
            #    admitted request is reading it, and none can be admitted until
            #    the gate reopens.
            if old_retriever is not None:
                if forced:
                    # ...but a forced swap is the case the gate could not make
                    # safe: a reader may still be holding the old retriever, and
                    # closing it is exactly what would kill that reader
                    # (`close()` nulls ``_adapter``). So it is left open — this
                    # response is the last thing that holds it besides those
                    # readers, so it goes away with the last of them.
                    logger.warning(
                        "reload-table: forced swap — the old retriever is left "
                        "open for the requests still reading it"
                    )
                else:
                    asyncio.create_task(_safe_close(old_retriever))

            total_ms = (time.monotonic() - start) * 1000.0

        return ReloadTableResponse(
            active_table=payload.table_name,
            drained=drained,
            drain_ms=drain_ms,
            total_ms=total_ms,
            in_flight_at_swap=in_flight_at_swap,
            forced_while_in_flight=forced,
        )


# ============================================================================
# /admin/rollback — two-level rollback to the last good release (ticket 22)
#
# The ticket: "rollback cả dữ liệu lẫn retrieval về bản tốt gần nhất, mỗi lần
# rollback là một sự kiện có người chịu trách nhiệm". The primitives already
# exist (lancedb_versioning.rollback_to/promote_to_prod, canonical_store.
# publish_pointer); this route wires them to the swap seam Part 0 extracted.
#
# Both levels move together, deliberately: a half-rollback leaves canonical and
# retrieval on different releases, which is the mixed state the ticket forbids.
# So there is no ``level`` field — the contract is one route that moves both.
#
# Authorization is the RECORDS_ROLLBACK grant, which the matrix gives to
# ``admin`` alone: a rollback retracts live data, and the actor is recorded in
# the audit event.
# ============================================================================


class RollbackRequest(BaseModel):
    """Body for POST /admin/rollback.

    ``reason`` is mandatory and non-blank. The event the rollback emits refuses
    to exist without one (``AuditAction.ROLLBACK`` is in ``REASON_REQUIRED``),
    so a blank reason is refused here — 422, before any store, pointer or table
    is touched.

    ``target_release_id`` is optional. Without it the target is the previous
    release in the lineage (``parent_release_id``, walked back from the active
    one); with it, a rollback can go further back than one release.

    ``force_while_in_flight`` is the emergency override for the drain gate. A
    rollback whose drain times out is refused with 409 like a reload is; this
    flag is how an operator says "roll back anyway, knowing a request still
    reading the release I am rolling away from may fail". It is a separate,
    explicitly-named decision — never a meaning read into the timeout.
    """

    reason: str = Field(..., min_length=1, max_length=2000)
    target_release_id: str | None = Field(default=None, max_length=128)
    force_while_in_flight: bool = Field(
        default=False,
        description=(
            "Emergency override: roll back even though the drain timed out. A "
            "request already reading the old release may then fail; the "
            "response reports ``forced_while_in_flight``."
        ),
    )

    @field_validator("reason")
    @classmethod
    def _reason_must_say_something(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                "reason must not be blank — a rollback nobody can explain is not accountable"
            )
        return cleaned


class RollbackResponse(BaseModel):
    """What the rollback did, per level, plus the audit event it emitted."""

    release_id: str
    previous_release_id: str
    lancedb_tag: str
    lancedb_tag_version: int | None
    lancedb_version: int | None
    steps: dict[str, str]
    event_id: str | None
    forced_while_in_flight: bool


def _rollback_service(request: Request) -> AdminRollbackService:
    """The app's rollback service, or 503.

    No lazy fallback, unlike the lifecycle service: a service built here would
    have to guess the releases root, the live retriever and the swap seam, and a
    rollback that guessed wrong would move canonical data while the process kept
    serving another release. Startup wiring
    (``app.lifecycle.configure_canonical_record_store``) builds it once.
    """
    service = getattr(request.app.state, "admin_rollback_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "rollback is not configured: app.state.admin_rollback_service is missing"
            ),
        )
    return service


# A validation failure is 4xx and left nothing mutated; anything else is a 5xx
# whose detail names every failure of the attempt, compensation included.
_ROLLBACK_STATUS: dict[type, int] = {
    ReleaseNotFoundError: 404,
    NoPreviousReleaseError: 409,
    TargetIsActiveError: 409,
    TargetNotTaggedError: 409,
    ReleaseManifestMismatchError: 409,
    InvalidReleaseIdError: 422,
    RollbackValidationError: 422,
}


class _rollback_errors:
    """Map rollback failures onto HTTP status codes."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        if isinstance(exc, RetrievalUnavailableError):
            # Nothing was mutated: the retrieval level does not exist yet.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, RollbackValidationError):
            raise HTTPException(
                status_code=_ROLLBACK_STATUS.get(type(exc), 422), detail=str(exc)
            ) from exc
        if isinstance(exc, RollbackError):
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return False


@router.post(
    "/rollback",
    response_model=RollbackResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_ROLLBACK))],
)
async def rollback_release(
    payload: RollbackRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> RollbackResponse:
    """Roll canonical + retrieval back to the last good release, together.

    Both levels move as one unit: the canonical pointer (step C), then the
    LanceDB tag and the running runtime (step D). If step D fails, the pre-call
    state is restored (step E) and the 5xx names both failures. The ROLLBACK
    audit event is emitted only after both levels have moved, so the ledger
    never records a rollback that did not happen.

    The same ``_reload_lock`` a blue-green reload holds is taken here: a reload
    and a rollback both change what every query reads, so they must not overlap.

    The same ``request_tracker.gate`` a blue-green reload holds is taken here
    too, for the same reason: step D rebinds the runtime every query path reads,
    so an in-flight query still holding the old retriever would be answered from
    a release the ledger just recorded as rolled back. Admission reopens once
    step D has completed.

    A drain that times out is refused with 409 (FIX-01), exactly as the reload
    route refuses one: the old "log and proceed" policy answered a request from
    the release being rolled away from, and closed the table under it. The
    rollback that must happen anyway is ``force_while_in_flight=true`` — an
    explicit emergency decision by the caller, recorded in the response.

    Idempotency (ticket 27): ``target_release_id`` is optional, and without it
    the target is resolved from the *current* pointer. A replayed implicit
    rollback therefore resolves a *different* target — R3→R2, then R2→R1 — and
    walks back two releases for what the caller believed was one action. No
    state comparison catches that (the two calls legitimately differ), so an
    ``Idempotency-Key`` is the only thing that can. With one, a replay returns
    the first call's stored response and does not run again.
    """
    # Service first: a missing service is a configuration error and must report
    # as such even when the tracker is also missing.
    service = _rollback_service(request)
    tracker = getattr(request.app.state, "request_tracker", None)
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="App state missing request_tracker",
        )

    guard = IdempotencyGuard(request, identity, "records.rollback", payload)
    if (replayed := guard.replay()) is not None:
        return replayed
    forced = False
    try:
        async with _reload_lock:
            with _rollback_errors():
                async with tracker.gate(timeout=DEFAULT_DRAIN_TIMEOUT_S) as drained:
                    if not drained:
                        # FAIL CLOSED (FIX-01). The old "proceed anyway" branch
                        # here answered a query from a release the ledger is
                        # about to record as rolled back, and step D closes /
                        # retags the table that query is reading. An operator
                        # who genuinely needs to move under load says so with
                        # ``force_while_in_flight`` — the timeout never means
                        # "force" by itself.
                        in_flight_at_timeout = tracker.in_flight
                        if not payload.force_while_in_flight:
                            logger.warning(
                                f"rollback: drain timeout after "
                                f"{DEFAULT_DRAIN_TIMEOUT_S}s with "
                                f"in_flight={in_flight_at_timeout}; refusing the "
                                "rollback (the active release is unchanged)"
                            )
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"Refusing to roll back: drain timed out "
                                    f"after {DEFAULT_DRAIN_TIMEOUT_S}s with "
                                    f"{in_flight_at_timeout} request(s) still in "
                                    "flight. Nothing was moved — the active "
                                    "release and its retrieval are unchanged — "
                                    "and admission has been reopened. Retry "
                                    "once the load has drained, or resend with "
                                    "force_while_in_flight=true to roll back "
                                    "anyway (a request already reading the old "
                                    "release may then fail)."
                                ),
                                headers={"Retry-After": "1"},
                            )
                        forced = True
                        logger.warning(
                            f"rollback: drain timeout after "
                            f"{DEFAULT_DRAIN_TIMEOUT_S}s with "
                            f"in_flight={in_flight_at_timeout}; "
                            "force_while_in_flight=true — rolling back anyway"
                        )
                    outcome = service.rollback(
                        reason=payload.reason,
                        actor_id=identity.actor_id,
                        role=identity.role.value,
                        target_release_id=payload.target_release_id,
                    )
    except BaseException:
        # The key is freed on failure: a 5xx must not consume the operator's
        # key, or the retry that fixes the problem would be refused as a replay
        # of the attempt that failed.
        guard.abandon()
        raise

    response = RollbackResponse(
        release_id=outcome.release_id,
        previous_release_id=outcome.previous_release_id,
        lancedb_tag=outcome.lancedb_tag,
        lancedb_tag_version=outcome.lancedb_tag_version,
        lancedb_version=outcome.lancedb_version,
        steps=dict(outcome.steps),
        event_id=outcome.event_id,
        forced_while_in_flight=forced,
    )
    guard.complete(response)
    return response


# ============================================================================
# /admin/records — canonical admissions lifecycle (ticket 12)
#
# State machine (spec, verbatim):
#   draft --submit--> review --approve--> approved --publish--> published
#                                                              --supersede--> archived
#     ^                  |
#     |                  +--reject(reason bắt buộc)--> draft
#   Cấm: draft -> published, approved -> archived trực tiếp, publish khi chưa approved.
#
# Writes are POST rather than PUT/PATCH: the app's CORS middleware allows only
# GET/POST/OPTIONS, so a PUT endpoint would be unreachable from the admin UI
# without a cross-cutting CORS change.
#
# AUTHORIZATION (ticket 13): each route names the AdminAction it performs and
# the role x action matrix (src/domain/admin_identity.py) decides who may
# perform it. The credential is resolved to an Identity server-side
# (src/routers/admin_auth.py), so the actor_id in the audit event is the
# authenticated caller, never something the body asked for. The shared ops
# token keeps serving /admin/warmup and /admin/reload-table (role 'ops') and is
# granted no records.* action, so it cannot publish. The guard ticket 12 placed
# ("no credential -> no publish") is untouched: the header stays required
# (missing -> 422) and an unrecognised credential is refused (-> 403).
# ============================================================================


class CreateRecordRequest(BaseModel):
    """Body for POST /admin/records — a full canonical record, version 1.

    No ``actor_id``: the actor is whoever the credential resolves to, so a
    caller cannot write a version under someone else's name.
    """

    record: CanonicalRecord


class EditRecordRequest(BaseModel):
    """Body for POST /admin/records/{record_id}/edits.

    Only the listed fields are editable: identity (``record_id``), the
    ``version`` counter and lifecycle ``status`` are owned by the store and the
    state machine, never by a caller. ``expected_version`` is the optimistic
    concurrency token — the version the caller believes is current.

    ``record_type`` is absent on purpose: the record's kind is part of what the
    fact *is*, and the id embeds it (``CHI_TIEU:nganh:...``), so re-typing a
    fact is a new record rather than an edit of this one.

    ``values`` is the record's structured payload (the canonical schema is
    additive, ``extra="allow"``). It **replaces** the payload wholesale, like
    every other field here; a caller that edits one structured field merges the
    rest itself. The admin UI does exactly that, so the extractor's other keys
    survive a hand edit.
    """

    expected_version: int
    reason: str | None = None
    admission_year: int | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    authority_level: AuthorityLevel | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    source_page: int | None = None
    supersedes: str | None = None
    admission_round: str | None = None
    major_codes: list[str] | None = None
    admission_methods: list[str] | None = None
    values: dict[str, Any] | None = None
    text: str | None = None


class TransitionRequest(BaseModel):
    """Body for POST /admin/records/{record_id}/transitions."""

    action: LifecycleAction
    expected_version: int | None = None
    reason: str | None = None


class ClaimRequest(BaseModel):
    """Body for POST /admin/records/{record_id}/claim."""

    expected_version: int | None = None
    reason: str | None = None


class LifecycleRecordResponse(BaseModel):
    """The stored record version plus the audit event the write emitted."""

    record: CanonicalRecord
    event: dict


class RecordHistoryResponse(BaseModel):
    """Every stored version of one record, newest first."""

    record_id: str
    versions: list[CanonicalRecord]


def _lifecycle_service(request: Request) -> AdminLifecycleService:
    """The app's lifecycle service, or a lazily built one over its store.

    503 is reserved for genuinely absent configuration — an app that was never
    wired to a canonical store (see ``app.lifecycle.configure_canonical_record_store``).

    The four-eyes switch is passed as the config *callable*, so the service
    consults the environment at each check rather than at build time: turning
    the control off is still a deliberate deployment change (only an operator
    editing the environment can do it), but a service built lazily on first
    request reflects the environment as it is, not as it was at import.

    The ledger is threaded in for the same reason the service refuses to write
    unaudited: a version that is stored without an event cannot be explained or
    rolled back later, and the caller would have no way to tell that from a
    write that never happened. An app that has a store but no audit sink is
    therefore refused rather than served — the read-only audit router makes the
    same choice when its ledger is missing.
    """
    service = getattr(request.app.state, "admin_lifecycle_service", None)
    if service is not None:
        return service
    store = getattr(request.app.state, "canonical_record_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="canonical record store is not configured",
        )
    ledger = getattr(request.app.state, "audit_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=503,
            detail="audit ledger is not configured",
        )
    registry = getattr(request.app.state, "source_document_registry", None)
    service = AdminLifecycleService(
        store, ledger=ledger, four_eyes=four_eyes_enabled, registry=registry
    )
    request.app.state.admin_lifecycle_service = service
    return service


class _lifecycle_errors:
    """Map lifecycle failures onto 4xx without changing stored data.

    Every one of these is raised before a version is written, so a rejected
    request leaves the store byte-identical — the ticket's "không đổi dữ liệu"
    requirement holds by construction rather than by cleanup.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        if isinstance(exc, RejectReasonRequiredError):
            # 422: the request is well-formed but lacks a required justification.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if isinstance(exc, IllegalTransitionError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, (VersionConflictError, NotSupersededError, ClaimRefusedError)):
            # 409: the request is well-formed, but the record is in a state that
            # does not allow the move. ``claim`` refuses a non-draft, or a draft
            # that already has an author, for the same reason ``supersede``
            # refuses a published record that nothing replaces.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, ValidationError):
            # A transition the state machine allows but the schema refuses —
            # publishing a record with no source document is the reachable one
            # (``CanonicalRecord`` requires provenance for ``published``). The
            # request was well-formed and the *state* refused it, so this is a
            # 422 the caller can act on, not a 500 that hides which field is
            # missing.
            raise HTTPException(
                status_code=422,
                detail="; ".join(
                    error.get("msg", "") for error in exc.errors()
                ).strip(),
            ) from exc
        if isinstance(exc, FourEyesViolation):
            # 403: the caller is authenticated and holds the approve grant, but
            # is not permitted to exercise it on this version.
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if isinstance(exc, RecordNotFound):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return False


@router.post(
    "/records",
    response_model=LifecycleRecordResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_CREATE))],
)
async def create_record(
    payload: CreateRecordRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> LifecycleRecordResponse:
    """Store a new canonical record as version 1 (draft)."""
    service = _lifecycle_service(request)
    with _lifecycle_errors():
        stored = service.create(
            payload.record, actor_id=identity.actor_id, role=identity.role.value
        )
    return LifecycleRecordResponse(record=stored, event=service.emitted_events()[-1].to_dict())


@router.get(
    "/records/{record_id}",
    response_model=CanonicalRecord,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def get_record(record_id: str, request: Request) -> CanonicalRecord:
    """Current version of one record."""
    service = _lifecycle_service(request)
    with _lifecycle_errors():
        return service.get(record_id)


@router.get(
    "/records/{record_id}/versions",
    response_model=RecordHistoryResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def get_record_versions(record_id: str, request: Request) -> RecordHistoryResponse:
    """Every stored version of one record, newest first."""
    service = _lifecycle_service(request)
    with _lifecycle_errors():
        versions = service.history(record_id)
    return RecordHistoryResponse(record_id=record_id, versions=list(versions))


@router.post(
    "/records/{record_id}/edits",
    response_model=LifecycleRecordResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_EDIT))],
)
async def edit_record(
    record_id: str,
    payload: EditRecordRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> LifecycleRecordResponse:
    """Append a new version of a record.

    The appended version returns to ``draft``: the previous version's review no
    longer describes this content. A published version is never rewritten.
    """
    service = _lifecycle_service(request)
    # ``expected_version`` steers the write but is not a record field;
    # everything else the caller actually sent is the change set. The actor is
    # the authenticated identity, never a body field.
    changes = payload.model_dump(exclude_unset=True, exclude={"expected_version"})
    reason = changes.pop("reason", None)
    with _lifecycle_errors():
        stored = service.edit(
            record_id,
            actor_id=identity.actor_id,
            role=identity.role.value,
            expected_version=payload.expected_version,
            reason=reason,
            **changes,
        )
    return LifecycleRecordResponse(record=stored, event=service.emitted_events()[-1].to_dict())



@router.post(
    "/records/{record_id}/claim",
    response_model=LifecycleRecordResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_EDIT))],
)
async def claim_record(
    record_id: str,
    payload: ClaimRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> LifecycleRecordResponse:
    """Attach the caller as author of an unclaimed draft (ticket T15).

    The seed drafts carry ``authored_by = null`` (spec 02 §108), and a version
    with no recorded author cannot be approved — four-eyes refuses the
    approval rather than approve for an unknown author. This route is the fix
    that keeps the seed version: it appends a new version whose ``authored_by``
    is the caller, the seed version and its audit history stay as written.

    Only legal on a draft that has no author yet: a non-draft is a 409
    (claiming would demote a version a decision has been made on) and an
    already-claimed draft is a 409 too — claim is not the way to re-author
    someone else's work, that is an edit.
    """
    service = _lifecycle_service(request)
    with _lifecycle_errors():
        stored = service.claim(
            record_id,
            actor_id=identity.actor_id,
            role=identity.role.value,
            expected_version=payload.expected_version,
            reason=payload.reason,
        )
    return LifecycleRecordResponse(record=stored, event=service.emitted_events()[-1].to_dict())


@router.post(
    "/records/{record_id}/transitions",
    response_model=LifecycleRecordResponse,
)
async def transition_record(
    record_id: str,
    payload: TransitionRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> LifecycleRecordResponse:
    """Move a record along the lifecycle state machine.

    One route carries five grants, so the required action is resolved from the
    payload and checked before the store is touched: ``submit`` is an editor's,
    ``approve``/``reject`` a reviewer's, ``publish``/``supersede`` a
    publisher's. A lifecycle action the matrix does not map is refused — an
    action nobody has been granted is not one this route may perform.

    An illegal move is refused with 409 and writes nothing; a reject without a
    reason is refused with 422; an approve by the version's author is refused
    with 403 (four-eyes). ``supersede`` archives a published record only when
    another published record already names it in ``supersedes``.

    Idempotency (ticket 27): the state machine already makes a *second* publish
    impossible — ``_TRANSITIONS`` has no self-loop, so a replay is refused 409
    rather than applied twice. That is state-based protection, and it is not the
    same as a replay returning the first call's answer: an operator whose network
    dropped mid-publish would get a 409 on retry and no way to tell "my publish
    landed" from "someone else published". With an ``Idempotency-Key`` the retry
    returns the original response instead.
    """
    enforce(identity, action_for_transition(payload.action))
    service = _lifecycle_service(request)

    guard = IdempotencyGuard(
        request,
        identity,
        f"records.transition.{payload.action}",
        # The record is addressed by the URL path, so it is not in the body the
        # guard fingerprints by default. Fold it in explicitly: without it the
        # same key on record B would replay record A's stored response and
        # return it as B's, leaving B untouched while the caller believes it
        # acted. The record is part of *what was requested*, so it belongs in
        # the fingerprint.
        {"record_id": record_id, **payload.model_dump(mode="json")},
    )
    if (replayed := guard.replay()) is not None:
        return replayed
    try:
        with _lifecycle_errors():
            stored = service.transition(
                record_id,
                payload.action,
                actor_id=identity.actor_id,
                role=identity.role.value,
                expected_version=payload.expected_version,
                reason=payload.reason,
            )
    except BaseException:
        guard.abandon()
        raise

    response = LifecycleRecordResponse(
        record=stored, event=service.emitted_events()[-1].to_dict()
    )
    guard.complete(response)
    return response

# ============================================================================
# /admin/identity + GET /admin/records — the admin CRUD surface (ticket 19)
#
# Two additions the admissions-office UI needs and the ticket-12 surface did
# not carry:
#
# * ``GET /admin/identity`` — the caller's own role and grants, so the UI can
#   *hide* an action it may not perform. A button that is rendered and then
#   403s is a worse surface than no button, and the alternative — a hardcoded
#   role→button table in the frontend — would be a second copy of the policy
#   that drifts from ``src/domain/admin_identity.py``.
# * ``GET /admin/records`` — the filterable list. Without it the office can
#   only reach a record it already knows the id of.
#
# Neither route weakens the ticket-13 rule: identity still comes from the
# credential, and every records route still names the action it performs.
# ============================================================================

class AdminIdentityResponse(BaseModel):
    """The caller's own identity plus the actions its role grants."""

    actor_id: str
    role: str
    actions: list[str]

@router.get("/identity", response_model=AdminIdentityResponse)
async def get_identity(
    identity: Identity = Depends(current_identity),
) -> AdminIdentityResponse:
    """Who the presented credential is, and what it may do.

    No action is required to read this, deliberately: the response exposes only
    the caller's *own* grants — never another role's — and a role granted
    nothing must still be able to discover that it was granted nothing. A
    required action here would make that case unanswerable (403 on the very
    call that reports the 403s).
    """
    return AdminIdentityResponse(
        actor_id=identity.actor_id,
        role=identity.role.value,
        actions=sorted(action.value for action in granted_actions(identity.role)),
    )

class RecordListResponse(BaseModel):
    """Latest version of every stored record that matched the filter."""

    records: list[CanonicalRecord]
    total: int

def _fold(text: str) -> str:
    """Case- and diacritic-folded search key for the ngành filter.

    Deliberately local rather than imported from
    ``src.services.vietnamese_analyzer``: that module imports its siblings as
    ``services.*`` (the layout the app runs in), which the ``src.*`` layout the
    router and its tests use cannot resolve. The rule is that module's rule,
    ``đ`` included — ``str.lower()`` maps ``Đ`` to ``đ`` first, so the single
    mapping below covers both cases.
    """
    out: list[str] = []
    for char in unicodedata.normalize("NFD", text.lower()):
        if char == "đ":
            out.append("d")
        elif ord(char) < 128:
            out.append(char)
        elif unicodedata.category(char) != "Mn":
            out.append(char)
    return "".join(out)

def _nganh_candidates(record: CanonicalRecord) -> tuple[str, ...]:
    """The strings that name the ngành (major) a record is about.

    The canonical schema declares no ``nganh`` field: a record states one fact,
    and the major it is about is carried either in its ``record_id`` (the seed
    convention is ``<TYPE>:nganh:<ma_nganh>:<year>:<source>``) or in its
    structured payload under the two keys the chunker already treats as
    schema-owned (``semantic_units.FIELD_LABELS``). Both spellings are read, so
    an extractor-written record that nests them in ``values`` and a
    hand-authored one that sets them at the top level filter identically.
    """
    payload = structured_payload(record)
    extra = record.model_extra or {}
    candidates = [record.record_id]
    for key in ("ma_nganh", "ten_nganh"):
        for source in (payload, extra):
            value = source.get(key)
            if isinstance(value, str):
                candidates.append(value)
    return tuple(candidates)

def _matches_nganh(record: CanonicalRecord, query: str) -> bool:
    """Whether ``record`` is about the major ``query`` names.

    Substring, case- and diacritic-insensitive, so "han nom", "Hán Nôm" and
    "7220104" all find the same record — the office types whichever of the
    three it has at hand.
    """
    needle = _fold(query).strip()
    if not needle:
        return True
    return any(needle in _fold(candidate) for candidate in _nganh_candidates(record))

def _latest_versions(store: CanonicalRecordStore) -> list[CanonicalRecord]:
    """Latest version of every record the store holds.

    A record directory with no version file — a write that died between
    ``mkdir`` and ``os.replace`` — is skipped rather than raising: this is a
    browse surface, and one half-written record must not take the whole list
    down.
    """
    latest: list[CanonicalRecord] = []
    for record_id in store.record_ids():
        versions = store.versions(record_id)
        if versions:
            latest.append(store.get_version(record_id, versions[-1]))
    return latest

@router.get(
    "/records",
    response_model=RecordListResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_records(
    request: Request,
    nganh: str | None = Query(
        default=None,
        max_length=128,
        description="Ngành: mã ngành, tên ngành, hoặc phần record_id nêu nó.",
    ),
    year: int | None = Query(
        default=None, ge=2000, le=2100, description="Năm tuyển sinh."
    ),
    status: RecordStatus | None = Query(
        default=None, description="Trạng thái lifecycle của version mới nhất."
    ),
) -> RecordListResponse:
    """Every stored record's latest version, filtered by ngành / năm / trạng thái.

    Filters combine with AND; an absent filter matches everything. Newest
    admission year first, then ``record_id`` — a total order, so the same store
    lists in the same order on every call.
    """
    store = _lifecycle_service(request).store
    matched = [
        record
        for record in _latest_versions(store)
        if (year is None or record.admission_year == year)
        and (status is None or record.status is status)
        and (nganh is None or _matches_nganh(record, nganh))
    ]
    matched.sort(key=lambda record: (-record.admission_year, record.record_id))
    return RecordListResponse(records=matched, total=len(matched))



# ============================================================================
# Ticket 26 — read model for releases and index jobs (release + job screen).
#
# Additive: two read-only GETs, appended. Nothing above is touched. Both read
# the same on-disk state the rollback path moves (the pointer + manifests) and
# the job's own state machine — no parallel registry, so the screen cannot
# disagree with what a rollback or a job would actually do.
# ============================================================================


class ReleaseViewResponse(BaseModel):
    """One release, as its own manifest states it."""

    release_id: str
    created_at: str | None
    parent_release_id: str | None
    lancedb_tag: str | None
    lancedb_version: int | None
    document_count: int
    chunk_count: int


class ReleaseStatusResponse(BaseModel):
    """GET /admin/releases — the active release and the one behind it.

    ``active_release_id`` is what the serving path reads (the pointer) and is
    present even when the release it names cannot be read; ``active``/``previous``
    are the manifest-backed views. ``problems`` carries anything that could not
    be read, so an unreadable pointer is never rendered as "nothing published".
    """

    active_release_id: str | None
    previous_release_id: str | None
    active: ReleaseViewResponse | None
    previous: ReleaseViewResponse | None
    problems: list[str]


class RetryStepResponse(BaseModel):
    """One step of retrying a failed job, and whether this process can run it."""

    step: str
    runnable: bool
    command: str | None
    reason: str | None


class IndexJobResponse(BaseModel):
    """One index-job row.

    ``state`` is a JobState value (pending/running/failed/active) or null when
    the row is a build on disk rather than a run. ``saved`` and ``serving`` are
    separate on purpose: ``saved`` means a build is persisted, ``serving`` means
    it is the corpus the index answers from, and only ``active`` sets ``serving``.
    """

    job_id: str
    state: str | None
    saved: bool
    serving: bool
    label: str
    release_id: str | None
    staged_dir: str | None
    reason: str | None
    retry: list[RetryStepResponse]


class IndexJobsResponse(BaseModel):
    """GET /admin/index-jobs — job state, staged builds, and what is owed."""

    jobs: list[IndexJobResponse]
    reindex_needed: bool
    reindex_reason: str
    pending_outbox_entries: int
    pending_affected_record_ids: list[str]
    problems: list[str]


def _release_status_service(request: Request) -> ReleaseStatusService:
    """The read model over the app's own releases root, or 503.

    Built from ``app.state.admin_rollback_service`` rather than from settings:
    that service owns the releases root a rollback actually moves, and reading a
    different root would let this screen report a release the rollback path does
    not know about. An app without it refuses (503) instead of guessing a root.
    """
    rollback = getattr(request.app.state, "admin_rollback_service", None)
    if rollback is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "release status is not configured: app.state.admin_rollback_service "
                "is missing, so the releases root cannot be resolved"
            ),
        )
    return ReleaseStatusService(releases_root=rollback.releases_root)


def _index_jobs_service(request: Request) -> IndexJobsService:
    """The job read model over the app's outbox, corpus and (if any) live job.

    ``app.state.reindex_job`` is read when a runner exists; the runner's
    ``started`` flag is what makes the rebuild step of a retry runnable. The
    corpus dir is anchored to ``rag2025/data/canonical_corpus`` — the same
    directory ``scripts/ingest_lancedb.py`` ingests.
    """
    state = request.app.state
    return IndexJobsService(
        outbox=getattr(state, "reindex_outbox", None),
        job=getattr(state, "reindex_job", None),
        runner=getattr(state, "reindex_runner", None),
    )


@router.get(
    "/releases",
    response_model=ReleaseStatusResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def get_release_status(request: Request) -> ReleaseStatusResponse:
    """Which release is active, and which one is behind it.

    Read-only: the pointer and the manifests, straight off disk. The previous
    release is the one a targetless ``POST /admin/rollback`` would move to.
    """
    status = _release_status_service(request).status()
    return ReleaseStatusResponse(
        active_release_id=status.active_release_id,
        previous_release_id=status.previous_release_id,
        active=_release_view(status.active),
        previous=_release_view(status.previous),
        problems=list(status.problems),
    )


@router.get(
    "/index-jobs",
    response_model=IndexJobsResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_index_jobs(request: Request) -> IndexJobsResponse:
    """Index jobs: pending/running/failed/active, plus builds saved but not serving.

    ``serving`` is true only for an ``active`` job. A build that exists in the
    staging area is reported ``saved=true, serving=false`` — never as published.
    """
    status = _index_jobs_service(request).status()
    return IndexJobsResponse(
        jobs=[
            IndexJobResponse(
                job_id=job.job_id,
                state=job.state,
                saved=job.saved,
                serving=job.serving,
                label=job.label,
                release_id=job.release_id,
                staged_dir=job.staged_dir,
                reason=job.reason,
                retry=[
                    RetryStepResponse(
                        step=step.step,
                        runnable=step.runnable,
                        command=step.command,
                        reason=step.reason,
                    )
                    for step in job.retry
                ],
            )
            for job in status.jobs
        ],
        reindex_needed=status.reindex_needed,
        reindex_reason=status.reindex_reason,
        pending_outbox_entries=status.pending_outbox_entries,
        pending_affected_record_ids=list(status.pending_affected_record_ids),
        problems=list(status.problems),
    )


def _release_view(view: ReleaseView | None) -> ReleaseViewResponse | None:
    """One manifest-backed release view, or ``None``."""
    if view is None:
        return None
    return ReleaseViewResponse(
        release_id=view.release_id,
        created_at=view.created_at,
        parent_release_id=view.parent_release_id,
        lancedb_tag=view.lancedb_tag,
        lancedb_version=view.lancedb_version,
        document_count=view.document_count,
        chunk_count=view.chunk_count,
    )

# Ticket 38: the conflict queue and its override route. Mounted here rather
# than in main.py so the admin surface stays one include: the child router
# carries its own ``/conflicts`` prefix and inherits this router's rate-limit
# dependency, and main.py never learns the ticket exists.
from src.routers.admin_conflicts import router as conflicts_router  # noqa: E402

router.include_router(conflicts_router)

# Ticket 39: the source-document registry. Mounted the same way as the
# conflict queue above — the child carries its own ``/source-documents``
# prefix, inherits this router's rate-limit dependency, and main.py never
# learns the ticket exists.
from src.routers.admin_source_documents import router as source_documents_router  # noqa: E402

router.include_router(source_documents_router)

# Ticket T04: document upload. Mounted the same way as the registry above —
# its own ``/source-documents`` prefix, the parent's rate limit, plus its own
# per-actor upload budget; main.py still never learns the ticket exists.
from src.routers.admin_document_upload import (  # noqa: E402
    router as document_upload_router,
)

router.include_router(document_upload_router)

# Ticket 37: the identity-management surface. Mounted the same way as the two
# child routers above — the child carries its own ``/identities`` prefix,
# inherits this router's rate-limit dependency, and main.py never learns the
# ticket exists.
from src.routers.admin_identities import router as identities_router  # noqa: E402

router.include_router(identities_router)
# Ticket T07: the unanswered-questions queue surface. Mounted the same way as
# the child routers above — the child carries its own ``/unanswered`` prefix,
# inherits this router's rate-limit dependency, and main.py never learns the
# ticket exists.
from src.routers.admin_unanswered import router as unanswered_router  # noqa: E402

router.include_router(unanswered_router)

# Ticket T21: the paste-a-URL → draft surface. Mounted the same way as the child
# routers above - the child carries its own ``/source-documents`` prefix, which
# it shares with the ticket-T04 upload surface (FastAPI allows two routers to
# contribute to one path prefix; only full paths must be unique), inherits this
# router's rate-limit dependency, and main.py never learns the ticket exists.
from src.routers.admin_url_ingest import router as url_ingest_router  # noqa: E402

router.include_router(url_ingest_router)
