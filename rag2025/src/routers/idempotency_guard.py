"""HTTP glue for request-level idempotency on the mutating admin routes.

``src/services/idempotency.py`` owns the storage and the rules; this module owns
the HTTP shape of them — reading the ``Idempotency-Key`` header, turning the
three failure modes into status codes, and marking a replayed response so a
client can tell a replay from a fresh execution.

The header is optional on purpose. A caller that sends no key gets exactly the
behaviour it had before this module existed (the state guards in the lifecycle
state machine and ``TargetIsActiveError``); a caller that sends one gets the
stronger, request-level guarantee. Nothing about existing callers changes.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.domain.admin_identity import Identity
from src.services.idempotency import (
    IdempotencyConflict,
    IdempotencyInProgress,
    IdempotencyKeyInvalid,
    IdempotencyStore,
    ReplayResponse,
)

IDEMPOTENCY_HEADER = "Idempotency-Key"

# Marks a response served from a stored record rather than from a fresh run, so
# a client can tell "your retry was absorbed" from "your retry ran again".
REPLAY_HEADER = "Idempotent-Replay"

__all__ = ["IDEMPOTENCY_HEADER", "REPLAY_HEADER", "IdempotencyGuard"]

class IdempotencyGuard:
    """One request's idempotency state, from claim to completion.

    Used as::

        guard = IdempotencyGuard(request, identity, "records.rollback", payload)
        if (replayed := guard.replay()) is not None:
            return replayed
        try:
            ...run the operation...
        except BaseException:
            guard.abandon()
            raise
        guard.complete(result)
        return result

    A request without the header is inert: :meth:`replay` returns ``None``,
    :meth:`complete` and :meth:`abandon` do nothing.
    """

    def __init__(
        self,
        request: Request,
        identity: Identity,
        operation: str,
        body: object,
    ) -> None:
        self._identity = identity
        self._operation = operation
        # Normalized to plain data before fingerprinting: a pydantic model
        # stringifies to a repr whose exact form is the library's business, and
        # a fingerprint that depends on it would change under a pydantic
        # upgrade — silently turning every replay into a 422.
        self._body = (
            body.model_dump(mode="json") if hasattr(body, "model_dump") else body
        )
        self._store: IdempotencyStore | None = getattr(
            request.app.state, "idempotency_store", None
        )
        self._key: str | None = None
        self._claimed = False

        raw = request.headers.get(IDEMPOTENCY_HEADER)
        if raw is None:
            return
        if self._store is None:
            # Refused rather than ignored. A caller that asked for a replay
            # guarantee must not be told it got one by a deployment that cannot
            # provide it — silently running twice is the exact failure the
            # header was sent to prevent.
            raise HTTPException(
                status_code=503,
                detail=(
                    "idempotency_store is not configured; retry without "
                    f"{IDEMPOTENCY_HEADER} or configure the store"
                ),
            )
        self._key = raw

    def replay(self) -> JSONResponse | None:
        """Claim the key, or return the stored response for a replay."""
        if self._key is None or self._store is None:
            return None
        try:
            stored = self._store.begin(
                actor_id=self._identity.actor_id,
                key=self._key,
                operation=self._operation,
                body=self._body,
            )
        except IdempotencyInProgress as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except IdempotencyKeyInvalid as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if stored is None:
            self._claimed = True
            return None
        return _as_response(stored)

    def complete(self, response: Any, *, status_code: int = 200) -> None:
        """Record the finished response. A no-op for an inert guard."""
        if not self._claimed or self._key is None or self._store is None:
            return
        self._store.complete(
            actor_id=self._identity.actor_id,
            key=self._key,
            status_code=status_code,
            response=jsonable_encoder(response),
        )

    def abandon(self) -> None:
        """Release the claim after a failure, so the operation can be retried.

        Only a completed 2xx consumes a key. A 4xx/5xx leaves the key free: the
        operator fixes the cause and retries the *same* action, which is the
        whole point of having sent a key.
        """
        if not self._claimed or self._key is None or self._store is None:
            return
        self._store.abandon(actor_id=self._identity.actor_id, key=self._key)

def _as_response(stored: ReplayResponse) -> JSONResponse:
    return JSONResponse(
        status_code=stored.status_code,
        content=stored.body,
        headers={REPLAY_HEADER: "true"},
    )
