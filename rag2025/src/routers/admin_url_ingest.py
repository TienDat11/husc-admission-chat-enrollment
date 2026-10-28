"""URL → draft record endpoint — ticket T21.

One route, mounted into the admin surface beside the upload it shares its
``/source-documents`` prefix with. It is the HTTP seam only: the SSRF guard, the
fetch, the SHA-256 comparison and the record write all live in
:mod:`src.services.url_ingest_service`, exactly as ticket T04's
``admin_document_upload`` keeps its work in ``source_document_service``.

Status codes:

- ``201`` when a draft was written; ``200`` when the page was unchanged
  (nothing written) or when an existing draft gained a version. The body always
  says which of the three happened — ``status: created | updated | unchanged`` —
  so "bỏ qua" is a value the caller reads, not an absence it has to infer.
- ``400`` for a URL this route will not fetch (non-http(s), or a host that is
  not publicly routable) and for a page past the byte cap — the same 400 the
  upload route answers for a file past its cap;
- ``422`` for a page with no readable text: the request was fine, the content
  cannot become a record;
- ``502`` when the upstream refused or failed after the retry budget;
- ``409`` when the record moved under us between read and write;
- ``403``/``422`` for a caller without the ``records.create`` grant or with no
  credential at all, exactly as the upload route answers.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.domain.admin_identity import AdminAction, Identity
from src.routers.admin_auth import current_identity, require_action
from src.routers.admin_rate_limit import upload_rate_limit
from src.services.admin_lifecycle import VersionConflictError
from src.services.url_ingest_service import (
    UrlIngestEmpty,
    UrlIngestResult,
    UrlIngestService,
)
from src.services.web_fetch import UnsafeUrlError, WebFetchError, WebFetchTooLarge

router = APIRouter(prefix="/source-documents", tags=["admin-url-ingest"])


class UrlIngestRequest(BaseModel):
    """Body for POST /source-documents/from-url."""

    url: str = Field(
        min_length=1,
        description="the notification page to fetch; http(s) and publicly routable only",
    )


class UrlIngestReport(BaseModel):
    """What the ingest did, and what it produced.

    ``status`` is the idempotency answer: ``unchanged`` means the page's bytes
    matched the stored digest and *no record was written* — ``version`` is then
    the version already on disk.
    """

    status: Literal["created", "updated", "unchanged"]
    record_id: str
    source_url: str
    sha256: str
    version: int
    chars: int


def _view(result: UrlIngestResult) -> UrlIngestReport:
    return UrlIngestReport(
        status=result.status,
        record_id=result.record_id,
        source_url=result.source_url,
        sha256=result.sha256,
        version=result.version,
        chars=result.chars,
    )


def _url_ingest_service(request: Request) -> UrlIngestService:
    """The app's URL-ingest service, or one over the parent's lifecycle service.

    ``app.state.url_ingest_service`` wins (that is the test seam, and where a
    deployment that wants a different fetcher would put one). Otherwise the
    parent admin router's lifecycle accessor is used, so this surface shares
    ticket 12's 503-on-unwired semantics rather than inventing its own.

    The import is deferred: this module is imported *by* ``admin.py`` at the
    bottom (which mounts it), so a top-level import back into ``admin.py`` would
    depend on which name happened to exist when the cycle closed. Deferred, it
    runs on the first request, when ``admin.py`` is fully built.
    """
    service = getattr(request.app.state, "url_ingest_service", None)
    if isinstance(service, UrlIngestService):
        return service
    from src.routers.admin import _lifecycle_service

    return UrlIngestService(_lifecycle_service(request))


@router.post(
    "/from-url",
    response_model=UrlIngestReport,
    status_code=200,
    dependencies=[
        Depends(require_action(AdminAction.RECORDS_CREATE)),
        Depends(upload_rate_limit),
    ],
)
async def ingest_from_url(
    request: Request,
    payload: UrlIngestRequest,
    response: Response,
    identity: Annotated[Identity, Depends(current_identity)],
) -> UrlIngestReport:
    """Fetch a pasted notification URL and draft / refresh its record.

    - the same URL with the same bytes again → ``unchanged``, nothing written;
    - changed bytes → a new version of the same record, back at ``draft``;
    - localhost / private / link-local hosts and non-http(s) schemes are 400
      before any socket is opened;
    - the page is streamed under a byte cap, and the fetch is retried with
      backoff on 5xx and transport errors, then 502.
    """
    service = _url_ingest_service(request)
    try:
        result = await asyncio.to_thread(
            service.ingest,
            payload.url,
            actor_id=identity.actor_id,
            role=identity.role.value,
        )
    except (UnsafeUrlError, WebFetchTooLarge) as exc:
        # Both are "the request names something this route will not hold" —
        # the same 400 the upload route gives for a file past its cap.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UrlIngestEmpty as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WebFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VersionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result.status == "created":
        response.status_code = 201
    return _view(result)
