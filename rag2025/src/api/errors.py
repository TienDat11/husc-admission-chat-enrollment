"""Error contract (plan PL-7) and request-id middleware for the FastAPI app.

Every 4xx/5xx body is ``{code, message, field_errors, request_id}``. The legacy
``detail`` key is kept so existing clients keep working. ``X-Request-ID`` is
echoed (or generated) on every response.

While ``WRITE_FREEZE`` is active, admin write requests (POST/PUT/PATCH/DELETE under
``/admin/``, except login/logout) are answered 503 ``WRITES_FROZEN`` before routing.
"""

from __future__ import annotations

import re
import uuid
from http import HTTPStatus
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from infrastructure.db import WRITE_FREEZE

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ADMIN_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_ADMIN_WRITE_EXEMPT = {"/admin/login", "/admin/logout"}
_LOCATION_PREFIXES = {"body", "query", "path", "header", "cookie"}
_STATUS_CODES = {
    400: "VALIDATION_FAILED",
    401: "SESSION_EXPIRED",
    403: "FORBIDDEN_ACTION",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_FAILED",
    423: "ACCOUNT_LOCKED",
    429: "RATE_LIMITED",
    503: "UNAVAILABLE",
}


def error_envelope(
    code: str,
    message: str,
    request_id: str,
    field_errors: Optional[list[dict[str, Any]]] = None,
    detail: Any = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "field_errors": field_errors or [],
        "request_id": request_id,
    }
    if detail is not None:
        body["detail"] = detail
    return body


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


def _http_code(status: int, detail: Any) -> str:
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return detail["code"]
    return _STATUS_CODES.get(status, f"HTTP_{status}")


def _http_message(status: int, detail: Any) -> str:
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, str):
        return detail
    return HTTPStatus(status).phrase


def _retry_after_ms(headers: dict[str, str]) -> int:
    raw = headers.get("Retry-After") or headers.get("retry-after")
    try:
        seconds = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        seconds = 0
    return seconds * 1000 if seconds else 2000


def _field_name(loc: tuple) -> str:
    parts = [str(part) for part in loc]
    if parts and parts[0] in _LOCATION_PREFIXES:
        parts = parts[1:]
    return ".".join(parts) or "body"


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = _request_id(request)
    headers = {k: v for k, v in dict(exc.headers or {}).items()}
    body = error_envelope(
        _http_code(exc.status_code, exc.detail),
        _http_message(exc.status_code, exc.detail),
        request_id,
        detail=exc.detail,
    )
    if exc.status_code == 503:
        body["retry_after_ms"] = _retry_after_ms(headers)
    headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id(request)
    errors = jsonable_encoder(exc.errors())
    field_errors = [
        {"field": _field_name(tuple(err.get("loc", ()))), "code": err.get("type", "invalid"), "message": err.get("msg", "")}
        for err in errors
    ]
    body = error_envelope("VALIDATION_FAILED", "Dữ liệu gửi lên không hợp lệ", request_id, field_errors, detail=errors)
    return JSONResponse(status_code=422, content=body, headers={REQUEST_ID_HEADER: request_id})


def _is_frozen_admin_write(request: Request) -> bool:
    return (
        WRITE_FREEZE.active
        and request.method in _ADMIN_WRITE_METHODS
        and request.url.path.startswith("/admin/")
        and request.url.path not in _ADMIN_WRITE_EXEMPT
    )


async def request_context_middleware(request: Request, call_next):
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id

    if _is_frozen_admin_write(request):
        # Only the coarse reason goes out: the revision ids in WRITE_FREEZE.detail
        # are schema internals that an unauthenticated caller must not see.
        detail = {"code": "WRITES_FROZEN", "message": "Ghi dữ liệu đang tạm dừng", "reason": WRITE_FREEZE.reason}
        response = JSONResponse(
            status_code=503,
            content=error_envelope("WRITES_FROZEN", detail["message"], request_id, detail=detail),
            headers={"Retry-After": "60"},
        )
    else:
        response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def install_error_handling(app: FastAPI) -> None:
    """Register the PL-7 exception handlers and the X-Request-ID / write-freeze middleware.

    Middleware added here is inner to CORS, so CORS headers also cover 503 responses.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.middleware("http")(request_context_middleware)
