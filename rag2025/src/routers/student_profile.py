"""§PL-7.2 Frame 12 — anonymous student profile flow.

``GET /v2/student-profile``, ``PUT /v2/student-profile``,
``DELETE /v2/student-profile`` and ``GET /v2/admission-roadmap``.

Design constraints (non-negotiable, per spec):
  - ``X-Student-Session`` MUST be a UUID v4, client-generated. Missing or
    invalid → 400 ``STUDENT_SESSION_REQUIRED`` with the PL-7 error envelope
    and an ``X-Request-ID`` header. The session id is never logged together
    with query text.
  - ``raw_score`` ∈ [0, 30] (Pydantic + DB CHECK), ``target_majors`` ≤ 10
    items, every stored text field run through :func:`redact_pii`.
  - ``PUT``/``DELETE`` rate-limited per client IP: 30 req/min; 31st → 429
    ``RATE_LIMITED`` with ``Retry-After`` header and ``retry_after_ms`` body.
  - Expired rows (``expires_at < now()``) ignored on ``GET`` (404) and
    purged. :func:`purge_expired_profiles` is the daily purge helper.
  - All errors use the PL-7 envelope plus ``X-Request-ID``.
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.db import make_session_factory
from services.admission_roadmap import build_admission_roadmap
from services.log_redaction import redact_pii
from services.priority_score import compute_priority
from src.config.constants import (
    STUDENT_PROFILE_RATE_LIMIT_PER_MINUTE,
    STUDENT_PROFILE_RATE_LIMIT_WINDOW_S,
    TRUSTED_PROXY_COUNT,
)
from src.utils.http import client_ip

router = APIRouter(tags=["student-profile"])

_STUDENT_SESSION_HEADER = "X-Student-Session"
_RATE_LIMIT_PER_MINUTE = STUDENT_PROFILE_RATE_LIMIT_PER_MINUTE
_RATE_LIMIT_WINDOW_S = STUDENT_PROFILE_RATE_LIMIT_WINDOW_S
_MAX_TARGET_MAJORS = 10


def _decode_json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return loaded
        except (json.JSONDecodeError, TypeError):
            pass
        return [value]
    return [value]


def _is_uuid_v4(value: str | None) -> bool:
    if not value:
        return False
    try:
        return uuid.UUID(value).version == 4
    except (ValueError, AttributeError, TypeError):
        return False


def _session_required(request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    body = {
        "code": "STUDENT_SESSION_REQUIRED",
        "message": "Thiếu hoặc sai mã phiên học viên (X-Student-Session phải là UUID v4).",
        "field_errors": [
            {
                "field": "header.X-Student-Session",
                "code": "invalid",
                "message": "Phải là UUID v4 do client sinh.",
            }
        ],
        "request_id": request_id,
    }
    return JSONResponse(status_code=400, content=body, headers={"X-Request-ID": request_id})


from services.rate_limit import FixedWindowLimiter  # noqa: E402


def _client_ip(request: Request) -> str:
    return client_ip(request, TRUSTED_PROXY_COUNT)


_student_profile_limiter = FixedWindowLimiter(
    limit=_RATE_LIMIT_PER_MINUTE,
    key_fn=_client_ip,
    window_s=_RATE_LIMIT_WINDOW_S,
)


def _rate_limited_response(request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    body = {
        "code": "RATE_LIMITED",
        "message": "Quá nhiều yêu cầu cập nhật hồ sơ. Vui lòng đợi rồi thử lại.",
        "field_errors": [],
        "request_id": request_id,
        "retry_after_ms": _RATE_LIMIT_WINDOW_S * 1000,
    }
    return JSONResponse(
        status_code=429,
        content=body,
        headers={"X-Request-ID": request_id, "Retry-After": str(_RATE_LIMIT_WINDOW_S)},
    )


def _enforce_student_rate_limit(request: Request) -> JSONResponse | None:
    try:
        _student_profile_limiter.enforce(_client_ip(request))
    except Exception:
        return _rate_limited_response(request)
    return None


class StudentProfileUpsert(BaseModel):
    raw_score: Decimal | None = Field(None, description="Điểm thi tốt nghiệp thang 30 (0–30)")
    method: str | None = Field(None, description="Phương thức xét tuyển")
    to_hop: str | None = Field(None, description="Tổ hợp môn")
    uu_tien_kv: str | None = Field(None, description="Ưu tiên khu vực")
    uu_tien_dt: str | None = Field(None, description="Ưu tiên đối tượng")
    target_majors: list[str] = Field(default=[], description="Danh sách mã ngành mục tiêu")
    admission_year: int | None = Field(None, description="Năm tuyển sinh")

    @field_validator("raw_score")
    @classmethod
    def _validate_raw_score(cls, v):
        if v is None:
            return None
        try:
            score = Decimal(str(v))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("raw_score phải là số") from exc
        if score < 0 or score > 30:
            raise ValueError("raw_score phải nằm trong khoảng 0–30")
        return score

    @field_validator("target_majors")
    @classmethod
    def _validate_target_majors(cls, v):
        if len(v) > _MAX_TARGET_MAJORS:
            raise ValueError(f"target_majors tối đa {_MAX_TARGET_MAJORS} phần tử")
        return [str(m) for m in v]

    @field_validator("method", "to_hop", "uu_tien_kv", "uu_tien_dt")
    @classmethod
    def _redact_text(cls, v):
        if v is None:
            return None
        return redact_pii(v)


class StudentProfileOut(BaseModel):
    student_session_id: str
    raw_score: float | None
    method: str | None
    to_hop: str | None
    uu_tien_kv: str | None
    uu_tien_dt: str | None
    target_majors: list[str]
    admission_year: int | None
    created_at: str | None
    updated_at: str | None
    expires_at: str | None


@asynccontextmanager
async def _session(request: Request):
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        from infrastructure.db import create_db_engine

        engine = create_db_engine()
    async with make_session_factory(engine)() as sess:
        yield sess


@router.get("/v2/student-profile", response_model=StudentProfileOut)
async def get_student_profile(request: Request):
    if not _is_uuid_v4(request.headers.get(_STUDENT_SESSION_HEADER)):
        return _session_required(request)

    session_id = request.headers[_STUDENT_SESSION_HEADER]
    async with _session(request) as sess:
        await sess.execute(
            text("DELETE FROM student_profiles WHERE expires_at < now()")
        )
        await sess.commit()
        row = await sess.execute(
            text(
                """
                SELECT student_session_id, raw_score, method, to_hop,
                       uu_tien_kv, uu_tien_dt, target_majors, admission_year,
                       created_at, updated_at, expires_at
                FROM student_profiles
                WHERE student_session_id = :sid
                  AND expires_at >= now()
                """
            ),
            {"sid": session_id},
        )
        rec = row.mappings().first()
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "NOT_FOUND",
                    "message": "Không tìm thấy hồ sơ học viên (hoặc hồ sơ đã hết hạn).",
                },
            )
        return StudentProfileOut(
            student_session_id=str(rec["student_session_id"]),
            raw_score=(
                float(rec["raw_score"]) if rec["raw_score"] is not None else None
            ),
            method=rec["method"],
            to_hop=rec["to_hop"],
            uu_tien_kv=rec["uu_tien_kv"],
            uu_tien_dt=rec["uu_tien_dt"],
            target_majors=_decode_json_list(rec["target_majors"]),
            admission_year=rec["admission_year"],
            created_at=rec["created_at"].isoformat() if rec["created_at"] else None,
            updated_at=rec["updated_at"].isoformat() if rec["updated_at"] else None,
            expires_at=rec["expires_at"].isoformat() if rec["expires_at"] else None,
        )


@router.put("/v2/student-profile", response_model=StudentProfileOut)
async def put_student_profile(request: Request, payload: StudentProfileUpsert):
    if not _is_uuid_v4(request.headers.get(_STUDENT_SESSION_HEADER)):
        return _session_required(request)

    limited = _enforce_student_rate_limit(request)
    if limited is not None:
        return limited

    session_id = request.headers[_STUDENT_SESSION_HEADER]
    data = payload.model_dump(exclude_none=True)
    if "raw_score" in data:
        data["raw_score"] = str(data["raw_score"])
    if "target_majors" in data and data["target_majors"] is not None:
        data["target_majors"] = json.dumps(data["target_majors"])

    async with _session(request) as sess:
        await sess.execute(
            text(
                """
                INSERT INTO student_profiles
                    (student_session_id, raw_score, method, to_hop,
                     uu_tien_kv, uu_tien_dt, target_majors, admission_year)
                VALUES (:sid, :raw_score, :method, :to_hop,
                        :uu_tien_kv, :uu_tien_dt, :target_majors, :admission_year)
                ON CONFLICT (student_session_id) DO UPDATE SET
                    raw_score = EXCLUDED.raw_score,
                    method = EXCLUDED.method,
                    to_hop = EXCLUDED.to_hop,
                    uu_tien_kv = EXCLUDED.uu_tien_kv,
                    uu_tien_dt = EXCLUDED.uu_tien_dt,
                    target_majors = EXCLUDED.target_majors,
                    admission_year = EXCLUDED.admission_year,
                    updated_at = now()
                """
            ),
            {
                "sid": session_id,
                "raw_score": data.get("raw_score"),
                "method": data.get("method"),
                "to_hop": data.get("to_hop"),
                "uu_tien_kv": data.get("uu_tien_kv"),
                "uu_tien_dt": data.get("uu_tien_dt"),
                "target_majors": data.get("target_majors", []),
                "admission_year": data.get("admission_year"),
            },
        )
        await sess.commit()
    return await get_student_profile(request)


@router.delete("/v2/student-profile", status_code=status.HTTP_204_NO_CONTENT)
async def delete_student_profile(request: Request):
    if not _is_uuid_v4(request.headers.get(_STUDENT_SESSION_HEADER)):
        return _session_required(request)

    limited = _enforce_student_rate_limit(request)
    if limited is not None:
        return limited

    session_id = request.headers[_STUDENT_SESSION_HEADER]
    async with _session(request) as sess:
        await sess.execute(
            text("DELETE FROM student_profiles WHERE student_session_id = :sid"),
            {"sid": session_id},
        )
        await sess.commit()
    return None


@router.get("/v2/admission-roadmap")
async def get_admission_roadmap(request: Request):
    if not _is_uuid_v4(request.headers.get(_STUDENT_SESSION_HEADER)):
        return _session_required(request)

    session_id = request.headers[_STUDENT_SESSION_HEADER]
    async with _session(request) as sess:
        await sess.execute(
            text("DELETE FROM student_profiles WHERE expires_at < now()")
        )
        await sess.commit()
        row = await sess.execute(
            text(
                """
                SELECT raw_score, method, to_hop, target_majors
                FROM student_profiles
                WHERE student_session_id = :sid
                  AND expires_at >= now()
                """
            ),
            {"sid": session_id},
        )
        rec = row.mappings().first()
        profile = (
            {
                "raw_score": rec["raw_score"],
                "method": rec["method"],
                "to_hop": rec["to_hop"],
                "target_majors": _decode_json_list(rec["target_majors"]),
            }
            if rec is not None
            else {}
        )

        thoi_gian = None
        rel = await sess.execute(
            text("SELECT release_id FROM releases WHERE status = 'active' LIMIT 1")
        )
        release_id = rel.scalar_one_or_none()
        if release_id:
            snap = await sess.execute(
                text(
                    "SELECT data_year FROM recommender_snapshots WHERE release_id = :rid"
                ),
                {"rid": release_id},
            )
            snap_row = snap.mappings().first()
            if snap_row:
                thoi_gian = {
                    "document_id": f"recommender_snapshots:{release_id}",
                    "page": snap_row["data_year"],
                }

    steps = build_admission_roadmap(profile, thoi_gian_source=thoi_gian)
    return {
        "steps": [
            {
                "n": s.n,
                "key": s.key,
                "status": s.status,
                "needs_input": s.needs_input,
                "source": s.source,
            }
            for s in steps
        ]
    }


async def purge_expired_profiles(request: Request | None = None) -> int:
    from infrastructure.db import create_db_engine

    engine = create_db_engine()
    async with make_session_factory(engine)() as sess:
        result = await sess.execute(
            text("DELETE FROM student_profiles WHERE expires_at < now()")
        )
        await sess.commit()
        return result.rowcount or 0
