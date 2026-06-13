# @spec(PHASE-A4) deterministic /v2/recommend + /v2/whatif endpoints
"""Pure-deterministic recommender endpoints (no LLM, no network).

Mounts under the SAME /v2 prefix as the existing ``/v2/query`` router so
the FE can reach it as ``POST /v2/recommend`` and ``POST /v2/whatif``.

HARD CONSTRAINTS (per spec, non-negotiable):
  - NO LLM calls on the hot path.
  - NO new httpx timeouts.
  - Single-digit ms response time.
  - Missing/invalid score → 422 or a clean error payload (NEVER 500).
  - Unknown major_code → graceful BLOCKED-style payload, NEVER 500.

These endpoints live in their OWN file/router so a future edit to the
existing ``/v2`` query path cannot accidentally drag this slice into the
4-layer guardrail or LLM call site.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

# Local import — uses the mtime-cached pure function from A2/A3.
from services.major_recommender import (
    Recommendation,
    WhatIfResult,
    recommend,
    whatif_probability,
)


router = APIRouter(prefix="/v2", tags=["recommender"])


# ---- Pydantic request/response schemas (no LLM round-trip needed) -----------

class RecommendRequest(BaseModel):
    score: float = Field(..., ge=0, le=30, description="Tổng điểm thang 30")
    to_hop: Optional[str] = Field(
        default=None,
        max_length=8,
        description="Tổ hợp môn xét tuyển, vd 'A00'. Optional.",
    )
    uu_tien: float = Field(
        default=0.0,
        ge=0,
        le=5.0,
        description="Điểm ưu tiên (khu vực / đối tượng / chứng chỉ). Mặc định 0.",
    )

    @field_validator("to_hop")
    @classmethod
    def _trim_to_hop(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        return v or None


class RecommendationOut(BaseModel):
    major_code: str
    major_name: str
    latest_diem_chuan: float
    latest_year: int
    delta: float
    label: str
    explanation: str
    to_hop: Optional[List[str]] = None


class RecommendResponse(BaseModel):
    score: float
    uu_tien: float
    to_hop: Optional[str] = None
    count: int
    recommendations: List[RecommendationOut]
    basis: str
    disclaimer: str


class WhatIfRequest(BaseModel):
    score: float = Field(..., ge=0, le=30, description="Tổng điểm thang 30")
    major_code: str = Field(..., min_length=1, max_length=20)
    to_hop: Optional[str] = Field(default=None, max_length=8)
    uu_tien: float = Field(default=0.0, ge=0, le=5.0)

    @field_validator("major_code")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("to_hop")
    @classmethod
    def _trim(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        return v or None


class WhatIfResponse(BaseModel):
    score: float
    major_code: str
    p_pass: str
    band: str
    basis: str
    disclaimer: str
    latest_diem_chuan: Optional[float] = None
    latest_year: Optional[int] = None
    delta: Optional[float] = None
    n_years: int = 0


# ---- Shared strings (single source of truth) -------------------------------

_BASIS_RECOMMEND = (
    "Dựa trên điểm chuẩn 1 năm (2025) phương thức THPT, chỉ mang tính tham khảo."
)
_DISCLAIMER = (
    "Đây là ước lượng heuristic, KHÔNG phải mô hình ML chính xác; "
    "kết quả thực tế còn phụ thuộc chỉ tiêu, số lượng nguyện vọng và phương thức xét tuyển."
)


def _recommendation_to_out(r: Recommendation) -> RecommendationOut:
    return RecommendationOut(
        major_code=r.major_code,
        major_name=r.major_name,
        latest_diem_chuan=r.latest_diem_chuan,
        latest_year=r.latest_year,
        delta=r.delta,
        label=r.label,
        explanation=r.explanation,
        to_hop=r.to_hop,
    )


def _validate_score(score: Any) -> Optional[str]:
    """Return an error message string if invalid; None if valid."""
    if isinstance(score, bool):
        return "score must be a number, not a bool"
    if not isinstance(score, (int, float)):
        return "score must be a number"
    if math.isnan(score) or math.isinf(score):
        return "score must be a finite number"
    if score < 0 or score > 30:
        return "score must be in [0, 30]"
    return None


# ---- Endpoints -------------------------------------------------------------

@router.post("/recommend", response_model=RecommendResponse)
async def post_recommend(req: RecommendRequest) -> RecommendResponse:
    """Rank HUSC majors by a candidate's (score + ưu_tiên) vs điểm chuẩn.

    Returns a deterministic, pure-arithmetic ranking — no LLM call, no
    network, no DB. Response time is single-digit ms.
    """
    err = _validate_score(req.score)
    if err is not None:
        # 422 path is handled by Pydantic; this is a defensive double-check.
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=err)
    if req.uu_tien is not None and (math.isnan(req.uu_tien) or math.isinf(req.uu_tien)):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="uu_tien must be finite")
    recs = recommend(score=req.score, to_hop=req.to_hop, uu_tien=req.uu_tien)
    return RecommendResponse(
        score=req.score,
        uu_tien=req.uu_tien,
        to_hop=req.to_hop,
        count=len(recs),
        recommendations=[_recommendation_to_out(r) for r in recs],
        basis=_BASIS_RECOMMEND,
        disclaimer=_DISCLAIMER,
    )


@router.post("/whatif", response_model=WhatIfResponse)
async def post_whatif(req: WhatIfRequest) -> WhatIfResponse:
    """Return a band-based probability estimate for one specific major.

    Unknown major_code → graceful BLOCKED-style payload (band=unknown_major,
    basis/disclaimer surfaced, NEVER 500).
    """
    err = _validate_score(req.score)
    if err is not None:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=err)
    res: WhatIfResult = whatif_probability(
        score=req.score,
        major_code=req.major_code,
        to_hop=req.to_hop,
        uu_tien=req.uu_tien,
    )
    return WhatIfResponse(
        score=req.score,
        major_code=req.major_code,
        p_pass=res.p_pass,
        band=res.band,
        basis=res.basis,
        disclaimer=res.disclaimer,
        latest_diem_chuan=res.latest_diem_chuan,
        latest_year=res.latest_year,
        delta=res.delta,
        n_years=res.n_years,
    )
