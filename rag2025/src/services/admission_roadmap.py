"""§PL-6.2 Roadmap 5 bước cho luồng thí sinh.

Steps:
  (1) hồ sơ điểm        → needs ``raw_score``
  (2) chọn phương thức  → needs ``method``
  (3) chọn tổ hợp hợp lệ cho ngành mục tiêu → needs ``to_hop``
  (4) so điểm chuẩn năm gần nhất → needs ``target_majors``
  (5) mốc thời gian đăng ký (THOI_GIAN record của release) → no input

First step with missing data is ``needs_input``; every step after it is
``blocked``. Steps up to and including the first complete one are ``done``.

Pure function: takes the in-memory profile + optional release snapshot, never
touches the DB or the clock. Designed so the router test can assert the state
machine without a database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["RoadmapStep", "build_admission_roadmap"]

# (n, key, field required to consider the step complete, human label)
_STEPS: tuple[tuple[int, str, str | None, str], ...] = (
    (1, "hoso_diem", "raw_score", "Hồ sơ điểm"),
    (2, "chon_phuong_thuc", "method", "Chọn phương thức xét tuyển"),
    (3, "chon_to_hop", "to_hop", "Chọn tổ hợp môn cho ngành mục tiêu"),
    (4, "so_diem_chuan", "target_majors", "So sánh điểm chuẩn năm gần nhất"),
    (5, "moc_thoi_gian", None, "Mốc thời gian đăng ký"),
)

_DONE = "done"
_NEEDS_INPUT = "needs_input"
_BLOCKED = "blocked"


@dataclass(frozen=True)
class RoadmapStep:
    n: int
    key: str
    status: str  # "done" | "needs_input" | "blocked"
    needs_input: dict[str, str | None]  # {"field": ..., "hint": ...}
    source: dict[str, Any]  # {"document_id": ..., "page": ...} when available


def _field_hint(field: str | None) -> str | None:
    if field == "raw_score":
        return "Nhập điểm thi tốt nghiệp (thang 30, 0–30)."
    if field == "method":
        return "Chọn phương thức: thi THPT, học bạ, điều chỉnh nguyện vọng hoặc xét tuyển thẳng."
    if field == "to_hop":
        return "Chọn tổ hợp môn phù hợp với ngành bạn đang nhắm tới."
    if field == "target_majors":
        return "Chọn ít nhất một ngành mục tiêu để so sánh điểm chuẩn."
    return None


def build_admission_roadmap(
    profile: dict[str, Any],
    *,
    thoi_gian_source: dict[str, Any] | None = None,
) -> list[RoadmapStep]:
    """Build the 5-step roadmap from a student profile.

    ``profile`` may carry ``raw_score``, ``method``, ``to_hop``,
    ``target_majors``. Missing/None fields make the first such step
    ``needs_input`` and every later step ``blocked``.

    ``thoi_gian_source`` is the release's THOI_GIAN record
    (``{"document_id": ..., "page": ...}``); when absent step 5 still renders
    with an empty source.
    """
    present = {
        "raw_score": profile.get("raw_score") is not None,
        "method": profile.get("method") is not None,
        "to_hop": profile.get("to_hop") is not None,
        "target_majors": bool(profile.get("target_majors")),
    }

    steps: list[RoadmapStep] = []
    hit_gap = False
    for n, key, field, _label in _STEPS:
        if hit_gap:
            status = _BLOCKED
            needs = {"field": None, "hint": None}
        elif field is None:
            # Step 5: no input field; done once every prior step is complete.
            status = _DONE
            needs = {"field": None, "hint": None}
        elif present[field]:
            status = _DONE
            needs = {"field": None, "hint": None}
        else:
            status = _NEEDS_INPUT
            needs = {"field": field, "hint": _field_hint(field)}
            hit_gap = True

        source: dict[str, Any] = {}
        if n == 5 and thoi_gian_source:
            source = {
                "document_id": thoi_gian_source.get("document_id"),
                "page": thoi_gian_source.get("page"),
            }
        steps.append(
            RoadmapStep(
                n=n,
                key=key,
                status=status,
                needs_input=needs,
                source=source,
            )
        )
    return steps
