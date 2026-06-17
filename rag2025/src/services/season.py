# @spec(S14.1)
"""Máy trạng thái mùa tuyển sinh thuần (pure season-phase machine).

Quyết định hành vi hệ thống từ hai trục:
  1. `has_current_year_data` (cờ cứng — dữ liệu năm hiện tại đã có trong index chưa).
  2. Cửa sổ tháng dương lịch (mềm — chỉ định hướng cách diễn đạt).

Cổ dữ liệu là công tắc cứng; cửa sổ tháng chỉ cập nhật cách hiển thị.
Điều này giúp hệ thống chuyển giao năm (2026 → 2027 → ...) mà không cần
sửa code — chỉ cần ingest dữ liệu năm mới.

Quy tắc quyết định (theo hợp đồng S14.1):
  - `has_current_year_data == True`            → IN_SEASON
  - `has_current_year_data == False` và
        (tháng < PUBLISH_MONTH  HOẶC  tháng >= NEW_SEASON_MONTH) → PRE_SEASON_GAP
  - ngược lại                                    → OFF_SEASON

Mô-đun này thuần: KHÔNG gọi mạng, KHÔNG truy cập DB, KHÔNG import
`temporal_authority` (giữ phụ thuộc bằng 0 để dễ test). Caller phải tự
truyền `current_year` đã được phân giải qua `temporal_authority`.
"""
from __future__ import annotations
import os
from datetime import datetime
from enum import Enum
from typing import Optional


class SeasonPhase(str, Enum):
    """Ba pha mùa tuyển sinh.

    - `PRE_SEASON_GAP`: lịch đã sang năm N nhưng index chưa có `data_year=N`.
                       Hệ thống phải hiển thị disclaimer + dữ liệu năm cũ gắn nhãn.
    - `IN_SEASON`:     dữ liệu năm hiện tại đã có — phục vụ bình thường, lọc cứng
                       các chunk cũ theo intent hiện tại.
    - `OFF_SEASON`:    dự phòng — không có dữ liệu mới, không nằm trong cửa sổ gap.
    """
    PRE_SEASON_GAP = "pre_season_gap"
    IN_SEASON = "in_season"
    OFF_SEASON = "off_season"


# Cửa sổ tháng dương lịch — đọc env một lần tại import. Defaults khớp với lịch
# tuyển sinh HUSC điển hình: kết quả ~tháng 8, mùa mới bắt đầu ~tháng 9, công bố
# ~tháng 3 năm sau. Cho phép vá runtime qua patch `season.NEW_SEASON_MONTH`
# trong test (xem `tests/services/test_season.py`).
RESULTS_MONTH = int(os.getenv("ADMISSION_RESULTS_MONTH", "8"))
NEW_SEASON_MONTH = int(os.getenv("ADMISSION_NEW_SEASON_MONTH", "9"))
PUBLISH_MONTH = int(os.getenv("ADMISSION_PUBLISH_MONTH", "3"))


def _now() -> datetime:
    """Đồng hồ có thể test — monkey-patch hàm này trong test để khoá thời gian."""
    return datetime.now()


def get_season_phase(
    current_year: int,
    has_current_year_data: bool,
    now: Optional[datetime] = None,
) -> SeasonPhase:
    """Quyết định pha mùa tuyển sinh hiện tại.

    Args:
        current_year: Năm tuyển sinh hiện hành (đã được caller phân giải qua
            `temporal_authority.get_current_admission_year()`). Tham số này
            mang tính thông tin — hàm KHÔNG tự tính lại năm.
        has_current_year_data: True nếu index chứa ít nhất một chunk có
            `data_year == current_year`. Đây là cổ cứng (hard switch).
        now: Thời điểm hiện tại để xét cửa sổ tháng. Nếu None, dùng `_now()`.
            Truyền tham số này để test deterministic.

    Returns:
        `SeasonPhase` ứng với một trong ba pha: IN_SEASON / PRE_SEASON_GAP / OFF_SEASON.
    """
    # --- Cổ cứng: có dữ liệu năm hiện tại → IN_SEASON. ---
    if has_current_year_data:
        return SeasonPhase.IN_SEASON

    # --- Không có dữ liệu: cửa sổ tháng quyết định giữa GAP và OFF_SEASON. ---
    effective_now = now if now is not None else _now()
    month = effective_now.month

    # Trước tháng công bố (1..PUBLISH_MONTH-1) HOẶC đã sang mùa mới
    # (NEW_SEASON_MONTH..12) → vẫn đang chờ dữ liệu năm hiện tại → PRE_SEASON_GAP.
    if month < PUBLISH_MONTH or month >= NEW_SEASON_MONTH:
        return SeasonPhase.PRE_SEASON_GAP

    # Cửa sổ giữa công bố và mùa mới (PUBLISH_MONTH..NEW_SEASON_MONTH-1):
    # không có dữ liệu mới nhưng cũng không phải gap → OFF_SEASON.
    return SeasonPhase.OFF_SEASON
