# @spec(S14.1) test_season — phase truth table
"""Test get_season_phase() boundary decisions.

Coverage:
1. has_data=True, any month -> IN_SEASON (HARD switch).
2. has_data=False, month in [NEW_SEASON_MONTH, 12] -> PRE_SEASON_GAP.
3. has_data=False, month in [1, PUBLISH_MONTH - 1] -> PRE_SEASON_GAP.
4. has_data=False, month in [PUBLISH_MONTH, NEW_SEASON_MONTH - 1] -> OFF_SEASON.
5. Rollover narrative: Dec 2026 (no 2027 data) -> PRE_SEASON_GAP;
   Apr 2027 (with 2027 data) -> IN_SEASON.
6. Env override: patching module constants re-evaluates boundaries.

Clock is controlled via monkeypatch on `season._now` (mirrors the
`temporal_authority._now` testable-clock convention).
"""
import pytest
from datetime import datetime
from unittest.mock import patch

from src.services import season
from src.services.season import (
    SeasonPhase,
    get_season_phase,
    RESULTS_MONTH,
    NEW_SEASON_MONTH,
    PUBLISH_MONTH,
)


def _at(month: int, year: int = 2026) -> datetime:
    """Build a deterministic clock for a given month/year (15th, noon)."""
    return datetime(year, month, 15, 12, 0, 0)


class TestHardSwitch:
    """has_current_year_data=True is the unconditional IN_SEASON trigger."""

    @pytest.mark.parametrize("month", list(range(1, 13)))
    def test_has_data_true_any_month_is_in_season(self, month):
        # Toạn bộ 12 tháng đều phải là IN_SEASON nếu đã có dữ liệu năm hiện tại.
        for year in (2025, 2026, 2027, 2030):
            assert get_season_phase(
                current_year=year,
                has_current_year_data=True,
                now=_at(month, year),
            ) == SeasonPhase.IN_SEASON, (
                f"year={year} month={month} should be IN_SEASON when data present"
            )


class TestNoDataGap:
    """has_current_year_data=False + month in [NEW_SEASON_MONTH..12] or [1..PUBLISH_MONTH-1]
    -> PRE_SEASON_GAP."""

    def test_no_data_october_is_gap(self):
        # Tháng 10 (>= NEW_SEASON_MONTH=9): chu kỳ mới đã bắt đầu, dữ liệu năm mới chưa về.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(10, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP

    def test_no_data_december_is_gap(self):
        # Tháng 12: vẫn đang chờ dữ liệu năm mới.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(12, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP

    def test_no_data_february_is_gap(self):
        # Tháng 2 (< PUBLISH_MONTH=3): dữ liệu năm hiện tại vẫn chưa công bố.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(2, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP

    def test_no_data_january_is_gap(self):
        # Tháng 1 (< PUBLISH_MONTH=3): biên giới đầu năm vẫn là gap.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(1, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP

    @pytest.mark.parametrize("month", [9, 10, 11, 12])
    def test_no_data_new_season_window_is_gap(self, month):
        # Toạn bộ cửa sổ [NEW_SEASON_MONTH=9 .. 12] đều phải là PRE_SEASON_GAP.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(month, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP

    @pytest.mark.parametrize("month", [1, 2])
    def test_no_data_publish_window_is_gap(self, month):
        # Toạn bộ cửa sổ [1 .. PUBLISH_MONTH-1=2] đều phải là PRE_SEASON_GAP.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(month, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP


class TestNoDataOffSeason:
    """has_current_year_data=False + month in [PUBLISH_MONTH..NEW_SEASON_MONTH-1]
    -> OFF_SEASON (trầm lặng, không có dữ liệu mới nhưng cũng chưa vào gap)."""

    def test_no_data_june_is_off_season(self):
        # Tháng 6: sau công bố (3), trước mùa mới (9) — không có dữ liệu mới.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(6, 2026),
        ) == SeasonPhase.OFF_SEASON

    @pytest.mark.parametrize("month", [3, 4, 5, 6, 7, 8])
    def test_no_data_mid_year_window_is_off_season(self, month):
        # Toạn bộ cửa sổ [PUBLISH_MONTH=3 .. NEW_SEASON_MONTH-1=8] đều là OFF_SEASON.
        assert get_season_phase(
            current_year=2026,
            has_current_year_data=False,
            now=_at(month, 2026),
        ) == SeasonPhase.OFF_SEASON


class TestRolloverNarrative:
    """10-year contract: cùng một hệ thống, khác mốc thời gian, hành vi khác."""

    def test_dec_2026_no_2027_data_is_gap(self):
        # Cuối năm 2026, dữ liệu 2027 chưa về → phải rơi vào gap.
        assert get_season_phase(
            current_year=2027,
            has_current_year_data=False,
            now=_at(12, 2026),
        ) == SeasonPhase.PRE_SEASON_GAP

    def test_apr_2027_with_2027_data_is_in_season(self):
        # Đầu năm 2027, dữ liệu 2027 đã có → IN_SEASON.
        assert get_season_phase(
            current_year=2027,
            has_current_year_data=True,
            now=_at(4, 2027),
        ) == SeasonPhase.IN_SEASON

    def test_apr_2027_without_2027_data_is_gap(self):
        # Nếu dữ liệu 2027 vẫn chưa xuất hiện vào tháng 4 (bất thường nhưng hợp lệ)
        # → vẫn là gap vì tháng 4 < PUBLISH_MONTH không — 4 >= 3, nên thực ra là OFF_SEASON.
        # Trường hợp này kiểm tra rằng bộ quy tắc xử lý đúng: tháng 4 nằm giữa publish
        # và new_season, nên rơi vào OFF_SEASON khi không có dữ liệu.
        assert get_season_phase(
            current_year=2027,
            has_current_year_data=False,
            now=_at(4, 2027),
        ) == SeasonPhase.OFF_SEASON


class TestEnvOverride:
    """Mô-đun đọc hằng số ở cấp module — test có thể vá trực tiếp."""

    def test_patching_new_season_month_shifts_gap_window(self, monkeypatch):
        # Nếu nhà trường lùi lịch new_season xuống tháng 11,
        # thì tháng 10 (trước đó là gap) sẽ trở thành off_season.
        monkeypatch.setattr(season, "NEW_SEASON_MONTH", 11)
        try:
            assert get_season_phase(
                current_year=2026,
                has_current_year_data=False,
                now=_at(10, 2026),
            ) == SeasonPhase.OFF_SEASON
            # ...và tháng 11 (giờ mới là gap) phải rơi vào PRE_SEASON_GAP.
            assert get_season_phase(
                current_year=2026,
                has_current_year_data=False,
                now=_at(11, 2026),
            ) == SeasonPhase.PRE_SEASON_GAP
        finally:
            monkeypatch.undo()

    def test_patching_publish_month_shifts_off_season_window(self, monkeypatch):
        # Nếu publish lùi xuống tháng 5, thì tháng 4 (trước đó off_season) trở thành gap.
        monkeypatch.setattr(season, "PUBLISH_MONTH", 5)
        try:
            assert get_season_phase(
                current_year=2026,
                has_current_year_data=False,
                now=_at(4, 2026),
            ) == SeasonPhase.PRE_SEASON_GAP
        finally:
            monkeypatch.undo()


class TestClockControl:
    """Khi caller không truyền `now`, hàm phải đọc từ `_now()`."""

    def test_default_now_reads_season_now(self):
        # Tiêm _now trả về tháng 10/2026, kỳ vọng PRE_SEASON_GAP.
        with patch("src.services.season._now", return_value=_at(10, 2026)):
            assert get_season_phase(
                current_year=2026,
                has_current_year_data=False,
            ) == SeasonPhase.PRE_SEASON_GAP

    def test_explicit_now_overrides_clock(self):
        # Caller truyền now tháng 6 (sẽ là OFF_SEASON) nhưng _now trả tháng 10
        # (sẽ là PRE_SEASON_GAP) — giá trị truyền vào phải thắng.
        with patch("src.services.season._now", return_value=_at(10, 2026)):
            assert get_season_phase(
                current_year=2026,
                has_current_year_data=False,
                now=_at(6, 2026),
            ) == SeasonPhase.OFF_SEASON


class TestPureModule:
    """Không phụ thuộc I/O — đảm bảo tính thuần khiết."""

    def test_no_temporal_authority_import(self):
        # Module phải đứng độc lập để test được mà không cần stub nặng.
        import src.services.season as s
        src_attrs = set(dir(s))
        assert "temporal_authority" not in src_attrs, (
            "season.py phải không import temporal_authority (giữ module thuần)."
        )
        assert "requests" not in src_attrs
        assert "lancedb" not in src_attrs

    def test_season_phase_is_str_enum(self):
        # Giá trị enum phải là chuỗi ổn định để lưu trữ/log mà không cần import lại.
        assert SeasonPhase.PRE_SEASON_GAP.value == "pre_season_gap"
        assert SeasonPhase.IN_SEASON.value == "in_season"
        assert SeasonPhase.OFF_SEASON.value == "off_season"
        # Enum cũng là str → an toàn khi serialize.
        assert isinstance(SeasonPhase.IN_SEASON, str)
