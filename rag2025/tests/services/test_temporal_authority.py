# @spec(S13) test_temporal_authority — verify year authority precedence
"""Test get_current_admission_year() precedence and fail-loud behavior."""
import os
import pytest
from datetime import datetime
from unittest.mock import patch

from src.services.temporal_authority import get_current_admission_year, _now


class TestTemporalAuthority:
    """Test current admission year resolution."""

    def test_env_override_wins(self, monkeypatch):
        """CURRENT_ADMISSION_YEAR env is authoritative."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2030")
        monkeypatch.delenv("APP_ENV", raising=False)
        assert get_current_admission_year() == 2030

    def test_cycle_fallback_month_9(self, monkeypatch):
        """Dev/test fallback: month=9 (Sep) → clock_year+1."""
        monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)
        
        # Mock September 2026
        mock_time = datetime(2026, 9, 15)
        with patch("src.services.temporal_authority._now", return_value=mock_time):
            assert get_current_admission_year() == 2027

    def test_cycle_fallback_month_3(self, monkeypatch):
        """Dev/test fallback: month=3 (Mar) → clock_year."""
        monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)
        
        # Mock March 2026
        mock_time = datetime(2026, 3, 15)
        with patch("src.services.temporal_authority._now", return_value=mock_time):
            assert get_current_admission_year() == 2026

    def test_prod_env_unset_raises(self, monkeypatch):
        """APP_ENV=prod + CURRENT_ADMISSION_YEAR unset → RuntimeError."""
        monkeypatch.setenv("APP_ENV", "prod")
        monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
        
        with pytest.raises(RuntimeError, match="required in production"):
            get_current_admission_year()

    def test_prod_env_invalid_raises(self, monkeypatch):
        """APP_ENV=prod + CURRENT_ADMISSION_YEAR invalid → RuntimeError."""
        monkeypatch.setenv("APP_ENV", "prod")
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "invalid")
        
        with pytest.raises(RuntimeError, match="invalid"):
            get_current_admission_year()

    def test_retriever_does_not_change_authority(self, monkeypatch):
        """Retriever arg is EVIDENCE-ONLY — proven ignored, not just out-voted.

        Uses a MagicMock retriever whose max_data_year (2025) differs from the
        env year (2028). Asserts (a) the returned year is the env year, AND
        (b) the retriever's data accessor was NEVER called. This test FAILS if
        anyone later wires the retriever into the year decision.
        """
        from unittest.mock import MagicMock

        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2028")
        monkeypatch.delenv("APP_ENV", raising=False)

        mock_retriever = MagicMock()
        mock_retriever.get_max_data_year.return_value = 2025  # differs from env

        assert get_current_admission_year(retriever=mock_retriever) == 2028
        # The whole point: the retriever must NOT be consulted for authority.
        mock_retriever.get_max_data_year.assert_not_called()
        assert get_current_admission_year(retriever=None) == 2028

    def test_env_out_of_range_dev_fallback(self, monkeypatch):
        """Dev mode: invalid env year → falls back to cycle calendar."""
        monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "1999")  # out of range
        monkeypatch.delenv("APP_ENV", raising=False)
        
        # Should fall back to cycle calendar
        mock_time = datetime(2026, 10, 1)
        with patch("src.services.temporal_authority._now", return_value=mock_time):
            assert get_current_admission_year() == 2027

    def test_cycle_boundary_august(self, monkeypatch):
        """Cycle boundary: August (month=8) → clock_year+1."""
        monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)
        
        # Mock August 1, 2026 (exactly at boundary)
        mock_time = datetime(2026, 8, 1)
        with patch("src.services.temporal_authority._now", return_value=mock_time):
            assert get_current_admission_year() == 2027

    def test_cycle_boundary_july(self, monkeypatch):
        """Cycle boundary: July (month=7) → clock_year."""
        monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)
        
        # Mock July 31, 2026 (just before boundary)
        mock_time = datetime(2026, 7, 31)
        with patch("src.services.temporal_authority._now", return_value=mock_time):
            assert get_current_admission_year() == 2026
