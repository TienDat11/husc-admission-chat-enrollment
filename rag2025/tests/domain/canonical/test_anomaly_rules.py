"""T19: anomaly detection over structured ``values`` — unit tests.

Covers all three rules:
  a. zero-implausible (tuition/scores of 0)
  b. negative values
  c. sector-outlier (silent without peers; fires with a peer cohort)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.anomaly_rules import (  # noqa: E402
    ANOMALY_RULES_VERSION,
    Anomaly,
    find_anomalies,
)


# ---------------------------------------------------------------------------
# rule a — zero-implausible
# ---------------------------------------------------------------------------


def test_zero_tuition_fires_warning():
    anomalies = find_anomalies(
        "HOC_PHI",
        {"hoc_phi_tin_chi_vnd": 0, "ten_nganh": "CNTT"},
    )
    assert any(a.rule == "zero-implausible" and a.field == "hoc_phi_tin_chi_vnd" for a in anomalies)
    matched = next(a for a in anomalies if a.field == "hoc_phi_tin_chi_vnd")
    assert "0" in matched.message
    assert matched.severity == "warning"


def test_zero_chi_tieu_fires_warning():
    anomalies = find_anomalies("CHI_TIEU", {"chi_tieu": 0})
    assert any(a.rule == "zero-implausible" and a.field == "chi_tieu" for a in anomalies)


def test_zero_diem_trung_tuyen_fires_warning():
    anomalies = find_anomalies("DIEM_CHUAN", {"diem_trung_tuyen_num": 0})
    assert any(a.rule == "zero-implausible" and a.field == "diem_trung_tuyen_num" for a in anomalies)


def test_normal_tuition_does_not_fire():
    anomalies = find_anomalies("HOC_PHI", {"hoc_phi_tin_chi_vnd": 150_000})
    assert not any(a.rule == "zero-implausible" for a in anomalies)


def test_non_zero_numeric_no_warning():
    anomalies = find_anomalies("HOC_PHI", {"hoc_phi_nam_vnd": 5_000_000})
    assert anomalies == ()


# ---------------------------------------------------------------------------
# rule b — negative
# ---------------------------------------------------------------------------


def test_negative_value_fires_warning():
    anomalies = find_anomalies("HOC_PHI", {"hoc_phi_tin_chi_vnd": -1})
    assert any(a.rule == "negative" and a.field == "hoc_phi_tin_chi_vnd" for a in anomalies)


def test_negative_diem_fires_warning():
    anomalies = find_anomalies("DIEM_CHUAN", {"diem_trung_tuyen_num": -5.0})
    assert any(a.rule == "negative" for a in anomalies)


# ---------------------------------------------------------------------------
# rule c — sector-outlier
# ---------------------------------------------------------------------------


def test_sector_outlier_fires_with_peers():
    # peer median = 20_000; record value = 200_000 → ratio 10x → fires
    peers = [{"hoc_phi_tin_chi_vnd": 15_000}, {"hoc_phi_tin_chi_vnd": 25_000}]
    anomalies = find_anomalies(
        "HOC_PHI",
        {"hoc_phi_tin_chi_vnd": 200_000, "khoi_nganh": "A00"},
        peer_values=peers,
    )
    assert any(a.rule == "sector-outlier" and a.field == "hoc_phi_tin_chi_vnd" for a in anomalies)

def test_sector_outlier_silent_without_peers():
    anomalies = find_anomalies(
        "HOC_PHI",
        {"hoc_phi_tin_chi_vnd": 100_000, "khoi_nganh": "A00"},
    )
    assert not any(a.rule == "sector-outlier" for a in anomalies)


def test_sector_outlier_silent_without_khoi_nganh():
    peers = [{"hoc_phi_tin_chi_vnd": 15_000}]
    anomalies = find_anomalies(
        "HOC_PHI",
        {"hoc_phi_tin_chi_vnd": 100_000},  # no khoi_nganh
        peer_values=peers,
    )
    assert not any(a.rule == "sector-outlier" for a in anomalies)


def test_sector_within_bounds_no_warning():
    # peer median = 20; value = 10 → ratio 0.5, within [0.2, 5] → no fire
    peers = [{"hoc_phi_tin_chi_vnd": 10_000}, {"hoc_phi_tin_chi_vnd": 30_000}]
    anomalies = find_anomalies(
        "HOC_PHI",
        {"hoc_phi_tin_chi_vnd": 10_000, "khoi_nganh": "A00"},
        peer_values=peers,
    )
    assert not any(a.rule == "sector-outlier" for a in anomalies)


# ---------------------------------------------------------------------------
# safety — unknown keys / missing keys / unknown record types never raise
# ---------------------------------------------------------------------------


def test_missing_keys_no_error():
    assert find_anomalies("UNKNOWN_TYPE", {}) == ()


def test_unknown_record_type_no_error():
    anomalies = find_anomalies("TOTALLY_UNKNOWN", {"hoc_phi_tin_chi_vnd": 0})
    # zero rule still fires (field-based, not record-type-gated)
    assert any(a.rule == "zero-implausible" for a in anomalies)


def test_string_values_ignored_by_numeric_rules():
    anomalies = find_anomalies("HOC_PHI", {"hoc_phi_tin_chi_vnd": "abc"})
    assert anomalies == ()


def test_boolean_not_treated_as_numeric():
    anomalies = find_anomalies("HOC_PHI", {"hoc_phi_tin_chi_vnd": False})
    assert not any(a.rule == "negative" for a in anomalies)


def test_anomaly_is_frozen_dataclass():
    a = Anomaly(field="x", value=0, message="m", rule="zero-implausible")
    d = a.to_dict()
    assert d["field"] == "x"
    assert d["rule"] == "zero-implausible"
    assert d["severity"] == "warning"


def test_version_is_string():
    assert isinstance(ANOMALY_RULES_VERSION, str)
