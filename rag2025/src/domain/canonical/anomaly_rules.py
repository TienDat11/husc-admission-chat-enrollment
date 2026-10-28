"""Value anomaly detection — ticket T19, layer (b+).

Structured ``values`` payloads can carry impossible readings (e.g.
``hoc_phi_tin_chi_vnd = 0``) that no student-facing answer should silently
repeat.  The spec's rule: a simple check on ``values``; anomalies surface as
warnings on the admin review screen — they never block publication.

Like ``high_impact.py`` the rules live in code and are versioned: a warning
stamped with ``ANOMALY_RULES_VERSION`` is reproducible by replaying the same
version of this module, and bumping the version is a deliberate code change.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping

ANOMALY_RULES_VERSION = "1"

# Fields whose value is 0 VNĐ / 0 units — not a real answer (rule a).
_ZERO_IMPOSSIBLE: frozenset[str] = frozenset(
    {
        "hoc_phi_tin_chi_vnd",
        "hoc_phi_nam_vnd",
        "hoc_phi_tin_chi",
        "diem_trung_tuyen_num",
        "chi_tieu",
    }
)

# Unit labels for the zero-implausible warning messages.
_UNITS: dict[str, str] = {
    "hoc_phi_tin_chi_vnd": "VNĐ/tín chỉ",
    "hoc_phi_nam_vnd": "VNĐ/năm",
    "hoc_phi_tin_chi": "VNĐ/tín chỉ",
    "diem_trung_tuyen_num": "điểm",
    "chi_tieu": "sinh viên",
}


@dataclass(frozen=True)
class Anomaly:
    """One anomalous reading in a record's structured values."""

    field: str
    value: object
    message: str
    rule: str
    severity: str = "warning"

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "message": self.message,
            "rule": self.rule,
            "severity": self.severity,
        }


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_items(values: Mapping[str, object]) -> list[tuple[str, float]]:
    return [(k, float(v)) for k, v in values.items() if _is_num(v)]


def find_anomalies(
    record_type: str,
    values: Mapping[str, object],
    *,
    peer_values: Iterable[Mapping[str, object]] = (),
) -> tuple[Anomaly, ...]:
    """Run the anomaly rules over ``values``; return the anomalies found.

    Unknown record types or missing keys never raise — the rules simply do
    not fire.  When ``peer_values`` is empty the sector-outlier rule stays
    silent (there is no peer cohort to deviate from).
    """

    out: list[Anomaly] = []
    numeric = _numeric_items(values)

    # rule a — zero-implausible
    for field, raw in values.items():
        if _is_num(raw) and raw == 0 and field in _ZERO_IMPOSSIBLE:
            unit = _UNITS.get(field, "đơn vị")
            out.append(
                Anomaly(
                    field=str(field),
                    value=raw,
                    message=f"{field} bằng 0 {unit} — kiểm tra lại",
                    rule="zero-implausible",
                )
            )

    # rule b — negative
    for field, num in numeric:
        if num < 0:
            out.append(
                Anomaly(
                    field=str(field),
                    value=num,
                    message=f"{field} = {num} — giá trị âm, kiểm tra lại",
                    rule="negative",
                )
            )

    # rule c — sector-outlier (silent when no peers or no khoi_nganh)
    peer_dicts = list(peer_values)
    if peer_dicts and "khoi_nganh" in values:
        for field, num in numeric:
            if field == "khoi_nganh":
                continue
            peer_nums = [
                float(p[field])
                for p in peer_dicts
                if _is_num(p.get(field, None))
            ]
            if not peer_nums:
                continue
            median = statistics.median(peer_nums)
            if median <= 0:
                continue
            ratio = num / median
            if ratio > 5.0 or ratio < 0.2:
                out.append(
                    Anomaly(
                        field=str(field),
                        value=num,
                        message=(
                            f"{field} = {num} lệch hơn {max(ratio, 1/ratio):.1f}x "
                            f"so với trung vị ngành ({median:g}) — kiểm tra lại"
                        ),
                        rule="sector-outlier",
                    )
                )

    return tuple(out)
