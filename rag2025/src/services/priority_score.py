"""§PL-6.1 Điểm ưu tiên Thông tư 06 (TT06).

Pure arithmetic over a raw thang-30 score plus ưu tiên KV (khu vực) and
ưu tiên DT (đối tượng). Mirrors ``recommender_snapshots.priority_rules`` of the
active release when that row is readable; the module-level ``KV_BONUS`` /
``DT_BONUS`` constants are the fallback used by tests and by any caller that
does not supply ``rules``.

Formula (exact):
    base = KV_BONUS[kv] + DT_BONUS[dt]
    if raw_score >= 22.5:
        bonus = ((30 - raw_score) / 7.5) * base        # linear decay
    else:
        bonus = base                                   # full bonus
    bonus = max(0, bonus) quantized to 0.01 ROUND_HALF_UP
    total = min(30, raw_score + bonus)

Design constraints (non-negotiable):
  - Pure function: no DB, no network, no clock. Deterministic for fixed inputs.
  - ``ROUND_HALF_UP`` to ``0.01`` on the bonus (never banker's rounding).
  - ``total`` capped at 30.
  - Missing kv/dt default to the lowest band ("KV3" / "NONE") so every call
    produces a result; unknown codes fall back the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

__all__ = ["KV_BONUS", "DT_BONUS", "PriorityResult", "compute_priority"]

# Fallback tables — identical to the values stored in
# ``recommender_snapshots.priority_rules`` when no release snapshot is
# available. Keys are case-sensitive.
KV_BONUS: dict[str, Decimal] = {
    "KV1": Decimal("0.75"),
    "KV2NT": Decimal("0.50"),
    "KV2": Decimal("0.25"),
    "KV3": Decimal("0.0"),
}
DT_BONUS: dict[str, Decimal] = {
    "UT1": Decimal("2.0"),
    "UT2": Decimal("1.0"),
    "NONE": Decimal("0.0"),
}

_QUANT = Decimal("0.01")
_DECAY_THRESHOLD = Decimal("22.5")
_DECAY_SPAN = Decimal("7.5")
_CAP = Decimal("30")
_TT06_FULL = "TT06_full"
_TT06_DECAY = "TT06_decay"


@dataclass(frozen=True)
class PriorityResult:
    """Result of :func:`compute_priority`.

    All monetary fields are :class:`decimal.Decimal` so no float noise can
    creep into a comparison against ``diem_chuan`` downstream.
    """

    raw_score: Decimal
    base: Decimal
    bonus: Decimal
    total: Decimal
    rule: str


def _bonus_map(rules: dict | None, kind: str) -> dict[str, Decimal]:
    """Resolve a bonus mapping from ``rules`` or the module fallback."""
    if not isinstance(rules, dict):
        return KV_BONUS if kind == "kv" else DT_BONUS
    nested = rules.get(kind)
    if isinstance(nested, dict):
        return {str(k): Decimal(str(v)) for k, v in nested.items()}
    return KV_BONUS if kind == "kv" else DT_BONUS


def compute_priority(
    raw_score: Decimal,
    kv: str | None,
    dt: str | None,
    rules: dict | None = None,
) -> PriorityResult:
    """Compute the TT06 priority total for a raw score.

    ``rules`` mirrors ``recommender_snapshots.priority_rules`` of the active
    release and may carry ``kv_bonus`` / ``dt_bonus`` mappings. When omitted
    (or lacking a band), the module-level ``KV_BONUS`` / ``DT_BONUS``
    constants apply.
    """
    score = Decimal(str(raw_score))
    kv_bonus_map = _bonus_map(rules, "kv")
    dt_bonus_map = _bonus_map(rules, "dt")

    base = kv_bonus_map.get(kv or "KV3", Decimal("0")) + dt_bonus_map.get(
        dt or "NONE", Decimal("0")
    )

    if score >= _DECAY_THRESHOLD:
        bonus = ((_CAP - score) / _DECAY_SPAN) * base
        rule = _TT06_DECAY
    else:
        bonus = base
        rule = _TT06_FULL

    bonus = max(Decimal(0), bonus).quantize(_QUANT, rounding=ROUND_HALF_UP)
    total = min(_CAP, score + bonus)

    return PriorityResult(raw_score=score, base=base, bonus=bonus, total=total, rule=rule)
