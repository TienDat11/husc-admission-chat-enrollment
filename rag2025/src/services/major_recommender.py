# @spec(PHASE-A2) deterministic major recommender + what-if probability
"""Pure-deterministic admission recommender for HUSC Phase A.

Two public functions, both pure-functional over a JSON dataset:

  * :func:`recommend` — returns a ranked list of :class:`Recommendation`
    objects for a given score (+optional tổ hợp + ưu tiên).
  * :func:`whatif_probability` — returns an HONEST band-based probability
    estimate (NOT an ML model) for a specific major given the same inputs.

Design contract (non-negotiable, per spec):
  - Read a local JSON dataset (parsed in A1).
  - Do arithmetic. Return.
  - ZERO LLM calls. ZERO network. ZERO new httpx timeouts.
  - Single-digit millisecond response time.

Label thresholds (delta = score + ưu_tiên − diem_chuan):
  - delta ≥  2            → "an_toan"
  - −1 ≤ delta <  2       → "can_nhac"
  - delta < −1            → "mao_hiem"

Probability bands (whatif):
  - delta ≥  3            → "rất cao ~90%"
  - 1 ≤ delta < 3         → "cao"
  - −1 ≤ delta < 1        → "trung bình 50/50"
  - delta < −1            → "thấp"

Mtime cache mirrors the pattern in ``services.guardrail._load_major_scope``
so a freshly rewritten ``diem_chuan_history.json`` is picked up without
restart.
"""
from __future__ import annotations

from decimal import Decimal
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from services.priority_score import compute_priority

# Resolve repo-root / src import path the same way the A1 parser does.
_HERE = Path(__file__).resolve().parent
# parents: services -> src -> rag2025 -> repo root
_RAG_ROOT = _HERE.parent.parent
import sys
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

_HISTORY_PATH = _RAG_ROOT / "data" / "major_codes" / "diem_chuan_history.json"

# ---- Cache (mtime → loaded dataset) -----------------------------------------
_CACHE_MTIME: Optional[float] = None
_CACHE_BY_KEY: Dict[str, list] = {}        # (major_code, method) → sorted list of (year, diem_chuan)
_CACHE_LATEST: Dict[str, tuple] = {}        # (major_code, method) → (year, diem_chuan)  # the freshest

# ---- Snapshot injection (U8: serving reads RAM, not data/major_codes/*.json) --
_SNAPSHOT: Any = None   # ReleaseSnapshot | None — set by the release reconciler


def set_snapshot(snapshot: Any) -> None:
    """Inject a ReleaseSnapshot so serving reads from RAM, never disk.

    Called by the release reconciler after each hot-swap. When set, ``recommend``
    and ``whatif_probability`` read ``recommender_snapshot`` instead of opening
    ``data/major_codes/*.json``.
    """
    global _SNAPSHOT
    _SNAPSHOT = snapshot


@dataclass
class Recommendation:
    major_code: str
    major_name: str
    latest_diem_chuan: float
    latest_year: int
    delta: float
    label: str
    explanation: str
    to_hop: Optional[list] = None


@dataclass
class WhatIfResult:
    p_pass: str
    band: str
    basis: str
    disclaimer: str
    latest_diem_chuan: Optional[float] = None
    latest_year: Optional[int] = None
    delta: Optional[float] = None
    n_years: int = 0


# ---- Load + cache ----------------------------------------------------------

def _load_history() -> List[dict]:
    """Read the A1 history JSON, reload on mtime change (mirrors guardrail).

    Returns the (cached) raw entries list for callers that want name
    metadata; the hot path in ``recommend``/``whatif_probability`` uses
    the derived ``_CACHE_LATEST`` / ``_CACHE_BY_KEY`` views instead.

    When a ReleaseSnapshot has been injected (U8), its ``recommender_snapshot``
    dict is used instead of touching ``data/major_codes/*.json``.
    """
    global _CACHE_MTIME, _CACHE_BY_KEY, _CACHE_LATEST, _cache_entries_cache

    # Snapshot takes precedence: serving reads RAM, never disk.
    if _SNAPSHOT is not None:
        snap = _SNAPSHOT.recommender_snapshot or {}
        entries = snap.get("entries", []) if isinstance(snap, dict) else []
        by_key: Dict[str, list] = {}
        latest: Dict[str, tuple] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            code = e.get("major_code")
            method = e.get("method")
            year = e.get("year")
            score = e.get("diem_chuan")
            if not code or not method or year is None or score is None:
                continue
            key = f"{code}::{method}"
            by_key.setdefault(key, []).append((int(year), float(score)))
            cur = latest.get(key)
            if cur is None or int(year) > cur[0]:
                latest[key] = (int(year), float(score))
        _CACHE_BY_KEY = by_key
        _CACHE_LATEST = latest
        _cache_entries_cache = list(entries)
        return list(entries)

    if not _HISTORY_PATH.exists():
        return []
    try:
        mtime = _HISTORY_PATH.stat().st_mtime
    except OSError:
        return []
    if _CACHE_MTIME is not None and mtime == _CACHE_MTIME and _CACHE_BY_KEY:
        return list(_cache_entries_cache)
    try:
        raw = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    by_key: Dict[str, list] = {}
    latest: Dict[str, tuple] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        code = e.get("major_code")
        method = e.get("method")
        year = e.get("year")
        score = e.get("diem_chuan")
        if not code or not method or year is None or score is None:
            continue
        key = f"{code}::{method}"
        by_key.setdefault(key, []).append((int(year), float(score)))
        cur = latest.get(key)
        if cur is None or int(year) > cur[0]:
            latest[key] = (int(year), float(score))
    _CACHE_MTIME = mtime
    _CACHE_BY_KEY = by_key
    _CACHE_LATEST = latest
    _cache_entries_cache = list(entries)
    return list(entries)


def _latest_for(major_code: str, method: str) -> Optional[tuple]:
    """Return (year, diem_chuan) of the freshest entry, or None."""
    # Make sure the cache is warm
    if not _CACHE_LATEST:
        _load_history()
    return _CACHE_LATEST.get(f"{major_code}::{method}")


def _history_for(major_code: str, method: str) -> list:
    """Return list[(year, diem_chuan)] newest-first for one major+method."""
    if not _CACHE_BY_KEY:
        _load_history()
    return list(_CACHE_BY_KEY.get(f"{major_code}::{method}", []))


# ---- Public API ------------------------------------------------------------

def _label_for(delta: float) -> str:
    if delta >= 2.0:
        return "an_toan"
    if delta >= -1.0:
        return "can_nhac"
    return "mao_hiem"


def _band_for(delta: float) -> tuple[str, str]:
    """Return (band_label, p_pass_text) for whatif."""
    if delta >= 3.0:
        return ("rất cao", "khoảng 90% (ước lượng tham khảo)")
    if delta >= 1.0:
        return ("cao", "khoảng 70% (ước lượng tham khảo)")
    if delta >= -1.0:
        return ("trung bình", "khoảng 50% (ước lượng tham khảo)")
    return ("thấp", "khoảng 20% (ước lượng tham khảo)")


def recommend(
    score: float,
    to_hop: Optional[str] = None,
    uu_tien: float = 0.0,
    kv: str | None = None,
    dt: str | None = None,
) -> List[Recommendation]:
    """Return ranked recommendations for a given score.

    ``score`` is the raw thang-30 score. ``uu_tien`` is the legacy ưu tiên
    bonus, used only when ``kv``/``dt`` are not provided (existing callers).
    Otherwise the TT06 priority total from :func:`compute_priority` drives
    the ranking.
    """
    if not isinstance(score, (int, float)) or math.isnan(score) or score < 0:
        return []
    # Warm cache (uses snapshot when injected, otherwise disk).
    _load_history()
    if not _CACHE_LATEST:
        return []

    # TT06 priority total (replaces the legacy score + uu_tien arithmetic).
    # Fall back to the legacy ``uu_tien`` bonus when kv/dt are not provided so
    # existing callers stay compatible.
    if kv is None and dt is None:
        priority_total = Decimal(str(score)) + Decimal(str(uu_tien))
        rule = "legacy_uu_tien"
    else:
        priority = compute_priority(Decimal(str(score)), kv, dt)
        priority_total = priority.total
        rule = priority.rule
    effective_bonus = float(priority_total - Decimal(str(score)))

    # Friendly names + to_hop from the entries already loaded by _load_history
    # (snapshot-backed when injected — never touches data/major_codes/*.json).
    by_code_meta: Dict[str, dict] = {}
    for e in _cache_entries_cache:
        if not isinstance(e, dict):
            continue
        code = e.get("major_code")
        if not code or code in by_code_meta:
            continue
        by_code_meta[code] = {
            "major_name": e.get("major_name") or code,
            "to_hop": e.get("to_hop"),
        }

    method = "thi_thpt"
    out: List[Recommendation] = []
    for code, (year, diem) in _CACHE_LATEST.items():
        major_code, m = code.split("::", 1)
        if m != method:
            continue
        meta = by_code_meta.get(major_code, {"major_name": major_code, "to_hop": None})
        if to_hop and meta.get("to_hop") and to_hop not in meta["to_hop"]:
            continue
        delta = float(priority_total) - float(diem)
        label = _label_for(delta)
        if to_hop and (not meta.get("to_hop") or to_hop not in meta["to_hop"]):
            # The user asked for a tổ hợp but this major's known tổ hợp
            # doesn't include it. The earlier ``continue`` already removed
            # explicit-mismatch majors; majors with no to_hop map get
            # a soft "không rõ tổ hợp" annotation.
            to_hop_note = f"; tổ hợp {to_hop} (chưa đối chiếu được)"
        else:
            to_hop_note = ""
        explanation = (
            f"Điểm chuẩn {year} ngành {meta['major_name']} là {diem:.2f}, "
            f"điểm của bạn {float(score):.2f}{(' + ưu tiên ' + format(effective_bonus, '.2f')) if effective_bonus else ''} "
            f"→ {label.replace('_', ' ')}{to_hop_note}"
        )
        out.append(Recommendation(
            major_code=major_code,
            major_name=meta["major_name"],
            latest_diem_chuan=float(diem),
            latest_year=int(year),
            delta=delta,
            label=label,
            explanation=explanation,
            to_hop=meta.get("to_hop"),
        ))
    out.sort(key=lambda r: (-r.delta, r.major_code))
    return out


def whatif_probability(
    score: float,
    major_code: str,
    to_hop: Optional[str] = None,  # noqa: ARG001 — kept for API parity
    uu_tien: float = 0.0,
    kv: str | None = None,
    dt: str | None = None,
) -> WhatIfResult:
    """Honest band-based probability estimate for a specific major.

    Returns a :class:`WhatIfResult` with ``p_pass`` (text band), ``band``,
    ``basis`` (always cites the year count + disclaimer), and ``disclaimer``.

    Unknown ``major_code`` → graceful BLOCKED-style result, NEVER raises.
    """
    if not isinstance(score, (int, float)) or math.isnan(score) or score < 0:
        return WhatIfResult(
            p_pass="không xác định",
            band="invalid_score",
            basis="Điểm đầu vào không hợp lệ (cần số ≥ 0).",
            disclaimer=(
                "Đây chỉ là ước lượng tham khảo dựa trên điểm chuẩn các năm trước; "
                "kết quả thực tế còn phụ thuộc chỉ tiêu, nguyện vọng và phương thức xét tuyển."
            ),
        )
    # Warm cache
    _load_history()
    # Method-agnostic: pick whichever method (THPT vs HB) we have data for;
    # prefer thi_thpt as it's the more common / flagship channel.
    latest = _latest_for(major_code, "thi_thpt")
    method_used = "thi_thpt"
    if latest is None:
        latest = _latest_for(major_code, "hoc_ba")
        method_used = "hoc_ba"
    history = _history_for(major_code, method_used) if latest else []
    if latest is None:
        # Unknown / unmapped major
        return WhatIfResult(
            p_pass="không xác định",
            band="unknown_major",
            basis=(
                f"Không tìm thấy dữ liệu điểm chuẩn cho mã ngành '{major_code}' "
                "(có thể ngành mới hoặc không nằm trong danh sách tuyển sinh hiện hành)."
            ),
            disclaimer=(
                "Vui lòng kiểm tra lại mã ngành hoặc tham khảo thông báo tuyển sinh chính thức."
            ),
        )
    year, diem = latest
    if kv is None and dt is None:
        priority_total = Decimal(str(score)) + Decimal(str(uu_tien))
    else:
        priority_total = compute_priority(Decimal(str(score)), kv, dt).total
    delta = float(priority_total) - float(diem)
    band, p_pass = _band_for(delta)
    n_years = len(history)
    if n_years >= 2:
        years_txt = f"{min(y for y, _ in history)}–{max(y for y, _ in history)}"
    elif n_years == 1:
        years_txt = f"{history[0][0]}"
    else:
        years_txt = "(không rõ)"
    basis = (
        f"Dựa trên điểm chuẩn {n_years} năm ({years_txt}), phương thức {method_used}, "
        f"chỉ mang tính tham khảo."
    )
    disclaimer = (
        "Đây là ước lượng heuristic, KHÔNG phải mô hình ML chính xác; "
        "kết quả thực tế còn phụ thuộc chỉ tiêu, số lượng nguyện vọng và phương thức xét tuyển."
    )
    return WhatIfResult(
        p_pass=p_pass,
        band=band,
        basis=basis,
        disclaimer=disclaimer,
        latest_diem_chuan=float(diem),
        latest_year=int(year),
        delta=float(delta),
        n_years=n_years,
    )
