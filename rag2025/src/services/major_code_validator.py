# @spec(S13.6) major code validator — C11 hard constraint (id=74 whitelist)
"""Validate Vietnamese major codes against the admissions whitelist.

Per spec C11: only majors listed in HUSC tuyển sinh notification id=74 are
valid. Any other code that the LLM mentions in its answer must be flagged
so the generator can post-process / refuse.

Codes follow the regex ^7\\d{6}([A-Z]{2})?$ (7-digit base + optional 2-letter
variant suffix like VJ for Việt-Nhật, IC for Vi mạch tích hợp, SC for Bán dẫn).

The whitelist is now YEAR-LOADABLE: prefer ``data/major_codes/<year>.json``
when present, else fall back to the legacy ``WHITELIST_2026`` frozenset. This
preserves backward compatibility for any caller that does not pass a year.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, Optional

# NOTE: ``services.temporal_authority`` is imported lazily inside get_whitelist()
# so the test loader (importlib.util.spec_from_file_location) — which does NOT
# add ``src/`` to sys.path — can still import this module without a hard
# dependency on the services package. The import is also fail-soft (returns
# the legacy 2026 default) to honor the "NEVER raises" contract.


# C11 — id=74 + nganhdaotao-dh.php live page (verified 2026-05-26).
# Total 28 codes for 2026 = 25 base + 3 new (KHDL 7460108, Vi mạch tích hợp
# 7510302IC, Bán dẫn 7440102SC). 7480201VJ (CNTT Việt-Nhật) IS included for 2026
# per https://tuyensinh.husc.edu.vn/nganhdaotao-dh.php (28 ngành) and TB74.
# REMOVED for 2026: 7340405 / 7480107TD (Quản trị và phân tích dữ liệu) — cô
# tuyển sinh confirmed năm 2026 không tuyển ngành này (tăng 3 mới, bỏ 1 cũ).
# This frozenset is the LEGACY fallback used by ``get_whitelist`` when no
# per-year JSON file is available. Callers reading the constant directly still
# see the historical 2026 snapshot (backward compat).
WHITELIST_2026: frozenset[str] = frozenset({
    # Khối V — Kỹ thuật / Công nghệ
    "7480101",   # Khoa học máy tính
    "7480103",   # Kỹ thuật phần mềm
    "7480201",   # Công nghệ thông tin (Cử nhân)
    "7480201VJ", # Công nghệ thông tin (Kỹ sư Việt-Nhật) — verified live page 2026-05-26
    "7480104",   # Hệ thống thông tin
    "7460108",   # Khoa học dữ liệu (mới 2026)
    "7510302IC", # Vi mạch tích hợp (mới 2026)
    "7440102SC", # Bán dẫn (mới 2026)
    # Khối Khoa học tự nhiên
    "7440101",   # Vật lý
    "7440102",   # Vật lý kỹ thuật
    "7440112",   # Hóa học
    "7460101",   # Toán học
    "7460112",   # Toán-Tin
    "7440113",   # Hóa dược
    "7440301",   # Khoa học môi trường
    "7460109",   # Toán ứng dụng
    "7510401",   # Công nghệ kỹ thuật hóa học
    # Khối Sinh-Y dược
    "7420201",   # Sinh học
    "7420101",   # Công nghệ sinh học
    "7420203",   # Sinh học ứng dụng
    "7720601",   # Y sinh
    # Khối Khoa học xã hội
    "7310201",   # Triết học
    "7310301",   # Xã hội học
    "7320201",   # Đông phương học
    "7320101",   # Báo chí
    "7220213",   # Đông Nam Á học
    "7320104",   # Truyền thông đa phương tiện
    "7310401",   # Tâm lý học
    "7310205",   # Quản lý nhà nước
})

# Codes confirmed REMOVED for 2026 (existed in id=49/2025 but NOT in id=74/2026
# nor on live nganhdaotao-dh.php). Used by validate_answer_majors to flag if LLM
# accidentally mentions a deprecated code.
REMOVED_FROM_2026: frozenset[str] = frozenset({
    "7340405",    # Quản trị và phân tích dữ liệu — bỏ năm 2026
    "7480107TD",  # Quản trị và phân tích dữ liệu (variant) — bỏ năm 2026
})

# Validate format only (does NOT imply 2026 validity).
_CODE_RX = re.compile(r"^7\d{6}([A-Z]{2})?$")
# Match codes embedded in text — same shape, but as a token.
_CODE_IN_TEXT_RX = re.compile(r"\b7\d{6}(?:[A-Z]{2})?\b")


# ---------------------------------------------------------------------------
# Year-loadable whitelist (S14.7 — de-snapshot the C11 registry)
# ---------------------------------------------------------------------------
# The historical frozenset above is the LEGACY fallback only. The new contract
# is:
#   get_whitelist(year=None) -> frozenset[str]
#     - year defaults to get_current_admission_year() (lazy import, fail-soft).
#     - prefers data/major_codes/<year>.json when present and well-formed.
#     - on any failure (file missing, malformed JSON, empty codes) falls back
#       to the legacy WHITELIST_2026 frozenset. NEVER raises.
#     - result is memoized per year for the lifetime of the module.

_MAJOR_CODES_DIR = Path(__file__).resolve().parents[2] / "data" / "major_codes"
_WHITELIST_CACHE: dict[int, frozenset[str]] = {}
_LEGACY_DEFAULT_YEAR = 2026


def get_whitelist(year: Optional[int] = None) -> frozenset[str]:
    """Return the canonical major-code whitelist for a given admission year.

    Resolution order:
      1. If ``year`` is None, resolve via ``get_current_admission_year()``;
         if that import/call fails for any reason, default to 2026.
      2. Load ``data/major_codes/<year>.json`` if present and well-formed.
      3. If the file is missing/malformed, fall back to the legacy
         ``WHITELIST_2026`` frozenset. **Never raises.**

    The result is memoized per-year; subsequent calls return the same frozenset.
    """
    if year is None:
        try:
            from services.temporal_authority import get_current_admission_year
            year = int(get_current_admission_year())
        except Exception:
            year = _LEGACY_DEFAULT_YEAR

    cached = _WHITELIST_CACHE.get(int(year))
    if cached is not None:
        return cached

    codes: frozenset[str]
    try:
        path = _MAJOR_CODES_DIR / f"{int(year)}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("codes", [])
        if not isinstance(raw, list) or not raw:
            raise ValueError("codes must be a non-empty list")
        valid = [c for c in raw if isinstance(c, str) and is_well_formed_major_code(c)]
        if not valid:
            codes = WHITELIST_2026
        else:
            codes = frozenset(valid)
    except Exception:
        codes = WHITELIST_2026

    _WHITELIST_CACHE[int(year)] = codes
    return codes


def is_well_formed_major_code(code: str) -> bool:
    """True if the string matches the major-code shape (7-digit + optional 2-letter)."""
    if not isinstance(code, str):
        return False
    return bool(_CODE_RX.match(code))


def is_valid_2026_major(code: str) -> bool:
    """True iff the code is in the current-year whitelist (C11).

    Backward-compat: signature unchanged. Resolves the current admission year
    internally via :func:`get_whitelist` which prefers
    ``data/major_codes/<year>.json`` and falls back to ``WHITELIST_2026``.
    """
    if not is_well_formed_major_code(code):
        return False
    return code in get_whitelist()


def get_2026_whitelist() -> frozenset[str]:
    """Return the canonical whitelist for the current admission year (immutable).

    Backward-compat: signature unchanged. Equivalent to ``get_whitelist()``
    (no year → current admission year).
    """
    return get_whitelist()


def validate_answer_majors(answer_text: str, year: Optional[int] = None) -> list[str]:
    """Scan answer text for major codes; return those NOT in the year's whitelist.

    Returns empty list when answer contains no codes or all codes are valid.
    The returned list preserves first-seen order and is de-duplicated.

    ``year`` defaults to the current admission year. Pass an int to override
    (e.g. when validating historical answers); backward compat preserved when
    omitted (existing callers pass only ``answer_text``).
    """
    if not isinstance(answer_text, str) or not answer_text:
        return []
    seen: set[str] = set()
    invalid: list[str] = []
    whitelist = get_whitelist(year)
    for m in _CODE_IN_TEXT_RX.finditer(answer_text):
        code = m.group(0)
        if code in seen:
            continue
        seen.add(code)
        if code not in whitelist:
            invalid.append(code)
    return invalid


def filter_chunks_by_major_whitelist(
    chunks: Iterable[dict], year: Optional[int] = None
) -> tuple[list[dict], list[dict]]:
    """Split chunks into (kept, rejected) by whether their major_code is in the year's whitelist.

    Chunks without a major_code field are ALWAYS kept (they're not subject to
    this validator — could be school-level info, FAQ, etc.).

    ``year`` defaults to the current admission year; pass an int to override.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    whitelist = get_whitelist(year)
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        code = meta.get("major_code")
        if code is None:
            kept.append(c)
            continue
        if code in whitelist:
            kept.append(c)
        else:
            rejected.append(c)
    return kept, rejected
