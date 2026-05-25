# @spec(S13.6) major code validator — C11 hard constraint (id=74 whitelist)
"""Validate Vietnamese major codes against the 2026 admissions whitelist.

Per spec C11: only majors listed in HUSC tuyển sinh notification id=74 are
valid for 2026. Any other code that the LLM mentions in its answer must be
flagged so the generator can post-process / refuse.

Codes follow the regex ^7\\d{6}([A-Z]{2})?$ (7-digit base + optional 2-letter
variant suffix like VJ for Việt-Nhật, IC for Vi mạch tích hợp).
"""
from __future__ import annotations
import re
from typing import Iterable


# C11 — id=74 whitelist. Total 28 codes = 25 base ngành + 3 new for 2026
# (KHDL 7460108, Vi mạch tích hợp 7510302IC, Bán dẫn 7440102SC).
# 7480201VJ (CNTT Việt-Nhật) is NOT included — it existed in id=59 (2025) but
# does NOT appear in the id=74 dossier per cô tuyển sinh's authoritative ruling.
# This is the SINGLE source of truth for "valid 2026 major code".
WHITELIST_2026: frozenset[str] = frozenset({
    # Khối V — Kỹ thuật / Công nghệ
    "7480101",   # Khoa học máy tính
    "7480103",   # Kỹ thuật phần mềm
    "7480201",   # Công nghệ thông tin
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

# Validate format only (does NOT imply 2026 validity).
_CODE_RX = re.compile(r"^7\d{6}([A-Z]{2})?$")
# Match codes embedded in text — same shape, but as a token.
_CODE_IN_TEXT_RX = re.compile(r"\b7\d{6}(?:[A-Z]{2})?\b")


def is_well_formed_major_code(code: str) -> bool:
    """True if the string matches the major-code shape (7-digit + optional 2-letter)."""
    if not isinstance(code, str):
        return False
    return bool(_CODE_RX.match(code))


def is_valid_2026_major(code: str) -> bool:
    """True iff the code is in the 2026 id=74 whitelist (C11)."""
    if not is_well_formed_major_code(code):
        return False
    return code in WHITELIST_2026


def get_2026_whitelist() -> frozenset[str]:
    """Return the canonical 2026 whitelist (immutable)."""
    return WHITELIST_2026


def validate_answer_majors(answer_text: str) -> list[str]:
    """Scan answer text for major codes; return list of those NOT in 2026 whitelist.

    Returns empty list when answer contains no codes or all codes are valid.
    The returned list preserves first-seen order and is de-duplicated.
    """
    if not isinstance(answer_text, str) or not answer_text:
        return []
    seen: set[str] = set()
    invalid: list[str] = []
    for m in _CODE_IN_TEXT_RX.finditer(answer_text):
        code = m.group(0)
        if code in seen:
            continue
        seen.add(code)
        if code not in WHITELIST_2026:
            invalid.append(code)
    return invalid


def filter_chunks_by_major_whitelist(chunks: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Split chunks into (kept, rejected) by whether their major_code is in the 2026 whitelist.

    Chunks without a major_code field are ALWAYS kept (they're not subject to
    this validator — could be school-level info, FAQ, etc.).
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        code = meta.get("major_code")
        if code is None:
            kept.append(c)
            continue
        if code in WHITELIST_2026:
            kept.append(c)
        else:
            rejected.append(c)
    return kept, rejected
