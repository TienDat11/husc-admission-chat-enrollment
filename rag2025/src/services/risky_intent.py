# @spec(S13.6) risky_intent — classify queries that risk numeric hallucination
"""Detect when a query asks for a numeric/lookup-style fact that we MUST NOT
let the LLM hallucinate (điểm chuẩn / học phí / chỉ tiêu / tổ hợp).

When the query is risky AND we lack a current-year chunk, the generator
returns a graceful fallback ("chưa có thông tin chính thức năm N") instead
of running a fabrication-forcing retry that would force the LLM to fabricate.

Pure regex; no LLM. Used by Phase 5 generation pipeline (S13.6).
"""
from __future__ import annotations
import re
from typing import Optional


# Each rule maps to a risky-intent label. Patterns are case-insensitive.
_RULES: list[tuple[re.Pattern[str], str]] = [
    # diem_chuan — admission cutoff scores
    (re.compile(r"\b(điểm\s+chuẩn|điểm\s+trúng\s+tuyển|điểm\s+sàn)\b", re.IGNORECASE), "diem_chuan"),
    # hoc_phi — tuition fees
    (re.compile(r"\b(học\s+phí|học\s*phí/tín\s*chỉ|học\s+phí\s+ngành)\b", re.IGNORECASE), "hoc_phi"),
    # chi_tieu — admission quotas
    (re.compile(r"\b(chỉ\s+tiêu|số\s+lượng\s+tuyển|quota|chỉ\s+tiêu\s+tuyển\s+sinh)\b", re.IGNORECASE), "chi_tieu"),
    # da_hop — combined / merged subject groups
    (re.compile(r"\b(tổ\s+hợp\s+đã\s+hợp|đã\s+hợp\s+tổ\s+hợp|tổ\s+hợp\s+xét\s+tuyển|tổ\s+hợp\s+môn|tổ\s+hợp\s+thi)\b", re.IGNORECASE), "da_hop"),
]

RISKY_INTENTS = frozenset({"diem_chuan", "hoc_phi", "chi_tieu", "da_hop"})


def infer_intent_from_query(query: str) -> Optional[str]:
    """Return one of {diem_chuan, hoc_phi, chi_tieu, da_hop} or None.

    First-match wins. Patterns are ordered by specificity. The output set is
    bounded (RISKY_INTENTS) so callers can use `intent in RISKY_INTENTS`
    without import gymnastics.
    """
    if not isinstance(query, str) or not query:
        return None
    for pattern, label in _RULES:
        if pattern.search(query):
            return label
    return None


def is_risky_intent(query: str) -> bool:
    """Convenience: True iff the query maps to a risky intent."""
    return infer_intent_from_query(query) is not None
