"""Vietnamese supersession detector - rule-based, no NLP model.
See rag2025/.omc/plans/data_lifecycle_plan_v2.md section 7, M2 lines 436-449.
Boilerplate Vietnamese legal text is highly repetitive -> regex gives high
precision (section 7.1). The single most dangerous failure mode is mistaking
a REFERENCE sentence ("Căn cứ Thông tư...") for a SUPERSESSION sentence -
section 7.7 negative-context guard exists specifically to prevent this
(gate G2c, M2-T10, must be exactly 0 false positives).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.domain.release import TransitionKind

RX_NEGATIVE = re.compile(
    r"^\s*(?:"
    r"Căn\s+cứ"
    r"|Thực\s+hiện"
    r"|Theo\s+(?:đề\s+nghị|hướng\s+dẫn|quy\s+định)"
    r"|Trên\s+cơ\s+sở"
    r"|Xét\s+đề\s+nghị"
    r")\b",
    re.I | re.M,
)

RX_CITATION_CTX = re.compile(
    r"(?:(?<!một số )quy\s+định|nêu|ghi)\s+tại\b", re.I,
)

RX_EFFECTIVE = re.compile(
    r"có\s+hiệu\s+lực(?:\s+(?:thi\s+hành|pháp\s+luật))?\s*"
    r"(?:kể\s+)?từ\s+ngày\s+"
    r"(\d{1,2})\s*(?:tháng|[/\-.])\s*(\d{1,2})\s*(?:năm|[/\-.])\s*(\d{4})",
    re.I,
)


def normalize(text: str) -> str:
    """NFC-normalize, strip nbsp, collapse repeated space/tab and 3+ newlines."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _is_negative_context(text: str, span: tuple[int, int]) -> bool:
    """Return whether span sits in citation/reference boilerplate."""
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    line = text[start:end if end != -1 else len(text)]
    return bool(RX_NEGATIVE.search(line) or RX_CITATION_CTX.search(line))


def extract_effective_date(text: str) -> Optional[date]:
    """Parse effective date from Vietnamese legal boilerplate."""
    match = RX_EFFECTIVE.search(text)
    if not match:
        return None
    day, month, year = (int(group) for group in match.groups())
    return date(year, month, day)


RX_REPLACES = re.compile(
    r"(?P<verb>thay\s+thế)\s+"
    r"(?:cho\s+)?"
    r"(?P<obj>(?:Thông\s+tư|Quyết\s+định|Nghị\s+định|Nghị\s+quyết|Quy\s+chế|"
    r"Đề\s+án|Công\s+văn|Thông\s+báo|Kế\s+hoạch|Hướng\s+dẫn)"
    r"[^.;\n]{0,120})",
    re.I,
)

RX_REPEALS = re.compile(
    r"(?P<verb>bãi\s+bỏ|hủy\s+bỏ|huỷ\s+bỏ|chấm\s+dứt\s+hiệu\s+lực(?:\s+của)?)\s+"
    r"(?P<obj>(?:toàn\s+bộ\s+|một\s+phần\s+)?"
    r"(?:Thông\s+tư|Quyết\s+định|Nghị\s+định|Nghị\s+quyết|Điều\s+\d+|khoản\s+\d+|"
    r"điểm\s+[a-zđăâêôơư]\)?|Đề\s+án|Quy\s+chế)[^.;\n]{0,120})",
    re.I,
)

RX_AMENDS = re.compile(
    r"(?P<verb>sửa\s+đổi(?:\s*,\s*bổ\s+sung)?|bổ\s+sung|bổ\s+sung\s+và\s+sửa\s+đổi|điều\s+chỉnh)\s+"
    r"(?:(?P<scope>một\s+số\s+(?:điều|quy\s+định)|khoản\s+\d+|Điều\s+\d+[a-zđ]?|"
    r"điểm\s+[a-zđăâêôơư]\)?)\s+)?"
    r"(?:của\s+|tại\s+)?"
    r"(?P<obj>(?:Thông\s+tư|Quyết\s+định|Nghị\s+định|Nghị\s+quyết|Đề\s+án|Quy\s+chế|Quy\s+định)"
    r"[^.;\n]{0,120})",
    re.I,
)

RX_CORRECTS = re.compile(
    r"(?P<verb>đính\s+chính|cải\s+chính|sửa\s+lỗi)\s+"
    r"(?:nội\s+dung\s+)?"
    r"(?:tại\s+|của\s+|trong\s+)?"
    r"(?P<obj>[^.;\n]{0,120})",
    re.I,
)

RX_REPLACE_PHRASE = re.compile(
    r"thay\s+(?:cụm\s+từ|từ|chữ)\s+[\"“'']([^\"”'']{1,200})[\"”'']\s+"
    r"(?:bằng|thành)\s+(?:cụm\s+từ|từ|chữ)\s+[\"“'']([^\"”'']{1,200})[\"”'']",
    re.I,
)

RX_EXPIRES = re.compile(
    r"(?P<obj>[^.;\n]{0,120}?)\s*"
    r"(?P<verb>hết\s+hiệu\s+lực)"
    r"(?:\s+(?P<extent>toàn\s+bộ|một\s+phần))?"
    r"(?:\s+(?:kể\s+)?từ\s+ngày\s+"
    r"(?P<d>\d{1,2})\s*(?:tháng|[/\-.])\s*(?P<m>\d{1,2})\s*(?:năm|[/\-.])\s*(?P<y>\d{4}))?",
    re.I,
)

RX_AFFECTED = re.compile(
    r"(?:Điều\s+(\d{1,3}[a-zđ]?)"
    r"(?:\s+khoản\s+(\d{1,2}))?"
    r"(?:\s+điểm\s+([a-zđăâêôơư]))?)",
    re.I,
)


@dataclass(frozen=True)
class TransitionCandidate:
    """M2 detector output, treated as a CANDIDATE, distinct from
    VersionTransition. Promoted to VersionTransition only after M5 human
    review (detector="rule" is never auto-applied per M2-R3 mitigation)."""

    kind: "TransitionKind"
    object_doc_number: Optional[str]
    effective_from: Optional[date]
    affected_articles: list
    evidence_span: str
    evidence_offset: tuple
    confidence: float


def _confidence(kind, obj_doc_number, in_negative, has_effective_date) -> float:
    """Section 7.8 scoring. in_negative short-circuits to 0.0 (the caller
    must still SKIP negative-context matches entirely; this is defense in
    depth, not the primary guard)."""
    if in_negative:
        return 0.0
    c = 0.55
    if obj_doc_number:
        c += 0.30
    if has_effective_date:
        c += 0.10
    if kind in (TransitionKind.REPLACES, TransitionKind.REPEALS):
        c += 0.05
    return min(c, 1.0)


def _extract_first_doc_number(fragment: str) -> Optional[str]:
    from src.services.vn_legal_structure import extract_doc_numbers

    nums = extract_doc_numbers(fragment)
    return nums[0] if nums else None


RX_AMENDS_PASSIVE = re.compile(
    r"(?P<obj>(?:Quy\s+chế|Thông\s+tư|Quyết\s+định|Nghị\s+định|Nghị\s+quyết|Đề\s+án|Quy\s+định)"
    r"[^.;\n]{0,80}?)\s+"
    r"được\s+(?P<verb>sửa\s+đổi(?:\s*,\s*bổ\s+sung)?)\s+theo\s+"
    r"(?:Quyết\s+định|Thông\s+tư|Nghị\s+định|Nghị\s+quyết)",
    re.I,
)

_KIND_PATTERNS = (
    (TransitionKind.REPLACES, RX_REPLACES),
    (TransitionKind.REPEALS, RX_REPEALS),
    (TransitionKind.AMENDS, RX_AMENDS),
    (TransitionKind.AMENDS, RX_AMENDS_PASSIVE),
    (TransitionKind.CORRECTS, RX_CORRECTS),
)


def detect(text: str, *, source_doc_id: str) -> list:
    """Run every kind pattern over normalized text, skip matches inside a
    negative (reference or citation) context, return one
    TransitionCandidate per surviving match. source_doc_id is accepted for
    API-shape parity with the spec but not yet embedded in the returned
    dataclass (evidence_doc_id wiring is deferred, not stubbed out with a
    placeholder branch)."""
    normalized = normalize(text)
    effective_from = extract_effective_date(normalized)
    out = []
    for kind, rx in _KIND_PATTERNS:
        for m in rx.finditer(normalized):
            span = m.span()
            if _is_negative_context(normalized, span):
                continue
            obj_fragment = m.group("obj")
            obj_num = _extract_first_doc_number(obj_fragment)
            conf = _confidence(kind, obj_num, False, effective_from is not None)
            out.append(
                TransitionCandidate(
                    kind=kind,
                    object_doc_number=obj_num,
                    effective_from=effective_from,
                    affected_articles=[],
                    evidence_span=m.group(0).strip(),
                    evidence_offset=span,
                    confidence=conf,
                )
            )
    return out
