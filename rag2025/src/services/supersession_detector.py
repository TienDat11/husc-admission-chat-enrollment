"""Vietnamese supersession detector - rule-based, no NLP model.
See rag2025/.omc/plans/data_lifecycle_plan_v2.md section 7, M2 lines 436-449.
Boilerplate Vietnamese legal text is highly repetitive -> regex gives high
precision (section 7.1). The single most dangerous failure mode is mistaking
a REFERENCE sentence ("Căn cứ Thông tư...") for a SUPERSESSION sentence -
section 7.7 negative-context guard exists specifically to prevent this
(gate G2c, M2-T10, must be exactly 0 false positives).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Optional
from collections import Counter
from collections.abc import Sequence

from src.domain.release import TransitionKind
from src.domain.canonical.identity import parse_coordinate_key

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


def _affected_articles(text: str, span: tuple[int, int]) -> list[str]:
    """Distinct ``Điều N [khoản M] [điểm x]`` references on the line holding ``span``."""
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    line = text[start:end if end != -1 else len(text)]
    found: list[str] = []
    for match in RX_AFFECTED.finditer(line):
        ref = " ".join(match.group(0).split())
        if ref not in found:
            found.append(ref)
    return found


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
                    affected_articles=_affected_articles(normalized, span),
                    evidence_span=m.group(0).strip(),
                    evidence_offset=span,
                    confidence=conf,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Supersession tiers (plan §PL-6.7)
# ---------------------------------------------------------------------------
#: Relations that move a head forward: the old coordinate is replaced.
SUCCESSION_RELATIONS = frozenset({"REPLACES", "AMENDS", "CORRECTS", "SPLIT", "MERGE", "CHAIN", "HANDOFF"})
#: Relations that end a coordinate with no successor.
WITHDRAWAL_RELATIONS = frozenset({"REPEALS", "EXPIRES"})
#: Relations a reviewer decides; text alone cannot establish them.
DECISION_RELATIONS = frozenset({"COEXIST", "REVOKE_SUPERSESSION", "HANDOFF"})
ALL_RELATIONS = SUCCESSION_RELATIONS | WITHDRAWAL_RELATIONS | {"COEXIST", "REVOKE_SUPERSESSION"}
SEMANTIC_COSINE_MIN = 0.6
SEMANTIC_JACCARD_MIN = 0.3
_WORD_RX = re.compile(r"\w+")


@dataclass(frozen=True)
class SupersessionEdge:
    old_key: str | None
    new_key: str | None
    relation: str
    tier: str  # explicit_ref | coordinate_key | semantic | decision


def _subject(key: str) -> str:
    return parse_coordinate_key(key).subject


def _year(key: str) -> int:
    return parse_coordinate_key(key).admission_year


def link_keys(
    old_keys: Sequence[str], new_keys: Sequence[str], *, relation: str
) -> tuple[SupersessionEdge, ...]:
    """Edges for one supersession event; the shape comes from the keys (§PL-6.7).

    1→1 is REPLACES/AMENDS/CORRECTS (coordinate_key when the subject is the same,
    else explicit_ref); 1→N is SPLIT; N→1 is MERGE, or CHAIN when all old keys
    share one subject; 1→0 is a withdrawal. COEXIST, REVOKE_SUPERSESSION and
    HANDOFF are reviewer decisions and carry tier ``decision``.
    """
    if relation not in ALL_RELATIONS:
        raise ValueError(f"unknown supersession relation: {relation!r}")
    old = sorted(dict.fromkeys(old_keys), key=_year)
    new = sorted(dict.fromkeys(new_keys), key=_year)
    if not old:
        raise ValueError("a supersession event needs at least one old key")

    if relation in DECISION_RELATIONS:
        if not new:
            raise ValueError(f"{relation} needs a successor key")
        if relation == "REVOKE_SUPERSESSION":  # reverses the new→old link
            return tuple(SupersessionEdge(n, o, relation, "decision") for o in old for n in new)
        return tuple(SupersessionEdge(o, n, relation, "decision") for o in old for n in new)

    if not new:
        if relation not in WITHDRAWAL_RELATIONS:
            raise ValueError(f"{relation} needs a successor key")
        return tuple(SupersessionEdge(o, None, relation, "explicit_ref") for o in old)
    if relation in WITHDRAWAL_RELATIONS:
        raise ValueError(f"{relation} ends a coordinate and takes no successor")

    if relation == "CHAIN":
        if len(new) != 1 or len({_subject(k) for k in old + new}) != 1:
            raise ValueError("a CHAIN is one subject, several years, one successor")
        path = [*old, new[0]]
        return tuple(
            SupersessionEdge(a, b, relation, "coordinate_key") for a, b in zip(path, path[1:])
        )
    if len(old) == 1 and len(new) == 1:
        old_key, new_key = old[0], new[0]
        same_subject = _subject(old_key) == _subject(new_key)
        tier = "coordinate_key" if same_subject and relation != "AMENDS" else "explicit_ref"
        return (SupersessionEdge(old_key, new_key, relation, tier),)
    if len(old) == 1:
        if relation != "SPLIT":
            raise ValueError("1→N supersession is a SPLIT")
        return tuple(SupersessionEdge(old[0], n, relation, "explicit_ref") for n in new)
    if len(new) == 1:
        if relation != "MERGE":
            raise ValueError("N→1 supersession is a MERGE or a CHAIN")
        return tuple(SupersessionEdge(o, new[0], relation, "explicit_ref") for o in old)
    raise ValueError("N→M supersession is not a supported shape")


def apply_revocations(edges: Sequence[SupersessionEdge]) -> tuple[SupersessionEdge, ...]:
    """Drop each succession edge a REVOKE_SUPERSESSION edge reverses; keep the revokes out."""
    revoked = {(e.new_key, e.old_key) for e in edges if e.relation == "REVOKE_SUPERSESSION"}
    return tuple(
        e
        for e in edges
        if e.relation != "REVOKE_SUPERSESSION" and (e.old_key, e.new_key) not in revoked
    )


def resolve_head(edges: Sequence[SupersessionEdge], key: str, *, max_depth: int = 32) -> str:
    """Follow applied succession edges from ``key`` to its current head (transitive).

    Semantic-tier edges are proposals awaiting review and are never followed.
    """
    successors: dict[str, list[str]] = {}
    for edge in edges:
        if edge.relation in SUCCESSION_RELATIONS and edge.tier != "semantic" and edge.old_key and edge.new_key:
            successors.setdefault(edge.old_key, []).append(edge.new_key)
    current = key
    for _ in range(max_depth):
        following = successors.get(current)
        if not following:
            return current
        current = max(following, key=_year)
    raise ValueError(f"supersession chain from {key!r} exceeds {max_depth} links")


@dataclass(frozen=True)
class SemanticRecord:
    key: str
    record_type: str
    major_codes: frozenset[str]
    text: str


def _tokens(text: str) -> list[str]:
    return _WORD_RX.findall(normalize(text).lower())


def semantic_scores(a: str, b: str) -> tuple[float, float]:
    """(TF-IDF cosine, Jaccard) of two texts. IDF is taken over the pair itself."""
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0, 0.0
    df: Counter[str] = Counter(set(tokens_a) | set(tokens_b))
    corpus = (tokens_a, tokens_b)
    vectors = []
    for tokens in corpus:
        counts = Counter(tokens)
        total = len(tokens)
        vectors.append(
            {t: (c / total) * (math.log(3 / (1 + df[t])) + 1) for t, c in counts.items()}
        )
    va, vb = vectors
    dot = sum(weight * vb.get(t, 0.0) for t, weight in va.items())
    norm = math.sqrt(sum(w * w for w in va.values())) * math.sqrt(sum(w * w for w in vb.values()))
    cosine = dot / norm if norm else 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    return cosine, len(sa & sb) / len(sa | sb)


def semantic_edge(old: SemanticRecord, new: SemanticRecord) -> SupersessionEdge | None:
    """Tier 3 proposal: same record type, a shared major code, cosine ≥ 0.6 and Jaccard ≥ 0.3.

    Returned as a proposal (tier ``semantic``) for review; never applied automatically.
    """
    if old.record_type != new.record_type or not (old.major_codes & new.major_codes):
        return None
    cosine, jaccard = semantic_scores(old.text, new.text)
    if cosine < SEMANTIC_COSINE_MIN or jaccard < SEMANTIC_JACCARD_MIN:
        return None
    return SupersessionEdge(old.key, new.key, "REPLACES", "semantic")
