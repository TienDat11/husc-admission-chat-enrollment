# @spec(T11) abstention_policy — evidence-sufficiency decision BEFORE generation
"""Ticket 11: decide whether retrieved evidence is sufficient to answer.

A wrong admissions answer (contact, deadline, score, quota) is worse than no
answer, so this policy runs *before* the LLM is called and refuses to let
generation proceed on thin evidence. Spec 4: "Generation chi chay khi evidence
du; thieu / lech nam / mau thuan chua resolve thi abstain hoac hien thi trang
thai conflict."

Rules, in priority order (first match wins):

1. ``out-of-scope-major``    — the query names a well-formed major code that
   is not in the target year's admissions whitelist -> abstain (criterion 5).
2. ``unresolved-conflict``   — an OPEN conflict on the same year (and major,
   when the query names one) -> ``conflict`` action carrying BOTH statements;
   the system never self-selects (criterion 3).
3. ``lineage-missing``       — a canonical-lineage chunk whose source document
   is not traceable -> abstain; an uncitable number is not an answer
   (criterion 4, "Blocked by: 10").
4. ``year-mismatch``         — the query is bound to year Y (explicit year or
   risky factual intent, which means "current"), Y is published (not
   pre-season-gap), and NO chunk carries Y while other-year chunks exist ->
   abstain, keeping the other-year data as labeled context (criterion 2).
5. ``no-evidence``           — a factual query with zero substantive chunks
   -> abstain (criterion 1).

Criterion 6: an abstain message always states what WAS verified (year
labels, both conflict statements — metadata only, never raw chunk text)
plus related sources when they exist — the response is never empty.

Season interaction (S14.4 / ADR-4, pinned by test_generator_season): during
``PRE_SEASON_GAP`` the current year is legitimately unpublished, and the
sanctioned behavior is the gap disclaimer over labeled prior-year data. The
year-mismatch rule therefore does NOT fire in that window; the existing season
block in ``llm_generator`` owns it.

The module is pure (no I/O at call time except the injectable conflict
provider) and is wired into ``LLMGenerator.generate_answer`` /
``generate_answer_stream`` as a pre-generation gate. See ticket 11's
"Follow-up hooks" note for the remaining main.py wiring (conflict provider).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from services._metadata_helpers import (
    get_legacy_source,
    get_notification_id,
    get_source_url,
)
from services.risky_intent import RISKY_INTENTS, infer_intent_from_query
from services.temporal_intent import TemporalIntent, classify_temporal, extract_explicit_year
from services.contact_block import _resolve_hotline

ACTION_ANSWER = "answer"
ACTION_ABSTAIN = "abstain"
ACTION_CONFLICT = "conflict"

REASON_NONE = ""
REASON_NO_EVIDENCE = "no-evidence"
REASON_YEAR_MISMATCH = "year-mismatch"
REASON_UNRESOLVED_CONFLICT = "unresolved-conflict"
REASON_LINEAGE_MISSING = "lineage-missing"
REASON_OUT_OF_SCOPE_MAJOR = "out-of-scope-major"

# Standard abstain phrasing — mirrors LLMGenerator._ABSTAIN_STRING. Duplicated
# as a plain constant so the policy stays importable without the LLM client
# stack; test_abstention_policy pins the two strings to stay identical.
ABSTAIN_STRING = (
    "Tôi không tìm thấy thông tin này trong tài liệu tuyển sinh hiện có. "
    "Bạn vui lòng liên hệ phòng tuyển sinh để được hỗ trợ: "
    "[tuyensinh.husc.edu.vn](https://tuyensinh.husc.edu.vn/thongbao.php)"
)
ABSTAIN_STRING_TOHOP = (
    "Mình chưa tìm thấy thông tin tổ hợp này trong dữ liệu tuyển sinh hiện có. "
    "Bạn vui lòng kiểm tra danh sách tổ hợp xét tuyển tại: "
    "[tuyensinh.husc.edu.vn](https://tuyensinh.husc.edu.vn/thongbao.php)"
)

_TOHOP_RX = re.compile(r"(?:tổ\s+hợp|khối)\s+[A-Z]\d{2}", re.IGNORECASE)
# Major-code token embedded in a query (same shape as major_code_validator).
_CODE_IN_TEXT_RX = re.compile(r"\b7\d{6}(?:[A-Z]{2})?\b")

# Human-friendly fact vocabulary for messages, keyed by the risky-intent label
# and the conflict-queue field vocabulary (conflict_queue.CONFLICT_FIELDS).
_FRIENDLY_INTENT = {
    "diem_chuan": "điểm chuẩn",
    "hoc_phi": "học phí",
    "chi_tieu": "chỉ tiêu tuyển sinh",
    "da_hop": "tổ hợp xét tuyển",
}
_FRIENDLY_FIELD = {
    "chi_tieu": "chỉ tiêu",
    "to_hop_codes": "tổ hợp xét tuyển",
    "hoc_phi_tin_chi_vnd": "học phí (VNĐ/tín chỉ)",
    "diem_trung_tuyen_num": "điểm trúng tuyển",
    "thoi_gian_dao_tao_nam": "thời gian đào tạo",
    "ten_nganh": "tên ngành",
}


@dataclass(frozen=True)
class EvidenceDecision:
    """Pre-generation verdict on the retrieved evidence.

    ``action`` is ``"answer"`` (generate normally), ``"abstain"`` (return
    ``message`` verbatim, no LLM call), or ``"conflict"`` (return ``message``
    presenting BOTH sides — never a pick). ``verified_parts`` holds the
    labeled evidence worth showing even in refusal (criterion 6);
    ``source_labels`` names the related sources when they exist.
    """

    action: str = ACTION_ANSWER
    reason_code: str = REASON_NONE
    message: str = ""
    verified_parts: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()


def _chunk_text(chunk: dict) -> str:
    return str(chunk.get("text") or chunk.get("summary") or "").strip()


def _chunk_year(chunk: dict) -> Optional[int]:
    """Parse the chunk's admission year; ``None`` means unknown.

    ``data_year == 0`` is the production sentinel for unknown-year rows
    (pinned by test_ingest_husc_chunks_schema / _probe_distinct_years), so
    it must NOT count as a real year: treating 0 as known would let Rule 4
    "prove" a mismatch from rows that carry no year at all.
    """
    md = chunk.get("metadata") or {}
    raw = md.get("data_year", chunk.get("data_year"))
    if raw is None or str(raw).strip() == "":
        return None
    try:
        year = int(str(raw))
    except (TypeError, ValueError):
        return None
    return year if year else None


def _has_canonical_markers(chunk: dict) -> bool:
    """True when the chunk was materialized from the canonical store.

    Legacy corpus chunks (pdf_* / qa_* / FAQ) carry no ``record_id`` at all;
    demanding canonical lineage from them would abstain on the entire legacy
    surface, which ticket 10's citation contract deliberately does not do
    (missing page/lineage is "optional o contract" there).
    """
    md = chunk.get("metadata") or {}
    return bool(md.get("record_id") or md.get("record_version_key"))


def _is_traceable(chunk: dict) -> bool:
    """A chunk is traceable when its provenance names a source document or a
    notification/URL the student can open.

    All legacy-``source`` reads go through ``_metadata_helpers`` (S13.2 grep
    gate); ``get_source_url``/``get_notification_id`` also cover the top-level
    fields that ``main._build_v2_chunks`` emits.
    """
    md = chunk.get("metadata") or {}
    return bool(
        md.get("source_document_id")
        or get_source_url(chunk)
        or get_notification_id(chunk) is not None
        or get_legacy_source(chunk)
    )


def _source_label(chunk: dict) -> str:
    """Human-readable provenance label: legacy source (e.g. "TB59 2025")
    first, then notification id, then URL. Empty string when untraceable."""
    label = (get_legacy_source(chunk) or "").strip()
    if label:
        return label
    nid = get_notification_id(chunk)
    if nid is not None:
        return f"thông báo {nid}"
    url = (get_source_url(chunk) or "").strip()
    if url:
        return url
    return ""



def _statement_text(value: Any) -> str:
    """Render one conflict statement without inventing structure."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


# ---------------------------------------------------------------------------
# Injectable conflict source (criterion 3 input at the generator seam)
# ---------------------------------------------------------------------------

_conflict_source: Optional[Callable[[], Iterable[Any]]] = None


def configure_conflict_source(source: Optional[Callable[[], Iterable[Any]]]) -> None:
    """Register a zero-arg callable returning open+resolved conflict items
    (e.g. ``ConflictQueueService.items``). ``None`` restores the default
    (no conflicts visible). Mounted at app startup by
    ``src/app/lifecycle.configure_canonical_record_store``."""
    global _conflict_source
    _conflict_source = source


def _conflicts_now() -> tuple:
    if _conflict_source is None:
        return ()
    try:
        return tuple(_conflict_source() or ())
    except Exception:
        # A broken conflict feed must not crash generation; but per criterion
        # 3's spirit, silently ignoring it would let generation self-select
        # past an unseen conflict. Callers that need hard guarantees pass
        # `unresolved_conflicts` explicitly.
        return ()


def _item_attr(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _matching_conflicts(
    conflicts: Iterable[Any],
    year: Optional[int],
    query_codes: set[str],
) -> list[Any]:
    """Open conflicts relevant to THIS query: same admission year (when the
    query is year-bound) and, when the query names major codes, a code the
    query actually mentions."""
    out: list[Any] = []
    for item in conflicts:
        if str(_item_attr(item, "status", "open")) != "open":
            continue
        if year is not None:
            try:
                if int(_item_attr(item, "admission_year", -1)) != int(year):
                    continue
            except (TypeError, ValueError):
                continue
        if query_codes:
            ma = str(_item_attr(item, "ma_nganh", "") or "")
            if ma and ma not in query_codes:
                continue
        statements = dict(_item_attr(item, "statements", {}) or {})
        if len(statements) < 2:
            continue
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Injectable release-date source (composition root only)
# ---------------------------------------------------------------------------

_release_date_source: Optional[Callable[[], Optional[str]]] = None


def configure_release_date_source(
    source: Optional[Callable[[], Optional[str]]],
) -> None:
    """Install the active-release date provider (composition root only).

    ``source`` is a zero-arg callable returning an ISO date string
    (``YYYY-MM-DD``) or a full ISO timestamp, or ``None`` when unknown.
    ``None`` (the default) restores "no date known" — the date line in
    ``refusal_footer`` is omitted entirely. Fail-soft: a provider that
    raises or returns a non-string yields ``None`` and never propagates.
    """
    global _release_date_source
    _release_date_source = source


def release_date() -> Optional[str]:
    """ISO date of the release currently being served, or None when unknown."""
    if _release_date_source is None:
        return None
    try:
        raw = _release_date_source()
    except Exception:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _format_release_date(raw: str) -> str:
    """Render ``raw`` (ISO date or full ISO timestamp) as ``dd/mm/yyyy``.

    Defensively: an unparseable value is returned as-is rather than raising.
    """
    from datetime import date, datetime

    s = raw.strip()
    try:
        d: date = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return raw
    return d.strftime("%d/%m/%Y")


def refusal_footer() -> str:
    """Uniform trailer appended to every refusal message.

    Renders the hotline + official link + optional last-updated date.
    Leading blank line included so it reads naturally when appended after
    the refusal body.
    """
    lines = [
        f"📞 Hotline: {_resolve_hotline()} (Phòng Đào tạo)",
        "🌐 https://tuyensinh.husc.edu.vn",
    ]
    date_raw = release_date()
    if date_raw:
        lines.append(f"_Thông tin cập nhật lần cuối: {_format_release_date(date_raw)}_")
    return "\n".join([""] + lines)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------



def evaluate_evidence(
    query: str,
    chunks: list[dict],
    *,
    current_year: Optional[int] = None,
    season_phase: Any = None,
    unresolved_conflicts: Optional[Iterable[Any]] = None,
    major_whitelist: Optional[Iterable[str]] = None,
) -> EvidenceDecision:
    """Decide whether ``chunks`` are sufficient evidence for ``query``.

    ``current_year`` is the live admission year; ``None`` resolves it lazily
    via ``temporal_authority`` (fail-soft). ``season_phase`` is the
    ``services.season.SeasonPhase`` for today; ``PRE_SEASON_GAP`` suppresses
    the year-mismatch rule (the gap disclaimer is the sanctioned behavior
    there). ``unresolved_conflicts`` overrides the injected conflict source.
    ``major_whitelist`` overrides the year's registry (tests inject a fake).
    """
    if current_year is None:
        try:
            from services.temporal_authority import get_current_admission_year

            current_year = int(get_current_admission_year())
        except Exception:
            current_year = 0  # unresolved: year-binding rules degrade to None

    query = query or ""
    substantive = [c for c in (chunks or []) if _chunk_text(c)]

    # Year binding (criterion 2): an explicit year, or a risky factual intent
    # which means "the current admission year". cross_year/comparison queries
    # deliberately span years -> never year-bound.
    explicit = extract_explicit_year(query)
    intent = infer_intent_from_query(query)
    temporal = classify_temporal(query, current_year) if current_year else TemporalIntent.ambiguous
    if temporal is TemporalIntent.cross_year:
        year_bound: Optional[int] = None
    elif explicit is not None:
        year_bound = explicit
    elif intent in RISKY_INTENTS:
        year_bound = current_year or None
    else:
        year_bound = None

    query_codes = {m.group(0) for m in _CODE_IN_TEXT_RX.finditer(query)}

    def _sources() -> tuple[str, ...]:
        labels = {_source_label(c) for c in substantive}
        return tuple(sorted(labels - {""}))

    # -- Rule 1: out-of-scope major (criterion 5) ----------------------------
    # Only when the query is bound to a year at-or-after the live admission
    # year: "ngành X năm 2024" is a legitimate historical question, not an
    # out-of-scope request.
    if query_codes and year_bound is not None and year_bound >= current_year:
        if major_whitelist is None:
            try:
                from services.major_code_validator import get_whitelist

                whitelist = set(get_whitelist(year_bound))
            except Exception:
                whitelist = set()
        else:
            whitelist = set(major_whitelist)
        unknown = sorted(query_codes - whitelist)
        if unknown and substantive:
            codes = ", ".join(unknown)
            return EvidenceDecision(
                action=ACTION_ABSTAIN,
                reason_code=REASON_OUT_OF_SCOPE_MAJOR,
                message=(
                    f"Mã ngành {codes} không thuộc phạm vi tuyển sinh năm "
                    f"{year_bound} hiện hành, nên hệ thống không trả lời về "
                    "ngành này. Vui lòng xem danh sách ngành tại "
                    "https://tuyensinh.husc.edu.vn/nganhdaotao-dh."
                )
                + refusal_footer(),
                verified_parts=(),
                source_labels=_sources(),
            )

    # -- Rule 2: unresolved same-year conflict (criterion 3) ------------------
    conflicts = (
        unresolved_conflicts if unresolved_conflicts is not None else _conflicts_now()
    )
    hits = _matching_conflicts(conflicts, year_bound, query_codes)
    if hits:
        parts: list[str] = []
        for item in hits:
            field = str(_item_attr(item, "field", "") or "")
            friendly = _FRIENDLY_FIELD.get(field, field or "thông tin")
            ma = str(_item_attr(item, "ma_nganh", "") or "")
            year = _item_attr(item, "admission_year", "")
            scope = f"ngành {ma} " if ma else ""
            statements = dict(_item_attr(item, "statements", {}) or {})
            sides = "; ".join(
                f"theo {src}: {_statement_text(val)}"
                for src, val in sorted(statements.items())
            )
            parts.append(f"{friendly} {scope}năm {year} — {sides}")
        labels: set[str] = set()
        for item in hits:
            labels.update(str(s) for s in (_item_attr(item, "statements", {}) or {}) if s)
        return EvidenceDecision(
            action=ACTION_CONFLICT,
            reason_code=REASON_UNRESOLVED_CONFLICT,
            message=(
                "Hai nguồn tuyển sinh chính thức đang mâu thuẫn về: "
                + " | ".join(parts)
                + ". Hệ thống không tự chọn một trong các nguồn; mâu thuẫn "
                "đang chờ cán bộ tuyển sinh xác quyết. Vui lòng liên hệ phòng "
                "tuyển sinh để có câu trả lời chính thức."
                + refusal_footer()
            ),
            verified_parts=tuple(parts),
            source_labels=tuple(sorted(labels)) or _sources(),
        )

    # -- Rule 3: missing citation lineage (criterion 4) -----------------------
    # Applies to canonical-lineage chunks only (see _has_canonical_markers).
    untraceable = [c for c in substantive if _has_canonical_markers(c) and not _is_traceable(c)]
    if untraceable:
        return EvidenceDecision(
            action=ACTION_ABSTAIN,
            reason_code=REASON_LINEAGE_MISSING,
            message=(
                ABSTAIN_STRING
                + " (Một số dữ liệu tìm được không truy vết được tới tài liệu "
                "nguồn nên không đủ điều kiện để trả lời.)"
                + refusal_footer()
            ),
            verified_parts=(),
            source_labels=(),
        )

    # -- Rule 4: year mismatch (criterion 2) ----------------------------------
    # Fires for EVERY year-bound factual query outside PRE_SEASON_GAP -
    # including risky current-year queries: the post-generation season block
    # still calls the model there, and criterion 2 requires refusing BEFORE
    # generation. The message is metadata-only (years + source labels, never
    # raw chunk text): the no-fabrication fixtures pin that a refusal must
    # not echo numerics found in old-year chunk text.
    from services.season import SeasonPhase  # local import: pure module

    in_gap = season_phase is SeasonPhase.PRE_SEASON_GAP
    if year_bound is not None and substantive and not in_gap:
        years = {_chunk_year(c) for c in substantive}
        # Fail-open when any substantive chunk carries no parseable year:
        # absence of a label is not proof of mismatch (pinned by the
        # multiturn/max-token fixtures, whose chunks have empty metadata).
        if None not in years and year_bound not in years:
            other_years = ", ".join(str(y) for y in sorted(years))
            labels = _sources()
            friendly = _FRIENDLY_INTENT.get(intent, "tuyển sinh")
            return EvidenceDecision(
                action=ACTION_ABSTAIN,
                reason_code=REASON_YEAR_MISMATCH,
                message=(
                    f"Chưa có căn cứ chính thức cho {friendly} năm "
                    f"{year_bound}; tài liệu chỉ có dữ liệu năm {other_years}"
                    + (f" (nguồn: {', '.join(labels)})" if labels else "")
                    + ". Hệ thống không tự suy đoán số liệu giữa các năm. "
                    + f"Vui lòng tham khảo {ABSTAIN_STRING.split(': ',1)[1]}"
                    + refusal_footer()
                ),
                verified_parts=tuple(f"năm {y}" for y in sorted(years)),
                source_labels=labels,
            )

    # -- Rule 5: zero evidence (criterion 1) ----------------------------------
    # Fires only for factual queries (year-bound or risky intent). A vague
    # non-factual query reaching the generator with no chunks is the
    # clarify/auto-answer path owned by the router + UW1 guard; overriding it
    # here would change a contract pinned elsewhere.
    if not substantive and (year_bound is not None or intent in RISKY_INTENTS):
        friendly = _FRIENDLY_INTENT.get(intent, "tuyển sinh")
        if year_bound is not None and current_year and year_bound < current_year:
            # Historical year absent from the corpus: honest "not found",
            # never "not published" (it WAS published, just not ingested).
            message = (
                f"Tôi không tìm thấy thông tin tuyển sinh năm {year_bound} "
                "trong tài liệu hiện có. Bạn vui lòng liên hệ phòng tuyển "
                "sinh để được hỗ trợ: [tuyensinh.husc.edu.vn]"
                "(https://tuyensinh.husc.edu.vn/thongbao.php)"
                + refusal_footer()
            )
        elif in_gap or (
            year_bound is not None and current_year and year_bound > current_year
        ):
            # Gap year or future year: the data genuinely is not published
            # yet - matches the season block's sanctioned phrasing (pinned
            # by test_generator_season cases 3/5 and durability 2099).
            year_text = year_bound if year_bound is not None else current_year
            message = (
                f"Thông tin {friendly} năm {year_text} chưa được công bố "
                "chính thức. Vui lòng tham khảo https://tuyensinh.husc.edu.vn "
                "để cập nhật."
                + refusal_footer()
            )
        elif _TOHOP_RX.search(query):
            message = ABSTAIN_STRING_TOHOP + refusal_footer()
        else:
            # Current year, in-season, nothing retrieved: the data exists in
            # the index; this query just found no grounding for it.
            message = ABSTAIN_STRING + refusal_footer()
        return EvidenceDecision(
            action=ACTION_ABSTAIN,
            reason_code=REASON_NO_EVIDENCE,
            message=message,
            verified_parts=(),
            source_labels=(),
        )

    return EvidenceDecision(action=ACTION_ANSWER, reason_code=REASON_NONE)
