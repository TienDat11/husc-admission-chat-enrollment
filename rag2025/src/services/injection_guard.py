# @spec(T20) prompt-injection guard — document content is UNTRUSTED DATA.
"""Deterministic prompt-injection defence for the HUSC admission RAG (ticket 20).

Threat model: retrieved chunks (web crawl, PDF extract, canonical records)
carry attacker-controlled text. That text must NEVER be able to change the
year filter, the lifecycle/policy decisions, or the system prompt. The user
query is likewise untrusted input and must not be able to exfiltrate system
information.

Three seams, all pure + offline (no I/O, no LLM):

1. ``detect_system_info_request`` — deterministic refusal for requests to
   leak the system prompt / instructions / API keys. Used by
   ``GuardrailService.precheck`` (chat.json path) and by
   ``SmartQueryRouter.route`` (the /v2 pre-routing gate), so the refusal
   fires regardless of the keyword fast-path.

2. ``escape_untrusted_text`` — neutralises chat-template control tokens
   (e.g. an "<|" ... "|>" style special token) and closes any dangling
   UNTRUSTED_DATA envelope, so a poisoned chunk cannot break out of the
   data block and impersonate a system/assistant turn.

3. ``restrict_to_allowed_corpus`` — drops chunks whose declared provenance
   (metadata.source_type / source_url host) is outside the allowed corpus,
   so retrieval can never leak non-HUSC content into the prompt. Chunks
   with NO provenance at all are kept (legacy rows predate schema v3);
   chunks with an EXPLICIT unknown/off-allowlist provenance are dropped.

``render_untrusted_block`` is the single place the generation prompt labels
source content: it stamps the source type and wraps the escaped text in an
UNTRUSTED_DATA marker, so the model can distinguish source content from
instructions (criterion 5).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# 1. System-information exfiltration refusal (criterion 2)
# ---------------------------------------------------------------------------

# Diacritic-insensitive fold of the query before matching, so "he thong",
# "hệ thống", "HE THỐNG" all hit the same pattern.
def _fold(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_only.lower()


# (leak-verb) + (system-object) must co-occur in the SAME query. Matching on
# a noun alone ("system prompt la gi trong tai lieu") would over-block; a
# verb alone ("chay thu lenh he thong") would too. Both together, with the
# object within ~40 chars after the verb, is the exfiltration shape.
_LEAK_VERB_RX = re.compile(
    r"(?:show|print|repeat|reveal|leak|output|display|paste|dump|"
    r"cho\s+toi\s+xem|cho\s+xem|cho\s+me\s+|hiển\s+thị|in\s+ra|liệt\s+kê|"
    r"lặp\s+lại|lặp\s+laị|bật\s+mí|tiết\s+lộ|công\s+bo|cho\s+biết)"
    r"[\s\S]{0,40}?"
    r"(?:system\s*prompt|prompt\s*h(?:ệ|e)\s*th(?:ố|o)ng|instruction|nhắc\s+lại|"
    r"quy\s*tắc\s*nội\s*bo|nội\s*quy|api\s*_?key|secret|b\u00ed\s*m(?:ật|at)|"
    r"m\u00e3\s*khau|mật\s+khẩu|password|he\s*thong\s*cua\s*ban|your\s*(?:system|instructions|prompt))",
    re.IGNORECASE,
)

# English "ignore previous instructions" and the Vietnamese override family.
# These are injection markers, not leak requests, but the same gate refuses
# them: answering "instructions ignored" is itself a system-info leak.
_OVERRIDE_RX = re.compile(
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?"
    r"(?:previous|prior|above|earlier|trước\s+đ[oó]|trc)\w*"
    r"[\s\S]{0,30}?(?:instruction|prompt|rule|quy\s*t|ch\u1ec9\s*th)",
    re.IGNORECASE,
)
_OVERRIDE_VI_RX = re.compile(
    r"(?:bỏ\s+qua|bo\s+qua|quên\s+(?:hết|tất|ca)\w*|quen\s+mat|phớt\s+lờ|phot\s+lo|"
    r"làm\s+theo\s+không|kh[oô]ng\s+cần\s+theo)\s+"
    r"[\s\S]{0,30}?(?:hướng\s+dẫn|chi\s+thi|chỉ\s+thi|quy\s+tắc|noi\s+quy|system|prompt|lệnh)",
    re.IGNORECASE,
)

# "Bạn là AI, vậy system prompt của bạn là gì?" style direct questions about
# the system itself, without needing a leak verb.
_SYSTEM_QUESTION_RX = re.compile(
    r"(?:system\s*prompt|prompt\s*h(?:ệ|e)\s*th(?:ố|o)ng|quy\s*tắc\s*(?:bảo\s*mật|an\s*toàn)|"
    r"nội\s*quy\s*(?:của|cua)\s*bot|instruction\s*của\s*bạn)"
    r"[\s\S]{0,30}?(?:là\s+gì|la\s+gi|gì\s*\?|hiển\s+thị|show|what|which|\?)|"
    r"(?:bạn|ban)\s*(?:được|duoc)?\s*(?:huấn\s+luyện|huan\s+luyen|lập\s+trình)\s+[\s\S]{0,20}?(?:như\s+thế\s+nào|the\s+nao|ra\s+sao)",
    re.IGNORECASE,
)


def detect_system_info_request(query: str) -> bool:
    """True when the query asks the bot to reveal/override system internals.

    Deterministic (regex on a diacritic-folded copy), so it fires before the
    admission-keyword fast-path and can never be talked out of it by an LLM.
    """
    if not isinstance(query, str) or not query.strip():
        return False
    folded = _fold(query)
    # Re-fold after stripping combining marks so patterns can be written
    # ASCII-only.
    if _LEAK_VERB_RX.search(folded) or _LEAK_VERB_RX.search(query):
        return True
    if _OVERRIDE_RX.search(folded) or _OVERRIDE_RX.search(query):
        return True
    if _OVERRIDE_VI_RX.search(folded) or _OVERRIDE_VI_RX.search(query):
        return True
    if _SYSTEM_QUESTION_RX.search(folded) or _SYSTEM_QUESTION_RX.search(query):
        return True
    return False


SYSTEM_INFO_REFUSAL_ANSWER = (
    "Mình không thể tiết lộ hoặc thay đổi nội dung hệ thống (system prompt, "
    "quy tắc nội bộ, khóa API). Mình chỉ trả lời câu hỏi tuyển sinh của "
    "Trường Đại học Khoa học — Đại học Huế (HUSC) dựa trên tài liệu công khai. "
    "Bạn cứ hỏi về ngành, điểm chuẩn, học phí, hồ sơ nhé!"
)

# ---------------------------------------------------------------------------
# 2. Untrusted-data escaping (criteria 1, 3, 5)
# ---------------------------------------------------------------------------

# Delimiter tags: ANY `<...>` token whose first character marks it as a tag
# (`letter`, `/`, or `|`) is neutralised. This covers the chat-template
# control tokens (`<|im_end|>`, ``), the envelope closer
# (`</UNTRUSTED_DATA>`), a fake opener (`<UNTRUSTED_DATA source="system">`),
# and the prompt's own structural tags (`<CONTEXT>`, `</QUESTION>`) — a
# poisoned chunk must never be able to close its block early or open a
# trusted-looking one. Legitimate admission documents contain no tag-shaped
# tokens worth preserving, so the replacement is lossless for behaviour.
DELIMITER_TAG_RX = re.compile(r"<[A-Za-z_/|][^<>]{0,120}>")

_TAG_REPLACEMENT = "[blocked-tag]"

# Zero-width / bidi characters are used to split control tokens past naive
# filters ("<\u200b|im_end|>") and are invisible to human reviewers. They
# carry no meaning in admission prose, so they are stripped BEFORE the tag
# filter, and the filter runs twice so stripping cannot itself create a tag.
_INVISIBLE_RX = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e]")


def escape_untrusted_text(text: str) -> str:
    """Neutralise every delimiter tag in source content (envelope-safe).

    Purely textual and idempotent: the replacement contains no angle
    brackets, so escaping already-escaped text is a no-op. That matters
    because the generation path may render a chunk twice (context +
    citation preview).
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    out = _INVISIBLE_RX.sub("", text)
    out = DELIMITER_TAG_RX.sub(_TAG_REPLACEMENT, out)
    out = DELIMITER_TAG_RX.sub(_TAG_REPLACEMENT, out)
    return out


def _sanitize_label(label: str) -> str:
    """Strip tag/quote characters from the source label in the opening tag."""
    return re.sub(r"[<>\"'\n\r]", "", str(label or "unknown"))


def render_untrusted_block(label: str, content: str) -> str:
    """Render one source excerpt as explicitly-marked untrusted data.

    The header states the source type so the model can distinguish source
    content from instructions (criterion 5); the body is escaped so it can
    neither close the envelope early nor open a fake one (criterion 1).
    """
    escaped = escape_untrusted_text(content)
    return (
        f"<UNTRUSTED_DATA source=\"{_sanitize_label(label)}\">\n"
        f"{escaped}\n"
        f"</UNTRUSTED_DATA>"
    )



# ---------------------------------------------------------------------------
# 3. Corpus containment (criterion 4)
# ---------------------------------------------------------------------------

# Provenance values the ingest parsers write (chunk_table_aware_v2.
# detect_source_type, normalize_data, canonical_chunk_builder). Anything
# else — "unknown", a typo, an attacker-forced value — is NOT trusted.
ALLOWED_SOURCE_TYPES: Tuple[str, ...] = (
    "web_crawl",
    "pdf_extract",
    "canonical_record",
    "jsonl",
    "json",
    "txt",
    "md",
    "faq",
)

# Hosts whose pages are HUSC corpus. Mirrors _URL_ALLOWLIST in
# llm_generator (tuyensinh.husc.edu.vn, husc.edu.vn) plus the ministry
# domains the contact-abstain guard already treats as canonical.
ALLOWED_SOURCE_HOSTS: Tuple[str, ...] = (
    "tuyensinh.husc.edu.vn",
    "husc.edu.vn",
    "dkxt.hueuni.edu.vn",
    "thisinh.thitotnghiepthpt.edu.vn",
)


def _chunk_source_type(chunk: Dict[str, Any]) -> Optional[str]:
    md = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    raw = md.get("source_type") or chunk.get("source_type")
    return str(raw).strip().lower() if raw else None


def _chunk_host(chunk: Dict[str, Any]) -> Optional[str]:
    # Local import: keeps this module import-clean for scripts that only
    # need the escape/refusal seams.
    from src.services._metadata_helpers import get_source_url

    url = get_source_url(chunk)
    if not url:
        return None
    try:
        host = urlsplit(url if "//" in url else f"//{url}").hostname or ""
    except ValueError:
        return None
    return host.lower()


def restrict_to_allowed_corpus(
    chunks: List[Dict[str, Any]],
    allowed_source_types: Tuple[str, ...] = ALLOWED_SOURCE_TYPES,
    allowed_hosts: Tuple[str, ...] = ALLOWED_SOURCE_HOSTS,
) -> List[Dict[str, Any]]:
    """Drop chunks whose DECLARED provenance is outside the allowed corpus.

    Rules (conservative on purpose — a false drop loses a legitimate source,
    a false keep leaks one chunk of data):
      * metadata.source_type present but not in the allowlist → drop.
      * A URL-like source (source_url / legacy source) whose host is not in
        the host allowlist → drop.
      * No source_type AND no host at all → keep (legacy rows predate the
        v3 schema; corpus-wide they are HUSC documents).

    Returns the kept chunks; the caller never sees the dropped list unless
    it asks (kept for audit logging via the returned dict shape below).
    """
    kept: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        st = _chunk_source_type(chunk)
        if st is not None and st not in allowed_source_types:
            continue
        host = _chunk_host(chunk)
        if host is not None and not any(host == h or host.endswith(f".{h}") for h in allowed_hosts):
            continue
        kept.append(chunk)
    return kept


# ---------------------------------------------------------------------------
# 4. Instruction-marker detection for fixtures (criterion 1/6 evidence)
# ---------------------------------------------------------------------------

# Phrases an adversarial document uses to hijack the model. Used by the eval
# runner to assert that a fixture really carries an injection payload (so a
# passing test can't be vacuous) and that the payload only ever appears in
# escaped/quarantined form.
INSTRUCTION_LIKE_RX = re.compile(
    r"(?:ignore|quên|bỏ\s+qua)[\s\S]{0,25}?(?:instruction|hướng\s+dẫn|chi\s+thi|prompt)|"
    r"(?:system\s+prompt|system_instruction)|"
    r"(?:bạn\s+(?:phải|hay|hãy)|you\s+(?:must|should))\s+[\s\S]{0,40}?"
    r"(?:trả\s+lời|tra\s+loi|answer|reply|respond|dùng|use)\s+[\s\S]{0,25}?(?:năm|year|20\d\d)|"
    r"(?:đặt|đổi|thay|override|set|change)\s+[\s\S]{0,25}?(?:năm|year|filter|lifecycle|chính\s+sách|policy)"
    r"[\s\S]{0,20}?(?:20\d\d|khác|other|mới|new)?",
    re.IGNORECASE,
)


def detect_instruction_like(text: str) -> bool:
    """True when document text looks like an instruction aimed at the model."""
    if not isinstance(text, str) or not text:
        return False
    return bool(INSTRUCTION_LIKE_RX.search(text) or INSTRUCTION_LIKE_RX.search(_fold(text)))


def find_control_tokens(text: str) -> List[str]:
    """List delimiter-tag occurrences (for eval assertions)."""
    if not isinstance(text, str):
        return []
    return DELIMITER_TAG_RX.findall(text)
