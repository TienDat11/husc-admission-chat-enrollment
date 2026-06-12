"""
Smart Query Router – HyDE + Step-Back Prompting + Query Classification

Pipeline (per query) — L3 single-call rewrite:
1. **HyDE Vague-Detection**: nếu câu hỏi quá mơ hồ (placeholder, độ dài <4 từ
   không có entity, "...") → reject hoặc trả meta-answer (link tuyensinh) thay vì
   chạy retrieval rỗng → tránh tốn 130-200s LLM cho câu user chưa cung cấp ý.
2. **HyDE Contact/Intro Block**: với câu contact/intro thuần (hotline, link
   website, địa chỉ, fanpage, "giới thiệu về trường", "thông tin chung")
   trả CONTACT_BLOCK year-agnostic (URL + hotline, NO majors/codes/tuition).
   Câu hỏi về NGÀNH ("trường có ngành gì", "tìm hiểu các ngành") rơi xuống
   retrieval thật (enumeration→graph_rag) — kills the structural hallucination
   previously emitted by the hardcoded 28-ngành list. (ADR-3 / S14.5)
3. Step-Back + Classify in ONE call: L3 merges step-back into the classify
   prompt; HyDE is dropped (the hypothetical_doc no longer exists, just
   echoes query so downstream embedding stays unchanged). Net: ONE LLM
   call per query instead of the pre-L3 chain (step_back → hyde +
   classify) — cuts wall-time by ~7s on weak net without re-bundling
   HyDE into classify (which was the S16 over-routing trigger).
4. Query Classification: decide routing
   - "padded"     → PaddedRAG (vector + BM25 + cross-encoder, 1-hop)
   - "hybrid"     → vector + PPR fusion + expander M=5
   - "graph"      → GraphRAG (PPR + entity graph traversal, M=8)

Vietnamese domain: HUSC admission system.

Design: Stateless service. All LLM calls go through UnifiedLLMClient.
Scalable: Adding new query types = extend the CLASSIFY_SYSTEM_PROMPT only.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import re

from loguru import logger

from src.services.llm_client import UnifiedLLMClient, get_llm_client

# Keywords that indicate enumeration/listing queries → always graph_rag
_ENUMERATION_PATTERNS = re.compile(
    r"(bao\s+nhiêu\s+ngành|tất\s+cả\s+các\s+ngành|danh\s+sách\s+ngành|liệt\s+kê|"
    r"các\s+ngành\s+có|ngành\s+nào\s+có|các\s+ngành\s+thuộc|bao\s+nhiêu\s+chương\s+trình|"
    r"toàn\s+bộ\s+ngành|danh\s+sách\s+các\s+ngành|các\s+ngành\s+đào\s+tạo|"
    r"ngành\s+nào\s+đào\s+tạo|có\s+mấy\s+ngành|có\s+bao\s+nhiêu\s+ngành)",
    re.IGNORECASE,
)

# Patterns signaling câu hỏi SO SÁNH / ĐA-THỰC-THỂ → always graph_rag
# (S15.3 / ADR-C). Covers so sánh / khác nhau / so với / hơn / kém / chênh lệch
# and the "A hay B" disjunction. Includes diacritic-stripped variants so common
# typo-input ("so sanh", "khac nhau") still trips the override. Word boundaries
# guard `hơn` / `kém` / `hay` from matching as substrings of other words.
_COMPARISON_PATTERNS = re.compile(
    r"(so\s+s[áa]nh|"            # so sánh / so sanh
    r"kh[áa]c\s+nhau|"            # khác nhau / khac nhau
    r"so\s+v[ớo]i|"               # so với / so voi
    r"ch[êe]nh\s+l[ệe]ch|"        # chênh lệch / chenh lech
    r"\bh[ơo]n\b|"                # hơn / hon (standalone word)
    r"\bk[éẻe]m\b|"               # kém / kem (standalone word)
    r"\bhay\b)",                   # "A hay B" disjunction (standalone word)
    re.IGNORECASE,
)

# Patterns signaling câu hỏi VỀ NGÀNH (majors) — must flow through RETRIEVAL
# (enumeration→graph_rag downstream). These are EXCLUDED from the contact/intro
# auto-answer gate so the pipeline does real retrieval+generation instead of
# returning a static fabricated list. Kills the structural Type-4/5/6
# hallucination previously emitted by HYDE_META_ANSWER_INTRO.
_MAJOR_QUESTION_PATTERNS = re.compile(
    r"(trường\s+có\s+ngành\s+gì|trường\s+có\s+những\s+ngành|"
    r"tìm\s+hiểu\s+(?:về\s+)?các\s+ngành|ngành\s+gì|có\s+những\s+ngành)",
    re.IGNORECASE,
)

# Patterns that signal câu hỏi GỢI Ý pure contact/intro — return year-agnostic
# CONTACT_BLOCK (no majors, no codes, no tuition, no year literal). This
# replaces the old HYDE_META_ANSWER_INTRO static list.
_HYDE_CONTACT_PATTERNS = re.compile(
    r"(trang\s+web\s+của\s+trường|"
    r"link\s+(?:của\s+)?trường|website\s+(?:của\s+)?trường|"
    r"thông\s+tin\s+chung|trường\s+ở\s+đâu|địa\s+chỉ\s+(?:của\s+)?trường|"
    r"liên\s+hệ\s+(?:với\s+)?trường|hotline|"
    r"có\s+group|có\s+zalo|có\s+facebook|fanpage|"
    r"giới\s+thiệu\s+về\s+trường)",
    re.IGNORECASE,
)

# Patterns chỉ rõ câu hỏi QUÁ MƠ HỒ (≤2 token có nghĩa, placeholder, không entity) → reject
_VAGUE_PLACEHOLDER_PATTERNS = re.compile(
    r"(\.{3,}|…|_{3,}|"
    r"^(?:em|e|cho\s+e|cho\s+em|hỏi|cho\s+hỏi|ad)\s*[?.!]*\s*$|"
    r"^(?:gì|cái\s+gì|sao|thế\s+nào|như\s+thế\s+nào)\s*[?]?\s*$|"
    r"^(?:hi|hello|chào|alo|test)\s*[?.!]*\s*$)",
    re.IGNORECASE,
)


# Year-agnostic contact block — replaces the old HYDE_META_ANSWER_INTRO that
# hardcoded majors, fabricated major codes, tuition numbers, and a wrong phone.
# Pure contact/intro queries (hotline, website, address, fanpage, "giới thiệu
# về trường", "thông tin chung") return this block instead. Kills both the
# current-year hallucination source AND next-year's stale-fact source.
#
# G1-T5 (durability S14.x): the hotline is no longer baked into the
# template. It is sourced (in priority order) from:
#   1. HUSC_HOTLINE env var (operator override, e.g. CI / staged deploy)
#   2. /api/meta payload (`MetaResponse.contact_hotline`) when available
#   3. _HOTLINE_FALLBACK below (the previous literal — preserves prod
#      behavior when the data-driven paths return nothing).
# Use `build_contact_block(meta=...)` to render. `CONTACT_BLOCK` is
# retained as a module-level pre-rendered fallback for callers that
# can't supply a meta payload (auto-answer path).
import os

_HOTLINE_FALLBACK = "0234.3823290"  # Phòng Đào tạo (HUSC)


def _resolve_hotline(meta: Optional[dict] = None) -> str:
    """Pick the best available hotline (env > meta > fallback).

    Args:
        meta: optional dict (e.g. /api/meta response). If it contains a
            non-empty `contact_hotline` key, that value wins over the
            fallback but is overridden by HUSC_HOTLINE env.

    Returns:
        The hotline string to embed in CONTACT_BLOCK.
    """
    env = os.getenv("HUSC_HOTLINE")
    if env and env.strip():
        return env.strip()
    if meta:
        v = meta.get("contact_hotline")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return _HOTLINE_FALLBACK


def build_contact_block(meta: Optional[dict] = None) -> str:
    """Render the CONTACT_BLOCK with the resolved hotline.

    `meta` is the /api/meta response (or a subset dict containing
    `contact_hotline`). When None or empty, the env-var or fallback
    literal is used.
    """
    hotline = _resolve_hotline(meta)
    return (
        "Bạn có thể tra cứu thông tin tuyển sinh chính thức của Trường Đại học Khoa học - "
        "Đại học Huế (HUSC) tại:\n\n"
        "🌐 Cổng tuyển sinh: https://tuyensinh.husc.edu.vn\n"
        "🌐 Trang chủ: https://husc.edu.vn\n"
        f"📞 Hotline: {hotline} (Phòng Đào tạo)\n\n"
        "Bạn muốn biết cụ thể về ngành nào, điểm chuẩn, học phí, tổ hợp xét tuyển hay phương thức xét tuyển? Mình sẽ tra cứu chi tiết giúp bạn."
    )


# Pre-rendered default (env not set, no meta) — matches the pre-fix
# hardcoded string so legacy callers see no behavior change.
CONTACT_BLOCK = build_contact_block()


HYDE_REJECT_VAGUE = (
    "Câu hỏi của bạn hiện chưa đủ rõ để mình tra cứu chính xác. "
    "Bạn vui lòng cho biết cụ thể hơn về:\n"
    "- **Ngành** quan tâm (VD: CNTT, Kỹ thuật phần mềm, Báo chí…)\n"
    "- **Phương thức xét tuyển** (THPT, học bạ, ĐGNL, kết hợp)\n"
    "- **Thông tin cần biết** (điểm chuẩn, học phí, tổ hợp, chỉ tiêu)\n\n"
    "Ví dụ: *\"Điểm chuẩn ngành CNTT năm hiện tại là bao nhiêu?\"* hoặc "
    "*\"Học phí ngành Kỹ thuật phần mềm năm nay?\"*"
)


class QueryRoute(str, Enum):
    """Routing decision for a query.

    Three branches (S16.1 / Wave A / AMF-1):
      - PADDED_RAG: 1-hop single-fact → pure vector (NO PPR fusion, NO expander).
      - HYBRID:     default for most admission queries → vector + PPR fusion +
                    expander M=5 (the prior graph behavior, default budget).
      - GRAPH_RAG:  genuine multi-hop / comparison / enumeration → vector +
                    PPR fusion + expander M=8 + multi-seed comparison bump.
    """
    PADDED_RAG = "padded_rag"     # Simple 1-hop → vector search sufficient
    HYBRID = "hybrid"             # Default → vector + light PPR fusion (M=5)
    GRAPH_RAG = "graph_rag"       # Multi-hop / comparative → PPR graph needed (M=8)


@dataclass
class RouterResult:
    """Full result of query routing.

    Attributes:
        original_query: Raw user input.
        step_back_query: Abstracted / principled version of the query.
        hypothetical_doc: LLM-generated fake answer document (for HyDE embedding).
        hyde_variants: List of query variants (original + step_back + rephrased).
        route: Routing decision (padded_rag or graph_rag).
        complexity: Complexity score 1-5 (higher → more graph-intensive).
        intent: Detected intent category.
        reasoning: Brief explanation for routing decision.
        auto_answer: Static answer trả ngay (HyDE auto-answer / vague reject)
            khi pipeline có thể bỏ qua retrieval/generation. None = chạy full pipeline.
        skip_retrieval: True nếu auto_answer được set → main pipeline trả luôn.
    """
    original_query: str
    step_back_query: str
    hypothetical_doc: str
    hyde_variants: List[str]
    route: QueryRoute
    complexity: int
    intent: str
    reasoning: str
    auto_answer: Optional[str] = None
    skip_retrieval: bool = False

    @property
    def all_queries_for_embedding(self) -> List[str]:
        """All text to embed for retrieval (original + hyde variants)."""
        seen = set()
        result = []
        for q in [self.original_query, self.step_back_query] + self.hyde_variants:
            if q and q not in seen:
                seen.add(q)
                result.append(q)
        return result


# ─── Entity-counter for CMF-5 padded-safety demote ─────────────────────────
# Reuses the same NGANH-keyword list and TO_HOP regex that NERService uses,
# so the router's notion of "≥2 NGANH/TO_HOP entities" matches the graph's
# seeding notion. Counts distinct normalized entity ids.
from src.services.ner_service import (  # noqa: E402
    _NGANH_KEYWORDS as _NER_NGANH_KEYWORDS,
    _TO_HOP_PATTERN as _NER_TO_HOP_PATTERN,
)
_NGANH_ENTITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _NER_NGANH_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _count_nganh_tohop_entities(query: str) -> int:
    """Return the number of DISTINCT NGANH + TO_HOP entities found in the
    query (case-insensitive, normalized).

    Used by CMF-5 padded-safety demote: padded + ≥2 such entities → hybrid.
    Same regex as NERService.extract_from_query — single source of truth.
    """
    found: set = set()
    for m in _NGANH_ENTITY_RE.finditer(query):
        found.add(("NGANH", m.group(0).lower()))
    for m in _NER_TO_HOP_PATTERN.finditer(query):
        found.add(("TOHOP", m.group(0).upper()))
    return len(found)


# ─── System Prompts ────────────────────────────────────────────────────────────

_STEP_BACK_SYSTEM = """Bạn là chuyên gia giáo dục đại học Việt Nam, am hiểu hệ thống tuyển sinh HUSC (Đại học Khoa học Huế).

**Kỹ thuật Step-Back Prompting**: Từ câu hỏi cụ thể, hãy rút ra câu hỏi TỔNG QUÁT hơn về nguyên lý/khái niệm nền tảng.

**QUY TẮC BẢO TOÀN THỰC THỂ** (BẮT BUỘC):
- PHẢI giữ nguyên các thực thể ràng buộc trong câu hỏi tổng quát:
  - MÃ NGÀNH (định dạng 7 chữ số bắt đầu bằng 7, ví dụ minh họa: <MÃ_NGÀNH>),
    NĂM tuyển sinh (ví dụ minh họa: <NĂM>),
    TỔ HỢP (VD: A00, B00, C01), ĐỐI TƯỢNG (VD: học sinh THPT, người đã tốt nghiệp).
- CHỈ trừu tượng hóa Ý ĐỊNH (intent), KHÔNG trừu tượng hóa các ràng buộc.

Ví dụ:
- Cụ thể: "Ngành CNTT tại HUSC điểm chuẩn năm <NĂM> là bao nhiêu?"
- Tổng quát: "Quy trình xác định điểm chuẩn xét tuyển đại học ngành CNTT năm <NĂM> tại Việt Nam hoạt động như thế nào?"

- Cụ thể: "Để học ngành Kỹ thuật phần mềm cần tổ hợp A00 năm <NĂM>?"
- Tổng quát: "Các tổ hợp A00 xét tuyển vào ngành Kỹ thuật phần mềm năm <NĂM> tại các trường đại học Việt Nam thường bao gồm những gì?"

CHỈ trả về câu hỏi tổng quát, không giải thích, không markdown."""


# Pattern for short single-fact lookups (≤5 word "tokens" with a major signal).
# Used by `_should_stepback` to decide if abstracting the query would HURT
# retrieval (i.e. lose the constraint that makes the lookup 1-hop).
_MAJOR_TOKEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _NER_NGANH_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_YEAR_4DIGIT_RE = re.compile(r"\b20\d{2}\b")


def _should_stepback(query: str) -> bool:
    """Decide whether to run the step-back abstraction on a query.

    Returns False (skip step-back, use raw query as step_back_query) when:
      - the query has BOTH a 4-digit year (20xx) AND a major signal, OR
      - the query is a short single-fact lookup (≤6 word tokens with a major).

    Returns True (run step-back) otherwise — for abstract concepts,
    procedures, generic policy questions.

    Why: for a 1-hop factual lookup like "học phí ngành CNTT 2026", the
    year + major IS the constraint that makes the lookup 1-hop. Stepping
    back to "học phí đại học tại Việt Nam" loses the constraint and the
    embedding becomes generic. We skip the abstraction to keep recall.
    """
    if not query:
        return True
    has_year = bool(_YEAR_4DIGIT_RE.search(query))
    has_major = bool(_MAJOR_TOKEN_RE.search(query))
    if has_year and has_major:
        return False
    # Short single-fact lookup (≤6 word tokens) with at least one major signal.
    # Long queries (procedures, multi-clause) still step back.
    word_count = len(re.findall(r"[\wÀ-ỹ]+", query, flags=re.UNICODE))
    if has_major and word_count <= 6:
        return False
    return True


_HYDE_SYSTEM = """Bạn là chuyên gia tư vấn tuyển sinh Đại học Khoa học Huế (HUSC).

**Kỹ thuật HyDE**: Tạo MỘT đoạn văn bản GIẢ ĐỊNH ngắn (≤40 từ) nghe như trích đoạn từ tài liệu tuyển sinh, có thể trả lời câu hỏi.

Đoạn này KHÔNG cần chính xác — mục tiêu là tạo embedding khớp với corpus.

QUY TẮC:
- Bằng tiếng Việt, phong cách hành chính/tuyển sinh.
- Dùng từ vựng tuyển sinh: "chỉ tiêu", "tổ hợp", "điểm chuẩn", "học phí tín chỉ"…
- TUYỆT ĐỐI KHÔNG bịa URL, số điện thoại, email, mã ngành cụ thể.
- KHÔNG dùng markdown, không giải thích, không tiêu đề.
- CHỈ trả về đúng đoạn văn bản giả định (một passage duy nhất)."""

_CLASSIFY_SYSTEM = """Bạn là hệ thống phân loại câu hỏi tuyển sinh đại học Việt Nam (HUSC - Đại học Khoa học Huế).

Phân tích câu hỏi và trả về JSON DUY NHẤT với cấu trúc:
{
  "route": "padded" | "hybrid" | "graph",
  "complexity": 1-5,
  "intent": "diem_chuan" | "hoc_phi" | "nganh_hoc" | "to_hop" | "chinh_sach" | "thu_tuc" | "so_sanh" | "da_hop" | "liet_ke" | "general",
  "reasoning": "lý do ngắn gọn (≤20 từ)"
}

QUY TẮC PHÂN LOẠI (theo thứ tự ưu tiên, rule nào khớp TRƯỚC thì thắng):
1) SO SÁNH / ĐA-THỰC-THỂ / QUAN HỆ / LIỆT KÊ nhiều ngành
   → route = "graph", complexity = 4-5
   Khớp: "so sánh", "khác nhau", "so với", "chênh lệch", "hơn", "kém",
   "A hay B", "so sánh ngành X và ngành Y", "liệt kê các ngành",
   "danh sách ngành", "bao nhiêu ngành", "các ngành đào tạo".
2) TRA CỨU 1 THỰC THỂ + 1 THUỘC TÍNH (câu hỏi 1-hop đơn giản)
   → route = "padded", complexity = 1
   Khớp: "học phí ngành X?", "địa chỉ trường?", "mã trường HUSC?",
   "hotline bao nhiêu?", "điểm chuẩn ngành X năm <NĂM>?" (nếu chỉ 1 ngành + 1 năm),
   "tổ hợp xét tuyển ngành Y?", "chỉ tiêu ngành Z năm nay?".
3) MỌI TRƯỜNG HỢP KHÁC (kể cả uncertain, multi-aspect policy, generic procedures)
   → route = "hybrid", complexity = 2-3
   Khớp: "cách xét học bạ + thời gian", "điều kiện xét tuyển kết hợp",
   "quy trình đăng ký", "thủ tục nộp hồ sơ", "thời gian xét tuyển",
   "lệ phí xét tuyển", "chính sách ưu tiên".

COMPLEXITY:
1 = tra cứu 1 thông tin         4 = liên kết 3+ thực thể
2 = tra cứu 2 thông tin         5 = so sánh / phân tích đa chiều / liệt kê nhiều thực thể
3 = liên kết 2 thực thể

VÍ DỤ FEW-SHOT (4 ví dụ - phân bố nhãn hợp lý, KHÔNG >40% cho 1 route):
- "học phí ngành CNTT?" → {"route":"padded","complexity":1,"intent":"hoc_phi","reasoning":"1 ngành, 1 thuộc tính"}
- "địa chỉ trường Đại học Khoa học Huế?" → {"route":"padded","complexity":1,"intent":"general","reasoning":"tra cứu địa chỉ"}
- "cách xét học bạ và thời gian xét tuyển?" → {"route":"hybrid","complexity":2,"intent":"chinh_sach","reasoning":"2 thuộc tính policy"}
- "so sánh ngành CNTT và ngành Khoa học dữ liệu?" → {"route":"graph","complexity":5,"intent":"so_sanh","reasoning":"so sánh đa thực thể"}
- "liệt kê các ngành đào tạo của trường?" → {"route":"graph","complexity":5,"intent":"liet_ke","reasoning":"enumeration"}

CHỈ trả về JSON hợp lệ, không có text bên ngoài. KHÔNG bịa thêm field ngoài schema."""


_CLASSIFY_STEPBACK_SYSTEM = """Bạn là hệ thống phân loại câu hỏi tuyển sinh đại học Việt Nam (HUSC - Đại học Khoa học Huế).

Phân tích câu hỏi và trả về JSON DUY NHẤT với cấu trúc:
{
  "route": "padded" | "hybrid" | "graph",
  "complexity": 1-5,
  "intent": "diem_chuan" | "hoc_phi" | "nganh_hoc" | "to_hop" | "chinh_sach" | "thu_tuc" | "so_sanh" | "da_hop" | "liet_ke" | "general",
  "reasoning": "lý do ngắn gọn (≤20 từ)",
  "step_back": "câu hỏi đã được trừu tượng hóa về nguyên lý/khái niệm nền tảng (tiếng Việt). Nếu câu hỏi đã là single-fact lookup cụ thể (đã có NGÀNH + NĂM ràng buộc), echo nguyên câu hỏi."
}

QUY TẮC ROUTE (theo thứ tự ưu tiên):
1) SO SÁNH / ĐA-THỰC-THỂ / QUAN HỆ / LIỆT KÊ nhiều ngành → route = "graph", complexity = 4-5
2) TRA CỨU 1 THỰC THỂ + 1 THUỘC TÍNH (1-hop đơn giản) → route = "padded", complexity = 1
3) MỌI TRƯỜNG HỢP KHÁC → route = "hybrid", complexity = 2-3

COMPLEXITY:
1 = tra cứu 1 thông tin         4 = liên kết 3+ thực thể
2 = tra cứu 2 thông tin         5 = so sánh / phân tích đa chiều / liệt kê nhiều thực thể
3 = liên kết 2 thực thể

QUY TẮC STEP_BACK (BẮT BUỘC — bảo toàn ràng buộc):
- PHẢI giữ nguyên các thực thể ràng buộc: MÃ NGÀNH (7 chữ số bắt đầu bằng 7), NĂM (20xx), TỔ HỢP (A00, B00, C01…), ĐỐI TƯỢNG.
- CHỈ trừu tượng hóa Ý ĐỊNH (intent), KHÔNG trừu tượng hóa ràng buộc.
- Nếu câu hỏi đã là single-fact lookup (1 NGÀNH + 1 NĂM) → echo nguyên câu hỏi, KHÔNG trừu tượng.

Ví dụ:
- "Ngành CNTT tại HUSC điểm chuẩn năm 2026?" → step_back = "Quy trình xác định điểm chuẩn xét tuyển ngành CNTT năm 2026 tại Việt Nam hoạt động như thế nào?"
- "học phí ngành CNTT 2026?" → step_back = "học phí ngành CNTT 2026?" (echo — đã có ràng buộc)
- "cách xét học bạ và thời gian xét tuyển?" → step_back = "Quy trình và thời gian xét tuyển bằng học bạ tại đại học Việt Nam"

CHỈ trả về JSON hợp lệ, không có text bên ngoài. KHÔNG bịa thêm field ngoài schema."""


class SmartQueryRouter:
    """Routes queries to PaddedRAG or GraphRAG using HyDE + Step-Back.

    Pipeline per query:
    1. Step-Back: abstract to principle-level question
    2. HyDE: generate hypothetical document for embedding
    3. Classify: route to padded_rag or graph_rag

    All LLM calls go through UnifiedLLMClient (gemini-2.5-flash primary).

    Args:
        llm: UnifiedLLMClient instance (injected or auto-created).
        simple_complexity_threshold: Queries with complexity ≤ this → padded_rag.
    """

    def __init__(
        self,
        llm: Optional[UnifiedLLMClient] = None,
        simple_complexity_threshold: int = 2,
    ) -> None:
        self._llm = llm or get_llm_client()
        self._threshold = simple_complexity_threshold

    async def _classify_combined(self, query: str) -> Dict:
        """Single LLM call: classify route + emit step_back in one JSON.

        L3: replaces the pre-L3 `_chain_stepback_hyde` (step_back) +
        `_generate_hyde_doc` (hyde) + `_classify` chain with a single
        `chat_json` round-trip. Halves the LLM cost (one call instead of
        three) and removes the HyDE-doc→classify injection that caused the
        S16 over-routing regression (re-bundling HyDE into classify is
        what triggered S16; we DROP HyDE entirely instead).

        Returns a dict with: route, complexity, intent, reasoning, step_back.
        On failure returns the safe-middle default (S16.2 / CMF-1 (d)).
        """
        user_msg = f"Câu hỏi gốc: {query}"
        try:
            data = await self._llm.chat_json(
                user_message=user_msg,
                system_message=_CLASSIFY_STEPBACK_SYSTEM,
                temperature=0.1,
                max_tokens=512,
            )
            return data
        except Exception as exc:
            logger.warning(
                f"Combined classify failed: {exc} – defaulting to hybrid "
                "(S16.2 fail-toward-safe-middle)"
            )
            return {
                "route": "hybrid",
                "complexity": 2,
                "intent": "general",
                "reasoning": "classification failed — default hybrid (S16.2)",
                "step_back": query,
            }

    async def _chain_stepback_hyde(self, query: str) -> tuple[str, str]:
        """DEPRECATED in L3: route() no longer calls this. Retained for
        back-compat with any external test/caller that still patches or
        awaits it. New code path: `_classify_combined` (single chat_json
        call) — see route() below.
        """
        """Run step-back then HyDE sequentially, returning (step_back, hyde).

        This is the dependent half of the router pipeline (step_back feeds
        into hyde). It is invoked as a single task by `route()` so the
        independent `_classify(query)` task can run concurrently via
        `asyncio.gather`. Behaviour is identical to the pre-L2 inline chain
        — only the scheduling changes.
        """
        step_back = await self._step_back(query)
        hyde_doc = await self._generate_hyde_doc(query, step_back)
        return step_back, hyde_doc

    async def _step_back(self, query: str) -> str:
        """Generate a step-back (abstracted) version of the query."""
        try:
            resp = await self._llm.chat(
                user_message=f"Câu hỏi cụ thể: {query}",
                system_message=_STEP_BACK_SYSTEM,
                temperature=0.2,
                max_tokens=128,
            )
            step_back = resp.content.strip()
            logger.debug(f"Step-back: {step_back[:80]}")
            return step_back
        except Exception as exc:
            logger.warning(f"Step-back failed: {exc}")
            return query  # fallback: use original

    async def _generate_hyde_doc(self, query: str, step_back: str) -> str:
        """Generate hypothetical document for HyDE embedding.

        Uses both original query and step-back for richer context.
        """
        combined = f"Câu hỏi: {query}\nNguyên lý: {step_back}"
        try:
            resp = await self._llm.chat(
                user_message=combined,
                system_message=_HYDE_SYSTEM,
                temperature=0.3,
                max_tokens=256,
            )
            doc = resp.content.strip()
            logger.debug(f"HyDE doc: {doc[:80]}...")
            return doc
        except Exception as exc:
            logger.warning(f"HyDE generation failed: {exc}")
            return query  # fallback

    async def _classify(self, query: str) -> Dict:
        """Classify query and get routing decision.

        S16.2 / AMF-2: takes ONLY the raw query (the HyDE doc concat was the
        #1 structural cause of over-routing to graph — see plan). The
        classifier sees what the user actually asked, not the hypothetical
        document we generated for embedding.

        Returns a dict with: route (padded|hybrid|graph), complexity (1-5),
        intent, reasoning.
        """
        user_msg = f"Câu hỏi gốc: {query}"
        try:
            data = await self._llm.chat_json(
                user_message=user_msg,
                system_message=_CLASSIFY_SYSTEM,
                temperature=0.1,
                max_tokens=512,
            )
            return data
        except Exception as exc:
            # S16.2 / CMF-1 (d): fail toward the SAFE MIDDLE (hybrid), not
            # graph. Graph is the most expensive and most prone to fabricate;
            # hybrid is a balanced default. The exception path is hit only
            # for non-vague queries (vague_reject short-circuits upstream).
            logger.warning(
                f"Classification failed: {exc} – defaulting to hybrid "
                "(S16.2 fail-toward-safe-middle)"
            )
            return {
                "route": "hybrid",
                "complexity": 2,
                "intent": "general",
                "reasoning": "classification failed — default hybrid (S16.2)",
                "hyde_variants": [query],
            }

    async def route(self, query: str) -> RouterResult:
        """Full routing pipeline: vague-check → auto-answer → step-back → HyDE → classify.

        Args:
            query: Raw user query (Vietnamese).

        Returns:
            RouterResult with all intermediate outputs and final route.
            Có thể trả về sớm với auto_answer/skip_retrieval=True nếu phát hiện
            câu hỏi mơ hồ hoặc câu common cần meta-answer.
        """
        # Phase E.3: in-memory LRU cache short-circuits identical queries
        from src.services.router_cache import get_router_cache
        cache = get_router_cache()
        cached = cache.get(query)
        if cached is not None:
            logger.debug(f"Router cache HIT for query: {query[:60]}")
            return cached

        logger.info(f"Router: processing query [{query[:60]}]")

        # ─── Pre-routing 1: Vague / placeholder rejection ───
        # Câu QUÁ mơ hồ (placeholder "...", chỉ "em hỏi", chỉ "?", quá ngắn không entity)
        # → reject ngay với hướng dẫn cụ thể, không tốn LLM call
        normalized_query = query.strip()
        word_count = len(re.findall(r"[\wÀ-ỹ]+", normalized_query, flags=re.UNICODE))
        # Test: nếu match vague pattern HOẶC (≤3 từ VÀ không có ngành/entity rõ ràng)
        is_vague_placeholder = bool(_VAGUE_PLACEHOLDER_PATTERNS.search(normalized_query))
        # Bổ sung: câu chỉ là dấu hỏi/dấu chấm — quá ngắn không thể đoán ý
        if word_count == 0 or (word_count == 1 and len(normalized_query) <= 3):
            is_vague_placeholder = True
        # Detect entity bằng heuristic đơn giản: có mã ngành/từ tuyển sinh/keyword
        _ENTITY_KEYWORDS = re.compile(
            r"(ngành|cntt|kỹ\s+thuật|báo\s+chí|truyền\s+thông|hán\s+nôm|triết|"
            r"lịch\s+sử|văn\s+học|hóa|vật\s+lý|sinh\s+học|môi\s+trường|kiến\s+trúc|"
            r"điểm|học\s+phí|chỉ\s+tiêu|tổ\s+hợp|xét\s+tuyển|đăng\s+ký|nguyện\s+vọng|"
            r"học\s+bạ|thpt|đgnl|husc|đại\s+học|trường|năm|\b20\d{2}\b|khối|"
            r"học\s+bổng|chính\s+sách|ưu\s+tiên|miễn|mã\s+ngành)",
            re.IGNORECASE,
        )
        has_entity = bool(_ENTITY_KEYWORDS.search(normalized_query))

        if is_vague_placeholder or (word_count <= 3 and not has_entity):
            logger.warning(
                f"Vague/placeholder query detected → rejecting with guidance. "
                f"query={normalized_query[:60]} words={word_count} has_entity={has_entity}"
            )
            result = RouterResult(
                original_query=query,
                step_back_query=query,
                hypothetical_doc="",
                hyde_variants=[query],
                route=QueryRoute.PADDED_RAG,
                complexity=1,
                intent="vague_reject",
                reasoning="Câu hỏi quá mơ hồ, không xác định được ý định cụ thể",
                auto_answer=HYDE_REJECT_VAGUE,
                skip_retrieval=True,
            )
            cache.put(query, result)
            return result

        # ─── Pre-routing 2: HyDE auto-answer cho câu contact/intro thuần ───
        # Match khi NGƯỜI DÙNG hỏi GỢI Ý contact/intro thuần (hotline, link,
        # address, fanpage, "giới thiệu về trường", "thông tin chung") mà KHÔNG
        # hỏi về NGÀNH cụ thể → trả CONTACT_BLOCK (year-agnostic, no majors).
        # Câu hỏi về NGÀNH ("trường có ngành gì", "tìm hiểu các ngành", ...)
        # bị LOẠI TRỪ ở đây → rơi xuống retrieval thật (enumeration→graph_rag
        # downstream). Kills the structural Type-4/5/6 hallucination previously
        # emitted by HYDE_META_ANSWER_INTRO (ADR-3).
        is_contact = bool(_HYDE_CONTACT_PATTERNS.search(normalized_query))
        is_major_question = bool(_MAJOR_QUESTION_PATTERNS.search(normalized_query))
        # Loại trừ: câu có hỏi điểm/học phí/chỉ tiêu CỤ THỂ → vẫn phải retrieval
        _SPECIFIC_QUERY = re.compile(
            r"(điểm\s+chuẩn|điểm\s+sàn|học\s+phí\s+ngành|chỉ\s+tiêu\s+ngành|"
            r"tổ\s+hợp\s+ngành|đăng\s+ký\s+xét|hồ\s+sơ|deadline|hạn\s+cuối)",
            re.IGNORECASE,
        )
        if is_contact and not is_major_question and not _SPECIFIC_QUERY.search(normalized_query):
            logger.info(
                f"HyDE contact/intro pattern matched → CONTACT_BLOCK (year-agnostic). "
                f"query={normalized_query[:60]}"
            )
            result = RouterResult(
                original_query=query,
                step_back_query=query,
                hypothetical_doc="",
                hyde_variants=[query],
                route=QueryRoute.PADDED_RAG,
                complexity=1,
                intent="hyde_contact_block",
                reasoning="Câu contact/intro thuần — trả CONTACT_BLOCK year-agnostic",
                auto_answer=CONTACT_BLOCK,
                skip_retrieval=True,
            )
            cache.put(query, result)
            return result

        # ─── Pre-routing 3: Regex fast-path for enum/comparison ─────────
        # Deterministic override (CMF-2, lines 703-713 below) already FORCES
        # route=graph for any query matching _ENUMERATION_PATTERNS or
        # _COMPARISON_PATTERNS. The LLM call is wasted work for that whole
        # class of query (~5s on the gateway). Short-circuit here, AFTER
        # the vague-placeholder and contact-block pre-routing gates (those
        # must still win first), but BEFORE `await self._classify_combined`.
        is_enum_fast = bool(_ENUMERATION_PATTERNS.search(normalized_query))
        is_comparison_fast = bool(_COMPARISON_PATTERNS.search(normalized_query))
        if is_enum_fast or is_comparison_fast:
            fast_intent = "liet_ke" if is_enum_fast else "so_sanh"
            logger.info(
                f"Fast-path: khớp pattern liệt kê/so sánh → graph (bỏ qua LLM classify). "
                f"query={normalized_query[:60]}"
            )
            result = RouterResult(
                original_query=query,
                step_back_query=query,
                hypothetical_doc=query,
                hyde_variants=[query],
                route=QueryRoute.GRAPH_RAG,
                complexity=4,
                intent=fast_intent,
                reasoning="Fast-path: khớp pattern liệt kê/so sánh → graph (bỏ qua LLM classify)",
                auto_answer=None,
                skip_retrieval=False,
            )
            cache.put(query, result)
            return result

        # L3: replace the pre-L3 `asyncio.gather(chain_task, classify_task)`
        # block with a SINGLE `await self._classify_combined(query)` call.
        # This drops HyDE entirely and folds step_back into the classify
        # prompt — net: ONE LLM round-trip per query (was 3: step_back,
        # hyde, classify) instead of 2 concurrent, saving ~one gateway
        # round-trip (~7s on weak net) without re-bundling HyDE into
        # classify (which caused the S16 over-routing regression).
        classification = await self._classify_combined(query)

        # S16.2 / CMF-2: parse the 3-way classify output. The schema accepts
        # both new tokens (padded|hybrid|graph) and the legacy
        # padded_rag/graph_rag (defensive — older prompts may still return
        # them; map legacy → new here so the rest of the file is uniform).
        raw_route = classification.get("route", "padded")
        if raw_route == "padded_rag":
            route_str = "padded"
        elif raw_route == "graph_rag":
            route_str = "graph"
        else:
            route_str = raw_route if raw_route in ("padded", "hybrid", "graph") else "padded"
        complexity = int(classification.get("complexity", 1))

        # step_back: from model output, but `_should_stepback` gate wins.
        # When the gate says False (year+major or short single-fact lookup),
        # echoing the raw query beats any abstraction the model produced
        # — same pre-L3 contract.
        model_step_back = classification.get("step_back", query) or query
        step_back = query if not _should_stepback(query) else model_step_back
        # HyDE dropped: hypothetical_doc is a no-op echo of the query so
        # downstream embedding consumers (`all_queries_for_embedding` /
        # vector averaging) keep working unchanged.
        hyde_doc = query

        # ─── CMF-2 precedence (terminal-marked) ────────────────────────────
        # 1) vague/contact auto-answer → already returned above (terminal).
        # 2) enum pattern → graph (terminal).
        # 3) comparison pattern → graph (terminal).
        # 4) classify route is AUTHORITATIVE.
        # 5) bump: padded + complexity >= threshold → hybrid (NEVER → graph).
        # 6) safety demote: padded + ≥2 NGANH/TO_HOP entities → hybrid
        #    (only demotion; never demote graph or hybrid).

        is_enum = bool(_ENUMERATION_PATTERNS.search(query))
        if is_enum and route_str != "graph":
            route_str = "graph"
            complexity = max(complexity, 4)
            logger.info(f"Enumeration pattern detected → forcing graph (complexity={complexity})")

        is_comparison = bool(_COMPARISON_PATTERNS.search(query))
        if is_comparison and route_str != "graph":
            route_str = "graph"
            complexity = max(complexity, 3)
            logger.info(f"Comparison pattern detected → forcing graph (complexity={complexity})")

        # 5) Bump padded→hybrid (NEVER escalates to graph).
        if route_str == "padded" and complexity >= self._threshold:
            route_str = "hybrid"
            logger.debug(
                f"Complexity={complexity} >= threshold={self._threshold}: "
                f"bumping padded → hybrid (S16.2 CMF-2)"
            )

        # 6) Safety demote: padded + ≥2 NGANH/TO_HOP entities → hybrid
        #    (CMF-5). Only the padded branch is touched.
        if route_str == "padded":
            n_nganh_tohop = _count_nganh_tohop_entities(query)
            if n_nganh_tohop >= 2:
                route_str = "hybrid"
                logger.debug(
                    f"CMF-5 safety demote: query has {n_nganh_tohop} "
                    f"NGANH/TO_HOP entities → padded demoted to hybrid"
                )

        # Map 3-way string → enum.
        if route_str == "graph":
            route = QueryRoute.GRAPH_RAG
        elif route_str == "hybrid":
            route = QueryRoute.HYBRID
        else:
            route = QueryRoute.PADDED_RAG

        # L3: HyDE dropped → hyde_variants is just [query]. Kept the
        # field on RouterResult for downstream embedding consumers; the
        # value is the same as pre-L3 fallback (single-element list).
        hyde_variants: List[str] = [query]

        result = RouterResult(
            original_query=query,
            step_back_query=step_back,
            hypothetical_doc=hyde_doc,
            hyde_variants=hyde_variants,
            route=route,
            complexity=complexity,
            intent=classification.get("intent", "general"),
            reasoning=classification.get("reasoning", ""),
            auto_answer=None,
            skip_retrieval=False,
        )

        logger.info(
            f"Router decision: route={route.value}, complexity={complexity}, "
            f"intent={result.intent}"
        )
        cache.put(query, result)
        return result

    async def route_batch(self, queries: List[str]) -> List[RouterResult]:
        """Route multiple queries sequentially (respects rate limits)."""
        results = []
        for q in queries:
            r = await self.route(q)
            results.append(r)
        return results


# Singleton
_router: Optional[SmartQueryRouter] = None


def get_router() -> SmartQueryRouter:
    """Get or create the singleton SmartQueryRouter."""
    global _router
    if _router is None:
        _router = SmartQueryRouter()
    return _router
