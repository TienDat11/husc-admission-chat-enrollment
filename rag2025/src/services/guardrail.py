from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from groq import AsyncGroq
from loguru import logger

from config.settings import RAGSettings


# Bounded multi-turn history cap. FE slices last 4 (2 user + 2 assistant);
# this helper enforces the SAME cap on the BE as a defense-in-depth measure
# so a malicious/legacy client cannot inject an unbounded history into the
# guardrail LLM message. Mirrors the cap in ChatLayout.tsx.
MAX_HISTORY_MSGS = 4


def _format_history_for_llm(history: List[Dict[str, str]]) -> str:
    """Render bounded chat history as plain text for the classifier LLM.

    Only the LLM-classifier user-message is allowed to see prior turns; the
    deterministic regex/folded paths in ``precheck`` MUST see the raw
    ``query`` alone (history injection there corrupts major matching and
    re-opens the denylist DoS holes).

    The format is a deterministic two-line block per turn:
        Người dùng: <content>
        Trợ lý: <content>
    The total message count is hard-capped at ``MAX_HISTORY_MSGS`` to keep
    the prompt bounded even when an upstream caller forgets to slice.
    """
    if not history:
        return ""
    rendered: List[str] = []
    for turn in history[-MAX_HISTORY_MSGS:]:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            rendered.append(f"Người dùng: {content}")
        elif role == "assistant":
            rendered.append(f"Trợ lý: {content}")
        else:
            # Unknown role — render generically rather than drop, so the
            # classifier still has full context.
            rendered.append(f"{role or 'unknown'}: {content}")
    return "\n".join(rendered)


@dataclass
class GuardrailDecision:
    is_in_scope: bool
    internal_code: str
    reason: str
    short_answer: str
    data_gap_hints: List[str] = field(default_factory=list)
    pii_detected: bool = False


class GuardrailService:
    # ADMISSION_KEYWORDS drives the deterministic fast-path. It is intentionally
    # generic so legitimate HUSC questions short-circuit without an LLM call.
    # The fast-path is disabled when the query names a DIFFERENT school and
    # does NOT mention HUSC — those cases fall through to the LLM (which is
    # now anchored to HUSC-only) so we don't blanket-allow other-university
    # queries just because they contain words like "điểm chuẩn" or "học phí".
    ADMISSION_KEYWORDS = [
        "tuyển sinh", "điểm chuẩn", "học phí", "ngành", "tổ hợp", "xét tuyển",
        "husc", "đại học khoa học huế", "chỉ tiêu", "học bổng", "hồ sơ",
        "đăng ký", "đăng kí", "nhập học", "thủ tục", "nộp hồ sơ",
        "điểm sàn", "điểm trúng tuyển", "điểm xét tuyển", "tổ hợp môn",
        "mã ngành", "đăng ký xét tuyển", "nguyện vọng", "chương trình đào tạo",
        "cổng thông tin tuyển sinh", "lệ phí", "miễn giảm", "xét tuyển thẳng",
        "hạn nộp", "đánh giá năng lực", "ký túc xá", "ktx", "liên thông",
        "vào trường", "thi vào", "học bạ",
        "thi vô", "thi vo", "vô trường", "vo truong",
        # Common no-diacritic / abbrev variants (cheap + unambiguous)
        "nguyen vong", "vao truong", "thi vao", "dgnl", "đgnl", "ky tuc xa",
    ]

    # Markers of OTHER universities/institutions. If ANY of these appears in
    # the query AND no HUSC alias is present, we MUST NOT take the keyword
    # fast-path — the LLM needs to see it and (with the hardened prompt) block.
    # "đại học huế" / "đh huế" is the PARENT of HUSC; treated as OTHER only
    # when NOT co-mentioned with a HUSC alias (handled in _mentions_other_school).
    OTHER_SCHOOL_MARKERS = [
        "bách khoa", "bach khoa",
        "fpt",
        "y hà nội", "y ha noi", "y dược", "y duoc",
        "ngoại thương", "ngoai thuong",
        "kinh tế quốc dân", "kinh te quoc dan",
        "tôn đức thắng", "ton duc thang",
        "rmit",
        "uel",
        "ueh",
        "đh huế", "dh hue",
    ]

    # Aliases for HUSC. If any of these appears in the query, the
    # "other-school" guard is bypassed — the query is treated as HUSC-related.
    HUSC_ALIASES = [
        "husc",
        "đại học khoa học huế",
        "dai hoc khoa hoc hue",
        "khoa học huế",
        "khoa hoc hue",
        "trường đại học khoa học",
        "truong dai hoc khoa hoc",
        "trường khoa học",
        # Additional low-collision aliases (people / search engines
        # sometimes shorten "Đại học Khoa học Huế" to "ĐHKH Huế" or
        # "trường ĐHKH" in informal writing)
        "đhkh huế", "dhkh hue", "đhkh", "dhkh",
        "đhkhh", "trường đhkh", "truong dhkh",
    ]

    PII_KEYWORDS = [
        "cccd", "cmnd", "số căn cước", "số tài khoản", "mật khẩu", "otp", "cvv",
    ]

    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?84|0)(?:[\s.-]?\d){8,10}\b"),
        "citizen_id": re.compile(r"\b\d{12}\b"),
    }

    # ------------------------------------------------------------------
    # Major-scope guardrail (S19 — block admission questions about majors
    # HUSC does NOT offer, deterministically, before retrieval / before
    # the LLM call).
    #
    # Design: hybrid positive allowlist + small negative denylist.
    # The POSITIVE allowlist is loaded DYNAMICALLY from
    # ``data/major_codes/husc_majors_current.json`` (overwritable by a
    # crawler), with mtime-based reload. The NEGATIVE denylist is a
    # small hardcoded set of field keywords HUSC has NEVER taught. We
    # block only when (a) a denylist field is named, OR (b) a "ngành
    # <X>" / "<X> điểm chuẩn" pattern names X that is not in the
    # allowlist AND X looks like a real major name (≥1 field word).
    # We DO NOT block generic admission questions that name no specific
    # major — those fall through to the existing keyword / LLM path.
    # ------------------------------------------------------------------

    # Hard-coded denylist — HUSC has NEVER offered these fields.
    # Cross-checked against husc_tuition_2026_official.json +
    # WHITELIST_2026 in major_code_validator.py:
    #   REMOVED from this list because HUSC ACTUALLY offers them:
    #     - "toán" / "toán học" / "toán ứng dụng" — kept allowlist
    #     - "hóa học" — kept allowlist
    #     - "hóa dược" — kept allowlist (HUSC offers Hóa dược 7440113)
    #     - "sinh học" / "công nghệ sinh học" — kept allowlist
    #     - "môi trường" / "khoa học môi trường" — kept allowlist
    #     - "cntt" / "công nghệ thông tin" / "phần mềm" — kept allowlist
    #     - "xây dựng" (Địa kỹ thuật xây dựng) — kept allowlist
    #     - "kiến trúc" — kept allowlist
    #     - "triết học" / "lịch sử" / "văn học" — kept allowlist
    #     - "báo chí" / "truyền thông" — kept allowlist
    #     - "tâm lý" / "tâm lý học" — kept allowlist
    #     - "xã hội học" / "công tác xã hội" — kept allowlist
    #   KEPT in denylist (HUSC has never offered these):
    #     - Kinh tế, Luật, Y, Y khoa, Y đa khoa, Dược (as a major — HUSC
    #       only has "Hóa dược" 7440113, a chemistry sub-discipline, not
    #       the pharmacy degree), Răng hàm mặt, Sư phạm, Điều dưỡng,
    #       Bách khoa (as a major), Tài chính ngân hàng, Kế toán, Quản
    #       trị kinh doanh, Marketing, Logistics, Ngôn ngữ Anh / Trung /
    #       Nhật (as standalone majors — HUSC teaches these only as
    #       modules / optional credits, not as full majors), Du lịch,
    #       Quản trị khách sạn, Nhà hàng khách sạn, Quản trị nhân lực.
    MAJOR_DENYLIST = [
        # Economics / Business
        "kinh tế", "kinh te", "kinh tế học", "tài chính", "tài chính ngân hàng",
        "kế toán", "ke toan", "kiểm toán", "quản trị kinh doanh", "qtkt",
        "marketing", "thương mại", "thuong mai", "logistics", "logistic",
        "quản trị nhân lực", "quản trị nhân sự", "quản trị khách sạn",
        "nhà hàng khách sạn", "du lịch", "du lich", "quản trị dịch vụ du lịch",
        "quản trị sự kiện",
        # Law
        "luật", "luat", "luật học", "luật kinh tế", "luật dân sự",
        # Medicine / Pharmacy (HUSC only has Hóa dược — chemistry subfield)
        "y khoa", "y da khoa", "đa khoa", "y đa khoa", "bác sĩ", "bac si",
        "răng hàm mặt", "rang ham mat", "nha khoa", "điều dưỡng", "dieu duong",
        "hộ sinh", "ho sinh", "dược", "duoc", "dược học", "duoc hoc",
        "kỹ thuật y học", "ky thuat y hoc", "y tế công cộng", "yte cong cong",
        # Education
        "sư phạm", "su pham", "giáo dục", "giao duc", "giáo viên", "giao vien",
        # Engineering-as-school (HUSC has individual engineering majors but
        # is NOT the polytechnic "ĐH Bách Khoa" itself). We do NOT include
        # bare "bach khoa" here — that's the school name handled by
        # _mentions_other_school, not a major HUSC doesn't offer.
        "đại học bách khoa", "dh bach khoa", "dai hoc bach khoa",
        "trường bách khoa", "truong bach khoa", "bách khoa hà nội", "bach khoa ha noi",
        # Foreign-language standalone majors (HUSC teaches languages only as
        # modules, not as full majors)
        "ngôn ngữ anh", "ngon ngu anh", "tiếng anh", "tieng anh",
        "ngôn ngữ trung", "ngon ngu trung", "tiếng trung", "tieng trung",
        "ngôn ngữ nhật", "ngon ngu nhat", "tiếng nhật", "tieng nhat",
        "ngôn ngữ hàn", "ngon ngu han", "tiếng hàn", "tieng han",
        "ngôn ngữ pháp", "ngon ngu phap", "tiếng pháp", "tieng phap",
        "ngôn ngữ nga", "ngon ngu nga", "tiếng nga", "tieng nga",
        "ngôn ngữ đức", "ngon ngu duc", "tiếng đức", "tieng duc",
        "ngôn ngữ trung quốc", "tieng trung quoc",
    ]

    # Major-scope file (crawler-updatable).
    MAJOR_SCOPE_FILE = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "major_codes"
        / "husc_majors_current.json"
    )
    # Fallback if the dynamic file is missing/corrupt: parse the canonical
    # tuition JSON (which is the source of truth for the 2026 majors) so
    # guardrail still works offline.
    _MAJOR_FALLBACK_FILES = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "major_codes"
        / "husc_tuition_2026_official.json",
    )

    # Cache (mtime -> (allowed_set, alias_to_canonical))
    _SCOPE_MTIME: Optional[float] = None
    _SCOPE_ALLOWED: Set[str] = set()
    _SCOPE_PATH: Optional[Path] = None

    def __init__(self, settings: RAGSettings):
        self._settings = settings
        self._enabled = settings.GUARDRAIL_ENABLED
        self._groq_key = os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=self._groq_key) if self._groq_key else None
        self._model = settings.GUARDRAIL_MODEL
        # Reset module-level cache so a fresh service gets the latest file.
        self._reset_major_scope_cache()

    @classmethod
    def _reset_major_scope_cache(cls) -> None:
        cls._SCOPE_MTIME = None
        cls._SCOPE_ALLOWED = set()
        cls._SCOPE_PATH = None

    # ----- Diacritic folding + major-scope loader -----

    @staticmethod
    def _fold_diacritics(text: str) -> str:
        """Lowercase + strip Vietnamese diacritics for fuzzy matching."""
        if not isinstance(text, str):
            return ""
        nfkd = unicodedata.normalize("NFKD", text)
        ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
        return ascii_only.lower().strip()

    @classmethod
    def _seed_minimal_allowed(cls) -> Set[str]:
        """Hard-coded minimum allowlist used as a last-resort seed when both
        the dynamic file and the canonical tuition JSON are unavailable.
        Mirrors the WHITELIST_2026 frozenset in major_code_validator.py.
        """
        return {
            "vat ly", "vat ly hoc", "vat ly ky thuat", "ban dan", "cong nghe ban dan",
            "cong nghe sinh hoc", "sinh hoc", "sinh hoc ung dung",
            "hoa hoc", "hoa duoc", "khoa hoc moi truong",
            "ky thuat phan mem", "cong nghe thong tin", "cntt",
            "khoa hoc may tinh", "he thong thong tin",
            "khoa hoc du lieu", "khdl", "khmt",
            "cnkt dien tu vien thong", "dien tu vien thong",
            "cnkt hoa hoc", "trac dia ban do",
            "kien truc", "dia ky thuat xay dung",
            "toan hoc", "toan", "toan ung dung", "toan tin",
            "y sinh", "han nom", "triet hoc", "lich su", "van hoc",
            "quan ly van hoa", "quan ly nha nuoc", "xa hoi hoc",
            "dong phuong hoc", "dong nam a hoc",
            "bao chi", "truyen thong so", "truyen thong da phuong tien",
            "tam ly hoc", "cong tac xa hoi",
            "quan ly tai nguyen va moi truong",
            "quan ly an toan suc khoe va moi truong",
        }

    @classmethod
    def _parse_scope_file(cls, path: Path) -> Set[str]:
        """Parse a husc_majors_current.json-style file and return the allowed
        token set (folded). NEVER raises — returns empty set on any error.
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return set()
        if not isinstance(data, dict):
            return set()
        majors = data.get("majors", [])
        if not isinstance(majors, list):
            return set()
        allowed: Set[str] = set()
        for entry in majors:
            if not isinstance(entry, dict):
                continue
            tokens: List[str] = []
            ten = entry.get("ten")
            if isinstance(ten, str) and ten.strip():
                tokens.append(ten)
            aliases = entry.get("aliases", [])
            if isinstance(aliases, list):
                for a in aliases:
                    if isinstance(a, str) and a.strip():
                        tokens.append(a)
            for tok in tokens:
                folded = cls._fold_diacritics(tok)
                if folded and len(folded) >= 2:
                    allowed.add(folded)
        return allowed

    @classmethod
    def _load_major_scope(cls) -> Set[str]:
        """Load the dynamic allowlist from disk, cached by file mtime.

        Resolution order:
          1. ``husc_majors_current.json`` (crawler-writable) — reloaded when
             mtime changes so a 2027 new major added by the crawler is
             picked up automatically.
          2. Canonical ``husc_tuition_2026_official.json`` (offline seed).
          3. Embedded hard-coded seed (last-resort).
        NEVER raises.
        """
        primary = cls.MAJOR_SCOPE_FILE
        try:
            if primary.exists():
                mtime = primary.stat().st_mtime
                if (
                    cls._SCOPE_MTIME == mtime
                    and cls._SCOPE_PATH == primary
                    and cls._SCOPE_ALLOWED
                ):
                    return cls._SCOPE_ALLOWED
                allowed = cls._parse_scope_file(primary)
                if allowed:
                    cls._SCOPE_MTIME = mtime
                    cls._SCOPE_PATH = primary
                    cls._SCOPE_ALLOWED = allowed
                    return allowed
        except Exception:
            pass

        # Fallback chain
        for fb in cls._MAJOR_FALLBACK_FILES:
            try:
                if fb.exists():
                    allowed = cls._parse_scope_file(fb)
                    if allowed:
                        cls._SCOPE_ALLOWED = allowed
                        cls._SCOPE_PATH = fb
                        cls._SCOPE_MTIME = fb.stat().st_mtime
                        return allowed
            except Exception:
                continue

        cls._SCOPE_ALLOWED = cls._seed_minimal_allowed()
        cls._SCOPE_PATH = None
        cls._SCOPE_MTIME = 0.0
        return cls._SCOPE_ALLOWED

    @classmethod
    def clear_major_scope_cache_for_testing(cls) -> None:
        """Test helper: force a reload on the next _load_major_scope() call."""
        cls._reset_major_scope_cache()

    def _looks_admission_related(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.ADMISSION_KEYWORDS)

    def _mentions_husc(self, query: str) -> bool:
        """Return True iff the query references HUSC (or any of its aliases)."""
        q = query.lower()
        return any(alias in q for alias in self.HUSC_ALIASES)

    def _mentions_other_school(self, query: str) -> bool:
        """Return True iff the query names a DIFFERENT university AND does NOT
        mention HUSC. The keyword fast-path must NOT short-circuit such queries
        — the LLM (with the hardened prompt) is the only safe way to block them.
        """
        q = query.lower()
        if self._mentions_husc(q):
            return False
        return any(marker in q for marker in self.OTHER_SCHOOL_MARKERS)

    def _redirect_answer(self) -> str:
        return (
            "Mình chỉ hỗ trợ tư vấn tuyển sinh HUSC. "
            "Bạn có thể hỏi về ngành học, điểm chuẩn, học phí, tổ hợp xét tuyển hoặc hồ sơ tuyển sinh."
        )

    def _pii_warning_answer(self) -> str:
        return (
            "Để bảo vệ thông tin cá nhân, bạn không nên gửi dữ liệu nhạy cảm như CCCD, "
            "email, số điện thoại hoặc số tài khoản đầy đủ. Vui lòng che bớt thông tin trước khi hỏi."
        )

    def _major_not_offered_answer(self) -> str:
        return (
            "HUSC (Trường Đại học Khoa học — Đại học Huế) hiện KHÔNG đào tạo ngành/ lĩnh vực này. "
            "HUSC tập trung vào nhóm ngành Khoa học tự nhiên, Công nghệ thông tin, Kỹ thuật công nghệ, "
            "Khoa học môi trường — Trái đất, và một số ngành Khoa học xã hội & Nhân văn "
            "(Toán, Vật lý, Hóa học, Sinh học, CNTT, Công nghệ bán dẫn, Khoa học dữ liệu, Kiến trúc, "
            "Địa kỹ thuật xây dựng, Hán-Nôm, Triết học, Báo chí, ...). "
            "Bạn có thể xem danh sách ngành HUSC đang đào tạo tại "
            "https://tuyensinh.husc.edu.vn/nganhdaotao-dh.php, hoặc hỏi mình về các ngành khác mà HUSC có."
        )

    def _contains_sensitive_pii(self, query: str) -> bool:
        q = query.lower()
        if any(k in q for k in self.PII_KEYWORDS):
            return True
        return any(pattern.search(query) is not None for pattern in self.PII_PATTERNS.values())

    # ----- Major-scope check (S19) -----

    _MAJOR_TAIL_RX = re.compile(
        r"(?:điểm chuẩn|diem chuan|học phí|hoc phi|"
        r"xét tuyển|xet tuyen|tổ hợp|to hop|chỉ tiêu|chi tieu|"
        r"tuyển sinh|tuyen sinh|học bổng|hoc bong|chương trình|chuong trinh|"
        r"ngành|nganh|mã ngành|ma nganh|thông tin|thong tin)",
        re.IGNORECASE,
    )

    # Common non-major words that, after diacritic-fold, could look like a
    # "head noun" before a tail. These should NOT be treated as majors.
    _MAJOR_PHRASE_BLOCKLIST = {
        "thoi gian", "thời gian", "thời hạn", "han", "hạn",
        "truong", "trường", "khoa", "nha", "nhà",
        "hoc", "học", "bao nhieu", "nhu the nao", "the nao",
        "cach", "cách", "khi nao", "khi nào", "o dau", "ở đâu",
        "hoc phi", "học phí", "diem chuan", "điểm chuẩn",
    }

    # Generic "ngành nào/ngành gì/ngành gi" → always generic, never a major.
    _GENERIC_NGANH_RX = re.compile(
        r"\b(?:ng[aà]nh|nganh)\s+(?:n[aà]o|g[iì]|gi)\b",
        re.IGNORECASE,
    )

    # Admission-process language that is also a "ngành <X>" / "<X> học"
    # head-noun extraction hazard — the major extractor must NOT
    # treat these as majors. They appear in queries like:
    #   "ngành miễn giảm học phí thế nào?" → "miễn giảm"
    #   "ngành ký túc xá có không?"         → "ký túc xá"
    #   "chương trình đào tạo ngành Hóa dược?" → "Hóa dược" (KEEP this!)
    # Adding to _BARE_GENERIC_MAJORS would over-block (a real major
    # "Văn" wouldn't be reachable from "ngành Văn"). Instead we use a
    # narrower blocklist of folded multi-word tokens.
    _ADMISSION_LANGUAGE_PHRASES = frozenset({
        "mien giam", "ky tuc xa", "hoc phi", "le phi", "nguyen vong",
        "thi vao", "vao truong", "xet tuyen thang", "danh gia nang luc",
        "han nop", "tong hop mon", "ma nganh", "cong thong tin tuyen sinh",
        "chuong trinh dao tao", "diem san", "diem trung tuyen",
        "diem xet tuyen", "dang ky xet tuyen", "lien thong", "hoc ba",
        "vien phi", "nganh nghe", "xet tuyen", "tuyen sinh", "nganh nghe",
    })

    # Single-syllable words that are REAL HUSC majors (Toán, Văn, Hóa, Lý,
    # Sử, KT[Kiến trúc]) but ALSO appear as bare generic language. The
    # major-scope extractor must only treat them as a NAMED major when
    # framed as "ngành <X>" or "<X> học" (compound), NOT as a bare one-
    # character token — otherwise "toán" in "học toán ở đâu?" gets
    # classified as the Toán học major and falls into the denylist/allowlist
    # path. Bare single-syllable tokens in non-compound context are
    # ignored — they are NOT treated as a major phrase.
    _BARE_GENERIC_MAJORS = frozenset({"toan", "van", "hoa", "ly", "su", "kt"})

    @classmethod
    def _is_admission_language_phrase(cls, phrase: str) -> bool:
        """True iff ``phrase`` (after diacritic-fold) is one of the
        multi-word admission-process language tokens. Used by the
        major extractor / _major_out_of_scope to bail out before
        classifying admission-process language as a major.
        """
        if not phrase:
            return False
        folded = cls._fold_diacritics(phrase).strip()
        return folded in cls._ADMISSION_LANGUAGE_PHRASES

    @classmethod
    def _is_bare_generic_major_token(cls, phrase: str) -> bool:
        """True iff ``phrase`` (after diacritic-fold) is exactly one of
        the single-syllable generic-language-but-also-major words from
        ``_BARE_GENERIC_MAJORS``. Used by the extractor to bail out
        before a bare token like "toán" / "văn" is misclassified as
        a major name.
        """
        if not phrase:
            return False
        folded = cls._fold_diacritics(phrase).strip()
        return folded in cls._BARE_GENERIC_MAJORS

    @staticmethod
    def _is_likely_major_phrase(phrase: str) -> bool:
        if not phrase or len(phrase) < 3:
            return False
        folded = GuardrailService._fold_diacritics(phrase)
        if not folded:
            return False
        if not any(ch.isalpha() for ch in folded):
            return False
        tokens = [t for t in re.split(r"[\s,;:/()\-]+", folded) if t]
        return any(len(t) >= 3 for t in tokens)

    # Process-verbs that signal a procedure / admission-process question,
    # NOT a question about a specific major. Used by _extract_major_phrase
    # to bail out before wrongly classifying the verb-phrase as a major.
    _PROCESS_VERB_PATTERNS = (
        r"\b(?:đăng\s*k[ýy]|dang\s*ky|dang\s*ki)\b",
        r"\b(?:đăng\s*k[íi]|dang\s*ki)\b",
        r"\bnh[ậa]p\s*h[ọo]c\b",
        r"\bth[ủu]\s*t[ụu]c\b",
        r"\bn[ộo]p\s*h[ồo]\s*s[ơo]\b",
        r"\bx[eé]t\s*tuy[ểe]n\b",
        r"\b(?:l[àa]m\s*sao|c[aá]ch)\b",
        r"\b(?:th[ủu]e|thue)\b",
    )
    _PROCESS_VERB_RX = re.compile(
        r"(?:" + "|".join(_PROCESS_VERB_PATTERNS) + r")",
        re.IGNORECASE,
    )

    @classmethod
    def _is_process_verb_phrase(cls, query: str) -> bool:
        """Return True if the query is dominated by process-verbs
        (đăng ký, nhập học, thủ tục, nộp hồ sơ, làm sao, cách, etc.).
        Such queries are admission-process questions, NOT major-scope
        questions, and must never be misclassified as a major.
        """
        if not query:
            return False
        return cls._PROCESS_VERB_RX.search(query) is not None

    def _is_generic_question_phrase(self, phrase: str) -> bool:
        folded = self._fold_diacritics(phrase)
        if folded in self._MAJOR_PHRASE_BLOCKLIST:
            return True
        # If the phrase is a generic interrogative tail (ba nhiêu, gì, nào,
        # nào, nhu the nao, o dau, etc.) it's not a major name.
        generic_tail = re.search(
            r"\b(b[aá]o nhi[eê]u|g[iì]|n[aà]o|th[eế] n[aà]o|ra sao|"
            r"nh[uư] th[eế] n[aà]o|[oở] đ[aâ]u|khi n[aà]o)\b",
            self._fold_diacritics(phrase),
            re.IGNORECASE,
        )
        return generic_tail is not None

    def _extract_major_phrase(self, query: str) -> Optional[str]:
        if not query:
            return None
        q = query.strip()
        # Process-verb guard: queries dominated by admission-process verbs
        # (đăng ký, nhập học, thủ tục, nộp hồ sơ, xét tuyển, làm sao, cách)
        # are NOT major-scope queries, even if a major-scope tail word
        # like "xét tuyển" / "tuyển sinh" is present. Bail out early so
        # Pattern 2/4 below does NOT back-track and grab the verb phrase
        # as a fake major name. The query then falls through to the
        # keyword fast-path (or LLM) for an in-scope verdict.
        if self._is_process_verb_phrase(q):
            return None
        # Bare single-syllable generic-major guard: real HUSC majors like
        # "Toán" / "Văn" / "Hóa" / "Lý" / "Sử" / "KT" also appear as
        # bare common-language words. A bare token ("học toán ở đâu?")
        # must NOT be classified as a major name — it only counts when
        # framed as "ngành Toán" or "<X> học" (compound). Compound
        # phrases (length >= 2 tokens after fold) still flow through to
        # the per-pattern logic below.
        if self._is_bare_generic_major_token(q):
            return None
        # Pattern 1: "ngành <X>" / "nganh <X>"
        m = re.search(r"\b(?:ngành|nganh)\s+([A-Za-zÀ-ỹĐđ\-\s]{3,})", q, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip(" ,.;:?!")
            # Trim at any known tail keyword that would have started a new clause
            phrase = self._MAJOR_TAIL_RX.split(phrase, maxsplit=1)[0].strip()
            # Trim trailing year / "là gì"
            phrase = re.sub(
                r"\s+(?:n[aă]m\s+\d{4}|\d{4}|l[aà]\s+g[iì])\s*$",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            # Trim a bare trailing "năm" (year prefix in "ngành Luật năm 2026?")
            phrase = re.sub(
                r"\s+n[aă]m\s*$",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            if self._is_likely_major_phrase(phrase) and not self._is_generic_question_phrase(phrase):
                if not self._is_bare_generic_major_token(phrase) and not self._is_admission_language_phrase(phrase):
                    return phrase
        # Pattern 2: "<X> điểm chuẩn" / "<X> học phí" — head noun BEFORE a tail
        m = re.search(
            r"([A-Za-zÀ-ỹĐđ\-\s]{3,}?)\s+(?:" + self._MAJOR_TAIL_RX.pattern + r")",
            q,
            re.IGNORECASE,
        )
        if m:
            phrase = m.group(1).strip(" ,.;:?!")
            phrase = re.sub(
                r"^(?:ngành|nganh|chuyên ngành|chuyen nganh|nghề|nghe|học|hoc|"
                r"trường|truong|khoa|đào tạo|dao tao|chương trình|chuong trinh)\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            phrase = re.sub(
                r"\s+(?:n[aă]m\s+\d{4}|\d{4})\s*$",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            if self._is_likely_major_phrase(phrase) and not self._is_generic_question_phrase(phrase):
                if not self._is_bare_generic_major_token(phrase) and not self._is_admission_language_phrase(phrase):
                    return phrase
        # Pattern 3: "<tail> ngành <X>" — common VI phrasing
        m = re.search(
            r"(?:" + self._MAJOR_TAIL_RX.pattern + r")\s+(?:ngành|nganh)\s+"
            r"([A-Za-zÀ-ỹĐđ\-\s]{2,})",
            q,
            re.IGNORECASE,
        )
        if m:
            phrase = m.group(1).strip(" ,.;:?!")
            phrase = re.sub(
                r"\s+(?:n[aă]m\s+\d{4}|\d{4})\s*$",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            if self._is_likely_major_phrase(phrase) and not self._is_generic_question_phrase(phrase):
                if not self._is_bare_generic_major_token(phrase) and not self._is_admission_language_phrase(phrase):
                    return phrase
        # Pattern 4: bare "<X> điểm chuẩn" / "<X> học phí" where X starts at the
        # beginning of the query (catches "Khoa học dữ liệu điểm chuẩn?").
        m = re.match(
            r"\s*([A-Za-zÀ-ỹĐđ\-\s]{4,}?)\s+(?:" + self._MAJOR_TAIL_RX.pattern + r")",
            q,
            re.IGNORECASE,
        )
        if m:
            phrase = m.group(1).strip(" ,.;:?!")
            phrase = re.sub(
                r"\s+(?:n[aă]m\s+\d{4}|\d{4})\s*$",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            if self._is_likely_major_phrase(phrase) and not self._is_generic_question_phrase(phrase):
                if not self._is_bare_generic_major_token(phrase) and not self._is_admission_language_phrase(phrase):
                    return phrase
        return None

    @classmethod
    def _major_in_denylist(cls, folded_phrase: str) -> bool:
        if not folded_phrase:
            return False
        for bad in cls.MAJOR_DENYLIST:
            if bad in folded_phrase:
                return True
        return False

    @classmethod
    def _major_in_allowlist(cls, folded_phrase: str, allowed: Set[str]) -> bool:
        if not folded_phrase or not allowed:
            return False
        if folded_phrase in allowed:
            return True
        # Substring match: "hoa hoc" is a real HUSC major; "sinh hoc"
        # / "sinh hoc ung dung" are also. We match if EITHER the folded
        # phrase contains an allowed token (>= 3 chars to avoid generic
        # single-syllable false positives) OR an allowed token fully
        # contains the phrase. A guarded exception prevents short
        # bare one-syllable tokens (see _BARE_GENERIC_MAJORS) from
        # being matched as majors in the allowlist — those are
        # disambiguated by the caller's own bare-token guard in
        # _extract_major_phrase.
        for tok in allowed:
            if len(tok) >= 3 and (tok in folded_phrase or folded_phrase in tok):
                # Reject when both are 1-2 char super-short tokens
                # (avoid "kt" matching everything with "kt" substring)
                if len(tok) < 4 and len(folded_phrase) < 4:
                    continue
                return True
        return False

    # Generic admission-intent guard: if a query is a clearly admission-
    # process question (đăng ký, nộp hồ sơ, xét tuyển, ký túc xá, miễn
    # giảm, etc.) AND a denylist substring is hit only because of a
    # generic-language false match (e.g. "học phí" → "phí" in denylist
    # via "kế toán" substring? No — but "học phí FPT" hits no denylist
    # entry, "miễn giảm học phí" contains "hoc phi" which is itself
    # admitted via the new keyword), the layer MUST return False.
    # The simplest robust rule: if a clear admission keyword is present,
    # do NOT block. The OTHER-SCHOOL guard already blocks non-HUSC
    # queries via a separate layer.
    @classmethod
    def _is_admission_intent_query(cls, query: str) -> bool:
        if not query:
            return False
        folded = cls._fold_diacritics(query)
        for kw in cls.ADMISSION_KEYWORDS:
            if cls._fold_diacritics(kw) in folded:
                return True
        return False

    # Denylist entries that are also legitimate admission-intent
    # language — the major-scope layer MUST NOT block a query just
    # because it contains one of these as a substring UNLESS a
    # specific major phrase can be extracted. The current set covers:
    #   - "kế toán" (Kế toán major) — appears inside "kế toán tiền học phí"
    #     which IS a process question
    #   - "dược" (Dược major) — appears inside "Hóa dược" (a HUSC major)
    #     and "lệ phí" (admission fee)
    #   - "phí" alone is NOT a denylist entry; the words are "học phí" /
    #     "lệ phí" and they are admission keywords.
    # Substring matching of denylist entries on the WHOLE query is a
    # known footgun. The fix: when a denylist substring is hit, also
    # try the extracted major phrase; if THAT phrase is in the
    # allowlist, do NOT block.
    def _major_out_of_scope(self, query: str) -> bool:
        if not query:
            return False
        # "HUSC có những ngành nào?" / "Các ngành của HUSC?" — generic
        if self._GENERIC_NGANH_RX.search(query):
            return False
        # Process / administrative admission questions that incidentally
        # contain a denylist substring must NOT be blocked. Examples:
        # "miễn giảm học phí thế nào?" (phí is generic, not a major);
        # "chương trình đào tạo ngành Hóa dược?" (hóa → Hóa dược allow).
        # The check: if the query is a process-style question (verb-
        # dominated) AND no specific major is extracted, return False.
        if self._is_process_verb_phrase(query) and not self._extract_major_phrase(query):
            return False
        # If the query is clearly an admission-intent question (one of
        # the ADMISSION_KEYWORDS hits) AND no specific major phrase
        # can be extracted, do NOT block. This stops generic
        # admission-process queries like "ký túc xá có không?" from
        # being mis-blocked by a denylist substring false positive.
        # SAFETY: this ONLY short-circuits when no major phrase was
        # extracted. "Ngành Kinh Tế điểm chuẩn?" extracts "Kinh Tế"
        # so the admission-intent bypass is NOT taken — denylist
        # still wins.
        phrase_early = self._extract_major_phrase(query)
        # SAFETY: the admission-intent bypass ONLY triggers when the
        # extracted phrase is either None OR a known admission-language
        # token (e.g. "ký túc xá", "miễn giảm"). A denylist phrase
        # like "Kinh Tế" / "Luật" / "Y đa khoa" extracted as a major
        # MUST NOT trigger the bypass — denylist must win.
        if self._is_admission_intent_query(query):
            if not phrase_early:
                return False
            if self._is_admission_language_phrase(phrase_early):
                return False
        folded_q = self._fold_diacritics(query)
        if self._major_in_denylist(folded_q):
            # If a denylist substring hits on the WHOLE query, check
            # whether a specific major phrase can be extracted. If the
            # extracted phrase is in the allowlist, do NOT block (this
            # is the Hóa dược case). Otherwise block (this is the
            # "Ngành Kinh Tế điểm chuẩn?" case — denylist substring
            # matches and the extracted phrase "Kinh Tế" is itself in
            # the denylist).
            phrase = phrase_early
            if phrase:
                folded_phrase = self._fold_diacritics(phrase)
                allowed = self._load_major_scope()
                if self._major_in_allowlist(folded_phrase, allowed):
                    return False
            return True
        phrase = phrase_early
        if not phrase:
            return False
        folded_phrase = self._fold_diacritics(phrase)
        if self._major_in_denylist(folded_phrase):
            # Even when the extracted phrase hits the denylist, we must
            # re-check the HUSC alias logic. The precheck() outer
            # already short-circuits "HUSC + denylist" but to be safe
            # at this layer we just return True; the outer layer will
            # then bypass it if the query is "HUSC <denylist>".
            return True
        allowed = self._load_major_scope()
        if not self._major_in_allowlist(folded_phrase, allowed):
            return True
        return False

    async def precheck(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> GuardrailDecision:
        if not self._enabled:
            return GuardrailDecision(True, "SUCCESS", "guardrail_disabled", "")

        if self._contains_sensitive_pii(query):
            return GuardrailDecision(
                False,
                "SENSITIVE_PII_DETECTED",
                "sensitive_pii_detected",
                self._pii_warning_answer(),
                ["Ẩn bớt thông tin định danh trước khi gửi câu hỏi", "Chỉ giữ lại phần nội dung liên quan tuyển sinh"],
                pii_detected=True,
            )

        # Major-scope check (S19): block admission questions about majors
        # HUSC does NOT offer, deterministically, BEFORE the keyword
        # fast-path and BEFORE any LLM call. Conservative: when the
        # query names no specific major, this returns False and the
        # normal flow proceeds unchanged.
        #
        # Two cases:
        #  (a) Query has no HUSC alias → block on any major-scope miss
        #      (this also catches the HUSC-vs-other case where the user
        #      asks "HUSC so với Bách Khoa ngành CNTT" because CNTT IS a
        #      real HUSC major, so no false positive there).
        #  (b) Query HAS a HUSC alias AND names a denylist major (e.g.
        #      "Ngành Luật HUSC") → still block, because the major-scope
        #      is a STRONGER signal than the school-scope. HUSC simply
        #      doesn't teach Luật, period.
        #  (c) Query HAS a HUSC alias AND names a real HUSC major
        #      (e.g. "Ngành CNTT HUSC") → fall through to the keyword
        #      path. The other-school guard handles the comparison case.
        if self._major_out_of_scope(query):
            if self._mentions_husc(query):
                # If the HUSC alias is present, only block when the named
                # major is in the denylist (case b). Otherwise let the
                # normal flow decide (case c).
                if not self._major_in_denylist(self._fold_diacritics(query)):
                    pass
                else:
                    return GuardrailDecision(
                        False,
                        "NOT_IN_HUSC_SCOPE",
                        "major_not_offered",
                        self._major_not_offered_answer(),
                        ["HUSC không đào tạo ngành/lĩnh vực này", "Xem danh sách ngành HUSC tại tuyensinh.husc.edu.vn/nganhdaotao-dh.php"],
                    )
            else:
                return GuardrailDecision(
                    False,
                    "NOT_IN_HUSC_SCOPE",
                    "major_not_offered",
                    self._major_not_offered_answer(),
                    ["HUSC không đào tạo ngành/lĩnh vực này", "Xem danh sách ngành HUSC tại tuyensinh.husc.edu.vn/nganhdaotao-dh.php"],
                )

        if self._looks_admission_related(query) and not self._mentions_other_school(query):
            return GuardrailDecision(True, "SUCCESS", "in_scope_keyword", "")

        if self._client is None:
            return GuardrailDecision(False, "NOT_IN_HUSC_SCOPE", "out_of_scope_heuristic", self._redirect_answer())

        prompt = (
            "Bạn là bộ lọc truy vấn cho chatbot tuyển sinh của Trường Đại học Khoa học (HUSC) — Đại học Huế. "
            "Bot CHỈ trả lời các câu hỏi TUYỂN SINH của HUSC (ngành, điểm chuẩn, học phí, tổ hợp xét tuyển, hồ sơ, học bổng, chỉ tiêu, chương trình đào tạo, so sánh ngành...). "
            "Câu hỏi về trường KHÁC (Bách Khoa, FPT, Y Hà Nội, Ngoại thương, Kinh tế Quốc dân, Tôn Đức Thắng, RMIT, UEH, UEL, Y Dược Huế, ĐH Huế đứng riêng không kèm HUSC...) là NGOÀI PHẠM VI. "
            "NGOẠI LỆ: nếu câu hỏi SO SÁNH trường khác với HUSC, hoặc hỏi về quan hệ/khác biệt giữa trường khác với HUSC, thì VẪN TRONG PHẠM VI (HUSC là chủ thể so sánh). "
            "HUSC có tên đầy đủ: Trường Đại học Khoa học — Đại học Huế (cũng gọi 'Đại học Khoa học Huế', 'Khoa học Huế', 'Trường Khoa học'). Mọi ngành HUSC (kể cả ngành mới 2026) đều hợp lệ. "
            "Trả về JSON: {\"is_in_scope\": bool, \"reason\": str}. "
            "Ví dụ in_scope: 'Học phí ngành CNTT HUSC?'; 'HUSC so với Bách Khoa ngành CNTT cái nào tốt hơn?'; 'Vật lý học - Công nghệ bán dẫn HUSC có gì hay?'; 'Đại học Huế khác HUSC như thế nào?'. "
            "Ví dụ out_of_scope: 'Điểm chuẩn ĐH Bách Khoa Hà Nội 2026?'; 'Học phí ĐH FPT?'; 'Trường FPT có những ngành gì?'; 'ĐH Y Hà Nội xét tuyển thế nào?'; 'Điểm chuẩn Đại học Huế Y Dược?'."
        )

        try:
            # Build the classifier user-message. ONLY the LLM-classifier path
            # gets the bounded prior-turn context — never the regex/folded
            # paths above, because folding prior turns corrupts major matching
            # and would re-introduce the DoS / false-block holes the
            # denylist/keyword fast-paths were hardened against.
            if history:
                history_text = _format_history_for_llm(history)
                user_content = (
                    f"Ngữ cảnh trước:\n{history_text}\n\n"
                    f"Câu hỏi hiện tại:\n{query}"
                )
            else:
                user_content = query
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content[content.find("{"):content.rfind("}") + 1])
            if bool(data.get("is_in_scope", False)):
                return GuardrailDecision(True, "SUCCESS", data.get("reason", "in_scope_llm"), "")
            return GuardrailDecision(False, "NOT_IN_HUSC_SCOPE", data.get("reason", "out_of_scope_llm"), self._redirect_answer())
        except Exception as exc:
            logger.warning(f"Guardrail precheck fallback: {exc}")
            return GuardrailDecision(False, "NOT_IN_HUSC_SCOPE", "out_of_scope_fallback", self._redirect_answer())

    async def classify_no_result(self, query: str) -> GuardrailDecision:
        if self._client is None:
            return GuardrailDecision(
                True,
                "INSUFFICIENT_DATA",
                "no_result_without_llm",
                "Mình chưa đủ dữ liệu để trả lời chính xác câu này trong kho tuyển sinh HUSC hiện tại.",
                ["Bổ sung chunk theo đúng ngành/sự việc được hỏi", "Thêm văn bản tuyển sinh mới nhất từ HUSC"],
            )

        prompt = (
            "Bạn là bộ phân tích lỗi truy vấn cho hệ thống tuyển sinh HUSC. "
            "Không có kết quả retrieval. Hãy phân loại 1 trong 2 mã: "
            "HUSC_ENTITY_NOT_FOUND hoặc INSUFFICIENT_DATA. "
            "Trả về JSON: {\"code\": string, \"reason\": string, \"hints\": [string, string]}."
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=180,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content[content.find("{"):content.rfind("}") + 1])
            code = data.get("code", "INSUFFICIENT_DATA")
            if code not in {"HUSC_ENTITY_NOT_FOUND", "INSUFFICIENT_DATA"}:
                code = "INSUFFICIENT_DATA"
            reason = data.get("reason", "no_result")
            hints = data.get("hints", []) if isinstance(data.get("hints", []), list) else []

            short_answer = (
                "Mình chưa tìm thấy ngành/sự việc này trong dữ liệu tuyển sinh HUSC hiện có."
                if code == "HUSC_ENTITY_NOT_FOUND"
                else "Mình chưa đủ dữ liệu để trả lời chính xác câu này trong kho tuyển sinh HUSC hiện tại."
            )
            return GuardrailDecision(True, code, reason, short_answer, hints)
        except Exception as exc:
            logger.warning(f"Guardrail no-result fallback: {exc}")
            return GuardrailDecision(
                True,
                "INSUFFICIENT_DATA",
                "no_result_fallback",
                "Mình chưa đủ dữ liệu để trả lời chính xác câu này trong kho tuyển sinh HUSC hiện tại.",
                ["Bổ sung thêm chunk liên quan truy vấn", "Cập nhật dữ liệu tuyển sinh mới nhất"],
            )

    def public_status(self, internal_code: str) -> str:
        mode = self._settings.ERROR_EXPOSURE_MODE.lower()
        if internal_code == "SUCCESS":
            return "SUCCESS"
        if mode == "prod":
            if internal_code == "SENSITIVE_PII_DETECTED":
                return "SENSITIVE_PII_DETECTED"
            return "NOT_IN_HUSC_SCOPE"
        return internal_code

    def expose_internal(self) -> bool:
        return self._settings.ERROR_EXPOSURE_MODE.lower() == "dev"
