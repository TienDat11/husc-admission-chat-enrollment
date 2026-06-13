from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from groq import AsyncGroq
from loguru import logger

from config.settings import RAGSettings


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
    ]

    PII_KEYWORDS = [
        "cccd", "cmnd", "số căn cước", "số tài khoản", "mật khẩu", "otp", "cvv",
    ]

    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?84|0)(?:[\s.-]?\d){8,10}\b"),
        "citizen_id": re.compile(r"\b\d{12}\b"),
    }

    def __init__(self, settings: RAGSettings):
        self._settings = settings
        self._enabled = settings.GUARDRAIL_ENABLED
        self._groq_key = os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=self._groq_key) if self._groq_key else None
        self._model = settings.GUARDRAIL_MODEL

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

    def _contains_sensitive_pii(self, query: str) -> bool:
        q = query.lower()
        if any(k in q for k in self.PII_KEYWORDS):
            return True
        return any(pattern.search(query) is not None for pattern in self.PII_PATTERNS.values())

    async def precheck(self, query: str) -> GuardrailDecision:
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
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
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
