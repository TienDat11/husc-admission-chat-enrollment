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
    ADMISSION_KEYWORDS = [
        "tuyển sinh", "điểm chuẩn", "học phí", "ngành", "tổ hợp", "xét tuyển",
        "husc", "đại học khoa học huế", "chỉ tiêu", "học bổng", "hồ sơ",
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

        if self._looks_admission_related(query):
            return GuardrailDecision(True, "SUCCESS", "in_scope_keyword", "")

        if self._client is None:
            return GuardrailDecision(False, "NOT_IN_HUSC_SCOPE", "out_of_scope_heuristic", self._redirect_answer())

        prompt = (
            "Bạn là bộ lọc truy vấn cho chatbot tuyển sinh HUSC. "
            "Phân loại câu hỏi thuộc phạm vi tuyển sinh HUSC hay không. "
            "Trả về JSON: {\"is_in_scope\": boolean, \"reason\": string}."
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
