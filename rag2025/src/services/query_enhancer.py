"""
HYDE Query Enhancement Service (Input Layer)

Hypothetical Document Embeddings (HYDE):
- Converts user query → 3-5 query variants → better retrieval
- Multi-LLM fallback: Gemini → GLM-4 → Groq
- Auto-classify query type for better routing
- Auto-estimate top_k based on query complexity
- Handles Vietnamese slang/abbreviations (đcm, hp, CNTT...)
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
import google.generativeai as genai
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


class HYDEQueryEnhancer:
    """
    HYDE: Hypothetical Document Embeddings
    Converts user query → 3-5 query variants (JSON) → better retrieval
    Uses optimized prompt from prompts/hyde_system_prompt.txt

    Priority: Groq (Llama 3.1) → GLM-4 → Gemini (fallback only)
    """

    def __init__(self):
        # Load HYDE system prompt from file
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "hyde_system_prompt.txt"
        if prompt_path.exists():
            self.hyde_system_prompt = prompt_path.read_text(encoding="utf-8")
            logger.info(f"Loaded HYDE prompt from {prompt_path}")
        else:
            logger.warning(f"HYDE prompt not found at {prompt_path}, using fallback")
            self.hyde_system_prompt = self._get_fallback_prompt()
        # Priority: Groq (Llama 3.1) → GLM-4 → Gemini (fallback only)
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.glm4_key = os.getenv("OPENAI_API_KEY")  # Z.AI endpoint
        self.gemini_key = os.getenv("GEMINI_API_KEY")  # Fallback only

        # Configure clients
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

        if self.glm4_key:
            self.glm4_client = AsyncOpenAI(
                api_key=self.glm4_key,
                base_url="https://open.bigmodel.cn/api/paas/v4"  # Z.AI/GLM-4 endpoint
            )

        if self.groq_key:
            self.groq_client = AsyncGroq(api_key=self.groq_key)
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def _get_fallback_prompt(self) -> str:
        """Fallback HYDE prompt if file not found"""
        return """Bạn là chuyên gia RAG cho hệ thống chatbot tuyển sinh đại học Việt Nam.

Nhiệm vụ: Phân tích câu hỏi của người dùng và tạo ra 3-5 biến thể câu hỏi khác nhau để cải thiện khả năng truy xuất thông tin.

Đầu vào: Một câu hỏi từ người dùng (có thể chứa tiếng lóng, viết tắt, từ ngữ suồng sã)

Đầu ra (chỉ trả về JSON):
{
  "original_query": "câu hỏi gốc",
  "detected_intent": "điểm chuẩn/học phí/điều kiện/tuyển sinh/thủ tục/...",
  "variants": [
    "biến thể 1 (trang trọng, đầy đủ)",
    "biến thể 2 (tập trung vào khía cạnh khác)",
    "biến thể 3 (thêm từ khóa liên quan)",
    ...
  ]
}

QUY TẮC:
1. Đúng JSON, không có markdown, không có văn bản ngoài JSON
2. Variants phải đa dạng về cấu trúc và từ khóa
3. Chuyển đổi tiếng lóng/viết tắt → từ ngữ chính thức:
   - "đcm" → "điểm chuẩn"
   - "hp" → "học phí"
   - "cntt" → "công nghệ thông tin"
   - "ut" → "uy tín" hoặc "điểm ưu tiên"
4. Variants tối đa 30 từ mỗi câu
5. Ít nhất 3 biến thể, tối đa 5 biến thể

Câu hỏi cần xử lý:"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_query_variants(self, query: str) -> Dict[str, Any]:
        """
        Generate 3-5 query variants using LLM with JSON output

        Priority: Groq (Llama 3.1) → GLM-4 → Gemini
        """
        prompt = f"{self.hyde_system_prompt}\n\n{query}"

        # Try Groq (Llama 3.1) first
        if self.groq_key:
            try:
                response = await self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": self.hyde_system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                logger.info("HYDE: Using Groq (Llama 3.1)")
                return self._parse_json_response(response.choices[0].message.content, query)
            except Exception as e:
                logger.warning(f"Groq failed: {e}, trying GLM-4...")

        # Fallback to GLM-4
        if self.glm4_key:
            try:
                response = await self.glm4_client.chat.completions.create(
                    model="glm-4-plus",
                    messages=[
                        {"role": "system", "content": self.hyde_system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                logger.info("HYDE: Using GLM-4")
                return self._parse_json_response(response.choices[0].message.content, query)
            except Exception as e:
                logger.warning(f"GLM-4 failed: {e}, trying Gemini...")

        # Fallback to Gemini (last resort)
        if self.gemini_key:
            try:
                response = await self.gemini_model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=500
                    )
                )
                logger.info("HYDE: Using Gemini (fallback)")
                return self._parse_json_response(response.text, query)
            except Exception as e:
                logger.warning(f"Gemini failed: {e}")

        raise Exception("All LLM providers failed")

    def _parse_json_response(self, response_text: str, original_query: str = "") -> Dict[str, Any]:
        """Parse LLM response to JSON, handling markdown code blocks"""
        text = response_text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            parsed = json.loads(text)

            # Ensure required fields exist
            if "original_query" not in parsed:
                parsed["original_query"] = original_query
            if "detected_intent" not in parsed:
                parsed["detected_intent"] = self._detect_intent_from_query(original_query)
            if "variants" not in parsed or not isinstance(parsed["variants"], list):
                parsed["variants"] = []

            # Ensure at least 1 variant
            if len(parsed["variants"]) == 0 and parsed["original_query"]:
                parsed["variants"] = [parsed["original_query"]]

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}, raw response: {text}")
            # Return smart fallback based on query analysis
            return self._create_fallback_response(original_query)

    def _detect_intent_from_query(self, query: str) -> str:
        """Detect intent from query keywords"""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["học phí", "hp", "tiền", "tín chỉ", "tín", "phí", "bao nhiêu"]):
            return "học phí"
        elif any(kw in query_lower for kw in ["điểm chuẩn", "đcm", "điểm"]):
            return "điểm chuẩn"
        elif any(kw in query_lower for kw in ["ngành", "chuyên ngành", "chỉ tiêu"]):
            return "ngành học"
        elif any(kw in query_lower for kw in ["tuyển sinh", "xét tuyển", "đăng ký"]):
            return "tuyển sinh"
        return "general"

    def _create_fallback_response(self, query: str) -> Dict[str, Any]:
        """Create intelligent fallback when JSON parse fails"""
        intent = self._detect_intent_from_query(query)

        # Generate smart variants based on detected intent
        variants = [query]

        if intent == "học phí":
            variants = [
                f"Học phí Đại học Khoa học Huế {query}",
                "Mức thu học phí tín chỉ Đại học Khoa học Huế năm 2025",
                "Đơn giá tín chỉ HUSC năm học 2025-2026",
                "Chi phí học tập Đại học Khoa học Huế"
            ]
        elif intent == "điểm chuẩn":
            variants = [
                f"Điểm chuẩn Đại học Khoa học Huế {query}",
                "Điểm chuẩn các ngành HUSC năm 2025",
                "Điểm xét tuyển Đại học Khoa học Huế"
            ]
        elif intent == "ngành học":
            variants = [
                f"Ngành đào tạo Đại học Khoa học Huế {query}",
                "Danh sách ngành tuyển sinh HUSC 2025",
                "Chỉ tiêu tuyển sinh Đại học Khoa học Huế"
            ]

        return {
            "original_query": query,
            "detected_intent": intent,
            "variants": variants
        }

    def classify_query_type(self, query: str) -> str:
        """
        Auto-classify query type for better routing
        """
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["điều kiện", "yêu cầu", "tuyển sinh"]):
            return "admission_criteria"
        elif any(kw in query_lower for kw in ["hồ sơ", "giấy tờ", "nộp"]):
            return "documents"
        elif any(kw in query_lower for kw in ["điểm", "thang điểm", "tính điểm"]):
            return "scoring"
        elif any(kw in query_lower for kw in ["ngành", "chuyên ngành", "đào tạo"]):
            return "major_info"
        elif any(kw in query_lower for kw in ["thời gian", "lịch", "deadline"]):
            return "timeline"
        else:
            return "general"

    def estimate_top_k(self, query: str, query_type: str) -> int:
        """
        Auto-adjust top_k based on query complexity
        """
        if query_type == "general" or len(query.split()) > 15:
            return 7  # Complex query needs more context
        elif query_type in ["admission_criteria", "documents"]:
            return 5  # Standard
        else:
            return 3  # Simple factual query

    async def enhance_query(
        self,
        user_query: str,
        force_rag_only: bool = False
    ) -> Dict[str, Any]:
        """
        Main method: Convert user string → QueryRequest dict with variants

        Returns:
            {
                "original_query": "user's original query",
                "detected_intent": "intent type",
                "variants": ["variant 1", "variant 2", ...],
                "top_k": 5,
                "force_rag_only": false,
                "query_type": "admission_criteria"
            }
        """
        logger.info(f"Enhancing query: {user_query}")

        # 1. Classify query type (for routing)
        query_type = self.classify_query_type(user_query)

        # 2. Estimate top_k based on query complexity
        top_k = self.estimate_top_k(user_query, query_type)

        # 3. Generate query variants using HYDE
        try:
            variants_result = await self.generate_query_variants(user_query)

            # Extract detected intent from variants result
            detected_intent = variants_result.get("detected_intent", query_type)
            variants = variants_result.get("variants", [])

            # Ensure at least original query in variants
            if len(variants) == 0:
                variants = [user_query]

        except Exception as e:
            logger.error(f"HYDE failed: {e}, using fallback")
            detected_intent = query_type
            variants = [user_query]

        return {
            "original_query": user_query,
            "detected_intent": detected_intent,
            "variants": variants,
            "top_k": top_k,
            "force_rag_only": force_rag_only,
            "query_type": query_type
        }


# Global instance
query_enhancer = HYDEQueryEnhancer()
