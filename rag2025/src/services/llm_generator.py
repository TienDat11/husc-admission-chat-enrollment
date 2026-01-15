"""
LLM Answer Generation Service (Generation Layer)

Generate final answer from retrieved chunks using:
- GLM-4.5 (Z.AI) - Smartest (NEW!)
- Groq Llama 3.1 (fallback - FREE)
- Anti-redundancy optimized prompt (50-120 words target)
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    logger.warning("zai-sdk not installed. GLM-4.5 will be disabled.")


class LLMGenerator:
    """
    Generate final answer from retrieved chunks
    Uses optimized anti-redundancy prompt from prompts/generation_system_prompt.txt

    Priority: GLM-4.5 (Z.AI) → Groq (Llama 3.1)
    """

    def __init__(self):
        # Load generation system prompt from file
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "generation_system_prompt.txt"
        if prompt_path.exists():
            self.generation_system_prompt = prompt_path.read_text(encoding="utf-8")
            logger.info(f"Loaded generation prompt from {prompt_path}")
        else:
            logger.warning(f"Generation prompt not found at {prompt_path}, using fallback")
            self.generation_system_prompt = self._get_fallback_prompt()

        # Configure GLM-4.5 client (Z.AI) - Priority 1
        self.zai_key = os.getenv("ZAI_API_KEY")
        if ZAI_AVAILABLE and self.zai_key:
            self.zai_client = ZaiClient(api_key=self.zai_key)
            logger.info("GLM-4.5 (Z.AI) client initialized")
        else:
            self.zai_client = None
            if not ZAI_AVAILABLE:
                logger.warning("zai-sdk not installed, GLM-4.5 disabled")
            else:
                logger.warning("ZAI_API_KEY not set, GLM-4.5 disabled")

        # Configure Groq client (Llama 3.1) - Priority 2 (fallback)
        self.groq_key = os.getenv("GROQ_API_KEY")
        if self.groq_key:
            self.groq_client = AsyncGroq(api_key=self.groq_key)
            logger.info("Groq (Llama 3.1) client initialized (fallback)")
        else:
            self.groq_client = None
            logger.warning("GROQ_API_KEY not set, Groq disabled")

        if not self.zai_client and not self.groq_client:
            raise Exception("At least one LLM API key required (ZAI_API_KEY or GROQ_API_KEY)!")

    def _get_fallback_prompt(self) -> str:
        """Fallback generation prompt if file not found"""
        return """Bạn là chuyên gia tư vấn tuyển sinh đại học Việt Nam 2025.

Nhiệm vụ: Trả lời câu hỏi dựa trên context được cung cấp.

QUY TẮC ANTI-REDUNDANCY (QUAN TRỌNG):
1. Đọc toàn bộ context trước khi trả lời
2. Xác định đâu là thông tin UNIQUE vs TRÙNG
3. GỘP thông tin trùng lặp thành 1 câu trả lời duy nhất

4. Độ dài câu trả lời: 50-120 từ (ngắn gọn, súc tích)

5. Ưu tiên thông tin từ summary - thường tóm tắt tốt nhất

6. Khi KHÔNG có thông tin:
   Trả lời: "Tôi không tìm thấy thông tin này trong tài liệu hiện có."
   KHÔNG bịa thông tin, KHÔNG dùng knowledge ngoài

7. Tone: Thân thiện, chuyên nghiệp như tư vấn viên tuyển sinh
"""

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context from retrieved chunks
        Include both summary AND text for complete information
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            # Get all available fields
            summary = chunk.get("summary", "")
            text = chunk.get("text", chunk.get("text_plain", ""))
            source = chunk.get("metadata", {}).get("source", "Không rõ nguồn")
            info_type = chunk.get("metadata", {}).get("info_type", "")

            # Include BOTH summary and full text for complete context
            # This ensures LLM has access to specific numbers (e.g., 545.000 VNĐ/tín chỉ)
            content_parts = []
            if summary:
                content_parts.append(f"Tóm tắt: {summary}")
            if text and text != summary:
                content_parts.append(f"Chi tiết: {text}")

            content = "\n".join(content_parts) if content_parts else text

            # Add metadata tags for context
            context_parts.append(
                f"[Đoạn {i}] (Nguồn: {source} | Loại: {info_type})\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        confidence: float,
        is_program_list_query: bool = False,  # Flag for program count/list queries
    ) -> Dict[str, Any]:
        """
        Generate answer with context from chunks using anti-redundancy prompt
        Target length: 50-120 words

        Args:
            query: User query
            chunks: Retrieved chunks
            confidence: Confidence score
            is_program_list_query: Whether this is a program list/count query (for optimization)
        """

        # Build context
        context = self._build_context(chunks)

        # Build full prompt with system instructions
        full_prompt = f'{self.generation_system_prompt}\n{context}\n\n**Câu hỏi**: {query}\n\n**Câu trả lời**:'

        # DEBUG: Log the full prompt for troubleshooting
        logger.debug(f"Full prompt length: {len(full_prompt)} chars")
        logger.debug(f"Context preview: {context[:500]}...")

        # Token limit for different query types
        max_tokens = 2000 if is_program_list_query else 800

        # Priority: Groq (Llama 3.1) → GLM-4.5 (Z.AI)
        answer = ""
        provider = ""

        try:
            # Try Groq first (Llama 3.3) - Primary
            if self.groq_client:
                logger.info("Generation: Using Groq (Llama 3.3)")

                # Use system + user messages for better instruction following
                response = await self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Use Llama 3.3 70B
                    messages=[
                        {"role": "system", "content": self.generation_system_prompt},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\n---\n\nCÂU HỎI: {query}\n\nHãy trả lời dựa trên context trên. Nếu context có thông tin về học phí (ví dụ: 545.000 VNĐ/tín chỉ), hãy sử dụng số liệu đó."}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                answer = response.choices[0].message.content.strip()
                provider = "Llama-3.3-70B (Groq)"

            # Fallback to GLM-4.5 (Z.AI) if Groq unavailable
            elif self.zai_client:
                logger.info("Generation: Falling back to GLM-4.5 (Z.AI)")
                response = self.zai_client.chat.completions.create(
                    model="glm-4-32b-0414-128k",
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    thinking={"type": "enabled"},  # Enable thinking for better reasoning
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
                provider = "GLM-4.5 (Z.AI)"

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback: return simple message
            answer = "Tôi không tìm thấy thông tin này trong tài liệu hiện có."
            provider = "Fallback"

        # Extract sources
        sources = list(set([
            chunk.get("metadata", {}).get("source", "Không rõ nguồn")
            for chunk in chunks
        ]))

        # Use first 3 chunks for chunks_used calculation
        chunks_used = min(len(chunks), 3)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "provider": provider,
            "chunks_used": chunks_used
        }


# Global instance
llm_generator = LLMGenerator()
