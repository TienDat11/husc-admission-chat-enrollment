"""
Smart Query Router – HyDE + Step-Back Prompting + Query Classification

Pipeline (per query):
1. Step-Back Prompting: abstract the query to find underlying principle/concept
2. HyDE (Hypothetical Document Embeddings): generate a fake answer document
   → embed both original + hypothetical → average vector for better retrieval
3. Query Classification: decide routing
   - "simple"      → PaddedRAG (vector + BM25 + cross-encoder, 1-hop)
   - "multihop"    → GraphRAG (PPR + entity graph traversal)
   - "comparative" → GraphRAG (multi-entity comparison via graph)

Vietnamese domain: HUSC admission system.

Design: Stateless service. All LLM calls go through UnifiedLLMClient.
Scalable: Adding new query types = extend the CLASSIFY_SYSTEM_PROMPT only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.services.llm_client import UnifiedLLMClient, get_llm_client


class QueryRoute(str, Enum):
    """Routing decision for a query."""
    PADDED_RAG = "padded_rag"     # Simple 1-hop → vector search sufficient
    GRAPH_RAG = "graph_rag"       # Multi-hop / comparative → PPR graph needed


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
    """
    original_query: str
    step_back_query: str
    hypothetical_doc: str
    hyde_variants: List[str]
    route: QueryRoute
    complexity: int
    intent: str
    reasoning: str

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


# ─── System Prompts ────────────────────────────────────────────────────────────

_STEP_BACK_SYSTEM = """Bạn là chuyên gia giáo dục đại học Việt Nam, am hiểu hệ thống tuyển sinh HUSC (Đại học Khoa học Huế).

**Kỹ thuật Step-Back Prompting**: Từ câu hỏi cụ thể, hãy rút ra câu hỏi TỔNG QUÁT hơn về nguyên lý/khái niệm nền tảng.

Ví dụ:
- Cụ thể: "Ngành CNTT tại HUSC điểm chuẩn 2025 là bao nhiêu?"
- Tổng quát: "Quy trình xác định điểm chuẩn xét tuyển đại học tại Việt Nam hoạt động như thế nào?"

- Cụ thể: "Để học ngành Kỹ thuật phần mềm cần tổ hợp nào?"
- Tổng quát: "Các tổ hợp môn xét tuyển vào ngành kỹ thuật tại các trường đại học Việt Nam thường bao gồm những gì?"

QUY TẮC: Chỉ trả về câu hỏi tổng quát, không giải thích, không markdown."""

_HYDE_SYSTEM = """Bạn là chuyên gia tư vấn tuyển sinh Đại học Khoa học Huế (HUSC).

**Kỹ thuật HyDE**: Tạo ra một đoạn văn bản GIẢ ĐỊNH nghe như đoạn văn bản từ tài liệu tuyển sinh chính thức, có thể trả lời câu hỏi được đặt ra.

Đoạn văn bản này KHÔNG cần chính xác 100% – mục tiêu là tạo ra embedding vector phù hợp với corpus tài liệu.

QUY TẮC:
- Viết bằng tiếng Việt, phong cách văn bản hành chính/tuyển sinh
- Dài 50-100 từ
- Dùng các thuật ngữ tuyển sinh: "chỉ tiêu", "tổ hợp xét tuyển", "điểm chuẩn", "học phí tín chỉ", v.v.
- KHÔNG dùng markdown, không giải thích
- Chỉ trả về đoạn văn bản giả định"""

_CLASSIFY_SYSTEM = """Bạn là hệ thống phân loại câu hỏi tuyển sinh đại học Việt Nam.

Phân tích câu hỏi và trả về JSON với cấu trúc:
{
  "route": "padded_rag" | "graph_rag",
  "complexity": 1-5,
  "intent": "diem_chuan" | "hoc_phi" | "nganh_hoc" | "to_hop" | "chinh_sach" | "thu_tuc" | "so_sanh" | "da_hop",
  "reasoning": "lý do ngắn gọn (≤20 từ)",
  "hyde_variants": ["variant 1", "variant 2", "variant 3"]
}

ROUTING RULES:
- "padded_rag": câu hỏi đơn giản, 1 thực thể, 1 thông tin cụ thể
  → Ví dụ: "điểm chuẩn ngành X", "học phí bao nhiêu", "tổ hợp ngành Y"
  
- "graph_rag": câu hỏi đa chiều, nhiều thực thể, so sánh, liên quan
  → Ví dụ: "so sánh ngành X và Y", "ngành nào phù hợp nếu điểm A với tổ hợp B",
             "điều kiện nào để được ưu tiên + xét tuyển ngành Z", "kết hợp nhiều tiêu chí"

COMPLEXITY:
1 = tra cứu 1 thông tin    4 = liên kết 3+ thực thể
2 = tra cứu 2 thông tin    5 = so sánh / phân tích đa chiều  
3 = liên kết 2 thực thể

Chỉ trả về JSON hợp lệ, không có text bên ngoài."""


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

    async def _classify(self, query: str, hyde_doc: str) -> Dict:
        """Classify query and get routing decision + variants."""
        user_msg = f"Câu hỏi gốc: {query}\n\nTài liệu giả định (HyDE): {hyde_doc}"
        try:
            data = await self._llm.chat_json(
                user_message=user_msg,
                system_message=_CLASSIFY_SYSTEM,
                temperature=0.1,
                max_tokens=512,
            )
            return data
        except Exception as exc:
            logger.warning(f"Classification failed: {exc} – defaulting to padded_rag")
            return {
                "route": "padded_rag",
                "complexity": 1,
                "intent": "general",
                "reasoning": "classification failed",
                "hyde_variants": [query],
            }

    async def route(self, query: str) -> RouterResult:
        """Full routing pipeline: step-back → HyDE → classify.

        Args:
            query: Raw user query (Vietnamese).

        Returns:
            RouterResult with all intermediate outputs and final route.
        """
        logger.info(f"Router: processing query [{query[:60]}]")

        step_back = await self._step_back(query)
        hyde_doc = await self._generate_hyde_doc(query, step_back)
        classification = await self._classify(query, hyde_doc)

        route_str = classification.get("route", "padded_rag")
        complexity = int(classification.get("complexity", 1))

        # Override: if complexity > threshold, force graph_rag
        if complexity > self._threshold and route_str == "padded_rag":
            route_str = "graph_rag"
            logger.debug(f"Complexity={complexity} > threshold={self._threshold}: upgrading to graph_rag")

        route = QueryRoute.GRAPH_RAG if route_str == "graph_rag" else QueryRoute.PADDED_RAG

        hyde_variants: List[str] = classification.get("hyde_variants", [])
        if not hyde_variants:
            hyde_variants = [query]

        result = RouterResult(
            original_query=query,
            step_back_query=step_back,
            hypothetical_doc=hyde_doc,
            hyde_variants=hyde_variants,
            route=route,
            complexity=complexity,
            intent=classification.get("intent", "general"),
            reasoning=classification.get("reasoning", ""),
        )

        logger.info(
            f"Router decision: route={route.value}, complexity={complexity}, "
            f"intent={result.intent}"
        )
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
