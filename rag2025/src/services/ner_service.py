"""
NER Service – Named Entity Recognition & Triple Extraction

Uses UnifiedLLMClient (gemini-2.5-flash via ramclouds.me/v1) as primary.
Auto-fallback to Groq if primary fails.

DDD: Application service that depends on domain entities, not infrastructure.
Scalable: Adding entity types = extend EntityType enum + update prompt only.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.domain.entities import (
    Chunk,
    Entity,
    EntityType,
    ExtractionResult,
    RelationType,
    Triple,
)
from src.services.llm_client import UnifiedLLMClient, get_llm_client


_NER_SYSTEM_PROMPT = """Bạn là hệ thống trích xuất thông tin cấu trúc từ văn bản tuyển sinh đại học Việt Nam.

Nhiệm vụ: Từ đoạn văn bản được cung cấp, trích xuất:
1. **Thực thể** (entities): tên ngành, mã ngành, tổ hợp môn, điểm chuẩn, học phí, thời gian đào tạo, tổ chức.
2. **Quan hệ** (triples): bộ ba (thực thể_đầu, quan_hệ, thực thể_cuối).

Loại thực thể hợp lệ: NGANH, TO_HOP, DIEM_CHUAN, HOC_PHI, THOI_GIAN, TO_CHUC, CHINH_SACH, UNKNOWN
Loại quan hệ hợp lệ: CO_TO_HOP, CO_DIEM, THUOC_TRUONG, YEU_CAU, LIEN_QUAN

QUY TẮC BẮT BUỘC:
- Chỉ trả về JSON hợp lệ, không có markdown, không có văn bản bên ngoài.
- "normalized": chuỗi viết thường, không dấu, dùng dấu gạch dưới thay khoảng trắng.
- Nếu không tìm thấy: {"entities": [], "triples": []}.

Ví dụ đầu ra:
{
  "entities": [
    {"text": "Công nghệ thông tin", "type": "NGANH", "normalized": "cong_nghe_thong_tin"},
    {"text": "A00", "type": "TO_HOP", "normalized": "a00"},
    {"text": "Đại học Khoa học Huế", "type": "TO_CHUC", "normalized": "dai_hoc_khoa_hoc_hue"}
  ],
  "triples": [
    {"head": "NGANH:cong_nghe_thong_tin", "relation": "CO_TO_HOP", "tail": "TO_HOP:a00"},
    {"head": "NGANH:cong_nghe_thong_tin", "relation": "THUOC_TRUONG", "tail": "TO_CHUC:dai_hoc_khoa_hoc_hue"}
  ]
}"""


class NERService:
    """Named Entity Recognition service for Vietnamese admission text.

    Uses UnifiedLLMClient (gemini-2.5-flash → Groq fallback).
    Fully swappable: inject any UnifiedLLMClient instance.

    Args:
        llm: UnifiedLLMClient instance (auto-created if None).
    """

    def __init__(self, llm: Optional[UnifiedLLMClient] = None) -> None:
        self._llm = llm or get_llm_client()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and triples from a single chunk.

        Args:
            chunk: Domain Chunk object.

        Returns:
            ExtractionResult with entities and triples, or error info.
        """
        try:
            data = await self._llm.chat_json(
                user_message=chunk.text,
                system_message=_NER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
            return self._parse(data, chunk.chunk_id)
        except Exception as exc:
            logger.warning(f"NER failed for {chunk.chunk_id}: {exc}")
            return ExtractionResult(chunk_id=chunk.chunk_id, error=str(exc))

    def _parse(self, data: Dict[str, Any], chunk_id: str) -> ExtractionResult:
        """Parse LLM JSON output into domain objects."""
        entities: List[Entity] = []
        for item in data.get("entities", []):
            try:
                entity_type = EntityType(item.get("type", "UNKNOWN"))
            except ValueError:
                entity_type = EntityType.UNKNOWN
            entity = Entity(
                text=item.get("text", ""),
                entity_type=entity_type,
                normalized=item.get("normalized", "").lower().replace(" ", "_"),
                chunk_id=chunk_id,
            )
            if entity.text:
                entities.append(entity)

        triples: List[Triple] = []
        for item in data.get("triples", []):
            try:
                relation = RelationType(item.get("relation", "LIEN_QUAN"))
            except ValueError:
                relation = RelationType.LIEN_QUAN
            triple = Triple(
                head=item.get("head", ""),
                relation=relation,
                tail=item.get("tail", ""),
                chunk_id=chunk_id,
            )
            if triple.head and triple.tail:
                triples.append(triple)

        return ExtractionResult(chunk_id=chunk_id, entities=entities, triples=triples)

    async def extract_batch(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        """Extract entities from a list of chunks sequentially."""
        results: List[ExtractionResult] = []
        for i, chunk in enumerate(chunks):
            logger.info(f"NER: {i+1}/{len(chunks)} – {chunk.chunk_id}")
            result = await self.extract(chunk)
            results.append(result)
        return results
