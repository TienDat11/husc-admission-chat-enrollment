"""
NER Service – Named Entity Recognition & Triple Extraction

Uses UnifiedLLMClient (gemini-2.5-flash via ramclouds.me/v1) as primary.
Auto-fallback to Groq if primary fails.

DDD: Application service that depends on domain entities, not infrastructure.
Scalable: Adding entity types = extend EntityType enum + update prompt only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import APIStatusError
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


# --- Regex Fallback Patterns ---
# Used when LLM-based NER fails or returns empty results.

_NGANH_KEYWORDS = [
    "công nghệ thông tin", "kỹ thuật phần mềm", "khoa học máy tính",
    "toán học", "vật lý", "hóa học", "sinh học", "địa lý",
    "lịch sử", "văn học", "hán nôm", "hán - nôm", "đông phương học",
    "triết học", "báo chí", "truyền thông số", "xã hội học",
    "công tác xã hội", "quản lý văn hóa", "khoa học môi trường",
    "kỹ thuật điện tử viễn thông", "công nghệ sinh học",
    "kiến trúc", "địa kỹ thuật xây dựng", "kỹ thuật trắc địa bản đồ",
    "quản lý tài nguyên và môi trường", "khoa học vật liệu",
    "quản trị và phân tích dữ liệu",
    "cnkt điện tử viễn thông", "cnkt hóa học",
    "cntt", "ktpm", "khmt",
    # 2026 additions
    "tin học", "khoa học dữ liệu", "trí tuệ nhân tạo",
    "điện tử viễn thông",
    "quản trị nhà nước", "quản lý nhà nước",
    "quản lý tài nguyên môi trường", "ql tài nguyên môi trường",
    "ql an toàn sức khỏe môi trường",
    "kt trắc địa bản đồ",
    "cnkt điện tử viễn thông", "cnkt hóa học",
    "vật lý học", "khoa học máy tính",
]

# Normalize mapping: keyword → normalized entity name
_NGANH_NORMALIZE_MAP = {
    "cntt": "cong_nghe_thong_tin",
    "ktpm": "ky_thuat_phan_mem",
    "khmt": "khoa_hoc_may_tinh",
    "tin học": "cong_nghe_thong_tin",
    "khoa học dữ liệu": "khoa_hoc_du_lieu",
    "trí tuệ nhân tạo": "tri_tue_nhan_tao",
    "điện tử viễn thông": "ky_thuat_dien_tu_vien_thong",
    "quản trị nhà nước": "quan_ly_nha_nuoc",
    "quản lý nhà nước": "quan_ly_nha_nuoc",
    "công tác xã hội": "cong_tac_xa_hoi",
    "quản lý văn hóa": "quan_ly_van_hoa",
    "truyền thông số": "truyen_thong_so",
    "hán nôm": "han_nom",
    "đông phương học": "dong_phuong_hoc",
    "cnkt điện tử viễn thông": "ky_thuat_dien_tu_vien_thong",
    "cnkt hóa học": "ky_thuat_hoa_hoc",
    "kt trắc địa bản đồ": "ky_thuat_trac_dia_ban_do",
    "địa kỹ thuật xây dựng": "dia_ky_thuat_xay_dung",
    "kiến trúc": "kien_truc",
    "khoa học môi trường": "khoa_hoc_moi_truong",
    "vật lý học": "vat_ly_hoc",
    "quản lý tài nguyên môi trường": "quan_ly_tai_nguyen_va_moi_truong",
    "ql tài nguyên môi trường": "quan_ly_tai_nguyen_va_moi_truong",
    "ql an toàn sức khỏe môi trường": "quan_ly_an_toan_suc_khoe_moi_truong",
}

_MA_NGANH_PATTERN = re.compile(
    r'\b(722|729|731|732|742|744|746|748|751|758|760|761|776|785|831)\d{4}(?:VJ|TD)?\b'
)

_TO_HOP_PATTERN = re.compile(r'\b(A00|A01|A02|A15|B00|C00|C01|C02|C03|C19|D01|D07|D09|D10|D14)\b', re.IGNORECASE)

_DIEM_PATTERN = re.compile(r'(\d{1,2}[.,]\d{1,2})\s*(điểm|đ|diem|d)\b', re.IGNORECASE)

_HOC_PHI_PATTERN = re.compile(r'(\d{3}\.000)\s*(VNĐ|đ|dong)?/tín chỉ', re.IGNORECASE)

_NGANH_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _NGANH_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


def _regex_extract(text: str) -> Dict[str, Any]:
    """Extract entities using regex patterns when LLM NER fails.

    Returns the same JSON structure as LLM NER output:
    {"entities": [...], "triples": [...]}
    """
    entities = []

    # Extract NGANH entities
    seen_nganh: set = set()
    for match in _NGANH_PATTERN.finditer(text):
        raw = match.group(0)
        norm = raw.lower().replace(" ", "_").replace("-", "_")
        key = f"NGANH:{norm}"
        if key not in seen_nganh:
            seen_nganh.add(key)
            entities.append({
                "text": raw,
                "type": "NGANH",
                "normalized": norm,
            })

    # Extract TO_HOP entities
    seen_th: set = set()
    for match in _TO_HOP_PATTERN.finditer(text):
        raw = match.group(0).upper()
        norm = raw.lower()
        key = f"TO_HOP:{norm}"
        if key not in seen_th:
            seen_th.add(key)
            entities.append({
                "text": raw,
                "type": "TO_HOP",
                "normalized": norm,
            })

    # Extract DIEM_CHUAN entities
    seen_diem: set = set()
    for match in _DIEM_PATTERN.finditer(text):
        raw = match.group(0).replace(",", ".")
        norm = re.sub(r'[^\d.]', '', raw)
        key = f"DIEM_CHUAN:{norm}"
        if key not in seen_diem:
            seen_diem.add(key)
            entities.append({
                "text": raw,
                "type": "DIEM_CHUAN",
                "normalized": norm,
            })

    # Extract HOC_PHI entities
    seen_hp: set = set()
    for match in _HOC_PHI_PATTERN.finditer(text):
        raw = match.group(0)
        norm = match.group(1).replace(".", "")
        key = f"HOC_PHI:{norm}"
        if key not in seen_hp:
            seen_hp.add(key)
            entities.append({
                "text": raw,
                "type": "HOC_PHI",
                "normalized": norm,
            })

    # Build simple triples: NGANH -> CO_TO_HOP -> TO_HOP
    triples = []
    nganh_list = [e for e in entities if e["type"] == "NGANH"]
    tohop_list = [e for e in entities if e["type"] == "TO_HOP"]
    if nganh_list and tohop_list:
        for ng in nganh_list[:3]:
            for th in tohop_list[:3]:
                triples.append({
                    "head": f"NGANH:{ng['normalized']}",
                    "relation": "CO_TO_HOP",
                    "tail": f"TO_HOP:{th['normalized']}",
                })

    return {"entities": entities, "triples": triples}


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

    Uses UnifiedLLMClient with RAMCLOUDS_NER_MODEL (deepseek-v4-flash by default).
    Fully swappable: inject any UnifiedLLMClient instance.

    Args:
        llm: UnifiedLLMClient instance (auto-created with NER model if None).
    """

    def __init__(
        self,
        llm: Optional[UnifiedLLMClient] = None,
        fallback_llm: Optional[UnifiedLLMClient] = None,
    ) -> None:
        # NER ladder: primary LLM → fallback LLM → regex.
        # The fallback model is sourced (in priority order) from
        # RAMCLOUDS_NER_FALLBACK_MODEL, then RAMCLOUDS_GEN_FALLBACK_MODEL.
        # Reason (2026-06-10): the primary NER model can degrade on the
        # gateway (e.g. gpt-5.5-chat returning a near-empty entity set for
        # major lists); a second LLM recovers rich entities before we drop
        # to the lower-recall regex extractor.
        if llm is None:
            ner_model = os.getenv("RAMCLOUDS_NER_MODEL", "deepseek-v4-flash")
            llm = UnifiedLLMClient(force_model=ner_model)
            logger.info(f"NER: using model={ner_model}")
            if fallback_llm is None:
                fb_model = (
                    os.getenv("RAMCLOUDS_NER_FALLBACK_MODEL")
                    or os.getenv("RAMCLOUDS_GEN_FALLBACK_MODEL")
                )
                if fb_model and fb_model != ner_model:
                    fallback_llm = UnifiedLLMClient(force_model=fb_model)
                    logger.info(f"NER: fallback model={fb_model}")
        self._llm = llm
        self._fallback_llm = fallback_llm

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and triples from a single chunk.

        Ladder: primary LLM → fallback LLM (if configured) → regex.
        The primary is retried per the decorator; a primary that raises or
        returns ZERO entities hands off to the fallback LLM, and only if that
        also fails/empties do we drop to the regex extractor.

        Args:
            chunk: Domain Chunk object.

        Returns:
            ExtractionResult with entities and triples, or error info.
        """
        primary = await self._extract_with_llm(self._llm, chunk, label="primary")
        if primary is not None and primary.entities:
            return primary

        # Primary failed or returned empty → try the fallback LLM if present.
        if self._fallback_llm is not None:
            fb = await self._extract_with_llm(self._fallback_llm, chunk, label="fallback")
            if fb is not None and fb.entities:
                logger.info(
                    f"NER fallback LLM recovered {len(fb.entities)} entities "
                    f"for {chunk.chunk_id} (primary empty/failed)"
                )
                return fb

        # Both LLMs failed/empty → regex extractor (lower recall, never raises).
        regex_result = self._regex_fallback(chunk)
        if regex_result.entities:
            return regex_result
        # Nothing recovered anywhere: return the primary result if we have one
        # (preserves prior shape), else the empty regex result.
        return primary if primary is not None else regex_result

    async def _extract_with_llm(
        self, llm: UnifiedLLMClient, chunk: Chunk, *, label: str
    ) -> Optional[ExtractionResult]:
        """Run NER via one LLM client. Returns the parsed result, or None when
        the call raised (caller then advances the ladder). Never raises."""
        try:
            data = await llm.chat_json(
                user_message=chunk.text,
                system_message=_NER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
            return self._parse(data, chunk.chunk_id)
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            logger.warning(
                f"NER {label} APIStatusError chunk={chunk.chunk_id} "
                f"status={status} body={body}"
            )
            return None
        except Exception as exc:
            logger.warning(f"NER {label} failed for {chunk.chunk_id}: {exc}")
            return None

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

    def extract_from_query(self, query: str) -> List[str]:
        """Regex-only entity extraction for short queries (no LLM call).

        Returns list of entity node_ids: ["NGANH:cong_nghe_thong_tin", "TO_HOP:a00", ...]
        Used by GraphRAG for fast PPR seeding.
        """
        data = _regex_extract(query)
        seeds: List[str] = []
        seen = set()
        for ent in data.get("entities", []):
            node_id = f"{ent['type']}:{ent['normalized']}"
            if node_id not in seen:
                seen.add(node_id)
                seeds.append(node_id)

        # Check for ma_nganh pattern directly in query
        for m in _MA_NGANH_PATTERN.finditer(query):
            code = m.group(0)
            node_id = f"MA_NGANH:{code}"
            if node_id not in seen:
                seen.add(node_id)
                seeds.append(node_id)

        # Check nganh keywords with normalization mapping
        query_lower = query.lower()
        for keyword, normalized in _NGANH_NORMALIZE_MAP.items():
            if keyword in query_lower:
                node_id = f"NGANH:{normalized}"
                if node_id not in seen:
                    seen.add(node_id)
                    seeds.append(node_id)

        return seeds

    def _regex_fallback(self, chunk: Chunk) -> ExtractionResult:
        """Fallback NER using regex patterns when LLM fails or returns empty."""
        data = _regex_extract(chunk.text)
        return self._parse(data, chunk.chunk_id)

    async def extract_batch(
        self, chunks: List[Chunk], concurrency: int = 5
    ) -> List[ExtractionResult]:
        """Extract entities from a list of chunks with concurrency."""
        import asyncio

        results: List[Optional[ExtractionResult]] = [None] * len(chunks)
        sem = asyncio.Semaphore(concurrency)

        async def _worker(idx: int, chunk: Chunk) -> None:
            async with sem:
                result = await self.extract(chunk)
                results[idx] = result
                if (idx + 1) % 50 == 0 or idx == len(chunks) - 1:
                    done = sum(1 for r in results if r is not None)
                    ok = sum(1 for r in results if r and r.is_success)
                    logger.info(f"NER progress: {done}/{len(chunks)} done ({ok} ok)")

        tasks = [_worker(i, c) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)
        return [r or ExtractionResult(chunk_id=chunks[i].chunk_id, error="timeout") for i, r in enumerate(results)]
