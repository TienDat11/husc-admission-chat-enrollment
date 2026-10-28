"""
NER Service – Named Entity Recognition & Triple Extraction

@spec(S22.A) NER disk cache (data/graph/ner_cache.jsonl) — keyed by
    (chunk_id, text_hash=sha256(text)[:16]). Cache hit → no LLM call.
    Successful (non-empty) extractions are PERSISTED incrementally so a
    crash / quota cutoff mid-run still preserves the work and lets the
    next run resume free. Failed/empty extractions are NOT cached.

@spec(S22.B) Same-gateway fallback: the ``ner`` role client holds the primary
    model (NS24_NER_MODEL) and its fallback model, both on the ns24 gateway.
    When both fail or return zero entities, the regex extractor runs.

Uses get_llm_client(role="ner") as the LLM client.

DDD: Application service that depends on domain entities, not infrastructure.
Scalable: Adding entity types = extend EntityType enum + update prompt only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import unicodedata
from pathlib import Path
import tempfile
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

# @spec(S22.A) — NER cache default location
DEFAULT_NER_CACHE_PATH = Path(
    os.getenv("NER_CACHE_PATH", "data/graph/ner_cache.jsonl")
)


def _text_hash(text: str) -> str:
    """Stable short hash for NER cache key. NFC-normalized so NFC/NFD forms match."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()[:16]


# Module-level per-path locks so two NERCache instances sharing the same
# file (same process) serialize put/compact operations, preventing lost
# updates from concurrent os.replace compaction.
_PATH_LOCKS: Dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _path_lock_for(path: Path) -> threading.Lock:
    """Return shared per-absolute-path lock for cross-instance concurrency."""
    abs_path = str(path.resolve())
    with _REGISTRY_LOCK:
        if abs_path not in _PATH_LOCKS:
            _PATH_LOCKS[abs_path] = threading.Lock()
        return _PATH_LOCKS[abs_path]


class NERCache:
    """@spec(S22.A) Persistent JSONL cache for NER results.

    Format: one JSON object per line:
      {"chunk_id": str, "text_hash": str, "entities": [...], "triples": [...], "model": str}

    Keying: (chunk_id, text_hash). The text_hash guard ensures we
    re-extract if a chunk's text is edited (id stays, content changes).

    Concurrency: writes (append + compaction) are serialized via
    threading.Lock — not by OS append atomicity.
    Bounded eviction: optional max_entries evicts oldest entries
    (dict insertion order) with atomic compaction on overflow.
    """

    def __init__(self, path: Path = DEFAULT_NER_CACHE_PATH, max_entries: Optional[int] = None) -> None:
        assert max_entries is None or max_entries >= 1
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._max_entries = max_entries
        self._lock = _path_lock_for(self._path)  # shared per-path lock
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue  # skip non-dict JSON (null, [], string)
                    cid = rec.get("chunk_id")
                    if cid:
                        self._index[cid] = rec
        except OSError as exc:
            logger.warning(f"NERCache: failed to load {self._path}: {exc}")
            return

        # @spec(S22.A) enforce max_entries after load
        if self._max_entries is not None and len(self._index) > self._max_entries:
            candidate = dict(self._index)
            excess = len(candidate) - self._max_entries
            for key in list(candidate.keys())[:excess]:
                del candidate[key]
            if self._compact(candidate):
                self._index = candidate
            # else: compact failed — keep full loaded index & file, no eviction

    def get(self, chunk_id: str, text: str) -> Optional[Dict[str, Any]]:
        """Return cached record if (chunk_id, text_hash, pattern_version) matches, else None."""
        rec = self._index.get(chunk_id)
        if rec is None:
            return None
        expected = _text_hash(text)  # current NFC hash
        actual = rec.get("text_hash")
        if actual != expected:
            # Backward compat: legacy entries stored raw/NFD hash.
            legacy_nfd = hashlib.sha256(
                unicodedata.normalize("NFD", text).encode("utf-8")
            ).hexdigest()[:16]
            if actual != legacy_nfd:
                return None
        # Accept legacy cache entries (pattern_version is None) as valid —
        # otherwise ALL 1712 legacy entries get rejected and re-extracted.
        pv = rec.get("pattern_version")
        if pv is not None and pv != _PATTERN_VERSION:
            return None
        return rec

    def put(self, chunk_id: str, text: str, entities: List[Dict[str, Any]],
            triples: List[Dict[str, Any]], model: str) -> None:
        """Persist a successful NER result. Skip if entities is empty."""
        if not entities:
            return
        rec = {
            "chunk_id": chunk_id,
            "text_hash": _text_hash(text),
            "pattern_version": _PATTERN_VERSION,
            "entities": entities,
            "triples": triples,
            "model": model,
        }
        with self._lock:
            # @spec(S22.C) Reload canonical disk state under shared path lock
            # so two NERCache instances (same process, same path) don't
            # independently compact and lose each other's entries via
            # os.replace collision.
            self._load()
            is_new_key = chunk_id not in self._index

            # At/over capacity with a NEW key: build candidate (incl. new record),
            # compact atomically, then publish index.
            if (
                is_new_key
                and self._max_entries is not None
                and len(self._index) >= self._max_entries
            ):
                candidate = dict(self._index)
                candidate[chunk_id] = rec
                # Evict oldest (dict insertion order)
                excess = len(candidate) - self._max_entries
                for key in list(candidate.keys())[:excess]:
                    del candidate[key]
                if self._compact(candidate):
                    self._index = candidate
                # else: candidate not published; index unchanged; file not written.
                # The append to disk is skipped too — caller can retry.
                return

            # Below capacity OR same-key update: append then compact if needed.
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning(f"NERCache: failed to persist {chunk_id}: {exc}")
                return
            self._index[chunk_id] = rec
            if self._max_entries is not None and len(self._index) > self._max_entries:
                candidate = dict(self._index)
                excess = len(candidate) - self._max_entries
                for key in list(candidate.keys())[:excess]:
                    del candidate[key]
                if self._compact(candidate):
                    self._index = candidate
                # else: index still has the entry (appended to file already).
                # The file has one extra un-compacted line — acceptable stale
                # data that _load can handle (oldest entry in file wins on reload).

    def _compact(self, candidate: Dict[str, Dict[str, Any]]) -> bool:
        """Atomically write candidate dict to disk via temp + os.replace.
        Never mutates self._index or self._path.
        Uses tempfile.NamedTemporaryFile with unique name in the canonical
        directory so multiple cache instances (same dir) don't collide.
        Returns True on success, False on failure.
        """
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False,
                dir=str(self._path.parent),
                prefix=".ner_cache_", suffix=".tmp",
            ) as f:
                tmp_path = f.name
                for rec in candidate.values():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self._path))
            tmp_path = None  # consumed — no longer needs cleanup
            return True
        except OSError as exc:
            logger.warning(f"NERCache: compaction failed: {exc}")
            return False
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


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

# Pattern version for cache invalidation: when TO_HOP_PATTERN changes,
# cached NER entries from the old pattern are no longer valid.
from src.data.to_hop_codes import TO_HOP_PATTERN_RE as _TO_HOP_PATTERN
_PATTERN_VERSION = hashlib.md5(str(_TO_HOP_PATTERN.pattern).encode()).hexdigest()[:8]
# WARNING: This pattern is auto-generated from src/data/to_hop_codes.py.
# Do NOT edit manually — update the canonical list there instead.

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

    Uses the ns24 ``ner`` role client (NS24_NER_MODEL, same-gateway fallback).
    Fully swappable: inject any UnifiedLLMClient instance.

    Args:
        llm: UnifiedLLMClient instance (auto-created with NER model if None).
    """

    def __init__(
        self,
        llm: Optional[UnifiedLLMClient] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        # NER ladder: ns24 ner client (primary model, then same-gateway fallback
        # inside the client) → regex extractor.
        self._llm = llm if llm is not None else get_llm_client(role="ner")

        # @spec(S22.A) — disk cache for NER results (idempotent, incremental).
        self._cache = NERCache(path=cache_path or DEFAULT_NER_CACHE_PATH)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and triples from a single chunk.

        Ladder: cache → ns24 ner client (primary model, then same-gateway fallback
        inside the client) → regex. An LLM failure or zero entities drops to the
        regex extractor.

        @spec(S22.A) Cache hit short-circuits ALL LLM calls.
        @spec(S22.A) Successful (non-empty) results are persisted BEFORE return.

        Args:
            chunk: Domain Chunk object.

        Returns:
            ExtractionResult with entities and triples, or error info.
        """
        # @spec(S22.A) — cache hit short-circuits everything.
        cached = self._cache.get(chunk.chunk_id, chunk.text)
        if cached is not None:
            logger.debug(f"NER cache HIT: {chunk.chunk_id} (model={cached.get('model', '?')})")
            return self._parse(
                {"entities": cached.get("entities", []), "triples": cached.get("triples", [])},
                chunk.chunk_id,
            )

        primary = await self._extract_with_llm(self._llm, chunk, label="primary")
        if primary is not None and primary.entities:
            self._cache_put(chunk, primary, model_label="primary")
            return primary


        # All LLMs failed/empty → regex extractor (lower recall, never raises).
        regex_result = self._regex_fallback(chunk)
        if regex_result.entities:
            # Regex results ARE persisted (they're deterministic for the
            # entities they find — ma_nganh/A00/diem/hoc_phi).
            self._cache_put(chunk, regex_result, model_label="regex")
            return regex_result
        # Nothing recovered anywhere: return the primary result if we have one
        # (preserves prior shape), else the empty regex result. NOT cached.
        return primary if primary is not None else regex_result

    def _cache_put(self, chunk: Chunk, result: ExtractionResult, *, model_label: str) -> None:
        """@spec(S22.A) Persist result to disk cache (skips empty/failed)."""
        try:
            self._cache.put(
                chunk.chunk_id,
                chunk.text,
                entities=[
                    {
                        "text": e.text,
                        "type": e.entity_type.value,
                        "normalized": e.normalized,
                    }
                    for e in result.entities
                ],
                triples=[
                    {
                        "head": t.head,
                        "relation": t.relation.value,
                        "tail": t.tail,
                    }
                    for t in result.triples
                ],
                model=model_label,
            )
        except Exception as exc:
            logger.warning(f"NER cache persist failed for {chunk.chunk_id}: {exc}")

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
