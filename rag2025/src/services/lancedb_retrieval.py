"""
Retrieval Layer - LanceDB Interface

Fail-safe dense retrieval for embedded LanceDB backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from config.settings import RAGSettings
from src.infrastructure.lancedb_adapter import LanceDBAdapter


class RetrievalError(Enum):
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    EMPTY_RESULT = "empty_result"
    INVALID_PAYLOAD = "invalid_payload"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class RetrievedDocument:
    text: str
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    point_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    documents: List[RetrievedDocument] = field(default_factory=list)
    error_type: Optional[RetrievalError] = None
    error_message: Optional[str] = None
    confidence: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.error_type is None and len(self.documents) > 0


class LanceDBRetrieverConfig:
    def __init__(
        self,
        uri: str,
        table_name: str,
        embedding_dim: int = 4096,
        default_top_k: int = 5,
    ):
        self.uri = uri
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.default_top_k = default_top_k


class LanceDBRetriever:
    def __init__(self, config: LanceDBRetrieverConfig) -> None:
        self._config = config
        self._adapter = LanceDBAdapter(uri=config.uri, table_name=config.table_name)
        self._adapter.ensure_table()

    @classmethod
    def from_env(cls) -> "LanceDBRetriever":
        settings = RAGSettings()
        config = LanceDBRetrieverConfig(
            uri=settings.LANCEDB_URI,
            table_name=settings.LANCEDB_TABLE,
            embedding_dim=settings.EMBEDDING_DIM,
            default_top_k=5,
        )
        return cls(config=config)

    def retrieve(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        if not query_vector:
            return RetrievalResult(
                error_type=RetrievalError.INVALID_PAYLOAD,
                error_message="Query vector is empty",
            )

        if len(query_vector) != self._config.embedding_dim:
            return RetrievalResult(
                error_type=RetrievalError.INVALID_PAYLOAD,
                error_message=(
                    f"Query vector dimension mismatch: "
                    f"expected {self._config.embedding_dim}, got {len(query_vector)}"
                ),
            )

        limit = top_k or self._config.default_top_k

        try:
            rows = self._adapter.search(query_vector=query_vector, top_k=max(limit * 5, limit))
            rows = self._apply_filter(rows, metadata_filter)
            docs = self._normalize_rows(rows[:limit])
            if not docs:
                return RetrievalResult(
                    documents=[],
                    error_type=RetrievalError.EMPTY_RESULT,
                    error_message="No matching documents found",
                    confidence=0.0,
                )
            return RetrievalResult(documents=docs, confidence=self._calculate_confidence(docs))
        except Exception as exc:
            logger.error(f"LanceDB retrieval failed: {exc}")
            return RetrievalResult(
                error_type=RetrievalError.UNKNOWN_ERROR,
                error_message=str(exc),
            )

    def _apply_filter(
        self,
        rows: List[Dict[str, Any]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not metadata_filter:
            return rows

        def match_condition(row: Dict[str, Any], cond: Dict[str, Any]) -> bool:
            raw = row.get("metadata_json") or row.get("metadata") or "{}"
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            for key, value in cond.items():
                if key == "or_conditions":
                    continue
                row_value = row.get(key)
                if row_value is None:
                    row_value = metadata.get(key)
                if row_value != value:
                    return False
            return True

        or_conditions = metadata_filter.get("or_conditions")
        if or_conditions:
            return [row for row in rows if any(match_condition(row, c) for c in or_conditions)]

        return [row for row in rows if match_condition(row, metadata_filter)]

    def _normalize_rows(self, rows: List[Dict[str, Any]]) -> List[RetrievedDocument]:
        documents: List[RetrievedDocument] = []
        for row in rows:
            text = row.get("text")
            if not text or not isinstance(text, str):
                continue

            raw = row.get("metadata_json") or row.get("metadata") or "{}"
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            distance = row.get("_distance")
            score = float(row.get("score", 0.0))
            if distance is not None:
                score = 1.0 / (1.0 + float(distance))

            documents.append(
                RetrievedDocument(
                    text=text,
                    source=row.get("source"),
                    chunk_id=row.get("chunk_id"),
                    metadata=metadata,
                    score=score,
                    point_id=str(row.get("id")) if row.get("id") is not None else None,
                )
            )
        return documents

    def _calculate_confidence(self, documents: List[RetrievedDocument]) -> float:
        top_scores = [doc.score for doc in documents[:3]]
        if len(top_scores) == 1:
            return top_scores[0]
        if len(top_scores) == 2:
            return 0.7 * top_scores[0] + 0.3 * top_scores[1]
        return 0.5 * top_scores[0] + 0.3 * top_scores[1] + 0.2 * top_scores[2]

    def check_collection(self) -> Dict[str, Any]:
        exists = self._adapter.table_exists()
        if not exists:
            return {"exists": False, "error": f"Table '{self._config.table_name}' not found"}
        return {
            "exists": True,
            "name": self._config.table_name,
            "vectors_count": self._adapter.count(),
            "status": "ready",
        }


_default_retriever: Optional[LanceDBRetriever] = None


def get_retriever() -> LanceDBRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = LanceDBRetriever.from_env()
    return _default_retriever
