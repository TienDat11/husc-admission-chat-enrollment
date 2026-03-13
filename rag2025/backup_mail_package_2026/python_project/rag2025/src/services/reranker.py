from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from config.settings import RAGSettings


class RerankerService:
    def __init__(self, settings: RAGSettings):
        self._enabled = settings.RERANKER_ENABLED
        self._model_name = settings.RERANKER_MODEL
        self._weight = settings.RERANKER_WEIGHT
        self._model = None

        if not self._enabled:
            logger.info("Reranker disabled by config")
            return

        try:
            self._model = CrossEncoder(self._model_name)
            logger.info(f"Reranker loaded: {self._model_name}")
        except Exception as exc:
            self._enabled = False
            logger.warning(f"Failed to load reranker {self._model_name}: {exc}")

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not chunks:
            return chunks[:top_k]

        pairs = [(query, c.get("text", "")) for c in chunks]
        rerank_scores = self._model.predict(pairs)

        rescored = []
        for chunk, rr_score in zip(chunks, rerank_scores):
            base = float(chunk.get("score", 0.0))
            rr = float(rr_score)
            fused = (1.0 - self._weight) * base + self._weight * rr
            updated = dict(chunk)
            updated["score"] = fused
            updated["rerank_score"] = rr
            rescored.append(updated)

        rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return rescored[:top_k]
