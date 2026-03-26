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
        self._max_rerank = settings.MAX_RERANK
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
        apply_lost_in_middle: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not chunks:
            return chunks[:top_k]

        # Pre-filter: limit candidates to MAX_RERANK before cross-encoder
        candidates = chunks[: self._max_rerank]

        pairs = [(query, c.get("text", "")) for c in candidates]
        rerank_scores = self._model.predict(pairs)

        rescored = []
        for chunk, rr_score in zip(candidates, rerank_scores):
            base = float(chunk.get("score", 0.0))
            rr = float(rr_score)
            fused = (1.0 - self._weight) * base + self._weight * rr
            updated = dict(chunk)
            updated["score"] = fused
            updated["rerank_score"] = rr
            rescored.append(updated)

        rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top_chunks = rescored[:top_k]

        if apply_lost_in_middle:
            return self._apply_lost_in_middle(top_chunks)
        return top_chunks

    def _apply_lost_in_middle(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reorder score-sorted chunks to mitigate the lost-in-the-middle effect.

        Chunks are placed alternately at the front and back of a result list
        so that the highest-scoring chunks end up at boundary positions where
        LLM attention is strongest.

        Example (5 chunks ranked 1st-5th by score):
          Input:   [1st, 2nd, 3rd, 4th, 5th]
          Output:  [1st, 3rd, 5th, 4th, 2nd]
                    ^                    ^
                   front               back  <- LLM pays most attention here

        Args:
            chunks: Score-sorted chunks (descending). Any length.

        Returns:
            Reordered list with boundary positions occupied by top-scored chunks.
            Returns the input unchanged if len <= 2.
        """
        n = len(chunks)
        if n <= 2:
            return chunks

        result = [None] * n
        head, tail = 0, n - 1

        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                result[head] = chunk
                head += 1
            else:
                result[tail] = chunk
                tail -= 1

        return result
