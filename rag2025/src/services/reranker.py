from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from config.settings import RAGSettings


# Public load-state vocabulary (ticket 40). Three distinct states, never
# compressed to a boolean: the config intentionally turned the reranker off,
# the model loaded, or the model was configured on but failed to load.
LOAD_STATE_LOADED = "loaded"
LOAD_STATE_LOAD_FAILED = "load_failed"
LOAD_STATE_DISABLED_BY_CONFIG = "disabled_by_config"


class RerankerService:
    # Class-level default so doubles built with ``RerankerService.__new__``
    # (existing unit tests) still answer ``load_state``/``skip_reason``
    # instead of raising AttributeError.
    _load_state = LOAD_STATE_DISABLED_BY_CONFIG

    def __init__(self, settings: RAGSettings):
        self._enabled = settings.RERANKER_ENABLED
        self._model_name = settings.RERANKER_MODEL
        self._weight = settings.RERANKER_WEIGHT
        self._max_rerank = settings.MAX_RERANK
        self._max_length = getattr(settings, "RERANKER_MAX_LENGTH", 512)
        self._pool = getattr(settings, "RERANKER_POOL", 5)
        self._adaptive_confidence = getattr(settings, "RERANKER_ADAPTIVE_CONFIDENCE", 0.5)
        self._model = None

        if not self._enabled:
            self._load_state = LOAD_STATE_DISABLED_BY_CONFIG
            logger.info("Reranker disabled by config")
            return

        try:
            self._model = CrossEncoder(self._model_name, max_length=self._max_length)
            # "loaded" means the model object really exists — a CrossEncoder
            # that returns None is not a loaded reranker.
            if self._model is not None:
                self._load_state = LOAD_STATE_LOADED
                logger.info(f"Reranker loaded: {self._model_name} (max_length={self._max_length})")
            else:
                self._enabled = False
                self._load_state = LOAD_STATE_LOAD_FAILED
                logger.warning(
                    f"Reranker load FAILED (model={self._model_name}): "
                    "CrossEncoder returned None"
                )
        except Exception as exc:
            self._enabled = False
            self._load_state = LOAD_STATE_LOAD_FAILED
            # Distinct from the "disabled by config" info line: a dead model
            # must be recognizable as a failure, not as an intentional off.
            logger.warning(
                f"Reranker load FAILED (model={self._model_name}): {exc}"
            )

    @property
    def load_state(self) -> str:
        """One of ``loaded`` / ``load_failed`` / ``disabled_by_config``."""
        return self._load_state

    @property
    def skip_reason(self) -> Optional[str]:
        """Why rerank cannot run: the load state, or None when it can run."""
        if self._load_state == LOAD_STATE_LOADED:
            return None
        return self._load_state

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    @property
    def adaptive_confidence(self) -> float:
        return self._adaptive_confidence

    @property
    def pool(self) -> int:
        return self._pool

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        apply_lost_in_middle: bool = True,
        max_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not chunks:
            return chunks[:top_k]

        # Pre-filter: limit candidates to MAX_RERANK before cross-encoder
        candidates = chunks[: self._max_rerank]

        # Truncation is SCORING-ONLY: the cross-encoder receives at most
        # `max_length` tokens of each chunk text for relevance scoring.
        # The original chunk text (including full text) is preserved in the
        # returned list — the generator always sees the complete chunk text.
        ml = max_length if max_length is not None else self._max_length
        pairs = [(query, c.get("text", "")) for c in candidates]
        rerank_scores = self._model.predict(pairs, show_progress_bar=False, max_length=ml)

        rescored = []
        for chunk, rr_score in zip(candidates, rerank_scores):
            base = float(chunk.get("score", 0.0))
            rr = float(rr_score)
            fused = (1.0 - self._weight) * base + self._weight * rr
            updated = dict(chunk)
            updated["score"] = fused
            updated["rerank_score"] = rr
            # Apply adaptive confidence gate: if max rerank score < threshold,
            # scale down the fused score to demote low-confidence results
            if float(rr_score) < self._adaptive_confidence:
                updated["score"] = fused * 0.5  # halve the fused score
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
