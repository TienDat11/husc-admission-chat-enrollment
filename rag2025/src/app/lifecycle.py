"""App lifecycle: lazy service registry + startup/shutdown hooks.
@spec(S10.B2)
"""
from dataclasses import dataclass, field
from typing import Optional

from rag2025.src.services.embedding import EmbeddingService
from rag2025.src.services.llm_client import UnifiedLLMClient, get_llm_client
from rag2025.config.settings import RAGSettings


@dataclass
class AppServices:
    """Lazy service container — services constructed on first access.
    @spec(S10.B2)
    """
    _settings: Optional[RAGSettings] = None
    embedding: Optional[EmbeddingService] = None
    llm_client: Optional[UnifiedLLMClient] = None

    def get_settings(self) -> RAGSettings:
        if self._settings is None:
            self._settings = RAGSettings()
        return self._settings

    def get_embedding(self) -> EmbeddingService:
        if self.embedding is None:
            self.embedding = EmbeddingService(self.get_settings())
        return self.embedding

    def get_llm_client(self, force_model: Optional[str] = None) -> UnifiedLLMClient:
        if self.llm_client is None:
            self.llm_client = get_llm_client(force_model=force_model)
        return self.llm_client


async def startup(services: AppServices) -> None:
    """Fast startup — services constructed lazily, no model load yet.
    @spec(S10.B2)
    """
    pass  # lazy: nothing to do


async def shutdown(services: AppServices) -> None:
    """Cleanup on shutdown."""
    # UnifiedLLMClient uses per-call AsyncOpenAI instances (no persistent pool to close).
    # If a close() method is added later, call it here.
    pass
