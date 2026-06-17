"""Tests for AppServices lifecycle.
@spec(S10.B2) — factory + startup/shutdown.
"""
import pytest

from rag2025.src.app.lifecycle import AppServices, startup, shutdown
from rag2025.src.services.embedding import EmbeddingService
from rag2025.src.services.llm_client import UnifiedLLMClient
from rag2025.config.settings import RAGSettings


class TestAppServicesFactory:
    """@spec(S10.B2): lazy factory creates services on first access."""

    def test_initial_state(self):
        svc = AppServices()
        assert svc.embedding is None
        assert svc.llm_client is None

    def test_get_settings_creates_singleton(self):
        svc = AppServices()
        s1 = svc.get_settings()
        s2 = svc.get_settings()
        assert isinstance(s1, RAGSettings)
        assert s1 is s2

    def test_get_embedding_creates_service(self):
        svc = AppServices()
        emb = svc.get_embedding()
        assert isinstance(emb, EmbeddingService)
        # Lazy: model not loaded yet
        assert not emb.is_loaded

    def test_get_embedding_returns_same_instance(self):
        svc = AppServices()
        emb1 = svc.get_embedding()
        emb2 = svc.get_embedding()
        assert emb1 is emb2

    def test_get_llm_client_creates_service(self):
        svc = AppServices()
        llm = svc.get_llm_client()
        assert isinstance(llm, UnifiedLLMClient)

    def test_get_llm_client_returns_same_instance(self):
        svc = AppServices()
        llm1 = svc.get_llm_client()
        llm2 = svc.get_llm_client()
        assert llm1 is llm2


class TestStartupShutdown:
    """@spec(S10.B2): startup is no-op (lazy), shutdown is safe."""

    @pytest.mark.asyncio
    async def test_startup_is_noop(self):
        svc = AppServices()
        await startup(svc)
        # No services loaded
        assert svc.embedding is None

    @pytest.mark.asyncio
    async def test_shutdown_is_safe(self):
        svc = AppServices()
        await shutdown(svc)
        # No error even with no services

    @pytest.mark.asyncio
    async def test_shutdown_with_services(self):
        svc = AppServices()
        svc.get_embedding()
        svc.get_llm_client()
        await shutdown(svc)
        # No error — graceful shutdown
