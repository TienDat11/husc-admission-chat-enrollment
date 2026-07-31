"""Tests for router-only Groq LLM client (fast-path router swap).

Validates:
- _build_router_providers() puts groq first when GROQ_API_KEY is set
- Graceful fallback to _build_providers() when GROQ_API_KEY is missing
- SmartQueryRouter uses router client (groq primary) by default
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
FAKE_GROQ_KEY = "gsk_test_fake_key_for_router_only"
FAKE_RAMCLOUDS_KEY = "rc_fake_key_router_test"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBuildRouterProviders:
    """_build_router_providers() provider ordering and fallback logic."""

    @patch.dict(os.environ, {
        "GROQ_API_KEY": FAKE_GROQ_KEY,
        "RAMCLOUDS_API_KEY": FAKE_RAMCLOUDS_KEY,
        "RAMCLOUDS_MODEL": "gemini-3.5-flash-low",
    }, clear=False)
    def test_router_providers_groq_first(self):
        """Groq is priority 0 when GROQ_API_KEY is present."""
        from src.services.llm_client import _build_router_providers

        providers = _build_router_providers()

        assert len(providers) >= 1
        # First provider must be groq
        assert providers[0].name == "groq"
        assert providers[0].model == "llama-3.3-70b-versatile"
        assert providers[0].priority == 0
        assert providers[0].base_url == "https://api.groq.com/openai/v1"

        # Second provider (if present) must be ramclouds fallback
        ramclouds = [p for p in providers if p.name == "ramclouds"]
        assert len(ramclouds) == 1
        assert ramclouds[0].priority == 1

    @patch.dict(os.environ, {
        "GROQ_ROUTER_MODEL": "deepseek-r1-distill-llama-70b",
        "GROQ_API_KEY": FAKE_GROQ_KEY,
    }, clear=False)
    def test_router_model_overridden_via_env(self):
        """GROQ_ROUTER_MODEL env var overrides the default model name."""
        from src.services.llm_client import _build_router_providers

        providers = _build_router_providers()
        assert providers[0].model == "deepseek-r1-distill-llama-70b"

    @patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False)
    @patch("src.services.llm_client._build_providers")
    def test_router_falls_back_without_groq(self, mock_build):
        """Without GROQ_API_KEY, _build_router_providers delegates to _build_providers."""
        from src.services.llm_client import _build_router_providers

        # Make _build_providers return a known list so we can assert identity
        fake_providers = [type("P", (), {"name": "ramclouds", "priority": 0})()]
        mock_build.return_value = fake_providers

        result = _build_router_providers()

        mock_build.assert_called_once()
        assert result is fake_providers


class TestRouterClientWiring:
    """SmartQueryRouter receives the router-only client by default."""

    @patch.dict(os.environ, {
        "GROQ_API_KEY": FAKE_GROQ_KEY,
        "RAMCLOUDS_API_KEY": FAKE_RAMCLOUDS_KEY,
    }, clear=False)
    def test_router_uses_router_client(self):
        """SmartQueryRouter()._llm has groq as first provider."""
        # Reset singletons so fresh instances are created
        import src.services.llm_client as _mod
        old_default = _mod._default_client
        old_router = _mod._router_client
        _mod._default_client = None
        _mod._router_client = None
        try:
            from src.services.query_router import SmartQueryRouter

            router = SmartQueryRouter()
            llm = router._llm

            # The router client's first provider must be groq
            assert llm._providers[0].name == "groq"
            assert llm._providers[0].model == "llama-3.3-70b-versatile"
        finally:
            _mod._default_client = old_default
            _mod._router_client = old_router

    def test_injected_llm_still_works(self):
        """Passing an explicit llm= param overrides the router client (test compat)."""
        from src.services.query_router import SmartQueryRouter
        from src.services.llm_client import ProviderConfig, UnifiedLLMClient

        fake_client = UnifiedLLMClient(providers=[
            ProviderConfig("fake", "fake-key", "http://fake/v1", "fake-model", priority=0),
        ])
        router = SmartQueryRouter(llm=fake_client)
        assert router._llm is fake_client
        assert router._llm._providers[0].name == "fake"

    @patch.dict(os.environ, {"GROQ_API_KEY": FAKE_GROQ_KEY}, clear=False)
    def test_get_router_llm_client_singleton(self):
        """get_router_llm_client() returns the same instance on repeated calls."""
        import src.services.llm_client as _mod
        old_router = _mod._router_client
        _mod._router_client = None
        try:
            from src.services.llm_client import get_router_llm_client

            c1 = get_router_llm_client()
            c2 = get_router_llm_client()
            assert c1 is c2
        finally:
            _mod._router_client = old_router
