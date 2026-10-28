"""Tests for LLM/NER provider permission handling + HTTP connection pool (B5)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from src.domain.entities import Chunk
from src.services.ner_service import NERService
from src.services.llm_client import UnifiedLLMClient, ProviderConfig


class _DummyResponse:
    request = None
    status_code = 403
    headers = {}


def _api_status_error(status_code: int = 403, body: object | None = None) -> APIStatusError:
    response = _DummyResponse()
    response.status_code = status_code
    return APIStatusError("forbidden", response=response, body=body)


# --- Existing NER test (preserved) ---

@pytest.mark.asyncio
async def test_ner_extract_maps_api_status_error() -> None:
    llm = AsyncMock()
    llm.chat_json.side_effect = _api_status_error(403, {"error": "permission denied"})

    svc = NERService(llm=llm)
    chunk = Chunk(chunk_id="c1", text="test", faq_type="", metadata={})

    result = await svc.extract(chunk)

    assert result.chunk_id == "c1"
    # NER service falls back to regex on APIStatusError, so error is None
    assert result.error is None
    assert result.entities == []
    assert result.triples == []


# --- B5 HTTP Connection Pool tests ---

def _make_provider(name: str = "test", api_key: str = "k") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        api_key=api_key,
        base_url="https://example.com/v1",
        model="test-model",
        priority=0,
    )


@pytest.mark.asyncio
async def test_llm_client_uses_persistent_pool() -> None:
    """Verify the same httpx.AsyncClient is reused across multiple _get_client calls."""
    with patch.dict("os.environ", {"NS24_API_KEY": "test-key"}):
        client = UnifiedLLMClient()

    p1 = _make_provider("p1")
    p2 = _make_provider("p2")

    openai1 = client._get_client(p1)
    openai2 = client._get_client(p2)

    # Both AsyncOpenAI instances share the same httpx client
    assert openai1._client is client._http_client
    assert openai2._client is client._http_client

    await client.close()


def test_llm_client_pool_max_connections() -> None:
    """Verify connection pool limits are configured correctly."""
    with patch.dict("os.environ", {"NS24_API_KEY": "test-key"}):
        client = UnifiedLLMClient()

    # httpx stores limits on the transport pool
    pool = client._http_client._transport._pool
    assert pool._max_connections == 20
    assert pool._max_keepalive_connections == 10


@pytest.mark.asyncio
async def test_llm_client_pool_close_on_shutdown() -> None:
    """Verify close() cleans up the HTTP pool without leaked connections."""
    with patch.dict("os.environ", {"NS24_API_KEY": "test-key"}):
        client = UnifiedLLMClient()

    assert not client._http_client.is_closed

    await client.close()

    assert client._http_client.is_closed


@pytest.mark.asyncio
async def test_llm_client_handles_concurrent_requests() -> None:
    """10 parallel _get_client calls should not create more than pool limit."""
    import asyncio

    with patch.dict("os.environ", {"NS24_API_KEY": "test-key"}):
        client = UnifiedLLMClient()

    async def _get() -> object:
        return client._get_client(_make_provider())

    results = await asyncio.gather(*[_get() for _ in range(10)])

    # All 10 share the exact same httpx pool
    for r in results:
        assert r._client is client._http_client

    await client.close()
