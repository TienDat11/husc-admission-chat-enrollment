"""Tests for LLM/NER provider permission handling."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from openai import APIStatusError

from src.domain.entities import Chunk
from src.services.ner_service import NERService


class _DummyResponse:
    request = None
    status_code = 403
    headers = {}


def _api_status_error(status_code: int = 403, body: object | None = None) -> APIStatusError:
    response = _DummyResponse()
    response.status_code = status_code
    return APIStatusError("forbidden", response=response, body=body)


@pytest.mark.asyncio
async def test_ner_extract_maps_api_status_error() -> None:
    llm = AsyncMock()
    llm.chat_json.side_effect = _api_status_error(403, {"error": "permission denied"})

    svc = NERService(llm=llm)
    chunk = Chunk(chunk_id="c1", text="test", faq_type="", metadata={})

    result = await svc.extract(chunk)

    assert result.chunk_id == "c1"
    assert result.error == "APIStatusError:403"
    assert result.entities == []
    assert result.triples == []
