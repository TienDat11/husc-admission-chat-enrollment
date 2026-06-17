"""TDD: NER model fallback (S18 — resilience when primary NER model fails).

Context (2026-06-10): the primary NER model (set via RAMCLOUDS_NER_MODEL) can
fail on the gateway — e.g. gpt-5.5-chat returning non-JSON, a 5xx, or empty
entities. Before this change the service degraded straight to regex, which
loses rich entities. Now it tries a SECONDARY LLM model first
(RAMCLOUDS_NER_FALLBACK_MODEL, defaulting to RAMCLOUDS_GEN_FALLBACK_MODEL),
and only falls to regex if BOTH LLM models fail/empty.

Ladder: primary LLM → fallback LLM → regex.

These tests inject fake LLM clients so no real gateway/model is touched.
"""
from __future__ import annotations

import asyncio
import importlib
from typing import Any, Dict, List, Optional

import pytest

from src.domain.entities import Chunk
from src.services import ner_service as ner_mod
from src.services.ner_service import NERService


# ---------------------------------------------------------------------------
# Fake LLM clients
# ---------------------------------------------------------------------------
class FakeLLM:
    """Async chat_json that either returns a dict or raises a chosen error."""

    def __init__(self, *, returns: Optional[Dict[str, Any]] = None, raises: Optional[Exception] = None):
        self._returns = returns
        self._raises = raises
        self.calls = 0

    async def chat_json(self, **kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._returns if self._returns is not None else {"entities": [], "triples": []}


def _good_payload() -> Dict[str, Any]:
    return {
        "entities": [
            {"type": "NGANH", "text": "Công nghệ thông tin", "normalized": "cong nghe thong tin"},
            {"type": "TO_HOP", "text": "A00", "normalized": "a00"},
        ],
        "triples": [
            {"head": "NGANH:cong_nghe_thong_tin", "relation": "CO_TO_HOP", "tail": "TO_HOP:a00"},
        ],
    }


def _chunk(text: str = "Ngành Công nghệ thông tin mã 7480201 tổ hợp A00.") -> Chunk:
    return Chunk(chunk_id="c1", text=text, faq_type="", metadata={})


def _run(coro):
    # Fresh event loop per call — a shared global loop gets closed/polluted by
    # other async tests in the suite (these tests pass in isolation but fail
    # after another async suite runs). Mirror the fix in
    # tests/routers/test_v2_uses_generate_answer.py::_run.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# 1. Primary succeeds → fallback is NOT called.
# ---------------------------------------------------------------------------
def test_primary_success_skips_fallback():
    primary = FakeLLM(returns=_good_payload())
    fallback = FakeLLM(returns=_good_payload())
    ner = NERService(llm=primary, fallback_llm=fallback)

    result = _run(ner.extract(_chunk()))

    assert result.is_success
    assert any(e.entity_type.value == "NGANH" for e in result.entities)
    assert primary.calls == 1
    assert fallback.calls == 0  # fallback untouched when primary works


# ---------------------------------------------------------------------------
# 2. Primary RAISES → fallback LLM is used and its entities win.
# ---------------------------------------------------------------------------
def test_primary_raises_uses_fallback_llm():
    primary = FakeLLM(raises=RuntimeError("gateway 503"))
    fallback = FakeLLM(returns=_good_payload())
    ner = NERService(llm=primary, fallback_llm=fallback)

    result = _run(ner.extract(_chunk()))

    assert fallback.calls == 1
    assert any(e.entity_type.value == "NGANH" for e in result.entities)


# ---------------------------------------------------------------------------
# 3. Primary returns EMPTY entities → fallback LLM is tried.
# ---------------------------------------------------------------------------
def test_primary_empty_uses_fallback_llm():
    primary = FakeLLM(returns={"entities": [], "triples": []})
    fallback = FakeLLM(returns=_good_payload())
    ner = NERService(llm=primary, fallback_llm=fallback)

    result = _run(ner.extract(_chunk()))

    assert fallback.calls == 1
    assert any(e.entity_type.value == "NGANH" for e in result.entities)


# ---------------------------------------------------------------------------
# 4. BOTH LLMs fail → regex fallback recovers entities from the text.
# ---------------------------------------------------------------------------
def test_both_llms_fail_uses_regex():
    primary = FakeLLM(raises=RuntimeError("primary dead"))
    fallback = FakeLLM(raises=RuntimeError("fallback dead"))
    ner = NERService(llm=primary, fallback_llm=fallback)

    # Text carries a major code + tổ hợp the regex extractor recognizes.
    result = _run(ner.extract(_chunk("Ngành mã 7480201 xét tổ hợp A00, A01.")))

    # regex fallback should still surface at least one entity (TO_HOP / MA_NGANH)
    assert result.entities, "regex fallback must recover entities when both LLMs fail"


# ---------------------------------------------------------------------------
# 5. No fallback configured (None) + primary raises → regex (back-compat).
# ---------------------------------------------------------------------------
def test_no_fallback_configured_falls_to_regex():
    primary = FakeLLM(raises=RuntimeError("primary dead"))
    ner = NERService(llm=primary, fallback_llm=None)

    result = _run(ner.extract(_chunk("Ngành mã 7480201 tổ hợp A00.")))

    assert result is not None
    # never raises; returns a (possibly regex-derived) ExtractionResult
    assert hasattr(result, "entities")


# ---------------------------------------------------------------------------
# 6. __init__ reads RAMCLOUDS_NER_FALLBACK_MODEL, else RAMCLOUDS_GEN_FALLBACK_MODEL.
# ---------------------------------------------------------------------------
def test_init_resolves_fallback_model_from_env(monkeypatch):
    created: List[str] = []

    class _RecordingClient:
        def __init__(self, force_model=None):
            created.append(force_model)

    monkeypatch.setattr(ner_mod, "UnifiedLLMClient", _RecordingClient)
    monkeypatch.setenv("RAMCLOUDS_NER_MODEL", "primary-model")
    monkeypatch.delenv("RAMCLOUDS_NER_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("RAMCLOUDS_GEN_FALLBACK_MODEL", "fallback-model")

    NERService()  # llm=None → both clients built from env

    assert "primary-model" in created
    assert "fallback-model" in created


def test_init_prefers_dedicated_ner_fallback_env(monkeypatch):
    created: List[str] = []

    class _RecordingClient:
        def __init__(self, force_model=None):
            created.append(force_model)

    monkeypatch.setattr(ner_mod, "UnifiedLLMClient", _RecordingClient)
    monkeypatch.setenv("RAMCLOUDS_NER_MODEL", "primary-model")
    monkeypatch.setenv("RAMCLOUDS_NER_FALLBACK_MODEL", "dedicated-ner-fb")
    monkeypatch.setenv("RAMCLOUDS_GEN_FALLBACK_MODEL", "gen-fb")

    NERService()

    assert "dedicated-ner-fb" in created
    assert "gen-fb" not in created  # dedicated NER fallback wins
