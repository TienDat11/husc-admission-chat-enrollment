"""
Unified LLM Client – OpenAI-compatible wrapper for ramclouds.me/v1

Single client that wraps ANY OpenAI-compatible endpoint.
Primary model: configurable via RAMCLOUDS_MODEL (default: gemini-2.5-flash) via https://ramclouds.me/v1
Fallback chain: GROQ_API_KEY → OPENAI_API_KEY (any provider)

Design principles:
- One class, any model, any compatible endpoint
- Retry with exponential backoff (tenacity)
- JSON-mode support for structured outputs
- Async-first, sync wrapper provided for compatibility
- Scalable: add new providers by adding env vars only (no code change)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class LLMResponse:
    """Typed wrapper for LLM chat completion response."""

    def __init__(self, content: str, model: str, provider: str) -> None:
        self.content = content
        self.model = model
        self.provider = provider

    def as_json(self) -> Dict[str, Any]:
        """Parse content as JSON, stripping markdown fences if present."""
        text = self.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # remove first line (```json or ```) and last ```
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        return json.loads(text.strip())

    def __str__(self) -> str:
        return self.content


class ProviderConfig:
    """Configuration for a single LLM provider endpoint."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        priority: int = 0,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.priority = priority


def _build_providers() -> List[ProviderConfig]:
    """Build ordered provider list from environment variables.

    Provider priority (lower = higher priority):
    0 → RAMCLOUDS (model from RAMCLOUDS_MODEL, primary - best quality for NER/GraphRAG)
    1 → GROQ (llama-3.1-8b-instant, fallback - free)
    2 → OPENAI_COMPAT (any provider, last resort)
    """
    providers: List[ProviderConfig] = []

    # Primary: ramclouds.me (configurable model — highest quality for NER/GraphRAG)
    ramclouds_key = os.getenv("RAMCLOUDS_API_KEY") or os.getenv("OPENAI_API_KEY")
    ramclouds_url = os.getenv("RAMCLOUDS_BASE_URL", "https://ramclouds.me/v1")
    ramclouds_model = os.getenv("RAMCLOUDS_MODEL", "gemini-2.5-flash")
    if ramclouds_key:
        providers.append(ProviderConfig(
            name="ramclouds",
            api_key=ramclouds_key,
            base_url=ramclouds_url,
            model=ramclouds_model,
            priority=0,
        ))

    # Fallback: Groq (free, fast)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        providers.append(ProviderConfig(
            name="groq",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            priority=1,
        ))

    # Last resort: generic OPENAI_COMPAT_* env vars
    compat_key = os.getenv("OPENAI_COMPAT_API_KEY")
    compat_url = os.getenv("OPENAI_COMPAT_BASE_URL")
    compat_model = os.getenv("OPENAI_COMPAT_MODEL", "gpt-3.5-turbo")
    if compat_key and compat_url:
        providers.append(ProviderConfig(
            name="compat",
            api_key=compat_key,
            base_url=compat_url,
            model=compat_model,
            priority=2,
        ))

    providers.sort(key=lambda p: p.priority)
    return providers


class UnifiedLLMClient:
    """OpenAI-compatible LLM client with automatic provider fallback.

    Primary: model from RAMCLOUDS_MODEL (default gemini-2.5-flash) via https://ramclouds.me/v1
    Fallback: Groq → any OpenAI-compatible endpoint

    Usage:
        client = UnifiedLLMClient()
        resp = await client.chat("Trả lời câu hỏi sau: ...")
        data = await client.chat_json("Extract JSON from: ...", system="...")
    """

    def __init__(self, force_model: Optional[str] = None) -> None:
        self._providers = _build_providers()
        self._force_model = force_model

        if not self._providers:
            logger.warning(
                "No LLM provider configured. Set RAMCLOUDS_API_KEY (or OPENAI_API_KEY) "
                "and RAMCLOUDS_BASE_URL=https://ramclouds.me/v1 in .env"
            )
        else:
            names = [p.name for p in self._providers]
            logger.info(f"UnifiedLLMClient initialized: providers={names}")

    def _get_client(self, provider: ProviderConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    async def _call_provider(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        client = self._get_client(provider)
        model = self._force_model or provider.model

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, model=model, provider=provider.name)

    async def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a chat completion request, trying providers in priority order.

        Args:
            user_message: The user turn content.
            system_message: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            LLMResponse with content, model, and provider name.

        Raises:
            RuntimeError: If all providers fail.
        """
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        for provider in self._providers:
            try:
                result = await self._call_provider(
                    provider, messages, temperature, max_tokens, json_mode=False
                )
                logger.debug(f"LLM [{provider.name}/{result.model}]: OK")
                return result
            except Exception as exc:
                logger.warning(f"LLM [{provider.name}] failed: {exc}")

        raise RuntimeError("All LLM providers failed")

    async def chat_json(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Chat completion with JSON output mode.

        Returns parsed dict. Falls back to manual JSON parsing if provider
        does not support response_format=json_object.
        """
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        for provider in self._providers:
            try:
                result = await self._call_provider(
                    provider, messages, temperature, max_tokens, json_mode=True
                )
                return result.as_json()
            except Exception as exc:
                logger.warning(f"LLM JSON [{provider.name}] json_mode=True failed: {exc}")

            try:
                # Fallback for providers/models that reject response_format=json_object
                result2 = await self._call_provider(
                    provider, messages, temperature, max_tokens, json_mode=False
                )
                return result2.as_json()
            except Exception as exc:
                logger.warning(f"LLM JSON [{provider.name}] json_mode=False fallback failed: {exc}")

        raise RuntimeError("All LLM providers failed for JSON mode")


# Singleton for module-level import
_default_client: Optional[UnifiedLLMClient] = None


def get_llm_client() -> UnifiedLLMClient:
    """Get or create the singleton UnifiedLLMClient."""
    global _default_client
    if _default_client is None:
        _default_client = UnifiedLLMClient()
    return _default_client
