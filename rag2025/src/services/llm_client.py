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
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


# --- Bounded retry/timeouts (S10 perf fix) ---
# chat (non-json, streaming generation) gets the wider retry budget because
# legitimate generation can hit transient 5xx; json_mode/classify gets a tight
# budget because empty-body responses from ramclouds.me are persistent and
# burning 4 attempts × 1-8s backoff drove router_classify p95=11.6s, max=60s.
CHAT_MAX_ATTEMPTS = 4
JSON_MODE_MAX_ATTEMPTS = 2

# httpx.Timeout: bound the read side so a stalled response fails at ~30s
# instead of hanging. 30s is safe for SSE streaming generation (gen p95 ~3.5s).
LLM_READ_TIMEOUT_S = 30.0
LLM_CONNECT_TIMEOUT_S = 10.0


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Retry transient transport/rate-limit/server errors.

    ramclouds.me intermittently returns `APIError: an internal error occurred`
    (a generic upstream 5xx surfaced without a status_code) — observed ~1 in 5
    gpt-5.4 calls. That is exactly the "LLM lỗi trả lời nửa chừng" symptom, so
    we treat bare APIError / InternalServerError as retryable too.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in {408, 409, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, APIError):
        # Generic APIError without a status_code (e.g. "an internal error occurred")
        msg = str(getattr(exc, "message", "") or exc).lower()
        return "internal error" in msg or "overloaded" in msg or "timeout" in msg
    return False


# ── G2: DeepSeek think-tag sanitization ──
# Strip <think>...</think> reasoning blocks from model output.
# deepseek-v4-flash may wrap reasoning in think tags AND/OR use the
# reasoning_content SSE field.  Pure function + hook in LLMResponse covers all
# paths (stream, non-stream, JSON mode).  Idempotent — safe to call twice.

_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LONE_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_LONE_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


class _StreamThinkSanitizer:
    """Stateful streaming sanitizer for DeepSeek think tags.

    Tags can split across chunks.  Buffers across chunk boundaries to suppress
    everything between <think> and </think> (case-insensitive) while flushing
    safe visible text incrementally.  Handles unmatched opening tag at stream
    end via flush().

    ponytail: If ramclouds adds native think-tag filtering at the SSE level,
    remove this class and the hook in _call_provider_stream.
    """

    _OPEN = re.compile(r"<think>", re.IGNORECASE)
    _CLOSE = re.compile(r"</think>", re.IGNORECASE)

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def process(self, chunk: str) -> str:
        """Feed a chunk; return safe visible text to emit (may be empty)."""
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._in_think:
                m = self._CLOSE.search(self._buf)
                if m is None:
                    break
                self._buf = self._buf[m.end() :]
                self._in_think = False
                continue
            m = self._OPEN.search(self._buf)
            if m is None:
                out.append(self._buf)
                self._buf = ""
                break
            out.append(self._buf[: m.start()])
            self._buf = self._buf[m.start() :]
            cm = self._CLOSE.search(self._buf)
            if cm is not None:
                self._buf = self._buf[cm.end() :]
                continue
            self._in_think = True
            self._buf = self._buf[m.end() :]
            break
        return "".join(out)

    def flush(self) -> str:
        """Finalize: return any remaining safe text; suppress open think."""
        if not self._buf:
            return ""
        remaining = self._buf
        self._buf = ""
        if self._in_think:
            return ""
        return strip_reasoning(remaining)


def strip_reasoning(text: str) -> str:
    """Strip DeepSeek think-tag reasoning blocks from model output.

    Handles:
    - Matched <think>...</think> blocks (content between is removed)
    - Unmatched opening <think (stream-fragment safe)
    - Unmatched closing </think>
    - Multiple tags in one string

    Normal non-think text, empty string, JSON bodies: all survive unchanged.
    Idempotent — applying twice yields the same result as once.

    ponytail: Provider-level stripping.  If ramclouds adds native think-tag
    filtering, remove this function and the hook in LLMResponse.__init__.
    """
    text = _THINK_PAIR_RE.sub("", text)
    text = _LONE_OPEN_RE.sub("", text)
    text = _LONE_CLOSE_RE.sub("", text)
    return text


class LLMResponse:
    """Typed wrapper for LLM chat completion response."""

    def __init__(self, content: str, model: str, provider: str) -> None:
        # G2: strip DeepSeek think tags — single chokepoint for all paths
        self.content = strip_reasoning(content)
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

    # Primary: AI Box (configurable model — highest quality for NER/GraphRAG)
    ramclouds_key = os.getenv("RAMCLOUDS_API_KEY")
    ramclouds_url = os.getenv("RAMCLOUDS_BASE_URL", "https://api.ai-box.vn/v1")
    ramclouds_model = os.getenv("RAMCLOUDS_MODEL", "deepseek-v4-flash")
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
            model=os.getenv("GROQ_NER_MODEL", "llama-3.1-8b-instant"),
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

    def __init__(self, force_model: Optional[str] = None, providers: Optional[List[ProviderConfig]] = None) -> None:
        self._providers = providers if providers is not None else _build_providers()
        self._force_model = force_model

        # @spec(S10.B5) — persistent HTTP connection pool
        # Bounded read timeout so a stalled gateway response fails at ~30s
        # rather than hanging indefinitely. Read cap is shared with SSE
        # generation but 30s ≫ gen p95 (~3.5s) so no risk of cut-off.
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
            timeout=httpx.Timeout(
                connect=LLM_CONNECT_TIMEOUT_S,
                read=LLM_READ_TIMEOUT_S,
                write=LLM_CONNECT_TIMEOUT_S,
                pool=LLM_CONNECT_TIMEOUT_S,
            ),
        )

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
            http_client=self._http_client,
        )

    async def close(self) -> None:
        """Cleanup HTTP pool — call on lifespan shutdown."""
        await self._http_client.aclose()

    @retry(
        stop=stop_after_attempt(CHAT_MAX_ATTEMPTS),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception(_is_retryable_llm_error),
        reraise=True,
    )
    async def _call_provider_chat(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Streaming/generation path — wide retry budget (4)."""
        return await self._call_provider(
            provider, messages, temperature, max_tokens, json_mode=False
        )

    @retry(
        stop=stop_after_attempt(JSON_MODE_MAX_ATTEMPTS),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception(_is_retryable_llm_error),
        reraise=True,
    )
    async def _call_provider_json(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """json_mode/classify path — tight retry budget (2).

        ramclouds.me intermittently returns an empty body for json_mode
        ("Expecting value: line 1 column 1"). 4 attempts × 1-8s exp backoff
        burned 30-60s on a single bad call, so we cap at 2 attempts and let
        chat_json's own per-provider fallback / RuntimeError handle the rest.
        """
        return await self._call_provider(
            provider, messages, temperature, max_tokens, json_mode=True
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
        # force_model is provider-scoped to ramclouds so a DeepSeek force_model
        # never bleeds into Groq (which would 404 on a non-Groq model name).
        model = self._force_model if provider.name == "ramclouds" and self._force_model else provider.model

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # SSE streaming path: gpt-5.x / mimo on ramclouds stream faster + survive
        # mid-answer cutoffs better (we accumulate deltas, so a late chunk drop
        # still yields a usable partial). JSON mode ALSO streams now (user asked
        # for HYDE on SSE) — we accumulate the full JSON string then the caller
        # parses via as_json(). Toggle via RAMCLOUDS_STREAM env.
        stream_enabled = os.getenv("RAMCLOUDS_STREAM", "true").lower() in {"1", "true", "yes"}
        if stream_enabled:
            kwargs["stream"] = True
            chunks: List[str] = []
            finish_reason = None
            stream = await client.chat.completions.create(**kwargs)
            async for ev in stream:
                if not ev.choices:
                    continue
                choice = ev.choices[0]
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    chunks.append(piece)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
            content = "".join(chunks)
            if finish_reason == "length":
                logger.warning(
                    f"LLM [{provider.name}/{model}] stream truncated (finish_reason=length, "
                    f"max_tokens={max_tokens}) — answer may be cut mid-sentence"
                )
            return LLMResponse(content=content, model=model, provider=provider.name)

        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, model=model, provider=provider.name)

    async def _call_provider_stream(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Core streaming call — always uses stream=True, yields token pieces.

        Does NOT apply retry; retry is handled by chat_stream's per-provider
        fallback loop. Each piece is a raw delta content string.
        """
        client = self._get_client(provider)
        # force_model is provider-scoped to ramclouds (see _call_provider).
        model = self._force_model if provider.name == "ramclouds" and self._force_model else provider.model

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        sanitizer = _StreamThinkSanitizer()
        stream = await client.chat.completions.create(**kwargs)
        async for ev in stream:
            if not ev.choices:
                continue
            choice = ev.choices[0]
            delta = getattr(choice, "delta", None)
            piece = getattr(delta, "content", None) if delta else None
            if piece:
                cleaned = sanitizer.process(piece)
                if cleaned:
                    yield cleaned
        remaining = sanitizer.flush()
        if remaining:
            yield remaining

    @retry(
        stop=stop_after_attempt(CHAT_MAX_ATTEMPTS),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception(_is_retryable_llm_error),
        reraise=True,
    )
    async def _call_provider_stream_retry(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        """Streaming call with retry — accumulates and returns token pieces.

        Same retry budget as _call_provider_chat (CHAT_MAX_ATTEMPTS=4).
        Returns the list of token strings so the caller can replay them
        after a provider fallback switch without losing earlier tokens.
        """
        pieces: List[str] = []
        async for piece in self._call_provider_stream(
            provider, messages, temperature, max_tokens
        ):
            pieces.append(piece)
        return pieces

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
                result = await self._call_provider_chat(
                    provider, messages, temperature, max_tokens
                )
                logger.debug(f"LLM [{provider.name}/{result.model}]: OK")
                return result
            except Exception as exc:
                logger.warning(f"LLM [{provider.name}] failed: {exc}")

        raise RuntimeError("All LLM providers failed")

    async def chat_stream(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Streaming chat completion — yields token pieces as they arrive.

        Provider-priority fallback chain same as chat(). Each provider gets
        CHAT_MAX_ATTEMPTS (4) retries with exp backoff via _call_provider_chat
        (which internally uses stream=True). Yields raw delta content strings.
        The caller accumulates if a full string is needed.

        Raises RuntimeError if ALL providers fail.
        """
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        for provider in self._providers:
            try:
                pieces = await self._call_provider_stream_retry(
                    provider, messages, temperature, max_tokens
                )
                for piece in pieces:
                    yield piece
                return  # success — exit
            except Exception as exc:
                logger.warning(f"LLM stream [{provider.name}] failed: {exc}")

        raise RuntimeError("All LLM providers failed for streaming")

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
                result = await self._call_provider_json(
                    provider, messages, temperature, max_tokens
                )
                return result.as_json()
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None)
                logger.warning(
                    f"LLM JSON [{provider.name}] json_mode=True APIStatusError "
                    f"status={status} model={provider.model} body={body}"
                )
            except Exception as exc:
                logger.warning(f"LLM JSON [{provider.name}] json_mode=True failed: {exc}")

            try:
                # Fallback for providers/models that reject response_format=json_object
                result2 = await self._call_provider_chat(
                    provider, messages, temperature, max_tokens
                )
                return result2.as_json()
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None)
                logger.warning(
                    f"LLM JSON [{provider.name}] json_mode=False fallback APIStatusError "
                    f"status={status} model={provider.model} body={body}"
                )
            except Exception as exc:
                logger.warning(f"LLM JSON [{provider.name}] json_mode=False fallback failed: {exc}")

        raise RuntimeError("All LLM providers failed for JSON mode")


# --- Router-only provider chain (fast path) ---
# Groq llama-3.3-70b for router classify/step-back (~315ms vs ~3.3s on ramclouds).
# Falls back to normal _build_providers() if GROQ_API_KEY is missing.


def _build_router_providers() -> List[ProviderConfig]:
    """Build router-specific provider list: groq primary, ramclouds fallback.

    Only used by SmartQueryRouter via get_router_llm_client(). Generation,
    HYDE, and NER clients continue to use _build_providers() / get_llm_client().
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        # No groq key → fall back to normal provider chain (nothing breaks)
        return _build_providers()

    providers: List[ProviderConfig] = []

    # Priority 0: Groq llama-3.3-70b (fast, JSON-capable)
    groq_model = os.getenv("GROQ_ROUTER_MODEL", "llama-3.3-70b-versatile")
    providers.append(ProviderConfig(
        name="groq",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        model=groq_model,
        priority=0,
    ))

    # Priority 1: ramclouds fallback (existing primary)
    ramclouds_key = os.getenv("RAMCLOUDS_API_KEY")
    if ramclouds_key:
        ramclouds_url = os.getenv("RAMCLOUDS_BASE_URL", "https://ramclouds.me/v1")
        ramclouds_model = os.getenv("RAMCLOUDS_MODEL", "gemini-2.5-flash")
        providers.append(ProviderConfig(
            name="ramclouds",
            api_key=ramclouds_key,
            base_url=ramclouds_url,
            model=ramclouds_model,
            priority=1,
        ))

    providers.sort(key=lambda p: p.priority)
    return providers


# Singleton for router-only LLM client
_router_client: Optional[UnifiedLLMClient] = None


def get_router_llm_client() -> UnifiedLLMClient:
    """Get or create the singleton router-only UnifiedLLMClient.

    Provider order: groq (llama-3.3-70b) → ramclouds fallback.
    Falls back to normal providers if GROQ_API_KEY is missing.
    """
    global _router_client
    if _router_client is None:
        _router_client = UnifiedLLMClient(providers=_build_router_providers())
    return _router_client


# Singleton for module-level import
_default_client: Optional[UnifiedLLMClient] = None


def get_llm_client(force_model: Optional[str] = None) -> UnifiedLLMClient:
    """Get or create the singleton UnifiedLLMClient.

    Args:
        force_model: Optional model override. If provided, creates a NEW client
            (not the singleton) with the specified model. Use for role-specific
            routing (NER, HYDE, GEN).
    """
    global _default_client
    if force_model:
        return UnifiedLLMClient(force_model=force_model)
    if _default_client is None:
        _default_client = UnifiedLLMClient()
    return _default_client
