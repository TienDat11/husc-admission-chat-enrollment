"""
Unified LLM Client – ns24 gateway (OpenAI-compatible)

One provider only: the ns24 gateway at ``NS24_BASE_URL``. Every call site picks
a *role* (router / ner / hyde / guardrail / gen / default) and the role resolves
to a primary model plus an optional same-gateway fallback model. There is no
cross-provider fallback: a role that cannot reach its models fails visibly
instead of silently answering on a different vendor.

Design principles:
- One class, one gateway, model-per-role
- Retry with exponential backoff (tenacity)
- JSON-mode support for structured outputs
- Async-first, sync wrapper provided for compatibility
- Model whitelist enforced at client construction (typo fails at startup, not
  on the first user question)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from config.request_context import get_request_id, reset_request_id, set_request_id
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
# budget because empty-body responses from the ns24 gateway are persistent and
# burning 4 attempts × 1-8s backoff drove router_classify p95=11.6s, max=60s.
CHAT_MAX_ATTEMPTS = 4
JSON_MODE_MAX_ATTEMPTS = 2

# httpx.Timeout: bound the read side so a stalled response fails at ~30s
# instead of hanging. 30s is safe for SSE streaming generation (gen p95 ~3.5s).
LLM_READ_TIMEOUT_S = 30.0
LLM_CONNECT_TIMEOUT_S = 10.0

# --- T12: per-call token accounting (R2 — LLM cost was never measured) ---
# One record per provider call: retries, stream, and sub-calls (router/hyde/
# ner/guard) all funnel through _call_provider / _call_provider_stream, so a
# ledger here covers every path without touching callers.


@dataclass
class TokenUsage:
    """One billable LLM call.

    ``prompt_tokens`` / ``completion_tokens`` are ``None`` when the gateway
    returned no ``usage`` object — null is honest, zero would read as "free".
    ``cost_usd`` is ``None`` in that case and in the case of a missing price;
    ``cost_partial`` marks a total that skipped a component (e.g. the gateway
    publishes no cache-write rate for the model).
    """

    provider: str
    model: str
    role: str = "default"
    request_id: Optional[str] = None
    attempt: int = 1
    fallback_from: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    outcome: str = "ok"
    usage_missing: bool = False
    cost_usd: Optional[float] = None
    cost_partial: bool = False


# Correlation id for the user request that triggered a chain of LLM calls.
# Defined in config.request_context (one module, one contextvar) and re-exported
# here so either import spelling of this module sees the same binding.



def estimate_cost(
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    cached_input_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
) -> Tuple[Optional[float], bool]:
    """Gateway cost in USD for one call.

    Returns ``(cost, partial)``. ``cost`` is ``None`` when tokens or the price
    row are unknown. ``partial`` is True when a billed component could not be
    priced — either the gateway does not publish a cache-write rate for the
    model, or cache counters were reported without a matching price.
    """
    from config.ns24_models import PRICE_USD_PER_MTOK

    if prompt_tokens is None or completion_tokens is None:
        return None, False
    price = PRICE_USD_PER_MTOK.get(model)
    if price is None:
        return None, False

    cached = cached_input_tokens or 0
    cache_write = cache_write_tokens or 0
    uncached = max(0, prompt_tokens - cached - cache_write)
    partial = False

    total = uncached * price.input + completion_tokens * price.output
    total += cached * price.cache_read
    if cache_write:
        if price.cache_write is None:
            partial = True
        else:
            total += cache_write * price.cache_write
    return total / 1_000_000, partial


class TokenLedger:
    """Thread-safe ledger of LLM calls: tokens, role, retry and cost.

    Callers read ``snapshot()`` / ``totals()`` at their eval-question boundary
    and ``reset()`` between questions. Missing ``usage`` is recorded as null
    tokens plus ``usage_missing``; nothing is ever estimated from text length.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: List[TokenUsage] = []

    def record(self, usage: TokenUsage) -> None:
        with self._lock:
            self._calls.append(usage)

    def snapshot(self) -> List[TokenUsage]:
        with self._lock:
            return list(self._calls)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    def totals(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate per (provider, model): tokens, calls, cost, missing usage."""
        agg: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            calls = list(self._calls)
        for c in calls:
            key = f"{c.provider}/{c.model}"
            slot = agg.setdefault(key, {
                "provider": c.provider,
                "model": c.model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "call_count": 0,
                "usage_missing_calls": 0,
                "cost_usd": 0.0,
                "cost_partial": False,
            })
            slot["prompt_tokens"] += c.prompt_tokens or 0
            slot["completion_tokens"] += c.completion_tokens or 0
            slot["cached_input_tokens"] += c.cached_input_tokens or 0
            slot["cache_write_tokens"] += c.cache_write_tokens or 0
            slot["call_count"] += 1
            if c.usage_missing:
                slot["usage_missing_calls"] += 1
            if c.cost_usd is None:
                slot["cost_partial"] = True
            else:
                slot["cost_usd"] += c.cost_usd
            if c.cost_partial:
                slot["cost_partial"] = True
        return agg

    def summary(self) -> Dict[str, Any]:
        """Per-role and per-request rollup plus a grand total.

        ``cost_usd`` at any level is ``None`` when at least one call in the
        group could not be priced; the group then also carries
        ``cost_partial=True`` so a report never presents a partial sum as final.
        """
        with self._lock:
            calls = list(self._calls)

        by_role: Dict[str, Dict[str, Any]] = {}
        by_request: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            for bucket, key in (
                (by_role, c.role or "default"),
                (by_request, c.request_id or "unattributed"),
            ):
                slot = bucket.setdefault(key, {
                    "call_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_partial": False,
                    "usage_missing_calls": 0,
                    "fallback_calls": 0,
                })
                slot["call_count"] += 1
                slot["prompt_tokens"] += c.prompt_tokens or 0
                slot["completion_tokens"] += c.completion_tokens or 0
                if c.usage_missing:
                    slot["usage_missing_calls"] += 1
                if c.fallback_from:
                    slot["fallback_calls"] += 1
                if c.cost_usd is None:
                    slot["cost_partial"] = True
                else:
                    slot["cost_usd"] += c.cost_usd
                if c.cost_partial:
                    slot["cost_partial"] = True

        total = {
            "call_count": len(calls),
            "prompt_tokens": sum(c.prompt_tokens or 0 for c in calls),
            "completion_tokens": sum(c.completion_tokens or 0 for c in calls),
            "cost_usd": 0.0,
            "cost_partial": False,
            "usage_missing_calls": sum(1 for c in calls if c.usage_missing),
        }
        for c in calls:
            if c.cost_usd is None:
                total["cost_partial"] = True
            else:
                total["cost_usd"] += c.cost_usd
            if c.cost_partial:
                total["cost_partial"] = True
        return {"by_role": by_role, "by_request": by_request, "total": total}


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Retry transient transport/rate-limit/server errors.

    the ns24 gateway intermittently returns `APIError: an internal error occurred`
    (a generic upstream 5xx surfaced without a status_code) — observed ~1 in 5
    calls on the previous gateway. That is exactly the "LLM lỗi trả lời nửa chừng" symptom, so
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

    ponytail: If the ns24 gateway adds native think-tag filtering at the SSE level,
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

    ponytail: Provider-level stripping.  If the ns24 gateway adds native think-tag
    filtering, remove this function and the hook in LLMResponse.__init__.
    """
    text = _THINK_PAIR_RE.sub("", text)
    text = _LONE_OPEN_RE.sub("", text)
    text = _LONE_CLOSE_RE.sub("", text)
    return text


class LLMResponse:
    """Typed wrapper for LLM chat completion response."""

    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        token_usage: Optional[TokenUsage] = None,
    ) -> None:
        # G2: strip DeepSeek think tags — single chokepoint for all paths
        self.content = strip_reasoning(content)
        self.model = model
        self.provider = provider
        # T12: tokens reported by the provider (None when the provider
        # did not send a usage object — callers default to 0).
        self.token_usage = token_usage

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
    """One candidate endpoint for a call: the ns24 gateway plus a model id."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        priority: int = 0,
        fallback_from: Optional[str] = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.priority = priority
        # Set on the fallback entry only: which primary model it replaced.
        self.fallback_from = fallback_from


# --- ns24 gateway ----------------------------------------------------------
# One gateway, one key, one OpenAI-compatible wire. Roles differ only in the
# model they name; the fallback entry is always another model on the SAME
# gateway so a vendor outage surfaces as an error instead of a silent swap.
NS24_PROVIDER_NAME = "ns24"
NS24_DEFAULT_BASE_URL = "https://api.ns24.io/v1"


def _validate_model(model: str, role: str) -> str:
    """Reject a model outside the whitelist at construction time."""
    from config.ns24_models import ALLOWED_MODELS

    cleaned = (model or "").strip()
    if cleaned not in ALLOWED_MODELS:
        raise ValueError(
            f"ns24 model {cleaned!r} (role={role}) is not in the allowed set: "
            f"{', '.join(ALLOWED_MODELS)}"
        )
    return cleaned


def _role_models(role: str) -> Tuple[str, str]:
    """Resolve ``(primary, fallback)`` for a role from env, then defaults.

    Env names come from ``config.ns24_models.ROLE_ENV`` (e.g. ``NS24_GEN_MODEL``
    / ``NS24_GEN_FALLBACK_MODEL``). An empty fallback means "no fallback".
    """
    from config.ns24_models import ROLE_ENV, ROLE_MODELS

    primary_default, fallback_default = ROLE_MODELS.get(
        role, ROLE_MODELS["default"]
    )
    env_primary, env_fallback = ROLE_ENV.get(role, ROLE_ENV["default"])
    primary = (os.getenv(env_primary) or primary_default).strip()
    fallback = (os.getenv(env_fallback) or fallback_default or "").strip()
    return primary, fallback


def _build_providers(
    role: str = "default", force_model: Optional[str] = None
) -> List[ProviderConfig]:
    """Build the provider list for one role: primary, then same-gateway fallback.

    Returns an empty list when ``NS24_API_KEY`` is unset — callers treat that as
    "LLM unavailable" and fail visibly rather than reaching for another vendor.
    """
    key = os.getenv("NS24_API_KEY")
    if not key:
        return []
    base_url = os.getenv("NS24_BASE_URL", NS24_DEFAULT_BASE_URL)
    primary, fallback = _role_models(role)
    if force_model:
        # An explicit model override replaces the primary and keeps the role's
        # fallback only when it differs from the override.
        primary = force_model
    primary = _validate_model(primary, role)

    providers = [
        ProviderConfig(
            name=NS24_PROVIDER_NAME,
            api_key=key,
            base_url=base_url,
            model=primary,
            priority=0,
        )
    ]
    if fallback and fallback != primary:
        providers.append(
            ProviderConfig(
                name=NS24_PROVIDER_NAME,
                api_key=key,
                base_url=base_url,
                model=_validate_model(fallback, role),
                priority=1,
                fallback_from=primary,
            )
        )
    return providers


def _usage_from(
    usage: Any,
    provider: str,
    model: str,
    *,
    role: str = "default",
    attempt: int = 1,
    fallback_from: Optional[str] = None,
    latency_ms: Optional[float] = None,
    outcome: str = "ok",
) -> TokenUsage:
    """Build a TokenUsage from an OpenAI ``usage`` object (or None).

    Missing fields → ``None`` plus ``usage_missing``: the gateway did not report
    them. We NEVER estimate from text length — null beats a guess.
    """
    prompt: Optional[int] = None
    completion: Optional[int] = None
    cached: Optional[int] = None
    cache_write: Optional[int] = None
    if usage is not None:
        p = getattr(usage, "prompt_tokens", None)
        c = getattr(usage, "completion_tokens", None)
        if isinstance(p, int):
            prompt = p
        if isinstance(c, int):
            completion = c
        details = getattr(usage, "prompt_tokens_details", None)
        cached_val = getattr(details, "cached_tokens", None) if details is not None else None
        if isinstance(cached_val, int):
            cached = cached_val
        write_val = getattr(details, "cache_write_tokens", None) if details is not None else None
        if isinstance(write_val, int):
            cache_write = write_val

    usage_missing = prompt is None and completion is None
    cost, cost_partial = estimate_cost(
        model, prompt, completion, cached, cache_write
    )
    return TokenUsage(
        provider=provider,
        model=model,
        role=role,
        request_id=get_request_id(),
        attempt=attempt,
        fallback_from=fallback_from,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_input_tokens=cached,
        cache_write_tokens=cache_write,
        latency_ms=latency_ms,
        outcome=outcome,
        usage_missing=usage_missing,
        cost_usd=cost,
        cost_partial=cost_partial,
    )


class UnifiedLLMClient:
    """ns24 gateway client with a role-selected model and same-gateway fallback.

    ``role`` picks the model pair (see ``config.ns24_models.ROLE_MODELS``):
    router / ner / hyde / guardrail / gen / default. ``force_model`` overrides
    the role's primary model — it must still be in the whitelist.

    With no ``NS24_API_KEY`` the client has no providers; every call then raises
    ``RuntimeError`` instead of quietly reaching another vendor.

    Usage:
        client = UnifiedLLMClient(role="gen")
        resp = await client.chat("Trả lời câu hỏi sau: ...")
        data = await client.chat_json("Extract JSON from: ...", system="...")
    """

    def __init__(
        self,
        force_model: Optional[str] = None,
        providers: Optional[List[ProviderConfig]] = None,
        role: str = "default",
    ) -> None:
        self._providers = (
            providers
            if providers is not None
            else _build_providers(role=role, force_model=force_model)
        )
        self._role = role
        self._force_model = force_model
        # Physical call counter for the current public call: incremented per
        # provider attempt (including tenacity retries) so the ledger can show
        # how many gateway hits one answer actually cost.
        self._attempt = 0
        # T12: every provider call (incl. retries + streams) records its
        # token usage here; eval harnesses read/drain it per question.
        self.token_ledger = TokenLedger()
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
                f"UnifiedLLMClient[{role}]: no provider — set NS24_API_KEY "
                f"(and NS24_BASE_URL if not {NS24_DEFAULT_BASE_URL})"
            )
        else:
            models = [p.model for p in self._providers]
            logger.info(f"UnifiedLLMClient[{role}] initialized: models={models}")

    def _get_client(self, provider: ProviderConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            http_client=self._http_client,
        )

    async def close(self) -> None:
        """Cleanup HTTP pool — call on lifespan shutdown."""
        await self._http_client.aclose()

    def get_token_ledger(self) -> TokenLedger:
        """T12: expose the ledger so eval harnesses can read per-question totals."""
        return self.token_ledger

    def get_token_usage_totals(self) -> Dict[str, Dict[str, int]]:
        """T12: aggregate token usage across all recorded calls.

        Returns a dict keyed by "provider/model" with totals.
        """
        return self.token_ledger.totals()

    def reset_token_ledger(self) -> None:
        """T12: clear all recorded calls — call between eval questions."""
        self.token_ledger.reset()


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

        The ns24 gateway intermittently returns an empty body for json_mode
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
        model = provider.model
        self._attempt += 1
        attempt = self._attempt
        started = time.perf_counter()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # T12: ask providers that support it for usage on streams (OpenAI
        # streams usage in a final chunk when stream_options.include_usage).
        kwargs["stream_options"] = {"include_usage": True}

        # SSE streaming path: gpt-6-luna (gen) and gemini-3.5-flash-lite (router/ner/hyde) on ns24 stream faster + survive
        # mid-answer cutoffs better (we accumulate deltas, so a late chunk drop
        # still yields a usable partial). JSON mode ALSO streams now (user asked
        # for HYDE on SSE) — we accumulate the full JSON string then the caller
        # parses via as_json(). Toggle via NS24_STREAM env.
        stream_enabled = os.getenv("NS24_STREAM", "true").lower() in {"1", "true", "yes"}
        try:
            if stream_enabled:
                kwargs["stream"] = True
                chunks: List[str] = []
                finish_reason = None
                stream_usage = None
                stream = await client.chat.completions.create(**kwargs)
                async for ev in stream:
                    # T12: the final chunk (choices empty) carries usage when
                    # the provider honors stream_options.include_usage.
                    ev_usage = getattr(ev, "usage", None)
                    if ev_usage is not None:
                        stream_usage = ev_usage
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
                        f"LLM [{provider.name}/{model}] stream truncated (finish_reason=length)"
                    )
                usage = _usage_from(
                    stream_usage,
                    provider.name,
                    model,
                    role=self._role,
                    attempt=attempt,
                    fallback_from=provider.fallback_from,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                self.token_ledger.record(usage)
                return LLMResponse(
                    content=content, model=model, provider=provider.name, token_usage=usage
                )

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            usage = _usage_from(
                getattr(response, "usage", None),
                provider.name,
                model,
                role=self._role,
                attempt=attempt,
                fallback_from=provider.fallback_from,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            self.token_ledger.record(usage)
            return LLMResponse(
                content=content, model=model, provider=provider.name, token_usage=usage
            )
        except Exception as exc:
            # A failed attempt is still a call: record it with null tokens so
            # the ledger shows the retry cost instead of hiding it.
            self.token_ledger.record(
                _usage_from(
                    None,
                    provider.name,
                    model,
                    role=self._role,
                    attempt=attempt,
                    fallback_from=provider.fallback_from,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    outcome=f"error:{type(exc).__name__}",
                )
            )
            raise

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
        model = provider.model
        self._attempt += 1
        attempt = self._attempt
        started = time.perf_counter()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # T12: capture usage in the final chunk when supported.
            "stream_options": {"include_usage": True},
        }

        sanitizer = _StreamThinkSanitizer()
        stream = await client.chat.completions.create(**kwargs)
        stream_usage = None
        try:
            async for ev in stream:
                # T12: final chunk (choices empty) carries usage when present.
                ev_usage = getattr(ev, "usage", None)
                if ev_usage is not None:
                    stream_usage = ev_usage
                if not ev.choices:
                    continue
                choice = ev.choices[0]
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    cleaned = sanitizer.process(piece)
                    if cleaned:
                        yield cleaned
        except Exception as exc:
            self.token_ledger.record(
                _usage_from(
                    stream_usage,
                    provider.name,
                    model,
                    role=self._role,
                    attempt=attempt,
                    fallback_from=provider.fallback_from,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    outcome=f"error:{type(exc).__name__}",
                )
            )
            raise
        remaining = sanitizer.flush()
        if remaining:
            yield remaining
        usage = _usage_from(
            stream_usage,
            provider.name,
            model,
            role=self._role,
            attempt=attempt,
            fallback_from=provider.fallback_from,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self.token_ledger.record(usage)

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

    def _exhausted_message(self, operation: str) -> str:
        """Error text that names the role and the models actually tried."""
        if not self._providers:
            return (
                f"LLM unavailable for role={self._role}: NS24_API_KEY is not set "
                f"({operation})"
            )
        tried = ", ".join(f"{p.model}" + (" (fallback)" if p.fallback_from else "") for p in self._providers)
        return f"All ns24 models failed for role={self._role} ({operation}): tried {tried}"

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

        self._attempt = 0
        for provider in self._providers:
            try:
                result = await self._call_provider_chat(
                    provider, messages, temperature, max_tokens
                )
                logger.debug(f"LLM [{provider.name}/{result.model}]: OK")
                return result
            except Exception as exc:
                logger.warning(f"LLM [{provider.name}/{provider.model}] failed: {exc}")

        raise RuntimeError(self._exhausted_message("chat"))

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

        self._attempt = 0
        for provider in self._providers:
            try:
                pieces = await self._call_provider_stream_retry(
                    provider, messages, temperature, max_tokens
                )
                for piece in pieces:
                    yield piece
                return  # success — exit
            except Exception as exc:
                logger.warning(f"LLM stream [{provider.name}/{provider.model}] failed: {exc}")

        raise RuntimeError(self._exhausted_message("chat_stream"))

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

        self._attempt = 0
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
                    f"LLM JSON [{provider.name}/{provider.model}] json_mode=True APIStatusError "
                    f"status={status} body={body}"
                )
            except Exception as exc:
                logger.warning(
                    f"LLM JSON [{provider.name}/{provider.model}] json_mode=True failed: {exc}"
                )

            try:
                # Fallback for models that reject response_format=json_object
                result2 = await self._call_provider_chat(
                    provider, messages, temperature, max_tokens
                )
                return result2.as_json()
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None)
                logger.warning(
                    f"LLM JSON [{provider.name}/{provider.model}] json_mode=False fallback "
                    f"APIStatusError status={status} body={body}"
                )
            except Exception as exc:
                logger.warning(
                    f"LLM JSON [{provider.name}/{provider.model}] json_mode=False fallback failed: {exc}"
                )

        raise RuntimeError(self._exhausted_message("chat_json"))


# --- Client singletons, one per call-site role ---
_clients: Dict[str, UnifiedLLMClient] = {}
_clients_lock = threading.Lock()


def get_llm_client(
    force_model: Optional[str] = None, role: str = "default"
) -> UnifiedLLMClient:
    """Get or create the per-role UnifiedLLMClient singleton.

    Args:
        force_model: Optional model override, still whitelist-checked. A forced
            model creates a NEW client so the shared per-role instance keeps
            the role's configured model.
        role: Call-site role — router / ner / hyde / guardrail / gen / default.
    """
    if force_model:
        return UnifiedLLMClient(force_model=force_model, role=role)
    with _clients_lock:
        client = _clients.get(role)
        if client is None:
            client = UnifiedLLMClient(role=role)
            _clients[role] = client
        return client


def get_router_llm_client() -> UnifiedLLMClient:
    """Router/classify client — its own role, its own model pair."""
    return get_llm_client(role="router")


def reset_llm_clients() -> None:
    """Drop cached clients (tests, or after a settings change)."""
    with _clients_lock:
        _clients.clear()


def aggregate_ledger_summary() -> Dict[str, Any]:
    """Merge every per-role client's ledger into one summary.

    ``TokenLedger`` lives on the client instance and ``get_llm_client`` caches
    one client per role, so a single request's calls are spread over up to six
    ledgers (router / ner / hyde / guardrail / gen / default). ``by_request``
    is therefore only meaningful when the buckets are merged: this is the
    function a cost report calls, not ``client.token_ledger.summary()``.

    Only clients already created are read — a role that never ran contributes
    nothing, and this never builds a client as a side effect. ``by_role`` is
    kept because the role name is the same key in every bucket.
    """
    by_role: Dict[str, Dict[str, Any]] = {}
    by_request: Dict[str, Dict[str, Any]] = {}
    total: Dict[str, Any] = {
        "call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "cost_partial": False,
        "usage_missing_calls": 0,
    }

    with _clients_lock:
        ledgers = [client.token_ledger for client in _clients.values()]

    for ledger in ledgers:
        summary = ledger.summary()
        for bucket, source in ((by_role, summary["by_role"]), (by_request, summary["by_request"])):
            for key, slot in source.items():
                acc = bucket.setdefault(key, {
                    "call_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_partial": False,
                    "usage_missing_calls": 0,
                    "fallback_calls": 0,
                })
                acc["call_count"] += slot["call_count"]
                acc["prompt_tokens"] += slot["prompt_tokens"]
                acc["completion_tokens"] += slot["completion_tokens"]
                acc["usage_missing_calls"] += slot["usage_missing_calls"]
                acc["fallback_calls"] += slot["fallback_calls"]
                acc["cost_usd"] += slot["cost_usd"]
                acc["cost_partial"] = acc["cost_partial"] or slot["cost_partial"]

        for field in ("call_count", "prompt_tokens", "completion_tokens", "usage_missing_calls"):
            total[field] += summary["total"][field]
        total["cost_usd"] += summary["total"]["cost_usd"]
        total["cost_partial"] = total["cost_partial"] or summary["total"]["cost_partial"]

    return {"by_role": by_role, "by_request": by_request, "total": total}
