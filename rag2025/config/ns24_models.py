"""ns24 gateway contract: allowed models, role mapping and price table.

Single source of truth shared by ``config.settings`` (typed, validated at
startup) and ``services.llm_client`` (env-only builders used by CLI/scripts).
Pure constants — no imports from either module, so there is no cycle.

Prices are the gateway's published rates (USD per 1M tokens), NOT the upstream
vendors'. A model whose cache-write rate is unknown carries ``None`` so the cost
report can mark the number partial instead of inventing a value.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# The five models this deployment is allowed to call. Anything else is a config
# error, not a runtime surprise.
ALLOWED_MODELS: Tuple[str, ...] = (
    "gpt-6-luna",
    "deepseek-v4.1-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "claude-haiku-5-5",
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# role -> (primary model, fallback model or "" for none). Both entries must be
# in ALLOWED_MODELS; the fallback stays inside the same gateway on purpose.
ROLE_MODELS: Dict[str, Tuple[str, str]] = {
    "default": (DEFAULT_MODEL, "deepseek-v4.1-flash"),
    "router": ("gemini-3.5-flash-lite", "deepseek-v4.1-flash"),
    "ner": ("gemini-3.5-flash-lite", "deepseek-v4.1-flash"),
    "hyde": ("gemini-3.5-flash-lite", "deepseek-v4.1-flash"),
    "guardrail": ("gemini-3.5-flash-lite", "deepseek-v4.1-flash"),
    "gen": ("gpt-6-luna", "claude-haiku-5-5"),
}

# role -> (env var holding the primary model, env var holding the fallback).
ROLE_ENV: Dict[str, Tuple[str, str]] = {
    role: (f"NS24_{role.upper()}_MODEL", f"NS24_{role.upper()}_FALLBACK_MODEL")
    for role in ROLE_MODELS
}
# ``default`` keeps the short NS24_MODEL name rather than NS24_DEFAULT_MODEL.
ROLE_ENV["default"] = ("NS24_MODEL", "NS24_FALLBACK_MODEL")


class ModelPrice:
    """USD per 1M tokens for one gateway model. ``None`` = not published."""

    __slots__ = ("input", "output", "cache_read", "cache_write")

    def __init__(
        self,
        input: float,
        output: float,
        cache_read: float,
        cache_write: Optional[float],
    ) -> None:
        self.input = input
        self.output = output
        self.cache_read = cache_read
        self.cache_write = cache_write

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        }


# Gateway rates supplied by the account owner on 2026-10-27.
PRICE_USD_PER_MTOK: Dict[str, ModelPrice] = {
    "gpt-6-luna": ModelPrice(0.008, 0.040, 0.00081, None),
    "deepseek-v4.1-flash": ModelPrice(0.031416, 0.094248, 0.003128, None),
    "gemini-3.8-flash": ModelPrice(0.03696, 0.11088, 0.00368, None),
    "gemini-3.5-flash-lite": ModelPrice(0.01848, 0.05544, 0.00184, None),
    "claude-haiku-5-5": ModelPrice(0.017, 0.085, 0.0017, 0.03375),
}
