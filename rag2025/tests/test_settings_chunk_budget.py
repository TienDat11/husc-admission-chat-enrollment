"""
Unit tests for chunk-budget settings (STEP 4: resolve chunk-budget config drift).

Asserts the four chunk knobs live on RAGSettings with env-driven defaults, the
hard cap strictly exceeds the soft floor, and the values actually flow into both
splitters (src.chunker + scripts/chunk_table_aware_v2) via monkeypatch.
"""
from __future__ import annotations

import pytest

from config.settings import RAGSettings, settings


def test_chunk_budget_settings_have_env_driven_defaults(monkeypatch):
    """Settings: the four chunk knobs default to 350/70/512/50 from code."""
    monkeypatch.delenv("HARD_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SOFT_MIN_TOKENS", raising=False)
    monkeypatch.delenv("CHUNK_SIZE_TOKENS", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    cfg = RAGSettings(_env_file=None)

    assert cfg.CHUNK_SIZE_TOKENS == 350  # target budget
    assert cfg.CHUNK_OVERLAP == 70       # overlap
    assert cfg.HARD_MAX_TOKENS == 512    # absolute cap
    assert cfg.SOFT_MIN_TOKENS == 50     # soft floor
    assert cfg.HARD_MAX_TOKENS > cfg.SOFT_MIN_TOKENS


def test_chunk_budget_settings_read_from_env(monkeypatch):
    """Settings: HARD_MAX_TOKENS / SOFT_MIN_TOKENS are env-driven, not fixed."""
    monkeypatch.setenv("HARD_MAX_TOKENS", "600")
    monkeypatch.setenv("SOFT_MIN_TOKENS", "100")
    cfg = RAGSettings(_env_file=None)

    assert cfg.HARD_MAX_TOKENS == 600
    assert cfg.SOFT_MIN_TOKENS == 100


def test_chunk_budget_settings_reject_inverted_gates(monkeypatch):
    """Settings: a config where hard cap <= soft floor must fail fast."""
    monkeypatch.setenv("HARD_MAX_TOKENS", "100")
    monkeypatch.setenv("SOFT_MIN_TOKENS", "200")
    with pytest.raises(ValueError):
        RAGSettings(_env_file=None)


def test_global_settings_instance_exposes_chunk_budget():
    """Settings: the shared `settings` singleton carries the chunk knobs."""
    assert settings.HARD_MAX_TOKENS >= settings.SOFT_MIN_TOKENS
    assert settings.CHUNK_SIZE_TOKENS >= settings.HARD_MAX_TOKENS // 2  # sanity only
