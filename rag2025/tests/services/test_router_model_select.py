"""Tests for S15.4 / ADR-D — RAMCLOUDS_ROUTER_MODEL effective resolution.

MUST-FIX #3 (architect): the default for the router LLM must equal the current
HYDE model — NOT the GEN model. The eval path uses a router LLM built from
``settings.RAMCLOUDS_HYDE_MODEL`` (``graphrag_retriever.py:259``). After S15.4
the router model is configurable via ``RAMCLOUDS_ROUTER_MODEL``; when unset
the no-op default resolves to the HYDE model so behaviour is identical to
pre-S15.4.

Note: the actual wire-up at ``graphrag_retriever.py:259`` is owned by Lane A.
This test pins ONLY the settings-level contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the package root is importable so `config.settings` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import RAGSettings  # noqa: E402


def test_effective_router_model_defaults_to_hyde_when_env_unset(monkeypatch):
    """Env unset → ``effective_router_model`` resolves to ``RAMCLOUDS_HYDE_MODEL``.

    True no-op default per MUST-FIX #3: the router stays on whatever the HYDE
    pipeline already uses, so simply rebuilding `from_disk` does not change
    routing behaviour unless the env is explicitly set.
    """
    monkeypatch.delenv("RAMCLOUDS_ROUTER_MODEL", raising=False)
    s = RAGSettings()
    assert s.effective_router_model == s.RAMCLOUDS_HYDE_MODEL


def test_effective_router_model_does_not_default_to_gen_model(monkeypatch):
    """Sanity: with env unset the resolved router model is NOT the GEN model.

    Pins the architect's MUST-FIX #3 — the original plan claim of "default =
    GEN model = zero change" was wrong; we explicitly assert the no-op default
    matches HYDE, not GEN, when the two are configured differently.
    """
    monkeypatch.delenv("RAMCLOUDS_ROUTER_MODEL", raising=False)
    s = RAGSettings()
    # The repo defaults set HYDE != GEN; this assertion guards against a future
    # change that accidentally pins the router to GEN under the no-op default.
    if s.RAMCLOUDS_HYDE_MODEL != s.RAMCLOUDS_GEN_MODEL:
        assert s.effective_router_model != s.RAMCLOUDS_GEN_MODEL


def test_effective_router_model_uses_env_override(monkeypatch):
    """Env ``RAMCLOUDS_ROUTER_MODEL=<sentinel>`` → ``effective_router_model`` matches it.

    Uses a sentinel that cannot collide with any real model name set in the
    repo's local ``.env`` (which may pin HYDE/GEN to specific provider models
    during development).
    """
    sentinel = "s15-router-override-sentinel"
    monkeypatch.setenv("RAMCLOUDS_ROUTER_MODEL", sentinel)
    s = RAGSettings()
    assert s.effective_router_model == sentinel
    # And the override must NOT silently fall back to HYDE.
    assert s.effective_router_model != s.RAMCLOUDS_HYDE_MODEL


def test_effective_router_model_empty_env_treated_as_unset(monkeypatch):
    """Explicit empty-string env value must resolve to HYDE (treated as unset).

    Operators sometimes set an empty value to "clear" an override; the
    resolution must mirror the unset path so nothing routes to a blank model
    name downstream.
    """
    monkeypatch.setenv("RAMCLOUDS_ROUTER_MODEL", "")
    s = RAGSettings()
    assert s.effective_router_model == s.RAMCLOUDS_HYDE_MODEL
    assert s.effective_router_model != ""
