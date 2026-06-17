"""TDD: query_enhancer must be settings-driven and lazily constructed.

BUG-3 HIGH: RAMCLOUDS_HYDE_MODEL must be read from config.settings, not
os.getenv at import time. The module-level singleton must be replaced by a
lazy factory with a backward-compat shim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

# Ensure a ramclouds provider exists for HYDEQueryEnhancer init.
os.environ.setdefault("RAMCLOUDS_API_KEY", "sk-test-fake")
os.environ.setdefault("RAMCLOUDS_BASE_URL", "https://example.test/v1")
os.environ.setdefault("RAMCLOUDS_MODEL", "deepseek-v4-pro")


def test_hyde_model_read_from_settings_not_process_env(monkeypatch):
    """If RAMCLOUDS_HYDE_MODEL env is 'gpt-5.3' but settings says
    'gemini-hyde-test', the enhancer must use the settings value.
    """
    monkeypatch.setenv("RAMCLOUDS_HYDE_MODEL", "gpt-5.3")

    # Import config.settings with a patched RAMCLOUDS_HYDE_MODEL.
    from config.settings import RAGSettings

    patched_settings = RAGSettings()
    monkeypatch.setattr(patched_settings, "RAMCLOUDS_HYDE_MODEL", "gemini-hyde-test")

    from services import query_enhancer as qe_mod

    # Force a fresh instance under patched settings.
    monkeypatch.setattr(qe_mod, "settings", patched_settings)
    qe_mod._query_enhancer_instance = None

    inst = qe_mod.get_query_enhancer()
    assert inst._hyde_client._force_model == "gemini-hyde-test", (
        f"HYDE used wrong model: {inst._hyde_client._force_model!r}"
    )


def test_query_enhancer_is_lazy_and_cached():
    """get_query_enhancer() must construct on first call and return the same
    object on subsequent calls.
    """
    from services import query_enhancer as qe_mod

    qe_mod._query_enhancer_instance = None
    first = qe_mod.get_query_enhancer()
    second = qe_mod.get_query_enhancer()
    assert first is second


def test_backward_compat_module_attr_resolves_methods():
    """The legacy `query_enhancer` module symbol must still proxy methods."""
    from services.query_enhancer import query_enhancer

    # Calling a method triggers lazy construction; must not raise AttributeError.
    assert callable(query_enhancer.classify_query_type)
    assert query_enhancer.classify_query_type("điều kiện tuyển sinh") == "admission_criteria"
