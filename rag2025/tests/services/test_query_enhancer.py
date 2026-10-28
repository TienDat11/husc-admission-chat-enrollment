"""TDD: query_enhancer must be settings-driven and lazily constructed.

BUG-3 HIGH: NS24_HYDE_MODEL must be read from config.settings, not
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

# Ensure an ns24 provider exists for HYDEQueryEnhancer init.
os.environ.setdefault("NS24_API_KEY", "sk-test-fake")


def test_hyde_role_model_pair_from_ns24_contract(monkeypatch):
    """The hyde role's primary/fallback come from the ns24 role contract, and the
    built HYDE client uses the primary model of that pair."""
    from config.ns24_models import ROLE_MODELS
    from services import query_enhancer as qe_mod

    qe_mod._query_enhancer_instance = None
    inst = qe_mod.get_query_enhancer()
    primary, _fallback = ROLE_MODELS["hyde"]
    models = [p.model for p in inst._hyde_client._providers]
    assert models[0] == primary


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
