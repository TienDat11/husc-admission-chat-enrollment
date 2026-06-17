"""TDD: dead-config LLM_MODEL must track RAMCLOUDS_MODEL.

BUG-2 CRITICAL/dead-config: LLM_MODEL must not be an orphan default that
 diverges from RAMCLOUDS_MODEL. It resolves to RAMCLOUDS_MODEL explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config"
if str(CONFIG_PATH) not in sys.path:
    sys.path.insert(0, str(CONFIG_PATH))


def test_llm_model_tracks_ramclouds_model():
    """LLM_MODEL must resolve to the live RAMCLOUDS_MODEL setting."""
    from config.settings import settings

    assert settings.LLM_MODEL == settings.RAMCLOUDS_MODEL, (
        f"LLM_MODEL ({settings.LLM_MODEL!r}) must track RAMCLOUDS_MODEL "
        f"({settings.RAMCLOUDS_MODEL!r})"
    )
