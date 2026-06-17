"""Pytest collection-time path fixes for rag2025 test suite.

Adds ``rag2025/src`` and the rag2025 package root to ``sys.path`` so that
the two import conventions used by the test suite resolve automatically
without requiring a manual ``PYTHONPATH`` environment variable.

- ``from services.X`` resolves when ``src`` is on path.
- ``from src.X`` resolves when ``src`` is on path.
- ``from rag2025.src.X`` and ``from rag2025.config.X`` resolve when the
  repo root is on path (package-root import style).
"""

import sys
from pathlib import Path

_RAG2025_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _RAG2025_ROOT.parent

sys.path.insert(0, str(_RAG2025_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))
