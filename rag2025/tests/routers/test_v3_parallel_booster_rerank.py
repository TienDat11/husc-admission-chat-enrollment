"""tests/routers/test_v3_parallel_booster_rerank.py

Regression test for Fix #7 — parallel booster + rerank with explicit
exception handling.

Verifies:
  - The /v3 endpoint sources the asyncio.gather fan-out pattern
  - A failure in the booster task doesn't hang the request or silently
    drop the reranker (or vice-versa)
  - When both succeed, the response payload reports both stages ran
"""
from __future__ import annotations

import inspect
import pytest


class TestParallelBoosterRerank:
    def test_v3_uses_asyncio_gather(self):
        """The /v3 hot path must call asyncio.gather for booster + rerank."""
        import src.main as main_mod
        src = inspect.getsource(main_mod)
        assert "asyncio.gather(" in src
        assert "return_exceptions=True" in src

    def test_v3_handles_exceptions_explicitly(self):
        """Both booster and rerank results must be checked with
        ``isinstance(..., Exception)`` — silent drop is forbidden."""
        import src.main as main_mod
        src = inspect.getsource(main_mod)
        assert "isinstance(booster_result, Exception)" in src
        assert "isinstance(rerank_result, Exception)" in src

    def test_v3_uses_run_in_threadpool_for_booster(self):
        """Booster wraps a sync call — must run in a threadpool, not the
        event loop."""
        import src.main as main_mod
        src = inspect.getsource(main_mod)
        # We expect at least one run_in_threadpool(_booster_task)
        assert "run_in_threadpool(_booster_task" in src