"""tests/services/test_chunk_dedup.py

Regression test for Fix #3 — dedup _chunk_answer_for_stream.

There must be ONE implementation of the SSE word-boundary chunker,
imported as an alias in main.py. A divergence here means the two
implementations could split on different boundaries and the streaming
deltas would diverge from the assembled `done` answer.
"""
from __future__ import annotations

import pytest


def test_chunk_answer_alias_is_chunk_text():
    """main._chunk_answer_for_stream must be the SAME object as
    services.llm_generator._chunk_text_for_stream (one implementation)."""
    from src.main import _chunk_answer_for_stream
    from src.services.llm_generator import _chunk_text_for_stream

    assert _chunk_answer_for_stream is _chunk_text_for_stream


def test_chunk_function_no_longer_defined_in_main():
    """The duplicate definition must be gone from main.py."""
    import inspect
    import src.main as main_mod

    src = inspect.getsource(main_mod)
    assert "def _chunk_answer_for_stream" not in src
    assert "def _chunk_text_for_stream" not in src


def test_chunk_function_still_callable():
    """The alias must produce sensible chunks for a Vietnamese sentence."""
    from src.main import _chunk_answer_for_stream

    out = _chunk_answer_for_stream("Hello world this is a test")
    assert isinstance(out, list)
    assert len(out) >= 1
    assert "".join(out) == "Hello world this is a test"