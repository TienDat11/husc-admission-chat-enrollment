# @spec(G1) test_query_router_contact_meta — hotline is data-driven
"""G1-T5: the hardcoded hotline "0234.3823290" in CONTACT_BLOCK must be
data-driven. Resolution order:
  1. HUSC_HOTLINE env var (operator override)
  2. `meta` dict's `contact_hotline` key (e.g. /api/meta payload)
  3. _HOTLINE_FALLBACK (the previous literal — preserves behavior when
     data-driven paths return nothing)

This test pins all three paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


def _build(meta=None):
    """Helper: import build_contact_block and render with optional meta."""
    from services.query_router import build_contact_block
    return build_contact_block(meta=meta)


class TestContactHotlineDataDriven:
    """The hotline in CONTACT_BLOCK must come from data, not a literal."""

    def test_fallback_is_pre_fix_literal(self):
        """No env, no meta → fallback to the previously-hardcoded
        literal (preserves prod behavior when nothing is supplied)."""
        from services.query_router import _HOTLINE_FALLBACK
        assert _HOTLINE_FALLBACK == "0234.3823290"
        # And the rendered block must contain the fallback.
        block = _build(meta=None)
        assert "0234.3823290" in block, (
            f"Fallback hotline missing from block: {block!r}"
        )

    def test_env_override_wins(self, monkeypatch):
        """HUSC_HOTLINE env var (operator override) wins over fallback."""
        monkeypatch.setenv("HUSC_HOTLINE", "0234.9999999")
        block = _build(meta=None)
        assert "0234.9999999" in block, (
            f"Env override not reflected: {block!r}"
        )

    def test_meta_dict_changes_hotline(self, monkeypatch):
        """A meta dict with a different `contact_hotline` must be
        reflected in the rendered CONTACT_BLOCK — this is the G1-T5
        data-driven path. (When the /api/meta payload supplies a new
        number, the answer text changes too.)"""
        # Wipe env override so meta wins by precedence (env > meta in
        # current design, so unset env).
        monkeypatch.delenv("HUSC_HOTLINE", raising=False)
        block = _build(meta={"contact_hotline": "0234.5550000"})
        assert "0234.5550000" in block, (
            f"Meta change not reflected: {block!r}"
        )
        # And the fallback is no longer in the rendered block.
        from services.query_router import _HOTLINE_FALLBACK
        assert _HOTLINE_FALLBACK not in block, (
            "Old fallback still present after meta override"
        )

    def test_env_wins_over_meta(self, monkeypatch):
        """HUSC_HOTLINE env > meta > fallback (operator override is
        authoritative for staged deploys)."""
        monkeypatch.setenv("HUSC_HOTLINE", "0234.0000001")
        block = _build(meta={"contact_hotline": "0234.5550000"})
        assert "0234.0000001" in block
        assert "0234.5550000" not in block

    def test_meta_empty_string_falls_through(self, monkeypatch):
        """If meta supplies an empty/whitespace hotline, ignore it and
        fall through to the fallback."""
        monkeypatch.delenv("HUSC_HOTLINE", raising=False)
        block = _build(meta={"contact_hotline": "   "})
        assert "0234.3823290" in block

    def test_meta_non_string_falls_through(self, monkeypatch):
        """Defensive: meta must contain a string hotline; non-string
        types are ignored."""
        monkeypatch.delenv("HUSC_HOTLINE", raising=False)
        block = _build(meta={"contact_hotline": 12345})
        assert "0234.3823290" in block
