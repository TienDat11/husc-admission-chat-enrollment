"""CORS / ENV guard: which ENV values start, and which origins they accept.

ENV is an allow-list. dev-like values get localhost defaults; prod/staging must
name their origins; anything else (typos included) refuses to start. ``*`` is
never accepted because credentials are enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.cors import resolve_cors_settings  # noqa: E402


@pytest.mark.parametrize("env", ["", "dev", "development", "local", "test", "  DEV  "])
def test_dev_like_env_gets_localhost_defaults(env):
    origins, regex = resolve_cors_settings(env, "")
    assert "http://localhost:5173" in origins
    assert regex is not None


@pytest.mark.parametrize("env", ["prod", "production", "staging", "PRODUCTION"])
def test_prod_like_env_without_origins_refuses_to_start(env):
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        resolve_cors_settings(env, "")


@pytest.mark.parametrize("env", ["prod", "staging"])
def test_prod_like_env_with_origins_uses_them_and_drops_regex(env):
    origins, regex = resolve_cors_settings(env, "https://a.example, https://b.example")
    assert origins == ["https://a.example", "https://b.example"]
    assert regex is None


@pytest.mark.parametrize("env", ["prdo", "producton", "stage", "qa"])
def test_unknown_env_refuses_to_start(env):
    with pytest.raises(RuntimeError, match="ENV"):
        resolve_cors_settings(env, "https://a.example")


@pytest.mark.parametrize("env", ["dev", "production"])
def test_wildcard_origin_is_refused_in_every_env(env):
    with pytest.raises(RuntimeError, match="\\*"):
        resolve_cors_settings(env, "https://a.example,*")


def test_unset_env_is_logged_as_dev_fallback(caplog) -> None:
    # An unset ENV is treated as dev; that must be visible in logs so a deploy that forgot it is noticed.
    import logging

    from src.api.cors import resolve_cors_settings

    with caplog.at_level(logging.WARNING, logger="src.api.cors"):
        resolve_cors_settings("", "")
    assert any("ENV is unset" in r.getMessage() for r in caplog.records)
