# @spec(DEMO-FIX) regression: /api/meta is mounted on the real app + CORS
# allows the Vite dev origin on ANY localhost port.
"""These two demo-blockers were live on 2026-06-08:

  1. /api/meta returned 404 — the meta router was never `include_router`'d
     into the real app, so the FE YearBanner silently showed its error state.
  2. CORS preflight (OPTIONS) from the Vite dev server returned 400 because
     the FE runs on port 8080 (and auto-increments to 8081+ when busy), but
     the BE allowlist only had 3000/5173. Every browser fetch failed.

This test loads the REAL src/main.py (the same importlib pattern the other
router contract tests use) and asserts both are fixed, so a future edit that
drops the router mount or tightens CORS back to a fixed port fails loudly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_main(monkeypatch) -> object:
    """Import src/main.py as a standalone module without booting heavy
    services. We never trigger the startup event (TestClient is used only
    for routing + middleware, and we don't enter its context manager), so
    LanceDB / embeddings / generator are not required.
    """
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("RAMCLOUDS_API_KEY", "")
    # Ensure the default localhost allowlist + regex path is exercised
    # (no explicit ALLOWED_ORIGINS override).
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    spec = importlib.util.spec_from_file_location("app_cors_meta_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_meta_route_is_mounted(monkeypatch) -> None:
    """GET /api/meta must exist on the real app (not 404) and return the
    banner contract with current_admission_year."""
    main = _load_main(monkeypatch)
    client = TestClient(main.app)
    res = client.get("/api/meta")
    assert res.status_code == 200, res.text
    body = res.json()
    assert "current_admission_year" in body
    assert isinstance(body["current_admission_year"], int)
    assert "freshness_alert" in body


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8080",   # Vite default
        "http://localhost:8081",   # Vite fallback when 8080 busy
        "http://localhost:5173",   # explicit allowlist entry
        "http://127.0.0.1:8080",
    ],
)
def test_cors_preflight_allows_localhost_any_port(monkeypatch, origin: str) -> None:
    """An OPTIONS preflight from any localhost origin must be allowed so the
    Vite dev server reaches the BE regardless of which port it lands on."""
    main = _load_main(monkeypatch)
    client = TestClient(main.app)
    res = client.options(
        "/v2/query",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res.status_code == 200, f"{origin} preflight got {res.status_code}: {res.text}"
    assert res.headers.get("access-control-allow-origin") == origin


def test_cors_rejects_foreign_origin(monkeypatch) -> None:
    """A non-localhost origin must NOT be echoed back (regex is localhost-only;
    production should pin ALLOWED_ORIGINS explicitly)."""
    main = _load_main(monkeypatch)
    client = TestClient(main.app)
    res = client.options(
        "/v2/query",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    # Starlette returns 400 for a disallowed preflight; crucially the
    # allow-origin header must not echo the foreign origin.
    assert res.headers.get("access-control-allow-origin") != "https://evil.example.com"
