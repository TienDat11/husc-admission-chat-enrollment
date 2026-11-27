"""main.py hardening: client IP trust, /metrics gating, debug preview auth and cap."""
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from config.constants import CHUNK_PREVIEW_MAX_TEXT_LENGTH
from src import main as main_mod
from src.main import app
from src.utils.http import client_ip


def _req(headers=None, peer=("10.0.0.9", 5555)):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({
        "type": "http", "method": "GET", "path": "/", "headers": raw,
        "client": peer, "query_string": b"",
    })


def test_client_ip_ignores_spoofed_leftmost_entry():
    req = _req({"x-forwarded-for": "6.6.6.6, 203.0.113.7"})
    assert client_ip(req, 1) == "203.0.113.7"


def test_client_ip_uses_peer_without_header():
    assert client_ip(_req(), 1) == "10.0.0.9"


def test_main_client_ip_uses_trusted_proxy_count(monkeypatch):
    monkeypatch.setattr(main_mod, "TRUSTED_PROXY_COUNT", 1)
    assert main_mod._client_ip(_req({"x-forwarded-for": "6.6.6.6, 203.0.113.7"})) == "203.0.113.7"


def test_metrics_denied_for_public_client(monkeypatch):
    monkeypatch.delenv("METRICS_TOKEN", raising=False)
    assert TestClient(app).get("/metrics").status_code == 403


def test_metrics_allowed_with_matching_token(monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "s3cret")
    client = TestClient(app)
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 403


def test_metrics_allowed_for_loopback_only(monkeypatch):
    monkeypatch.delenv("METRICS_TOKEN", raising=False)
    monkeypatch.setattr(main_mod, "TRUSTED_PROXY_COUNT", 1)
    assert main_mod._metrics_access_allowed(_req(peer=("127.0.0.1", 1))) is True
    # Forwarded public client behind a loopback proxy hop is not loopback.
    assert main_mod._metrics_access_allowed(
        _req({"x-forwarded-for": "203.0.113.7"}, peer=("127.0.0.1", 1))
    ) is False


def test_preview_chunks_requires_admin():
    resp = TestClient(app).post("/debug/preview-chunks", json={"text": "x" * 50})
    # This API reports a missing admin credential as 422 VALIDATION_FAILED (see require_action).
    assert resp.status_code in (401, 403, 422)
    assert "credential" in resp.text


def test_preview_chunks_rejects_oversize_text():
    route = next(r for r in app.routes if getattr(r, "path", None) == "/debug/preview-chunks")
    auth_dep = route.dependant.dependencies[0].call
    app.dependency_overrides[auth_dep] = lambda: None
    try:
        resp = TestClient(app).post(
            "/debug/preview-chunks",
            json={"text": "x" * (CHUNK_PREVIEW_MAX_TEXT_LENGTH + 1)},
        )
        # The app's validation handler maps request validation errors to 4xx (400/422).
        assert resp.status_code in (400, 422)
    finally:
        app.dependency_overrides.pop(auth_dep, None)
