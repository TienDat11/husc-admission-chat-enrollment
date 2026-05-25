"""Tests for slack_notify (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "observability" / "slack_notify.py"


@pytest.fixture
def sn():
    spec = importlib.util.spec_from_file_location("slack_notify", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_returns_false_when_webhook_unset(sn, monkeypatch):
    monkeypatch.delenv("SLACK_RAG_OPS_WEBHOOK", raising=False)
    assert sn.send_slack("hello") is False


def test_posts_to_webhook_with_correct_payload(sn, monkeypatch):
    captured: dict = {}

    def fake_post(self, url, json=None, **_):
        captured["url"] = url
        captured["payload"] = json
        resp = MagicMock(status_code=200, text="ok")
        return resp

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        ok = sn.send_slack("rotation complete year=2027", level="info", webhook_url="https://hooks.slack.com/test")

    assert ok is True
    assert captured["url"] == "https://hooks.slack.com/test"
    payload = captured["payload"]
    assert "attachments" in payload
    att = payload["attachments"][0]
    assert att["color"] == sn._LEVEL_COLORS["info"]
    assert att["text"] == "rotation complete year=2027"


def test_warning_level_uses_orange_color(sn):
    captured: dict = {}

    def fake_post(self, url, json=None, **_):
        captured["payload"] = json
        return MagicMock(status_code=200, text="ok")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        sn.send_slack("stale data", level="warning", webhook_url="https://x")

    assert captured["payload"]["attachments"][0]["color"] == sn._LEVEL_COLORS["warning"]
    assert captured["payload"]["attachments"][0]["color"] == "#FFA500"


def test_error_level_uses_red_color(sn):
    captured: dict = {}

    def fake_post(self, url, json=None, **_):
        captured["payload"] = json
        return MagicMock(status_code=200, text="ok")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        sn.send_slack("rotation FAILED", level="error", webhook_url="https://x")

    assert captured["payload"]["attachments"][0]["color"] == "#FF0000"


def test_returns_false_on_non_2xx(sn):
    def fake_post(self, url, json=None, **_):
        return MagicMock(status_code=500, text="server error")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        assert sn.send_slack("test", webhook_url="https://x") is False


def test_returns_false_on_network_error(sn):
    def fake_post(self, url, json=None, **_):
        raise httpx.ConnectError("connection refused")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        assert sn.send_slack("test", webhook_url="https://x") is False


def test_handles_unexpected_error_gracefully(sn):
    def fake_post(self, url, json=None, **_):
        raise ValueError("malformed payload")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        assert sn.send_slack("test", webhook_url="https://x") is False


def test_uses_env_webhook_when_no_override(sn, monkeypatch):
    captured: dict = {}
    monkeypatch.setenv("SLACK_RAG_OPS_WEBHOOK", "https://hooks.slack.com/env-url")

    def fake_post(self, url, json=None, **_):
        captured["url"] = url
        return MagicMock(status_code=200, text="ok")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        ok = sn.send_slack("via env")

    assert ok is True
    assert captured["url"] == "https://hooks.slack.com/env-url"


def test_payload_contains_level_field(sn):
    captured: dict = {}

    def fake_post(self, url, json=None, **_):
        captured["payload"] = json
        return MagicMock(status_code=200, text="ok")

    with patch.object(sn.httpx.Client, "post", new=fake_post):
        sn.send_slack("hi", level="warning", webhook_url="https://x")

    fields = captured["payload"]["attachments"][0]["fields"]
    level_field = next((f for f in fields if f["title"] == "level"), None)
    assert level_field is not None
    assert level_field["value"] == "warning"
