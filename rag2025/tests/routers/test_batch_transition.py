"""U8: POST /admin/records/batch-transition — four-eyes, skipped, rejected."""

from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from config.settings import RAGSettings
from src.app.lifecycle import configure_canonical_record_store

MAIN_PATH = ROOT / "src" / "main.py"


ADMIN_TOKEN = "batch-test-admin-token"
ADMIN_ACTOR = "batch-test-admin"


def _make_app(tmp_path, monkeypatch):
    """Build the real app over a throwaway store root, with one admin identity."""
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps([{"token": ADMIN_TOKEN, "actor_id": ADMIN_ACTOR, "role": "admin"}]),
    )
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    monkeypatch.setenv("ADMIN_FOUR_EYES", "0")
    app_module = __import__("src.main", fromlist=["app"])
    configure_canonical_record_store(
        app_module.app,
        settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "store"),
    )
    return app_module.app


def _post_batch(client, payload):
    return client.post("/admin/records/batch-transition", json=payload)


# ------------------------------------------------------------------
# RBAC — route requires RECORDS_READ
# ------------------------------------------------------------------
def test_batch_transition_requires_records_read(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app, headers={"X-Admin-Token": "not-a-registered-credential"})
    # Unregistered credential -> 401/403
    r = _post_batch(client, {"items": []})
    assert r.status_code in (401, 403)


# ------------------------------------------------------------------
# Empty batch is a no-op
# ------------------------------------------------------------------
def test_empty_batch_returns_empty_lists(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app, headers={"X-Admin-Token": ADMIN_TOKEN})
    r = _post_batch(client, {"items": []})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] == []
    assert body["rejected"] == []
    assert body["skipped"] == []


# ------------------------------------------------------------------
# Unknown action -> FORBIDDEN_ACTION
# ------------------------------------------------------------------
def test_unknown_action_rejected(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app, headers={"X-Admin-Token": ADMIN_TOKEN})
    r = _post_batch(client, {"items": [{"record_id": "r1", "action": "nope"}]})
    assert r.status_code == 200
    rej = r.json()["rejected"]
    assert len(rej) == 1
    assert rej[0]["code"] == "FORBIDDEN_ACTION"
