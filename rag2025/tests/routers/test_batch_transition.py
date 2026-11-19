"""U8: POST /admin/records/batch-transition — four-eyes, skipped, rejected."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from config.settings import RAGSettings
from rag2025.src.app.lifecycle import configure_canonical_record_store

MAIN_PATH = ROOT / "src" / "main.py"


def _make_app(tmp_path, monkeypatch):
    """Build the real app over a throwaway store root."""
    settings = RAGSettings()
    app_module = __import__("src.main", fromlist=["app"])
    store = configure_canonical_record_store(settings, tmp_path / "store")
    app_module.app.state.record_store = store
    app_module.app.state.lifecycle_service = None
    return app_module.app


def _post_batch(client, payload):
    return client.post("/admin/records/batch-transition", json=payload)


# ------------------------------------------------------------------
# RBAC — route requires RECORDS_READ
# ------------------------------------------------------------------
def test_batch_transition_requires_records_read(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    # No admin token -> 401
    r = _post_batch(client, {"items": []})
    assert r.status_code in (401, 403)


# ------------------------------------------------------------------
# Empty batch is a no-op
# ------------------------------------------------------------------
def test_empty_batch_returns_empty_lists(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app)
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
    client = TestClient(app)
    r = _post_batch(client, {"items": [{"record_id": "r1", "action": "nope"}]})
    assert r.status_code == 200
    rej = r.json()["rejected"]
    assert len(rej) == 1
    assert rej[0]["code"] == "FORBIDDEN_ACTION"
