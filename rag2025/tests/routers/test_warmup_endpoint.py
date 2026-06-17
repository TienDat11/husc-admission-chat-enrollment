"""Tests for GET /warmup — fire-and-forget endpoint to load the embedding
model BEFORE the user's first /v2/query, so the first query does not pay
the ~15s cold-load latency.

Two contracts pinned:
  1. When embedding_service is None → 503 (services not ready yet).
  2. When embedding_service is set → 200 + {"warmed": true, "ms": int} AND
     encode_query was actually called (forces _ensure_loaded).

Approach: import the REAL src/main.py via importlib so the route is
registered (same pattern as test_app_cors_and_meta_mount.py and
test_v2_uses_generate_answer.py). We then MOCK the module-level
`embedding_service` global with a fake that records `encode_query` calls
— the real Qwen3 model is never loaded.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_main(monkeypatch) -> object:
    """Import src/main.py as a standalone module without booting heavy
    services (LanceDB / generator). We never enter the TestClient
    context manager, so the startup event does not run.
    """
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("RAMCLOUDS_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    spec = importlib.util.spec_from_file_location("warmup_endpoint_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_embedding_service() -> SimpleNamespace:
    """Fake EmbeddingService — encode_query returns a dummy vector. The
    fact that `encode_query` was called is what we assert on; we do NOT
    want to load the real Qwen3 model (~15s, ~2GB RAM) inside unit tests.
    """
    fake = SimpleNamespace()
    fake.encode_query = lambda text: [0.0] * 4
    fake.encode_query_called_with: list[str] = []
    original = fake.encode_query

    def _record(text: str):
        fake.encode_query_called_with.append(text)
        return original(text)

    fake.encode_query = _record
    return fake


def test_warmup_returns_503_when_embedding_service_is_none(monkeypatch) -> None:
    """With embedding_service=None, GET /warmup must return 503 so the
    caller (FE) learns the BE isn't ready yet, rather than hanging.
    """
    main = _load_main(monkeypatch)
    main.embedding_service = None  # force the not-initialized path

    client = TestClient(main.app)
    res = client.get("/warmup")
    assert res.status_code == 503, res.text
    body = res.json()
    assert "detail" in body


def test_warmup_returns_200_and_calls_encode_query(monkeypatch) -> None:
    """With embedding_service set, GET /warmup returns 200 + warmed=True
    and forces the underlying _ensure_loaded() path via encode_query().
    """
    main = _load_main(monkeypatch)
    fake = _fake_embedding_service()
    main.embedding_service = fake

    client = TestClient(main.app)
    res = client.get("/warmup")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body.get("warmed") is True
    assert isinstance(body.get("ms"), int)
    assert body["ms"] >= 0
    # encode_query was actually called — that's what forces the lazy load.
    assert len(fake.encode_query_called_with) == 1
    assert fake.encode_query_called_with[0] == "warmup"


def test_warmup_is_idempotent(monkeypatch) -> None:
    """Second call to /warmup is a no-op (model already loaded). Both
    calls return 200 — the double-checked lock inside _ensure_loaded
    makes the second call cheap. We don't assert timing; we assert the
    route is safe to call repeatedly.
    """
    main = _load_main(monkeypatch)
    fake = _fake_embedding_service()
    main.embedding_service = fake

    client = TestClient(main.app)
    first = client.get("/warmup")
    second = client.get("/warmup")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["warmed"] is True
    assert second.json()["warmed"] is True
    # Both calls invoked encode_query (cheap once model is loaded).
    assert len(fake.encode_query_called_with) == 2
