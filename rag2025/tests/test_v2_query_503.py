"""TDD: POST /v2/query returns 503 + Retry-After header when the unified
pipeline has not finished warming.

Contract pinned by these tests:
  1. Response status == 503.
  2. Response includes `Retry-After` header (RFC 7231, integer seconds).
  3. Response body has `retry_after_ms` (parseable by FE for sub-second backoff).
  4. When the pipeline IS set, the 503 gate does NOT fire (sanity).

Approach: same importlib trick as the other main.py-touching tests —
we never enter the TestClient context, so the lifespan event does not
run; the unified_pipeline global stays None and the gate fires.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
MAIN_PATH = SRC / "main.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
assert MAIN_PATH.is_file(), MAIN_PATH


def _load_main(monkeypatch) -> object:
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    spec = importlib.util.spec_from_file_location("v2_query_503_test", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_query_returns_503_when_pipeline_is_none(monkeypatch) -> None:
    """Unified pipeline not warmed → 503 with the documented body shape."""
    main = _load_main(monkeypatch)
    main.unified_pipeline = None

    client = TestClient(main.app)
    res = client.post("/v2/query", json={"query": "test"})

    assert res.status_code == 503, res.text


def test_v2_query_503_includes_retry_after_header(monkeypatch) -> None:
    """The 503 response MUST include a Retry-After header so naive HTTP
    clients (browsers, fetch wrappers) back off without polling."""
    main = _load_main(monkeypatch)
    main.unified_pipeline = None

    client = TestClient(main.app)
    res = client.post("/v2/query", json={"query": "test"})
    assert res.status_code == 503

    header_keys = {k.lower() for k in res.headers.keys()}
    assert "retry-after" in header_keys, dict(res.headers)
    retry_after = int(res.headers.get("retry-after", "0"))
    assert retry_after >= 1, f"retry-after={retry_after!r} must be ≥ 1s"


def test_v2_query_503_body_has_retry_after_ms(monkeypatch) -> None:
    """The body must carry `retry_after_ms` so the FE retry wrapper can
    use sub-second backoff if needed."""
    main = _load_main(monkeypatch)
    main.unified_pipeline = None

    client = TestClient(main.app)
    res = client.post("/v2/query", json={"query": "test"})
    assert res.status_code == 503

    body = res.json()
    assert "detail" in body, body
    assert "retry_after_ms" in body, body
    assert isinstance(body["retry_after_ms"], int)
    assert body["retry_after_ms"] >= 1


def test_v2_query_does_not_503_when_pipeline_is_present(monkeypatch) -> None:
    """Sanity: when the pipeline is set, the warming gate must NOT fire."""
    main = _load_main(monkeypatch)
    fake = SimpleNamespace()
    fake._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 1})
    main.unified_pipeline = fake

    client = TestClient(main.app)
    res = client.post("/v2/query", json={"query": "test"})
    assert res.status_code != 503, f"unexpected 503 — gate fired despite pipeline set: {res.text}"