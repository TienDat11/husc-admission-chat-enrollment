"""Tests for /api/meta router (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


SRC = Path(__file__).resolve().parents[2] / "src"
META_PATH = SRC / "routers" / "meta.py"


@pytest.fixture
def meta_module():
    spec = importlib.util.spec_from_file_location("meta_router", META_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_app(meta_module, *, retriever=None, audit_payload: dict | None = None, audit_path: Path | None = None):
    app = FastAPI()
    app.state.retriever = retriever
    if audit_payload is not None and audit_path is not None:
        audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False), encoding="utf-8")
    app.include_router(meta_module.router)
    return app


def test_returns_200_with_required_keys(meta_module, monkeypatch):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", "/no/such/path")
    app = _build_app(meta_module)
    client = TestClient(app)
    res = client.get("/api/meta")
    assert res.status_code == 200
    body = res.json()
    for k in ("current_admission_year", "latest_crawl_date", "total_notifications", "freshness_lag_days", "freshness_alert"):
        assert k in body


def test_current_admission_year_reads_env(meta_module, monkeypatch, tmp_path):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2027")
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(tmp_path / "no.json"))
    app = _build_app(meta_module)
    client = TestClient(app)
    res = client.get("/api/meta")
    assert res.json()["current_admission_year"] == 2027


def test_default_year_when_env_unset(meta_module, monkeypatch, tmp_path):
    monkeypatch.delenv("CURRENT_ADMISSION_YEAR", raising=False)
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(tmp_path / "no.json"))
    app = _build_app(meta_module)
    client = TestClient(app)
    res = client.get("/api/meta")
    assert res.json()["current_admission_year"] == 2026


def test_audit_fallback_populates_freshness(meta_module, monkeypatch, tmp_path):
    """When app.state.retriever is None but audit JSON exists, freshness comes from audit."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    audit_iso = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    audit_path = tmp_path / "audit.json"
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(audit_path))
    app = _build_app(
        meta_module,
        audit_payload={"audit_date": audit_iso, "total_rows": 3500},
        audit_path=audit_path,
    )
    client = TestClient(app)
    res = client.get("/api/meta")
    body = res.json()
    assert body["latest_crawl_date"] is not None
    assert body["total_notifications"] == 3500
    assert body["freshness_lag_days"] is not None
    assert 9 <= body["freshness_lag_days"] <= 11  # allow ±1 day for clock skew
    assert body["freshness_alert"] is False


def test_freshness_alert_fires_when_lag_exceeds_threshold(meta_module, monkeypatch, tmp_path):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    audit_iso = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    audit_path = tmp_path / "audit.json"
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(audit_path))
    app = _build_app(
        meta_module,
        audit_payload={"audit_date": audit_iso, "total_rows": 100},
        audit_path=audit_path,
    )
    client = TestClient(app)
    res = client.get("/api/meta")
    body = res.json()
    assert body["freshness_alert"] is True
    assert body["freshness_lag_days"] is not None
    assert body["freshness_lag_days"] > 90


def test_no_state_no_audit_returns_year_only(meta_module, monkeypatch, tmp_path):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(tmp_path / "missing.json"))
    app = _build_app(meta_module)
    client = TestClient(app)
    res = client.get("/api/meta")
    body = res.json()
    assert body["current_admission_year"] == 2026
    assert body["latest_crawl_date"] is None
    assert body["total_notifications"] is None
    assert body["freshness_lag_days"] is None
    assert body["freshness_alert"] is False


def test_retriever_snapshot_takes_precedence_over_audit(meta_module, monkeypatch, tmp_path):
    """Path 1 (retriever.metadata_snapshot) wins over Path 2 (audit JSON)."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    retriever_iso = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    audit_iso = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    audit_path = tmp_path / "audit.json"
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(audit_path))

    retriever = MagicMock()
    retriever.metadata_snapshot = MagicMock(
        return_value={"latest_crawl_date": retriever_iso, "total_notifications": 4000}
    )
    app = _build_app(
        meta_module,
        retriever=retriever,
        audit_payload={"audit_date": audit_iso, "total_rows": 100},
        audit_path=audit_path,
    )
    client = TestClient(app)
    res = client.get("/api/meta")
    body = res.json()
    assert body["total_notifications"] == 4000
    # Lag must reflect the retriever date (~5 days), not the audit date (200 days).
    assert body["freshness_lag_days"] is not None
    assert 4 <= body["freshness_lag_days"] <= 6
    assert body["freshness_alert"] is False


def test_retriever_snapshot_exception_does_not_crash(meta_module, monkeypatch, tmp_path):
    """Public endpoint must not 500 when retriever throws inside snapshot()."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(tmp_path / "missing.json"))
    retriever = MagicMock()
    retriever.metadata_snapshot = MagicMock(side_effect=RuntimeError("retriever down"))
    app = _build_app(meta_module, retriever=retriever)
    client = TestClient(app)
    res = client.get("/api/meta")
    assert res.status_code == 200
    body = res.json()
    assert body["latest_crawl_date"] is None


def test_compute_freshness_lag_handles_z_suffix(meta_module):
    iso = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    lag = meta_module._compute_freshness_lag(iso)
    assert lag is not None
    assert 2 <= lag <= 4


def test_compute_freshness_lag_handles_invalid_string(meta_module):
    assert meta_module._compute_freshness_lag("not-a-date") is None
    assert meta_module._compute_freshness_lag(None) is None
    assert meta_module._compute_freshness_lag("") is None


def test_compute_freshness_lag_treats_naive_as_utc(meta_module):
    naive_iso = (datetime.now(timezone.utc) - timedelta(days=2)).replace(tzinfo=None).isoformat()
    lag = meta_module._compute_freshness_lag(naive_iso)
    assert lag is not None
    assert 1 <= lag <= 3


def test_total_notifications_zero_preserved_not_dropped(meta_module, monkeypatch, tmp_path):
    """HIGH-1 regression: retriever returning 0 must NOT be silently replaced with None.

    The previous `or` short-circuit treated 0 as falsy and fell through to None,
    masking an empty index as 'no data available'.
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(tmp_path / "missing.json"))
    retriever = MagicMock()
    retriever.metadata_snapshot = MagicMock(
        return_value={
            "latest_crawl_date": "2026-05-25T00:00:00+00:00",
            "total_notifications": 0,  # ← empty index after a failed reingest
        }
    )
    app = _build_app(meta_module, retriever=retriever)
    client = TestClient(app)
    res = client.get("/api/meta")
    body = res.json()
    assert body["total_notifications"] == 0


def test_audit_relative_path_resolved_to_repo_root(meta_module, monkeypatch):
    """HIGH-2 regression: when AUDIT_FALLBACK_PATH is unset and CWD differs from
    repo root, the default path must still resolve correctly.
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.delenv("AUDIT_FALLBACK_PATH", raising=False)
    # Swap CWD to verify default path is anchored, not relative.
    monkeypatch.chdir("/")
    app = _build_app(meta_module)
    client = TestClient(app)
    # Must NOT 500 even though CWD is "/" — endpoint resolves the path safely.
    res = client.get("/api/meta")
    assert res.status_code == 200


def test_custom_freshness_threshold_via_env(meta_module, monkeypatch, tmp_path):
    """MED-3 regression: FRESHNESS_ALERT_DAYS env overrides the default 90."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.setenv("FRESHNESS_ALERT_DAYS", "30")
    audit_iso = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    audit_path = tmp_path / "audit.json"
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(audit_path))
    app = _build_app(
        meta_module,
        audit_payload={"audit_date": audit_iso, "total_rows": 100},
        audit_path=audit_path,
    )
    client = TestClient(app)
    res = client.get("/api/meta")
    body = res.json()
    # 45 days lag > 30 day threshold → alert fires.
    assert body["freshness_alert"] is True


def test_malformed_current_admission_year_env_falls_back_to_2026(meta_module, monkeypatch, tmp_path):
    """LOW-10 regression: CURRENT_ADMISSION_YEAR='2026a' must NOT 500."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026a")
    monkeypatch.setenv("AUDIT_FALLBACK_PATH", str(tmp_path / "missing.json"))
    app = _build_app(meta_module)
    client = TestClient(app)
    res = client.get("/api/meta")
    assert res.status_code == 200
    assert res.json()["current_admission_year"] == 2026
