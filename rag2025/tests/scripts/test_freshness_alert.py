"""Tests for freshness_alert (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "freshness_alert.py"


@pytest.fixture
def fa():
    spec = importlib.util.spec_from_file_location("freshness_alert", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_audit(tmp_path: Path, days_ago: int) -> Path:
    iso = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"audit_date": iso, "total_rows": 1000}), encoding="utf-8")
    return path


def test_status_ok_when_lag_below_threshold(fa, tmp_path):
    path = _write_audit(tmp_path, days_ago=10)
    calls: list[tuple] = []
    def notifier(msg, level="info"):
        calls.append((msg, level))
        return True

    res = fa.check_freshness(audit_path=path, threshold_days=90, notifier=notifier)
    assert res["status"] == "ok"
    assert res["lag_days"] is not None and res["lag_days"] <= 11
    assert res["notified"] is False
    assert calls == []


def test_status_stale_when_lag_above_threshold(fa, tmp_path):
    path = _write_audit(tmp_path, days_ago=120)
    calls: list[tuple] = []
    def notifier(msg, level="info"):
        calls.append((msg, level))
        return True

    res = fa.check_freshness(audit_path=path, threshold_days=90, notifier=notifier)
    assert res["status"] == "stale"
    assert res["notified"] is True
    assert len(calls) == 1
    msg, level = calls[0]
    assert level == "warning"
    assert "120" in msg or "stale" in msg.lower()


def test_status_no_data_when_audit_missing(fa, tmp_path):
    missing = tmp_path / "nope.json"
    calls: list[tuple] = []
    def notifier(msg, level="info"):
        calls.append((msg, level))
        return True

    res = fa.check_freshness(audit_path=missing, threshold_days=90, notifier=notifier)
    assert res["status"] == "no_data"
    assert res["notified"] is False
    assert calls == []


def test_status_no_data_when_audit_lacks_date(fa, tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"total_rows": 100}), encoding="utf-8")
    res = fa.check_freshness(audit_path=path, threshold_days=90, notifier=lambda *a, **k: True)
    assert res["status"] == "no_data"
    assert res["notified"] is False


def test_status_no_data_when_audit_invalid_json(fa, tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("{not json", encoding="utf-8")
    res = fa.check_freshness(audit_path=path, threshold_days=90, notifier=lambda *a, **k: True)
    assert res["status"] == "no_data"


def test_notifier_failure_does_not_crash(fa, tmp_path):
    path = _write_audit(tmp_path, days_ago=120)
    def bad_notifier(msg, level="info"):
        raise RuntimeError("slack down")

    res = fa.check_freshness(audit_path=path, threshold_days=90, notifier=bad_notifier)
    assert res["status"] == "stale"
    assert res["notified"] is False  # gracefully degraded


def test_threshold_env_override(fa, tmp_path, monkeypatch):
    path = _write_audit(tmp_path, days_ago=45)
    monkeypatch.setenv("FRESHNESS_ALERT_DAYS", "30")
    res = fa.check_freshness(audit_path=path, notifier=lambda *a, **k: True)
    # 45 > 30 → stale.
    assert res["status"] == "stale"


def test_compute_lag_handles_z_suffix(fa):
    iso = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    lag = fa._compute_lag_days(iso)
    assert lag is not None and 2 <= lag <= 4


def test_compute_lag_handles_invalid_string(fa):
    assert fa._compute_lag_days("not-a-date") is None


def test_threshold_zero_means_anything_stale(fa, tmp_path):
    path = _write_audit(tmp_path, days_ago=1)
    calls: list[tuple] = []
    def notifier(msg, level="info"):
        calls.append((msg, level))
        return True

    res = fa.check_freshness(audit_path=path, threshold_days=0, notifier=notifier)
    assert res["status"] == "stale"
    assert res["notified"] is True


def test_default_threshold_when_env_unset(fa, tmp_path, monkeypatch):
    monkeypatch.delenv("FRESHNESS_ALERT_DAYS", raising=False)
    path = _write_audit(tmp_path, days_ago=30)
    res = fa.check_freshness(audit_path=path, notifier=lambda *a, **k: True)
    # 30 < default 90 → ok.
    assert res["status"] == "ok"
    assert res["threshold"] == 90
