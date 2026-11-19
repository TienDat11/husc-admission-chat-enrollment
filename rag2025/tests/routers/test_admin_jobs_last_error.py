"""Unit tests for last_error redaction in the admin jobs surface (no DB)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.routers.admin_jobs import _redact_last_error  # noqa: E402


def test_none_returns_none():
    assert _redact_last_error(None) is None


def test_non_dict_returns_none():
    assert _redact_last_error("boom") is None
    assert _redact_last_error(["code", "message"]) is None


def test_keeps_only_code_and_message():
    value = {
        "code": "E_FAIL",
        "message": "failed",
        "stack": "Traceback ...",
        "path": "/srv/secret/file.py",
    }
    assert _redact_last_error(value) == {"code": "E_FAIL", "message": "failed"}


def test_non_string_code_is_dropped():
    assert _redact_last_error({"code": 42, "message": "failed"}) == {"message": "failed"}


def test_only_stack_returns_none():
    assert _redact_last_error({"stack": "Traceback ..."}) is None
