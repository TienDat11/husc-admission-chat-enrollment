"""Tests for _metadata_helpers (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
from pathlib import Path
import sys

import pytest

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from services._metadata_helpers import (  # noqa: E402
    get_source_url,
    get_notification_id,
    get_legacy_source,
    get_source_label,
    get_source_breadcrumb,
    _FALLBACK_LABEL,
)


# get_source_url

def test_source_url_from_metadata():
    assert get_source_url({"metadata": {"source_url": "https://x"}}) == "https://x"


def test_source_url_from_top_level():
    assert get_source_url({"source_url": "https://y"}) == "https://y"


def test_source_url_missing_returns_none():
    assert get_source_url({"metadata": {}}) is None


def test_source_url_empty_string_returns_none():
    assert get_source_url({"metadata": {"source_url": ""}}) is None


# get_notification_id

def test_notification_id_int():
    assert get_notification_id({"metadata": {"notification_id": 74}}) == 74


def test_notification_id_string_coerced():
    assert get_notification_id({"metadata": {"notification_id": "59"}}) == 59


def test_notification_id_missing_returns_none():
    assert get_notification_id({"metadata": {}}) is None


def test_notification_id_invalid_returns_none():
    assert get_notification_id({"metadata": {"notification_id": "abc"}}) is None


# get_legacy_source

def test_legacy_source_from_metadata():
    assert get_legacy_source({"metadata": {"source": "tuyensinh.husc.edu.vn"}}) == "tuyensinh.husc.edu.vn"


def test_legacy_source_from_top_level():
    assert get_legacy_source({"source": "TB59"}) == "TB59"


def test_legacy_source_missing_returns_none():
    assert get_legacy_source({"metadata": {}}) is None


# get_source_label — full fallback chain

def test_label_prefers_source_url():
    chunk = {
        "metadata": {
            "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74",
            "notification_id": 74,
            "source": "TB74-legacy",
        }
    }
    assert get_source_label(chunk) == "https://tuyensinh.husc.edu.vn/thongbao.php?id=74"


def test_label_falls_back_to_notification_id():
    chunk = {"metadata": {"notification_id": 59, "source": "legacy"}}
    assert get_source_label(chunk) == "TB59"


def test_label_falls_back_to_legacy_source():
    chunk = {"metadata": {"source": "tuyensinh.husc.edu.vn"}}
    assert get_source_label(chunk) == "tuyensinh.husc.edu.vn"


def test_label_returns_default_when_all_missing():
    assert get_source_label({}) == _FALLBACK_LABEL
    assert get_source_label({"metadata": {}}) == _FALLBACK_LABEL


# get_source_breadcrumb — alias for label

def test_breadcrumb_matches_label():
    chunk = {"metadata": {"notification_id": 74}}
    assert get_source_breadcrumb(chunk) == get_source_label(chunk)
