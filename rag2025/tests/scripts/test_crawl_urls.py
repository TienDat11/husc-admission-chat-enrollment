"""Tests for crawl_urls (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "crawl_urls.py"


@pytest.fixture
def crawl_module():
    spec = importlib.util.spec_from_file_location("crawl_urls", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mock_session(status: int = 200, content: bytes = b"<html>OK</html>"):
    sess = MagicMock()
    resp = MagicMock(status_code=status, content=content)
    sess.get.return_value = resp
    return sess


def test_fetch_url_returns_bytes_on_200(crawl_module):
    sess = _mock_session(200, b"<html>hi</html>")
    body = crawl_module.fetch_url(74, session=sess, max_attempts=1)
    assert body == b"<html>hi</html>"
    sess.get.assert_called_once()


def test_fetch_url_retries_on_5xx_then_fails(crawl_module):
    sess = MagicMock()
    sess.get.return_value = MagicMock(status_code=500, content=b"")
    with patch.object(crawl_module.time, "sleep", return_value=None):
        with pytest.raises(RuntimeError, match="Failed to fetch id=74"):
            crawl_module.fetch_url(74, session=sess, max_attempts=3)
    assert sess.get.call_count == 3


def test_fetch_url_succeeds_after_transient_5xx(crawl_module):
    sess = MagicMock()
    sess.get.side_effect = [
        MagicMock(status_code=503, content=b""),
        MagicMock(status_code=200, content=b"<html>OK</html>"),
    ]
    with patch.object(crawl_module.time, "sleep", return_value=None):
        body = crawl_module.fetch_url(74, session=sess, max_attempts=3)
    assert body == b"<html>OK</html>"
    assert sess.get.call_count == 2


def test_fetch_url_does_not_retry_on_404(crawl_module):
    sess = MagicMock()
    sess.get.return_value = MagicMock(status_code=404, content=b"")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        crawl_module.fetch_url(99999, session=sess, max_attempts=3)
    assert sess.get.call_count == 1


def test_crawl_one_writes_html_file(crawl_module, tmp_path):
    sess = _mock_session(200, b"<html>id74</html>")
    res = crawl_module.crawl_one(74, tmp_path, session=sess)
    assert res["status"] in {"fetched", "replaced"}
    assert (tmp_path / "74.html").read_bytes() == b"<html>id74</html>"


def test_crawl_one_idempotent_skip_when_hash_matches(crawl_module, tmp_path):
    sess = _mock_session(200, b"<html>id74</html>")
    crawl_module.crawl_one(74, tmp_path, session=sess)
    # Second call with identical body → status "unchanged"
    res2 = crawl_module.crawl_one(74, tmp_path, session=sess)
    assert res2["status"] == "unchanged"


def test_crawl_one_replaces_when_hash_differs(crawl_module, tmp_path):
    sess1 = _mock_session(200, b"<html>v1</html>")
    crawl_module.crawl_one(74, tmp_path, session=sess1)
    sess2 = _mock_session(200, b"<html>v2</html>")
    res = crawl_module.crawl_one(74, tmp_path, session=sess2)
    assert res["status"] == "replaced"
    assert (tmp_path / "74.html").read_bytes() == b"<html>v2</html>"


def test_crawl_one_force_overwrites_even_on_match(crawl_module, tmp_path):
    sess = _mock_session(200, b"<html>same</html>")
    crawl_module.crawl_one(74, tmp_path, session=sess)
    res = crawl_module.crawl_one(74, tmp_path, session=sess, force=True)
    assert res["status"] == "fetched"


def test_crawl_many_continues_on_failure(crawl_module, tmp_path):
    sess = MagicMock()

    def get(url, **_):
        nid = int(url.split("=")[-1])
        if nid == 99999:
            return MagicMock(status_code=404, content=b"")
        return MagicMock(status_code=200, content=f"<html>{nid}</html>".encode())

    sess.get.side_effect = get
    results = crawl_module.crawl_many([74, 99999, 75], tmp_path, session=sess)
    assert len(results) == 3
    statuses = {r["id"]: r["status"] for r in results}
    assert statuses[74] in {"fetched", "replaced"}
    assert statuses[99999] == "failed"
    assert statuses[75] in {"fetched", "replaced"}


def test_parse_ids_default(crawl_module):
    ids = crawl_module._parse_ids("")
    assert 59 in ids
    assert all(63 <= n <= 74 for n in ids if n != 59)
    assert len(ids) == 13


def test_parse_ids_explicit_list(crawl_module):
    assert crawl_module._parse_ids("63,74,59") == [63, 74, 59]


def test_backoff_delay_capped(crawl_module):
    assert crawl_module._backoff_delay(0) == 1.0
    assert crawl_module._backoff_delay(1) == 2.0
    assert crawl_module._backoff_delay(10) == crawl_module.BACKOFF_CAP_S
