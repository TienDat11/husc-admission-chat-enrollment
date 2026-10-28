"""Tests for crawl_urls (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "crawl_urls.py"

# Ticket T21 moved the fetch policy (backoff, retry, byte cap) into
# ``src/services/web_fetch`` so the admin URL route and this CLI share one
# implementation; the script re-exports the knobs it still documents.
if str(SCRIPT_PATH.parent.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent))

from src.services import web_fetch  # noqa: E402


@pytest.fixture
def crawl_module():
    spec = importlib.util.spec_from_file_location("crawl_urls", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mock_response(status: int = 200, content: bytes = b"<html>OK</html>") -> MagicMock:
    """A response mock that streams like ``requests`` does.

    ``web_fetch`` reads the body with ``iter_content`` (so the byte cap can stop
    an oversized page before it is held in memory), which a bare ``MagicMock``
    cannot satisfy — it would hand back a mock instead of chunks.
    """
    resp = MagicMock(status_code=status, content=content)
    resp.headers = {"Content-Length": str(len(content))}
    resp.iter_content.return_value = iter([content]) if content else iter([])
    return resp


def _mock_session(status: int = 200, content: bytes = b"<html>OK</html>"):
    """A session mock that answers every call with a *fresh* response.

    Fresh matters: a response's body is consumed once, so a single reused mock
    would hand the second fetch an empty stream and make an unchanged page look
    like a changed one.
    """
    sess = MagicMock()
    sess.get.side_effect = lambda *args, **kwargs: _mock_response(status, content)
    return sess


def test_fetch_url_returns_bytes_on_200(crawl_module):
    sess = _mock_session(200, b"<html>hi</html>")
    body = crawl_module.fetch_url(74, session=sess, max_attempts=1)
    assert body == b"<html>hi</html>"
    sess.get.assert_called_once()


def test_fetch_url_retries_on_5xx_then_fails(crawl_module):
    sess = MagicMock()
    sess.get.return_value = _mock_response(500, b"")
    with patch.object(web_fetch.time, "sleep", return_value=None):
        with pytest.raises(RuntimeError, match="Failed to fetch id=74"):
            crawl_module.fetch_url(74, session=sess, max_attempts=3)
    assert sess.get.call_count == 3


def test_fetch_url_succeeds_after_transient_5xx(crawl_module):
    sess = MagicMock()
    sess.get.side_effect = [
        _mock_response(503, b""),
        _mock_response(200, b"<html>OK</html>"),
    ]
    with patch.object(web_fetch.time, "sleep", return_value=None):
        body = crawl_module.fetch_url(74, session=sess, max_attempts=3)
    assert body == b"<html>OK</html>"
    assert sess.get.call_count == 2


def test_fetch_url_does_not_retry_on_404(crawl_module):
    sess = MagicMock()
    sess.get.return_value = _mock_response(404, b"")
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
    # MED-8 fix: force-overwrite of existing file now returns "replaced" not "fetched"
    assert res["status"] == "replaced"


def test_crawl_many_continues_on_failure(crawl_module, tmp_path):
    sess = MagicMock()

    def get(url, **_):
        nid = int(url.split("=")[-1])
        if nid == 99999:
            return _mock_response(404, b"")
        return _mock_response(200, f"<html>{nid}</html>".encode())

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


def test_backoff_delay_capped():
    """The CLI's backoff is the shared core's, capped at 30s (ticket T21)."""
    assert web_fetch.backoff_delay(0) == 1.0
    assert web_fetch.backoff_delay(1) == 2.0
    assert web_fetch.backoff_delay(10) == web_fetch.BACKOFF_CAP_S
