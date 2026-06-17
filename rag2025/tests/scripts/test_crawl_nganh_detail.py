"""Tests for rag2025/scripts/crawl_nganh_detail.py

Covers:
- URL format is correct for each code
- Idempotent skip when SHA256 matches existing file
- Retry on 5xx, then succeed
- All 28 default codes are attempted
- 404 is treated as skipped_404 (not fatal)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure repo root is on path so the script is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rag2025.scripts.crawl_nganh_detail import (
    BASE_URL,
    DEFAULT_CODES,
    _sha256_bytes,
    _backoff_delay,
    crawl_one,
    crawl_many,
    fetch_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_response(status_code: int, content: bytes = b"<html>ok</html>") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# 1. URL format
# ---------------------------------------------------------------------------

class TestUrlFormat:
    def test_base_url_contains_manganh_placeholder(self):
        assert "{code}" in BASE_URL

    @pytest.mark.parametrize("code", DEFAULT_CODES)
    def test_url_format_per_code(self, code):
        url = BASE_URL.format(code=code)
        assert f"manganh={code}" in url
        assert url.startswith("https://tuyensinh.husc.edu.vn/")

    def test_all_28_codes_present(self):
        assert len(DEFAULT_CODES) == 28


# ---------------------------------------------------------------------------
# 2. Idempotent skip on hash match
# ---------------------------------------------------------------------------

class TestIdempotentSkip:
    def test_skip_when_hash_matches(self, tmp_path):
        content = b"<html>nganh content</html>"
        code = "7460108"
        target = tmp_path / f"nganh_{code}.html"
        target.write_bytes(content)

        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(200, content)

        result = crawl_one(code, tmp_path, client=mock_client)

        assert result["status"] == "unchanged"
        assert result["sha256"] == _sha256(content)
        # File should not be rewritten
        assert target.read_bytes() == content

    def test_replace_when_hash_differs(self, tmp_path):
        old_content = b"<html>old</html>"
        new_content = b"<html>new</html>"
        code = "7480201"
        target = tmp_path / f"nganh_{code}.html"
        target.write_bytes(old_content)

        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(200, new_content)

        result = crawl_one(code, tmp_path, client=mock_client)

        assert result["status"] == "replaced"
        assert target.read_bytes() == new_content

    def test_fetched_when_file_absent(self, tmp_path):
        content = b"<html>fresh</html>"
        code = "7480103"

        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(200, content)

        result = crawl_one(code, tmp_path, client=mock_client)

        assert result["status"] == "fetched"
        assert (tmp_path / f"nganh_{code}.html").read_bytes() == content


# ---------------------------------------------------------------------------
# 3. Retry on 5xx
# ---------------------------------------------------------------------------

class TestRetryOn5xx:
    def test_retries_on_503_then_succeeds(self, tmp_path):
        content = b"<html>ok after retry</html>"
        code = "7510302"

        fail_resp = _make_response(503)
        ok_resp = _make_response(200, content)

        mock_client = MagicMock()
        mock_client.get.side_effect = [fail_resp, ok_resp]

        with patch("rag2025.scripts.crawl_nganh_detail.time.sleep"):
            result = crawl_one(code, tmp_path, client=mock_client)

        assert result["status"] == "fetched"
        assert mock_client.get.call_count == 2

    def test_fails_after_max_attempts_on_5xx(self, tmp_path):
        code = "7440102"
        fail_resp = _make_response(500)

        mock_client = MagicMock()
        mock_client.get.return_value = fail_resp

        with patch("rag2025.scripts.crawl_nganh_detail.time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                fetch_url(code, client=mock_client, max_attempts=3)

        assert mock_client.get.call_count == 3

    def test_no_retry_on_4xx_non_404(self, tmp_path):
        code = "7440112"
        fail_resp = _make_response(403)

        mock_client = MagicMock()
        mock_client.get.return_value = fail_resp

        with pytest.raises(RuntimeError, match="HTTP 403"):
            fetch_url(code, client=mock_client)

        assert mock_client.get.call_count == 1


# ---------------------------------------------------------------------------
# 4. 404 treated as skipped_404 (not fatal)
# ---------------------------------------------------------------------------

class TestNotFoundHandling:
    def test_404_returns_skipped_status(self, tmp_path):
        code = "7760101"
        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(404)

        result = crawl_one(code, tmp_path, client=mock_client)

        assert result["status"] == "skipped_404"
        assert not (tmp_path / f"nganh_{code}.html").exists()

    def test_404_does_not_raise_in_crawl_many(self, tmp_path):
        codes = ["7760101", "7320101"]
        responses = {
            "7760101": _make_response(404),
            "7320101": _make_response(200, b"<html>ok</html>"),
        }

        mock_client = MagicMock()
        mock_client.get.side_effect = lambda url, **kw: (
            responses["7760101"] if "7760101" in url else responses["7320101"]
        )

        with patch("rag2025.scripts.crawl_nganh_detail.httpx") as mock_httpx:
            mock_httpx.Client.return_value.__enter__ = lambda s: mock_client
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.RequestError = Exception  # so isinstance check works

            results = crawl_many(codes, tmp_path)

        statuses = {r["code"]: r["status"] for r in results}
        assert statuses["7760101"] == "skipped_404"
        assert statuses["7320101"] == "fetched"


# ---------------------------------------------------------------------------
# 5. All 28 codes attempted in crawl_many
# ---------------------------------------------------------------------------

class TestAllCodesAttempted:
    def test_all_28_codes_attempted(self, tmp_path):
        content = b"<html>page</html>"
        attempted_codes: list[str] = []

        def fake_get(url, **kwargs):
            # Extract exact code from URL query string: manganh=<code>
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            code = qs.get("manganh", [None])[0]
            if code:
                attempted_codes.append(code)
            return _make_response(200, content)

        mock_client = MagicMock()
        mock_client.get.side_effect = fake_get

        with patch("rag2025.scripts.crawl_nganh_detail.httpx") as mock_httpx:
            mock_httpx.Client.return_value.__enter__ = lambda s: mock_client
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.RequestError = Exception

            results = crawl_many(DEFAULT_CODES, tmp_path)

        assert len(results) == 28
        assert set(attempted_codes) == set(DEFAULT_CODES)


# ---------------------------------------------------------------------------
# 6. Backoff helper
# ---------------------------------------------------------------------------

class TestBackoffHelper:
    def test_backoff_caps_at_30s(self):
        assert _backoff_delay(10) == 30.0

    def test_backoff_first_attempt_is_1s(self):
        assert _backoff_delay(0) == 1.0

    def test_backoff_second_attempt_is_2s(self):
        assert _backoff_delay(1) == 2.0


# ---------------------------------------------------------------------------
# 7. SHA256 helper
# ---------------------------------------------------------------------------

class TestSha256Helper:
    def test_sha256_deterministic(self):
        data = b"hello world"
        assert _sha256_bytes(data) == _sha256_bytes(data)

    def test_sha256_differs_for_different_data(self):
        assert _sha256_bytes(b"a") != _sha256_bytes(b"b")
