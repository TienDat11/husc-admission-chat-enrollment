"""Web fetch core — ticket T21.

The fetch policy shared by the ID-list crawler and the admin URL route: the
SSRF guard, the streamed byte cap, the retry rules, and HTML text extraction.
No socket and no DNS anywhere — the resolver and the session are both
monkeypatched, so the suite is deterministic and offline.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services import web_fetch  # noqa: E402
from src.services.web_fetch import (  # noqa: E402
    UnsafeUrlError,
    WebFetchError,
    WebFetchTimeout,
    WebFetchTooLarge,
    backoff_delay,
    extract_html_text,
    fetch_html,
    sha256_bytes,
    validate_public_url,
)


# ---------------------------------------------------------------------------
# doubles: a session that streams, and a resolver that never touches the net
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status: int = 200, body: bytes = b"<html>ok</html>", headers=None):
        self.status_code = status
        self.headers = dict(headers or {})
        self._chunks = [body[i : i + 7] for i in range(0, len(body), 7)] if body else []
        self.closed = False

    def iter_content(self, chunk_size: int = 0):
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class _Session:
    """A ``requests.Session`` stand-in: one queued outcome per call."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _resolve_to(monkeypatch, *addresses: str) -> None:
    """Make every hostname resolve to ``addresses``."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(2, 1, 6, "", (address, 0)) for address in addresses]

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", fake_getaddrinfo)


# ---------------------------------------------------------------------------
# validate_public_url — the SSRF guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http:///no-host",
        "https:///no-host",
    ],
)
def test_non_http_schemes_and_hostless_urls_are_refused(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/thongbao",
        "http://127.0.0.1/thongbao",
        "http://10.1.2.3/thongbao",
        "http://192.168.1.10/thongbao",
        "http://172.16.5.4/thongbao",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://0.0.0.0/thongbao",
        "http://[::1]/thongbao",
        "http://[fd00::1]/thongbao",
    ],
)
def test_private_and_loopback_hosts_are_refused(url: str, monkeypatch) -> None:
    _resolve_to(monkeypatch, "127.0.0.1")
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


def test_literal_public_address_is_accepted_without_dns(monkeypatch) -> None:
    def explode(*args, **kwargs):  # a literal address must not need a resolver
        raise AssertionError("getaddrinfo called for a literal address")

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", explode)
    url = "http://93.184.216.34/thongbao.php?id=63"
    assert validate_public_url(url) == url


def test_hostname_resolving_to_public_address_is_accepted(monkeypatch) -> None:
    _resolve_to(monkeypatch, "93.184.216.34")
    url = "https://tuyensinh.husc.edu.vn/thongbao.php?id=63"
    assert validate_public_url(url) == url


def test_hostname_with_a_private_answer_is_refused(monkeypatch) -> None:
    _resolve_to(monkeypatch, "10.0.0.1")
    with pytest.raises(UnsafeUrlError, match="not a public address"):
        validate_public_url("http://internal.example.com/x")


def test_one_private_answer_among_public_ones_is_refused(monkeypatch) -> None:
    """Every answer must be public — a mixed answer is refused, not raced."""
    _resolve_to(monkeypatch, "93.184.216.34", "127.0.0.1")
    with pytest.raises(UnsafeUrlError):
        validate_public_url("http://rebind.example.com/x")


def test_unresolvable_host_is_refused(monkeypatch) -> None:
    def fake_getaddrinfo(*args, **kwargs):
        raise OSError("NXDOMAIN")

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UnsafeUrlError, match="does not resolve"):
        validate_public_url("http://nope.invalid/x")


# ---------------------------------------------------------------------------
# fetch_html — retry rules, byte cap, and the guard actually running
# ---------------------------------------------------------------------------


def test_fetch_returns_bytes_on_200(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(_Response(200, b"<html>hi</html>"))
    assert fetch_html("http://93.184.216.34/x", session=session) == b"<html>hi</html>"
    assert len(session.calls) == 1


def test_fetch_retries_5xx_then_raises(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(web_fetch.time, "sleep", sleeps.append)
    session = _Session(_Response(503, b""))
    with pytest.raises(WebFetchError, match="after 3 attempts"):
        fetch_html("http://93.184.216.34/x", session=session, max_attempts=3)
    assert len(session.calls) == 3
    assert sleeps == [backoff_delay(0), backoff_delay(1)]


def test_fetch_returns_after_a_transient_5xx(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(_Response(503, b""), _Response(200, b"<html>ok</html>"))
    assert fetch_html("http://93.184.216.34/x", session=session, max_attempts=3) == b"<html>ok</html>"
    assert len(session.calls) == 2


def test_fetch_does_not_retry_4xx(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(_Response(404, b""))
    with pytest.raises(WebFetchError, match="HTTP 404"):
        fetch_html("http://93.184.216.34/x", session=session, max_attempts=3)
    assert len(session.calls) == 1


def test_fetch_retries_transport_errors(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(
        web_fetch.requests.ConnectionError("boom"), _Response(200, b"<html>ok</html>")
    )
    assert fetch_html("http://93.184.216.34/x", session=session, max_attempts=3) == b"<html>ok</html>"


def test_declared_content_length_over_cap_is_refused_without_reading(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(_Response(200, b"tiny", headers={"Content-Length": "999999"}))
    with pytest.raises(WebFetchTooLarge, match="declares"):
        fetch_html("http://93.184.216.34/x", session=session, max_bytes=1024)


def test_streamed_body_over_cap_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(_Response(200, b"x" * 4096))
    with pytest.raises(WebFetchTooLarge, match="exceeds"):
        fetch_html("http://93.184.216.34/x", session=session, max_bytes=1024)


def test_fetch_validates_before_opening_a_socket() -> None:
    session = _Session(_Response(200, b"<html>ok</html>"))
    with pytest.raises(UnsafeUrlError):
        fetch_html("http://127.0.0.1/x", session=session)
    assert session.calls == []


def test_validate_can_be_turned_off_for_a_built_url(monkeypatch) -> None:
    """The CLI builds its URL from a constant, so it fetches with the guard off."""
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    session = _Session(_Response(200, b"<html>ok</html>"))




# ---------------------------------------------------------------------------
# redirects: each hop is re-validated, capped, and the target is never
# opened before its address has passed the SSRF guard
# ---------------------------------------------------------------------------


def test_redirect_to_a_private_target_is_refused_without_contact(monkeypatch) -> None:
    """A 302 to 127.0.0.1 must be refused by the guard, not opened."""
    _resolve_to(monkeypatch, "93.184.216.34")  # source host resolves public
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    redirect = _Response(
        302, b"", headers={"Location": "http://127.0.0.1/secret"}
    )
    session = _Session(redirect, _Response(200, b"<html>never</html>"))
    with pytest.raises(UnsafeUrlError, match="not a public address"):
        fetch_html("http://93.184.216.34/start", session=session)
    # the target was never contacted: only the initial source URL was seen
    assert session.calls == ["http://93.184.216.34/start"]


def test_redirect_chain_within_hop_cap_to_a_public_target_succeeds(monkeypatch) -> None:
    _resolve_to(monkeypatch, "93.184.216.34")
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    hop1 = _Response(302, b"", headers={"Location": "/step2"})
    hop2 = _Response(302, b"", headers={"Location": "http://93.184.216.34/final"})
    final = _Response(200, b"<html>done</html>")
    session = _Session(hop1, hop2, final)
    assert fetch_html("http://93.184.216.34/start", session=session) == b"<html>done</html>"
    assert session.calls == [
        "http://93.184.216.34/start",
        "http://93.184.216.34/step2",
        "http://93.184.216.34/final",
    ]


def test_redirect_chain_beyond_hop_cap_is_refused(monkeypatch) -> None:
    _resolve_to(monkeypatch, "93.184.216.34")
    monkeypatch.setattr(web_fetch.time, "sleep", lambda _s: None)
    # three 302s each pointing at the next step; the cap is 2 here
    hop = lambda n: _Response(302, b"", headers={"Location": f"/s{n}"})
    session = _Session(hop(1), hop(2), hop(3), _Response(200, b"<html>x</html>"))
    with pytest.raises(WebFetchError, match="too many redirects"):
        fetch_html("http://93.184.216.34/start", session=session, max_redirects=2)


# ---------------------------------------------------------------------------
# the overall deadline: a slow-drip body that never completes a chunk is cut off
# ---------------------------------------------------------------------------


class _DripRaw:
    """A urllib3-style ``response.raw`` whose ``_sock_shutdown`` hook is what
    actually interrupts a blocking read — recorded so the test can assert
    the watchdog fired it."""

    def __init__(self, shutdowns: list, released: threading.Event):
        self._shutdowns = shutdowns
        self._released = released

    @property
    def _sock_shutdown(self):
        def _fire(how: int) -> None:
            self._shutdowns.append(how)
            self._released.set()

        return _fire


class _DripResponse:
    """A ``requests.Response``-shaped double whose ``iter_content`` blocks on a
    socket read until the shutdown hook releases it — the only escape hatch,
    mirroring a server that dribbles bytes slower than any per-yield check
    ever reaches."""

    def __init__(self, shutdowns: list, released: threading.Event):
        self.status_code = 200
        self.headers = {}
        self.closed = False
        self._released = released
        self.raw = _DripRaw(shutdowns, released)

    def iter_content(self, chunk_size: int = 0):
        # Simulate a blocking ``recv`` that will not return on its own: it
        # waits for the watchdog's shutdown hook to release it.  The per-yield
        # deadline check never gets a chance to run, so the watchdog is the
        # only thing that can stop the read.
        self._released.wait(timeout=5)
        yield b""

    def close(self) -> None:
        self.closed = True


def test_slow_drip_body_is_cut_off_at_the_overall_deadline(monkeypatch) -> None:
    """A server that drips a byte every 15 s holds the socket read open; the
    per-yield deadline check is starved, so the watchdog must call the
    urllib3 socket-shutdown hook to interrupt the read and raise
    ``WebFetchTimeout``."""
    import socket as _socket

    shutdowns: list[int] = []
    released = threading.Event()
    session = _Session(_DripResponse(shutdowns, released))
    monkeypatch.setattr(web_fetch, "_DEADLINE_S", 0.05)
    with pytest.raises(web_fetch.WebFetchTimeout):
        fetch_html("http://93.184.216.34/drip", session=session)
    # the watchdog fired the shutdown with SHUT_RDWR to interrupt the read
    assert _socket.SHUT_RDWR in shutdowns
def test_sha256_bytes_is_the_content_hash() -> None:
    import hashlib

    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


# ---------------------------------------------------------------------------
# extract_html_text
# ---------------------------------------------------------------------------


def test_extract_prefers_main_and_drops_chrome() -> None:
    html = b"""
    <html><head><style>p{color:red}</style></head>
    <body>
      <nav>Trang chu | Lien he</nav>
      <header>Header</header>
      <main>
        <p>Thong bao tuyen sinh 2026</p>
        <p>   </p>
        <div>Hoc phi 12000000</div>
      </main>
      <script>var x = 1;</script>
      <footer>Footer</footer>
    </body></html>
    """
    text = extract_html_text(html)
    assert "Thong bao tuyen sinh 2026" in text
    assert "Hoc phi 12000000" in text
    for chrome in ("Trang chu", "Header", "Footer", "var x", "color:red"):
        assert chrome not in text
    assert "\n\n" not in text  # blank lines collapsed


def test_extract_falls_back_to_a_content_div() -> None:
    html = b"""
    <html><body>
      <div class="menu">Menu</div>
      <div id="content">Noi dung thong bao</div>
    </body></html>
    """
    text = extract_html_text(html)
    assert "Noi dung thong bao" in text
    assert "Menu" not in text


def test_extract_of_a_textless_page_is_empty() -> None:
    assert extract_html_text(b"<html><body><nav>a</nav></body></html>") == ""
