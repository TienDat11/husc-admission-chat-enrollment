"""One importable HTML-fetch core — ticket T21.

The SSRF guard and the fetch-with-retry-and-cap logic that the admin URL-ingest
route and the ID-list crawler share, kept in one module so the two callers
cannot drift.  Three properties a caller that forgets them the hard way would
learn:

- Every URL — the initial one *and* every redirect hop — passes
  :func:`validate_public_url` before a socket is opened to it, and the
  redirect chain is capped.  A public URL that 302s to a private target is
  refused without ever contacting that target.
- The first hop's connection is pinned to the IP the guard actually validated
  (``Host`` header and TLS SNI keep naming the original host), so a DNS
  rebind between validation and connect cannot defeat the guard.  Redirect
  hops close that window by re-validating each hop.
- A monotonic overall deadline bounds DNS + connect + retries + redirects +
  body read.  A server that dribbles bytes slower than the socket timeout is
  cut off at the deadline, not at the per-socket inactivity window — a
  watchdog calls the urllib3 socket-shutdown hook to interrupt a blocking read.

No socket and no DNS at import time: :mod:`socket.getaddrinfo` is the only
resolver, and it is called only when a URL actually has a hostname to check.
"""
from __future__ import annotations

import hashlib
import ipaddress
import socket
import time
from typing import Callable
from urllib.parse import urlsplit, urljoin
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

#: One page, one timeout. The largest HUSC notification page is well under a
#: megabyte; the cap exists so a mis-pasted URL cannot pull a binary into the
#: admin worker's memory.
TIMEOUT_S = 30
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
MAX_BYTES = 5 * 1024 * 1024
USER_AGENT = "HUSC-Admission-Chat/1.0 (+temporal-reingest)"
#: Stream read size: bounded, and large enough that a 5 MB page is ~80 reads.
_CHUNK = 64 * 1024
_SCHEMES = frozenset({"http", "https"})
#: Bounded redirect chain: cap it, so a public URL that issues 302 → private
#: target is refused *before* the second socket is opened.
_MAX_REDIRECTS = 5
#: Monotonic overall deadline (DNS + connect + all hops + retries + body reads).
_DEADLINE_S = 60.0
#: Redirect status codes we follow manually.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class WebFetchError(Exception):
    """The page could not be fetched: transport failure or a 4xx/5xx status."""


class WebFetchTooLarge(WebFetchError):
    """The body exceeds ``max_bytes``; it was not read past the cap."""


class UnsafeUrlError(WebFetchError):
    """The URL is not one this process will fetch — see the module docstring."""

class WebFetchTimeout(WebFetchError):
    """The overall deadline expired before the fetch completed."""


def sha256_bytes(data: bytes) -> str:
    """Content hash used for the idempotency comparison."""
    return hashlib.sha256(data).hexdigest()


def backoff_delay(attempt: int) -> float:
    """Delay before retry ``attempt`` (0-based), capped at ``BACKOFF_CAP_S``."""
    return min(BACKOFF_BASE_S * (2**attempt), BACKOFF_CAP_S)


def _host_addresses(host: str) -> tuple[str, ...]:
    """Every address ``host`` resolves to; IPv4 and IPv6 alike.

    A literal address is returned as itself — no DNS round trip, so a URL that
    already names a private address is refused without a resolver in the loop.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return (host,)
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError as exc:  # NXDOMAIN, no resolver, …
        raise UnsafeUrlError(f"{host!r} does not resolve: {exc}") from exc
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


def validate_public_url(url: str) -> str:
    """Refuse anything but an http(s) URL whose host is globally routable.

    ``*`` every address a hostname resolves to must pass: a name with one
    public and one private answer is refused, not raced. Loopback, private,
    link-local (the cloud metadata endpoint lives at 169.254.169.254), CGNAT,
    reserved and multicast addresses are all rejected.

    Returns ``url`` unchanged so a caller can write
    ``url = validate_public_url(url)`` and get the string it validated.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _SCHEMES:
        raise UnsafeUrlError(f"{url!r} is not http(s); this route fetches web pages only")
    host = parts.hostname
    if not host:
        raise UnsafeUrlError(f"{url!r} names no host")
    _validated_host_addresses(host)
    return url
def _validated_host_addresses(host: str) -> tuple[str, ...]:
    """Every address ``host`` resolves to, *validated as global* — the one
    resolve+check pass the guard does.  Callers that pin a connection to the
    validated address use this result rather than re-resolving DNS, which is
    the whole point: the IP the socket dials is the IP that passed the guard,
    so a public→private rebind between validation and connect cannot land on
    a private address we did not check.
    """
    addresses = _host_addresses(host)
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeUrlError(f"{host!r} resolved to unreadable address {address!r}") from exc
        if not parsed.is_global:
            raise UnsafeUrlError(
                f"{host!r} resolves to {address}, which is not a public address; "
                "this route does not fetch localhost or private networks"
            )
    return addresses


def _pinned_session(url: str) -> requests.Session:
    """A :class:`requests.Session` whose adapter dials the validated IP of
    ``url``'s host rather than re-resolving DNS at connect time.

    The DNS-rebinding TOCTOU: ``validate_public_url`` resolves the host,
    checks every answer, then throws the result away; ``urllib3`` resolves
    the same hostname again at connect time.  A hostname that answered
    public during validation but rebinds to a private address between the
    two resolution passes defeats the guard.  Pinning the connection to the
    IP we actually validated closes that window: the socket goes to the
    address we checked, while the ``Host`` header and TLS SNI keep naming
    the original hostname, so the certificate check is unchanged.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    scheme = (parsed.scheme or "http").lower()
    # One resolve+validate pass; the pinned IP comes from that pass, never
    # from a second DNS lookup.
    pinned_ip = _validated_host_addresses(host)[0]
    adapter = _PinnedAdapter(pinned_host=host, pinned_ip=pinned_ip, scheme=scheme)
    sess = requests.Session()
    sess.mount(f"{scheme}://", adapter)
    return sess


class _PinnedAdapter(requests.adapters.HTTPAdapter):
    """Dial ``pinned_ip`` for requests whose host is ``pinned_host``.

    The URL's netloc is swapped for the IP literal, the ``Host`` header is
    restored to the original hostname, and (for https) the TLS SNI +
    certificate assertion are set on the urllib3 pool kwarg so the
    certificate validates against the original hostname.  The TCP/SSL
    connection is therefore made to the address that was validated, never
    to a re-resolved one.
    """

    def __init__(self, pinned_host: str, pinned_ip: str, scheme: str, **kwargs):
        # Thread SNI + cert assertion into the urllib3 pool so an https
        # connection to the pinned IP still validates the certificate against
        # the original hostname.  requests 2.31's HTTPAdapter.send does not
        # accept a ``connect`` kwarg; the pool manager's connection_pool_kw
        # is the supported seam.
        pool_kw = dict(kwargs.pop("pool_kwargs", None) or {})
        if scheme == "https":
            pool_kw["server_hostname"] = pinned_host
            pool_kw["assert_hostname"] = pinned_host
        super().__init__(**kwargs, pool_kwargs=pool_kw)
        self._pinned_host = pinned_host
        self._pinned_ip = pinned_ip
        self._scheme = scheme

    def send(self, request, **kwargs):
        host = urlparse(request.url).hostname
        if host != self._pinned_host:
            return super().send(request, **kwargs)
        parts = urlparse(request.url)
        default_port = 443 if self._scheme == "https" else 80
        netloc = (
            self._pinned_ip
            if parts.port in (None, default_port)
            else f"{self._pinned_ip}:{parts.port}"
        )
        request.url = parts._replace(netloc=netloc).geturl()
        request.headers["Host"] = self._pinned_host
        return super().send(request, **kwargs)

def _declared_length(response: requests.Response) -> int | None:
    """``Content-Length`` as an int, or ``None`` when absent or malformed.

    A malformed header is not a size claim; the streaming cap below is what
    actually holds the bound.
    """
    raw = response.headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _read_bounded(
    response: requests.Response,
    url: str,
    max_bytes: int,
    deadline: float | None = None,
) -> bytes:
    """Stream the body, refusing to hold more than ``max_bytes`` of it.

    When ``deadline`` is set, a watchdog thread calls the urllib3
    ``sock_shutdown`` callback (exposed as ``response.raw._sock_shutdown``)
    the moment the overall clock passes it.  This unblocks the blocking
    ``recv`` inside ``iter_content`` (which waits up to ``chunk_size``
    bytes before yielding, so a per-yield time check is starved by a server
    that dribbles one byte every 15 s).  The watchdog is a daemon that waits
    on a :class:`threading.Event`; the ``finally`` sets the event before
    joining so a completed read does not leave a sleeper that would later
    shutdown a reused pooled connection.
    """
    import threading

    declared = _declared_length(response)
    if declared is not None and declared > max_bytes:
        raise WebFetchTooLarge(f"{url!r} declares {declared} bytes, past the {max_bytes}-byte cap")

    watchdog: threading.Thread | None = None
    stop = threading.Event()
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WebFetchTimeout(f"overall deadline ({_DEADLINE_S:.0f}s) expired on {url}")
        sock_shutdown = _sock_shutdown_hook(response)
        if sock_shutdown is not None:
            def _watchdog() -> None:
                stop.wait(remaining)
                if not stop.is_set():
                    try:
                        sock_shutdown(socket.SHUT_RDWR)
                    except (OSError, ValueError):
                        pass

            watchdog = threading.Thread(target=_watchdog, daemon=True)
            watchdog.start()

    try:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=_CHUNK):
            if deadline is not None and time.monotonic() > deadline:
                raise WebFetchTimeout(f"overall deadline ({_DEADLINE_S:.0f}s) expired on {url}")
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise WebFetchTooLarge(f"{url!r} exceeds the {max_bytes}-byte cap")
            chunks.append(chunk)
        if deadline is not None and time.monotonic() > deadline:
            raise WebFetchTimeout(f"overall deadline ({_DEADLINE_S:.0f}s) expired on {url}")
        return b"".join(chunks)
    finally:
        # Signal the watchdog to stop *before* joining, so a completed read
        # does not leave a sleeper that would later shutdown a reused pooled
        # connection.  The join is bounded: a fired watchdog has already
        # returned; a not-yet-fired one is woken by the event.
        stop.set()
        if watchdog is not None:
            watchdog.join(timeout=1.0)


def _sock_shutdown_hook(response: requests.Response) -> Callable[[int], None] | None:
    """The urllib3 socket-shutdown callback behind a streaming response.

    ``requests`` exposes the connection via ``response.raw`` (a urllib3
    ``HTTPResponse``).  When the pool wraps the connection it passes a
    ``sock_shutdown`` callable into the response constructor; that callable
    is the supported seam for interrupting a blocking read without leaking
    the pooled connection.  Returns ``None`` when the attribute is not
    available (e.g. an injected test double with no ``raw``), so the
    watchdog is skipped and the per-yield check is the only guard.
    """
    raw = getattr(response, "raw", None)
    if raw is None:
        return None
    hook = getattr(raw, "_sock_shutdown", None)
    return hook if callable(hook) else None



def fetch_html(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
    max_bytes: int = MAX_BYTES,
    validate: bool = True,
    max_redirects: int = _MAX_REDIRECTS,
) -> bytes:
    """Fetch one page's bytes, with bounded backoff and a streamed byte cap.

    5xx and transport errors are retried up to ``max_attempts``; a 4xx is a
    permanent refusal and is not retried (asking again will not change the
    answer). ``validate`` runs :func:`validate_public_url` first — the default,
    because the caller who forgets it is the one handling a request field.

    Redirects are followed manually (``allow_redirects=False``): each 3xx hop's
    Location is resolved, re-validated, and the chain is capped at
    ``max_redirects`` hops, so a public URL that 302s to a private target is
    refused without ever opening a socket to that target.  A monotonic overall
    deadline (``_DEADLINE_S``) bounds DNS + connect + every retry, hop and
    body read; a server that dribbles bytes slower than the socket timeout is
    cut off at the deadline, not at the per-socket inactivity window.
    """
    deadline = time.monotonic() + _DEADLINE_S
    if validate:
        url = validate_public_url(url)
        sess = session or _pinned_session(url)
    else:
        sess = session or requests.Session()
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None
    redirects = 0
    attempt = 0
    while attempt < max_attempts:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WebFetchTimeout(f"overall deadline ({_DEADLINE_S:.0f}s) expired on {url}")
        try:
            response = sess.get(
                url,
                timeout=min(timeout, remaining),
                headers=headers,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(f"url={url} attempt={attempt + 1} error={exc} retrying")
            response = None
        else:
            status = response.status_code
            if status == 200:
                try:
                    return _read_bounded(response, url, max_bytes, deadline=deadline)
                finally:
                    response.close()
            if status in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                response.close()
                if location is None:
                    raise WebFetchError(f"HTTP {status} on {url}: no Location header")
                if redirects >= max_redirects:
                    raise WebFetchError(f"too many redirects on {url} (> {max_redirects})")
                next_url = urljoin(url, location)
                if validate:
                    validate_public_url(next_url)
                redirects += 1
                url = next_url
                continue  # redirect hop: not a failed attempt, no sleep, no retry burn
            if 500 <= status < 600:
                response.close()
                last_exc = WebFetchError(f"HTTP {status} on {url}")
                logger.warning(f"url={url} attempt={attempt + 1} status={status} retrying")
            else:
                # 4xx — do not retry: the answer will not change.
                response.close()
                raise WebFetchError(f"HTTP {status} on {url} (no retry)")
        if attempt + 1 < max_attempts:
            delay = min(backoff_delay(attempt), max(deadline - time.monotonic(), 0.0))
            time.sleep(delay)
        attempt += 1
    raise WebFetchError(f"Failed to fetch {url} after {max_attempts} attempts: {last_exc}")


#: Tags whose text is chrome, not the notification: dropped before extraction.
_STRIP_TAGS = ("nav", "footer", "header", "script", "style", "noscript", "aside", "form")
#: class/id fragments that mark the page's own content container.
_CONTENT_HINTS = ("content", "main", "detail", "thongbao", "article")


def _content_root(soup: BeautifulSoup):
    """The element holding the page's own text, best effort.

    ``<main>`` / ``<article>`` when the page marks them, else the first
    ``<div>`` whose class or id looks like a content container, else the body.
    The order is deliberate: a semantic element is the page telling us where
    its text is, which beats guessing from a class name.
    """
    for tag in ("main", "article"):
        found = soup.find(tag)
        if found is not None:
            return found
    for div in soup.find_all("div"):
        marker = " ".join(div.get("class", ())) + " " + (div.get("id") or "")
        if any(hint in marker.lower() for hint in _CONTENT_HINTS):
            return div
    return soup.find("body") or soup


def extract_html_text(html: bytes) -> str:
    """The readable text of one page: chrome stripped, blank lines collapsed.

    ``html.parser`` (not lxml) so extraction needs no compiled dependency
    beyond what the repo already pins — the same parser
    ``scripts/build_nganh_detail_jsonl.py`` uses.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    root = _content_root(soup)
    lines = (line.strip() for line in root.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)
