"""Redirect hops in fetch_html re-validate and pin a fresh session per hop.

DNS is faked through socket.getaddrinfo; no socket is opened.
"""
import socket

import pytest
import requests

from src.services import web_fetch

_DNS = {
    "a.example": "93.184.216.34",
    "b.example": "93.184.216.35",
    "priv.example": "10.0.0.5",
}


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_DNS[host], port or 0))]

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", fake_getaddrinfo)


class _Resp:
    def __init__(self, status, headers=None, body=b""):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.raw = None

    def iter_content(self, chunk_size=1):
        yield self._body

    def close(self):
        pass


class _Sess:
    def __init__(self, responses):
        self._responses = list(responses)
        self.gets = []
        self.closed = False

    def get(self, url, **kwargs):
        self.gets.append(url)
        return self._responses.pop(0)

    def close(self):
        self.closed = True


def _install_factory(monkeypatch, scripts):
    made = []

    def factory(url):
        sess = _Sess(scripts[len(made)])
        made.append((url, sess))
        return sess

    monkeypatch.setattr(web_fetch, "_pinned_session", factory)
    return made


def test_cross_host_redirect_gets_fresh_pinned_session(monkeypatch):
    made = _install_factory(monkeypatch, [
        [_Resp(302, {"Location": "https://b.example/x"})],
        [_Resp(200, body=b"<html>ok</html>")],
    ])
    body = web_fetch.fetch_html("https://a.example/")
    assert body == b"<html>ok</html>"
    assert [url for url, _ in made] == ["https://a.example/", "https://b.example/x"]
    assert made[1][1].gets == ["https://b.example/x"]
    assert made[0][1].closed is True


def test_scheme_change_on_redirect_pins_new_session_for_new_scheme(monkeypatch):
    made = _install_factory(monkeypatch, [
        [_Resp(301, {"Location": "http://b.example/x"})],
        [_Resp(200, body=b"ok")],
    ])
    assert web_fetch.fetch_html("https://a.example/") == b"ok"
    assert [url for url, _ in made] == ["https://a.example/", "http://b.example/x"]


def test_redirect_to_private_address_is_refused_before_any_request(monkeypatch):
    made = _install_factory(monkeypatch, [
        [_Resp(302, {"Location": "http://priv.example/secret"})],
    ])
    with pytest.raises(web_fetch.UnsafeUrlError):
        web_fetch.fetch_html("https://a.example/")
    assert [url for url, _ in made] == ["https://a.example/"]
    assert all("priv.example" not in g for _, s in made for g in s.gets)


def test_each_hop_adapter_dials_its_own_validated_address():
    sess = web_fetch._pinned_session("https://b.example/x")
    adapter = sess.get_adapter("https://b.example/x")
    assert adapter._pinned_ip == "93.184.216.35"
    assert adapter._pinned_host == "b.example"


def test_ipv6_pinned_address_is_bracketed_in_netloc(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1::1", port or 0, 0, 0))]

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", fake_getaddrinfo)
    seen = []

    def fake_send(self, request, **kwargs):
        seen.append((request.url, request.headers["Host"]))
        return None

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    for url, expected in [
        ("https://v6.example/x", "https://[2606:2800:220:1::1]/x"),
        ("https://v6.example:8443/x", "https://[2606:2800:220:1::1]:8443/x"),
    ]:
        sess = web_fetch._pinned_session(url)
        adapter = sess.get_adapter(url)
        adapter.send(requests.Request("GET", url).prepare())
        assert adapter._pinned_ip == "2606:2800:220:1::1"
    assert [u for u, _ in seen] == [
        "https://[2606:2800:220:1::1]/x",
        "https://[2606:2800:220:1::1]:8443/x",
    ]
    assert all(host == "v6.example" for _, host in seen)
