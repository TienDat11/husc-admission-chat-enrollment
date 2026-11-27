"""Tests for src.utils.http.client_ip: the Nth-from-right X-Forwarded-For hop."""
from types import SimpleNamespace

from src.utils.http import client_ip


class _Req:
    def __init__(self, xff=None, peer="5.5.5.5"):
        self.headers = {"x-forwarded-for": xff} if xff is not None else {}
        self.client = SimpleNamespace(host=peer) if peer is not None else None


def test_one_trusted_proxy_takes_rightmost_entry():
    # Client-supplied leftmost entry is ignored (spoof-proof).
    assert client_ip(_Req("9.9.9.9, 1.1.1.1, 2.2.2.2"), 1) == "2.2.2.2"


def test_two_trusted_proxies_take_second_from_right():
    assert client_ip(_Req("9.9.9.9, 1.1.1.1, 2.2.2.2"), 2) == "1.1.1.1"


def test_zero_trusted_proxies_ignores_header():
    assert client_ip(_Req("9.9.9.9"), 0) == "5.5.5.5"


def test_missing_header_uses_socket_peer():
    assert client_ip(_Req(None), 1) == "5.5.5.5"


def test_header_shorter_than_trusted_hops_falls_back_to_peer():
    assert client_ip(_Req("2.2.2.2"), 2) == "5.5.5.5"


def test_whitespace_is_trimmed():
    assert client_ip(_Req("  9.9.9.9 ,  2.2.2.2  "), 1) == "2.2.2.2"


def test_no_peer_and_no_header_is_unknown():
    assert client_ip(_Req(None, peer=None), 1) == "unknown"
