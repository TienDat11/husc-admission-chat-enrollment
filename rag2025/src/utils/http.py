"""Client address resolution behind the reverse proxy (Caddy fronts the app)."""
from __future__ import annotations

from typing import Any


def client_ip(request: Any, trusted_proxy_count: int) -> str:
    """Return the client address for ``request``.

    Each trusted proxy appends the address it received to X-Forwarded-For, so
    the entry ``trusted_proxy_count`` from the right is the client as seen by
    the outermost trusted proxy. Entries further left are client-supplied and
    ignored. With ``trusted_proxy_count`` 0 the header is ignored; if the
    header has fewer entries than trusted hops, the socket peer is used.
    """
    peer = request.client.host if request.client and request.client.host else "unknown"
    if trusted_proxy_count <= 0:
        return peer
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return peer
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    if len(hops) < trusted_proxy_count:
        return peer
    return hops[-trusted_proxy_count]
