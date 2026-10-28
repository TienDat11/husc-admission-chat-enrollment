"""Fixed-window rate limiting shared by the anonymous query path and the admin API.

Two properties matter, and both were previously wrong or absent:

1. **Rejection is a 429, not a 500.** The anonymous limiter in ``src/main.py``
   raised ``_RateLimited()``, a name that was never defined anywhere in the
   package. Every request past the limit therefore died with ``NameError``
   inside the threadpool instead of being rejected, so ``RATE_LIMIT_PER_MINUTE``
   never applied to anyone. ``RateLimited`` below is a real exception and
   ``enforce`` translates it into the HTTP response.

2. **Admin traffic is limited by actor, not by IP.** Admin callers share
   egress addresses (an office NAT, a reverse proxy), so an IP-keyed bucket
   makes one admin's burst throttle every other admin. Buckets here are keyed
   by whatever the caller passes — the admin path passes ``Identity.actor_id``.

Buckets are per-process. With multiple workers the effective limit is
``limit x workers``; that is acceptable for abuse control and keeps the
limiter dependency-free, matching the existing in-memory approach.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable

from fastapi import HTTPException, Request


class RateLimited(Exception):
    """Raised when a caller exceeds its budget for the current window."""


class FixedWindowLimiter:
    """Sliding-window counter over per-key timestamp deques.

    ``limit <= 0`` disables limiting entirely (the caller still pays one dict
    lookup). ``key_fn`` maps a request to a bucket key; returning ``None``
    skips limiting for that request, which is how the anonymous path keeps
    health checks and other internal probes unmetered.
    """

    def __init__(
        self,
        limit: int | Callable[[], int],
        window_s: float = 60.0,
        key_fn: Callable[[Request], str | None] | None = None,
    ) -> None:
        # ``limit`` may be a callable so a surface whose budget is
        # operator-configurable can read it per request instead of freezing it
        # at import — otherwise the value cannot be changed without a restart.
        self.limit = limit
        self.window_s = window_s
        self._key_fn = key_fn
        self._lock = Lock()
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def _budget(self) -> int:
        """This limiter's current budget. ``<= 0`` disables limiting."""
        return self.limit() if callable(self.limit) else self.limit

    def check(self, key: str) -> None:
        """Record one hit for ``key``; raise ``RateLimited`` if over budget."""
        limit = self._budget()
        if limit <= 0:
            return
        now = time.monotonic()
        window_start = now - self.window_s
        with self._lock:
            self._evict(window_start)
            bucket = self._buckets[key]
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= limit:
                raise RateLimited()
            bucket.append(now)

    def _evict(self, window_start: float) -> None:
        """Drop buckets with no live entries so the dict cannot grow forever.

        Caller holds ``self._lock``.
        """
        stale = [k for k, b in self._buckets.items() if not b or b[-1] < window_start]
        for k in stale:
            self._buckets.pop(k, None)

    def reset(self) -> None:
        """Clear every bucket. Test seam — production never needs this."""
        with self._lock:
            self._buckets.clear()

    def key_for(self, request: Request) -> str | None:
        """Resolve this request's bucket key, or ``None`` to skip limiting."""
        if self._key_fn is None:
            return None
        return self._key_fn(request)

    def enforce(self, key: str) -> None:
        """Spend one unit for ``key``; raise HTTP 429 when over budget.

        For callers that derive the key from something other than the request
        (the admin limiter keys on the authenticated actor). Shares this
        module's single 429 translation with :meth:`__call__`.
        """
        try:
            self.check(key)
        except RateLimited:
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(int(self.window_s))},
            ) from None

    async def __call__(self, request: Request) -> None:
        """FastAPI dependency: limit this request or raise 429."""
        key = self.key_for(request)
        if key is None:
            return
        self.enforce(key)
