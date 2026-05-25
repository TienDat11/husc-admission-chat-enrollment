# @spec(S13.4) RequestTracker — lifespan-tracked semaphore for drain/swap
"""In-flight request counter used by /admin/reload-table to drain before swap.

Used as middleware around request handlers (skipping admin endpoints to avoid
self-deadlock). Drain timeout is bounded; on timeout we log a warning and let
the caller decide whether to proceed (default policy: proceed — stale
connections will be reset by OS when the old retriever closes).
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from loguru import logger


class RequestTracker:
    """Lifespan-tracked counter for in-flight requests.

    Definition of *drained*: counter == 0 (every track() context manager has exited).

    Thread/coroutine safety: protected by asyncio.Lock for mutation; the wait
    primitive is asyncio.Event toggled when counter reaches 0.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._zero_event: asyncio.Event = asyncio.Event()
        self._zero_event.set()  # initially drained
        self._lock: asyncio.Lock = asyncio.Lock()

    @asynccontextmanager
    async def track(self) -> AsyncIterator[None]:
        """Context manager: increment on enter, decrement on exit."""
        async with self._lock:
            self._counter += 1
            self._zero_event.clear()
        try:
            yield
        finally:
            async with self._lock:
                self._counter -= 1
                if self._counter <= 0:
                    self._counter = 0
                    self._zero_event.set()

    async def drain(self, timeout: float) -> bool:
        """Wait for counter to reach 0, bounded by timeout (seconds).

        Returns:
            True when fully drained.
            False on timeout (caller decides whether to proceed).
        """
        try:
            await asyncio.wait_for(self._zero_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"RequestTracker.drain timeout after {timeout}s; in_flight={self._counter}"
            )
            return False

    @property
    def in_flight(self) -> int:
        """Number of requests currently being tracked. Read-only snapshot."""
        return self._counter
