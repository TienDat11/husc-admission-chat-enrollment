# @spec(S13.4) RequestTracker — lifespan-tracked semaphore for drain/swap
"""In-flight request counter used by /admin/reload-table to drain before swap.

Used as middleware around request handlers (skipping admin endpoints to avoid
self-deadlock). Drain timeout is bounded; on timeout we log a warning and let
the caller decide whether to proceed (default policy: proceed).

Closing a retriever is not survivable by a reader mid-flight: ``close()`` nulls
``LanceDBRetriever._adapter`` and the query path then dereferences it
(``self._adapter.search(...)``), so a request still reading the old retriever
fails with ``AttributeError`` — not the "connections reset by the OS" an earlier
version of this docstring claimed. That is why the routes below close the old
retriever *only* when the drain proved nothing was reading it.

Admission gate
--------------
A counter alone does not make a swap safe. ``drain()`` returning True only means
the counter *reached* zero: admission was still open, so a request arriving in
the window between "drained" and the swap incremented the counter, read the old
retriever, and was then still running when the route closed it
(``_safe_close(old_retriever)``). Draining is therefore two moves, and
:meth:`RequestTracker.gate` performs both:

1. :meth:`begin_drain` — close admission. New ``track()`` callers block before
   incrementing, so none of them can observe the pre-swap runtime.
2. :meth:`drain` — wait for the counter to reach zero. Only requests admitted
   *before* the gate are counted, and they all finish before it returns.
3. :meth:`resume` — reopen admission. Every caller released here reads the
   runtime the swap already rebound.

The wait in ``track()`` is unbounded by design: the caller that closes admission
owns reopening it, and both routes do so from a ``finally``, so cancellation or
an exception cannot strand a request. Worst-case wait is one swap, which the F4
budget bounds at 30s (drain ≤25s + swap ≤1s + healthcheck ≤4s).
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
        self._admit_event: asyncio.Event = asyncio.Event()
        self._admit_event.set()  # initially admitting

    @asynccontextmanager
    async def track(self) -> AsyncIterator[None]:
        """Context manager: increment on enter, decrement on exit.

        Blocks while admission is closed (a gate is mid-drain). The wait comes
        *before* the increment, which is what makes the gate safe: a caller that
        waited is not counted by the in-progress drain, and by the time it
        proceeds the swap has rebound the runtime it is about to read.
        """
        await self._admit_event.wait()
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

    def begin_drain(self) -> None:
        """Close admission: new ``track()`` callers block until :meth:`resume`.

        Synchronous and non-blocking — it only clears the event, so a caller can
        close the gate without yielding to the event loop first.
        """
        self._admit_event.clear()

    def resume(self) -> None:
        """Reopen admission, releasing every caller blocked in ``track()``."""
        self._admit_event.set()

    @asynccontextmanager
    async def gate(self, timeout: float) -> AsyncIterator[bool]:
        """Close admission, drain, and reopen it around a runtime swap.

        Yields whether the drain completed within ``timeout`` (the caller
        decides whether to proceed, per the module policy). Admission stays
        closed for the whole ``async with`` body, so whatever the body rebinds
        is rebound before any request is admitted, and reopens in ``finally``
        even if the body raises or is cancelled. What the body covers is the
        caller's call: both routes put the swap in it, and reload-table keeps
        the build and the healthcheck out of it, where they delay only the swap.

        Both routes that move the runtime (``/admin/reload-table`` and
        ``/admin/rollback``) swap inside this gate. Taking ``drain()`` alone
        leaves the window it was supposed to close.
        """
        self.begin_drain()
        try:
            drained = await self.drain(timeout=timeout)
            yield drained
        finally:
            self.resume()

    @property
    def admitting(self) -> bool:
        """Whether new requests are being admitted. Read-only snapshot."""
        return self._admit_event.is_set()

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
