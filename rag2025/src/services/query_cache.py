from __future__ import annotations

import time
from typing import Any, Dict, Optional


class QueryCache:
    # Status payloads that must NEVER be cached: they are guard-rail
    # responses (out-of-scope / PII) that have already been emitted to
    # the user — re-serving from cache would bypass the original
    # rejection and cause cache poisoning.
    _POISONED_STATUSES = frozenset({"NOT_IN_HUSC_SCOPE", "SENSITIVE_PII_DETECTED"})

    def __init__(self, ttl_seconds: int = 900):
        self._ttl = max(1, ttl_seconds)
        self._store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        # Sửa #2: defense-in-depth — skip poisoned status responses so
        # guard-rail rejections never get cached. Defense lives here so
        # call sites can't accidentally poison the cache.
        if isinstance(value, dict):
            status = value.get("status_code")
            if status in self._POISONED_STATUSES:
                return
            if value.get("pii_detected") is True:
                return
        self._store[key] = (time.time() + self._ttl, value)

    def clear(self) -> None:
        self._store.clear()
