"""In-memory LRU cache for SmartQueryRouter results.

Caches step-back, HyDE doc, and classification per query. Useful when an
evaluation suite contains duplicate or rephrased queries, or in production
where popular queries repeat.

Phase E — Slice E.3.
Refined by Group 4 (G4-T2): cache key normalizes case + collapsed whitespace
ONLY — digits and diacritics are preserved so year and major-code
distinctions never alias (see the load-bearing "year collision" test in
``tests/services/test_router_cache.py``).
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Optional

from src.services.query_router import RouterResult

_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Normalize a query for cache-keying.

    - Strips leading/trailing whitespace.
    - Collapses internal whitespace runs to a single space.
    - Lowercases (case-fold).

    Deliberately DOES NOT fold digits, diacritics, or punctuation — folding
    digits would alias ``"điểm chuẩn 2025"`` to ``"điểm chuẩn 2026"`` and
    wrong-route every subsequent year. Folding diacritics would alias
    major-name variants. The spec requires year/major-code distinctions
    to be preserved.
    """
    if not query:
        return ""
    return _WS_RE.sub(" ", query.strip().lower())


class RouterCache:
    """Simple bounded LRU cache, keyed by SHA256 of normalized query.

    The hot read path is pure: no LLM, no breaker, no retry, no metrics
    side-effects beyond ``hits``/``misses`` counters.
    """

    def __init__(self, max_size: int = 256):
        self._cache: "OrderedDict[str, RouterResult]" = OrderedDict()
        self._max = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, query: str) -> str:
        return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()[:16]

    def get(self, query: str) -> Optional[RouterResult]:
        k = self._key(query)
        if k in self._cache:
            self._cache.move_to_end(k)
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, query: str, result: RouterResult) -> None:
        k = self._key(query)
        if k in self._cache:
            self._cache.move_to_end(k)
            self._cache[k] = result
            return
        self._cache[k] = result
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        rate = (self.hits / total) if total else 0.0
        return {
            "size": len(self._cache),
            "max": self._max,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": rate,
        }


_router_cache: Optional[RouterCache] = None


def get_router_cache() -> RouterCache:
    global _router_cache
    if _router_cache is None:
        _router_cache = RouterCache()
    return _router_cache
