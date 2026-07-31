"""tests/services/test_query_cache_poisoning.py

Regression test for Fix #2 — QueryCache.set must skip poisoned payloads.

If the cache stored guard-rail rejection responses (NOT_IN_HUSC_SCOPE,
SENSITIVE_PII_DETECTED, or pii_detected=True), a follow-up benign query
hitting the same cache key would get back the rejected answer — bypass
of guard rails = cache poisoning.
"""
from __future__ import annotations

import pytest

from src.services.query_cache import QueryCache


class TestQueryCachePoisoning:
    def test_oos_status_not_stored(self):
        cache = QueryCache(ttl_seconds=60)
        payload = {"status_code": "NOT_IN_HUSC_SCOPE", "answer": "X"}
        cache.set("k1", payload)
        assert cache.get("k1") is None

    def test_pii_status_not_stored(self):
        cache = QueryCache(ttl_seconds=60)
        payload = {"status_code": "SENSITIVE_PII_DETECTED", "answer": "X"}
        cache.set("k1", payload)
        assert cache.get("k1") is None

    def test_pii_detected_flag_not_stored(self):
        cache = QueryCache(ttl_seconds=60)
        payload = {"answer": "X", "pii_detected": True}
        cache.set("k1", payload)
        assert cache.get("k1") is None

    def test_clean_payload_still_stored(self):
        cache = QueryCache(ttl_seconds=60)
        payload = {"status_code": "OK", "answer": "Hello"}
        cache.set("k1", payload)
        assert cache.get("k1") == payload

    def test_non_dict_payload_stored(self):
        """Plain values (lists, scalars) must still cache as before."""
        cache = QueryCache(ttl_seconds=60)
        cache.set("k1", [1, 2, 3])
        assert cache.get("k1") == [1, 2, 3]

    def test_dict_without_status_stored(self):
        cache = QueryCache(ttl_seconds=60)
        cache.set("k1", {"answer": "X"})
        assert cache.get("k1") == {"answer": "X"}


class TestQueryCacheTTL:
    """Negative/zero TTL must be clamped to at least 1 second."""

    def test_negative_ttl_clamped(self):
        cache = QueryCache(ttl_seconds=-5)
        assert cache._ttl >= 1

    def test_zero_ttl_clamped(self):
        cache = QueryCache(ttl_seconds=0)
        assert cache._ttl >= 1

    def test_valid_ttl_preserved(self):
        cache = QueryCache(ttl_seconds=300)
        assert cache._ttl == 300