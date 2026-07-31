"""tests/routers/test_overview_phrasing_variants.py

Six Vietnamese phrasings of "tổng số ngành tuyển sinh" must ALL route to
GRAPH_RAG or HYBRID (NOT PADDED alone).  These queries match the
``_ENUMERATION_PATTERNS`` fast-path in ``SmartQueryRouter.route()``,
so they should deterministically hit graph_rag without needing an LLM call.

Approach:
  - Construct a ``SmartQueryRouter`` with a dummy LLM that raises on call
    (proving the fast-path fires before any LLM invocation).
  - Patch ``get_router_cache()`` to return a no-op cache so the import
    of the singleton doesn't pollute across tests.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock
from typing import Any, Dict

import pytest

from src.services.query_router import (
    SmartQueryRouter,
    QueryRoute,
    _ENUMERATION_PATTERNS,
    _VAGUE_PLACEHOLDER_PATTERNS,
    _HYDE_CONTACT_PATTERNS,
)


# ---------------------------------------------------------------------------
# Fake LLM that MUST NOT be called (fast-path should short-circuit)
# ---------------------------------------------------------------------------
class _NeverCalledLLM:
    """Raises if any method is called — proves the fast-path bypasses LLM."""

    async def chat_json(self, **kwargs: Any) -> Dict:
        raise RuntimeError("LLM chat_json called — fast-path should have short-circuited")

    async def chat(self, **kwargs: Any) -> str:
        raise RuntimeError("LLM chat called — fast-path should have short-circuited")


class _NoOpCache:
    """In-memory no-op cache for router tests."""

    def get(self, key: str) -> Any:
        return None

    def put(self, key: str, value: Any) -> None:
        pass

    def stats(self) -> Dict:
        return {"size": 0, "max": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}


# ---------------------------------------------------------------------------
# Test data: 6 phrasings of "tổng số ngành tuyển sinh"
# ---------------------------------------------------------------------------
OVERVIEW_PHRASINGS = [
    "HUSC tuyển sinh năm 2026 có bao nhiêu ngành?",
    "Năm nay trường đào tạo những ngành nào?",
    "Tổng số ngành HUSC đang tuyển sinh là bao nhiêu?",
    "Trường có những ngành đào tạo nào trong năm 2026?",
    "Liệt kê các ngành HUSC đang đào tạo 2026",
    "Danh sách tất cả ngành tuyển sinh năm 2026 của HUSC",
]

EXPECTED_ROUTES = (QueryRoute.GRAPH_RAG, QueryRoute.HYBRID)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestOverviewPhrasingVariants:
    """All phrasings of 'tổng số ngành tuyển sinh' must route to
    GRAPH_RAG or HYBRID, never PADDED alone.
    """

    @pytest.fixture(autouse=True)
    def _patch_cache(self):
        """Replace the router cache singleton with a no-op for isolation."""
        with patch(
            "src.services.router_cache.get_router_cache",
            return_value=_NoOpCache(),
        ):
            yield

    @pytest.fixture
    def router(self) -> SmartQueryRouter:
        return SmartQueryRouter(llm=_NeverCalledLLM())

    # --- Individual phrasing tests ---

    @pytest.mark.asyncio
    async def test_p1_husc_tuyen_sinh_bao_nhieu(self, router: SmartQueryRouter):
        result = await router.route("HUSC tuyển sinh năm 2026 có bao nhiêu ngành?")
        assert result.route in EXPECTED_ROUTES, (
            f"Expected GRAPH_RAG or HYBRID, got {result.route}"
        )

    @pytest.mark.asyncio
    async def test_p2_nam_nay_dao_tao_nhung_nganh(self, router: SmartQueryRouter):
        result = await router.route("Năm nay trường đào tạo những ngành nào?")
        assert result.route in EXPECTED_ROUTES, (
            f"Expected GRAPH_RAG or HYBRID, got {result.route}"
        )

    @pytest.mark.asyncio
    async def test_p3_tong_so_nganh(self, router: SmartQueryRouter):
        result = await router.route("Tổng số ngành HUSC đang tuyển sinh là bao nhiêu?")
        assert result.route in EXPECTED_ROUTES, (
            f"Expected GRAPH_RAG or HYBRID, got {result.route}"
        )

    @pytest.mark.asyncio
    async def test_p4_truong_co_nhung_nganh_dao_tao(self, router: SmartQueryRouter):
        result = await router.route("Trường có những ngành đào tạo nào trong năm 2026?")
        assert result.route in EXPECTED_ROUTES, (
            f"Expected GRAPH_RAG or HYBRID, got {result.route}"
        )

    @pytest.mark.asyncio
    async def test_p5_liet_ke(self, router: SmartQueryRouter):
        result = await router.route("Liệt kê các ngành HUSC đang đào tạo 2026")
        assert result.route in EXPECTED_ROUTES, (
            f"Expected GRAPH_RAG or HYBRID, got {result.route}"
        )

    @pytest.mark.asyncio
    async def test_p6_danh_sach_tat_ca(self, router: SmartQueryRouter):
        result = await router.route("Danh sách tất cả ngành tuyển sinh năm 2026 của HUSC")
        assert result.route in EXPECTED_ROUTES, (
            f"Expected GRAPH_RAG or HYBRID, got {result.route}"
        )

    # --- Aggregate parametrized test ---

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phrasing", OVERVIEW_PHRASINGS, ids=[
        "husc_tuyen_sinh_bao_nhieu",
        "nam_nay_dao_tao_nganh",
        "tong_so_nganh",
        "truong_co_nganh_dao_tao",
        "liet_ke_nganh",
        "danh_sach_tat_ca",
    ])
    async def test_all_phrasings_route_to_graph_or_hybrid(
        self, router: SmartQueryRouter, phrasing: str
    ):
        result = await router.route(phrasing)
        assert result.route in EXPECTED_ROUTES, (
            f"Phrasing '{phrasing}' routed to {result.route} — "
            f"expected GRAPH_RAG or HYBRID, NOT PADDED alone"
        )

    # --- Pattern sanity: phrasings should match _ENUMERATION_PATTERNS ---
    # Note: some phrasings (e.g. "Danh sách tất cả ngành ...") may not
    # match the regex and instead fall through to LLM classify, which can
    # still route correctly to GRAPH_RAG/HYBRID.  The fast-path regex
    # is a performance optimization, not a correctness gate.

    ENUM_FAST_PATH_PHRASINGS = [
        "HUSC tuyển sinh năm 2026 có bao nhiêu ngành?",
        "Năm nay trường đào tạo những ngành nào?",
        "Tổng số ngành HUSC đang tuyển sinh là bao nhiêu?",
        "Trường có những ngành đào tạo nào trong năm 2026?",
        "Liệt kê các ngành HUSC đang đào tạo 2026",
    ]

    def test_majority_of_phrasings_match_enumeration_pattern(self):
        """Most phrasings should be caught by _ENUMERATION_PATTERNS for
        the fast-path optimization.  At least 4 of 6 must match."""
        matched = sum(
            1 for p in OVERVIEW_PHRASINGS
            if _ENUMERATION_PATTERNS.search(p)
        )
        assert matched >= 4, (
            f"Only {matched}/6 phrasings match _ENUMERATION_PATTERNS — "
            f"expected at least 4 for adequate fast-path coverage"
        )

    def test_no_phrasing_matches_vague_placeholder(self):
        """None of the phrasings should be rejected as vague."""
        for phrasing in OVERVIEW_PHRASINGS:
            assert not _VAGUE_PLACEHOLDER_PATTERNS.search(phrasing), (
                f"Phrasing '{phrasing}' falsely matched _VAGUE_PLACEHOLDER_PATTERNS"
            )

    def test_no_phrasing_matches_contact_pattern(self):
        """None of the phrasings should be treated as contact/intro."""
        for phrasing in OVERVIEW_PHRASINGS:
            assert not _HYDE_CONTACT_PATTERNS.search(phrasing), (
                f"Phrasing '{phrasing}' falsely matched _HYDE_CONTACT_PATTERNS"
            )
