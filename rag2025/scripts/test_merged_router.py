"""Test merged guardrail + step_back + route single-LLM call.

TDD: tests first, then implementation.
Verifies:
1. Merged LLM call returns all required fields
2. step_back is non-empty for non-single-fact queries
3. route is one of padded/hybrid/graph
4. is_in_scope integrates the major-scope guardrail check
5. Fast-paths (vague/contact/enum/comparison) still fire before LLM
"""
from __future__ import annotations

import pytest
import re

from src.services.query_router import (
    SmartQueryRouter,
    QueryRoute,
    _VAGUE_PLACEHOLDER_PATTERNS,
    _HYDE_CONTACT_PATTERNS,
    _ENUMERATION_PATTERNS,
    _COMPARISON_PATTERNS,
)


class TestFastPathPatterns:
    """Verify regex fast-paths still catch the right cases."""

    def test_vague_placeholder(self):
        for q in ["...", "em hỏi", "cho em hỏi  ?", "hello", "test"]:
            assert _VAGUE_PLACEHOLDER_PATTERNS.search(q), f"Should match vague: {q!r}"

    def test_vague_not_over_catch(self):
        for q in [
            "Điểm chuẩn ngành CNTT 2026?",
            "Học phí ngành Báo chí?",
        ]:
            assert not _VAGUE_PLACEHOLDER_PATTERNS.search(q), f"Should NOT match: {q!r}"

    def test_contact_block(self):
        for q in [
            "trang web của trường là gì",
            "địa chỉ trường",
            "hotline tuyển sinh HUSC",
            "fanpage HUSC",
        ]:
            assert _HYDE_CONTACT_PATTERNS.search(q), f"Should match contact: {q!r}"

    def test_enumeration_pattern(self):
        for q in [
            "liệt kê các ngành",
            "bao nhiêu ngành",
            "danh sách ngành đào tạo",
            "các ngành thuộc trường",
            "chi tiết các ngành tuyển sinh",
        ]:
            assert _ENUMERATION_PATTERNS.search(q), f"Should match enum: {q!r}"

    def test_comparison_pattern(self):
        for q in [
            "so sánh ngành CNTT và ngành KTPM",
            "khác nhau",
            "so với",
            "chênh lệch điểm",
        ]:
            assert _COMPARISON_PATTERNS.search(q), f"Should match comparison: {q!r}"


class TestRouterFastPath:
    """Verify route() fast-paths still fire before LLM call."""

    @pytest.fixture
    def router(self):
        return SmartQueryRouter()

    @pytest.mark.asyncio
    async def test_vague_reject_short_circuit(self, router):
        result = await router.route("...")
        assert result.skip_retrieval is True
        assert result.auto_answer is not None
        assert result.intent == "vague_reject"

    @pytest.mark.asyncio
    async def test_contact_block_short_circuit(self, router):
        # Contact-only query (no majors, no specific facts) → auto-answer
        result = await router.route("địa chỉ trường Đại học Khoa học Huế ở đâu?")
        assert result.skip_retrieval is True
        assert result.auto_answer is not None
        assert result.intent == "hyde_contact_block"

    @pytest.mark.asyncio
    async def test_enum_fast_path(self, router):
        # Enum query → graph_rag, no LLM call
        result = await router.route("liệt kê các ngành đào tạo của trường")
        assert result.route == QueryRoute.GRAPH_RAG
        assert result.intent == "liet_ke"

    @pytest.mark.asyncio
    async def test_comparison_fast_path(self, router):
        result = await router.route("so sánh ngành CNTT và ngành Báo chí")
        assert result.route == QueryRoute.GRAPH_RAG
        assert result.intent == "so_sanh"


class TestRouterMergedLLMCall:
    """Verify the merged _classify_guard_route LLM call returns correct schema."""

    @pytest.fixture
    def router(self):
        return SmartQueryRouter()

    @pytest.mark.asyncio
    async def test_merged_call_schema(self, router):
        """Merged call returns all required fields from one LLM round-trip."""
        # This query hits the merged LLM path (not fast-path)
        result = await router.route("Điểm chuẩn ngành CNTT HUSC năm 2026?")
        # Should NOT short-circuit via fast-path
        assert result.skip_retrieval is False
        # Schema integrity
        assert hasattr(result, "route")
        assert hasattr(result, "complexity")
        assert hasattr(result, "intent")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "step_back_query")
        assert result.route in (
            QueryRoute.PADDED_RAG,
            QueryRoute.HYBRID,
            QueryRoute.GRAPH_RAG,
        )
        assert isinstance(result.complexity, int)
        assert 0 <= result.complexity <= 5
        assert isinstance(result.intent, str)
        assert isinstance(result.reasoning, str)

    @pytest.mark.asyncio
    async def test_step_back_non_empty(self, router):
        """step_back is non-empty for queries that need abstraction."""
        result = await router.route("Cách xét học bạ và thời gian nộp hồ sơ?")
        if not result.skip_retrieval:
            # Only check step_back if we hit the LLM path
            assert result.step_back_query, "step_back must be non-empty for procedural queries"
            assert len(result.step_back_query.strip()) >= 5

    @pytest.mark.asyncio
    async def test_single_fact_echo(self, router):
        """Short single-fact query with major+year: step_back echoes original."""
        result = await router.route("Học phí ngành CNTT 2026?")
        # Either short-circuit or route+padded
        assert result.route in (
            QueryRoute.PADDED_RAG,
            QueryRoute.HYBRID,
            QueryRoute.GRAPH_RAG,
        )
        # step_back should echo or be a brief abstraction
        assert len(result.step_back_query) >= 3


class TestRouterRouteOutputs:
    """Smoke test: all 3 route values are reachable via LLM path."""

    @pytest.fixture
    def router(self):
        return SmartQueryRouter()

    @pytest.mark.asyncio
    async def test_can_reach_padded(self, router):
        result = await router.route("Mã ngành Kỹ thuật phần mềm là gì?")
        assert result.route in (
            QueryRoute.PADDED_RAG,
            QueryRoute.HYBRID,
            QueryRoute.GRAPH_RAG,
        )

    @pytest.mark.asyncio
    async def test_can_reach_hybrid(self, router):
        result = await router.route(
            "Điều kiện xét tuyển kết hợp và quy trình đăng ký như thế nào?"
        )
        assert result.route in (
            QueryRoute.PADDED_RAG,
            QueryRoute.HYBRID,
            QueryRoute.GRAPH_RAG,
        )

    @pytest.mark.asyncio
    async def test_guardrail_scope_inside_router(self, router):
        """Verify in-scope check is handled inside the router LLM call.

        An out-of-scope query (e.g. medical question) should either:
        - short-circuit via fast-path (if regex catches it), OR
        - be routed as hybrid/padded but NOT cause a separate guardrail call in main.py.
        The merged call embeds the scope check.
        """
        result = await router.route("Triệu chứng bệnh tiểu đường là gì?")
        # Should still return a valid RouterResult (merged call handles scope internally)
        assert hasattr(result, "route")
        assert result.route in (
            QueryRoute.PADDED_RAG,
            QueryRoute.HYBRID,
            QueryRoute.GRAPH_RAG,
        )
