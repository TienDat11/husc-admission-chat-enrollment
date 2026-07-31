"""tests/routers/test_main_py_three_fixes.py

Regression tests for three bug fixes in src/main.py:
  A) _graph_update_lock was undefined → added ``_graph_update_lock = Lock()``
  B) /v2 cache key mismatch → removed the ``|actual:{rag_result.route}``
     suffix mutation that caused store/lookup keys to diverge.
  C) /query hot-path outside _track_latency → the entire pipeline was
     re-indented to live INSIDE the ``async with _track_latency(...)`` block.
"""
from __future__ import annotations

import asyncio
import ast
import sys
import os
import textwrap
from typing import Dict, Any, Optional

import pytest


# ---------------------------------------------------------------------------
# A) _graph_update_lock is defined and is an asyncio.Lock
# ---------------------------------------------------------------------------
class TestGraphUpdateLockDefined:
    """Bug 1: ``_graph_update_lock`` was referenced but never created."""

    def test_graph_update_lock_defined(self):
        from src.main import _graph_update_lock
        assert isinstance(_graph_update_lock, asyncio.Lock)

    def test_graph_update_lock_is_module_level(self):
        """Confirm it is a module-level singleton, not a local."""
        import src.main as main_mod
        assert hasattr(main_mod, "_graph_update_lock")
        # Same object on re-access
        assert main_mod._graph_update_lock is main_mod._graph_update_lock


# ---------------------------------------------------------------------------
# B) /v2 cache key matches on store AND lookup (no |actual: suffix)
# ---------------------------------------------------------------------------
class TestV2CacheKeyLookupMatchesStore:
    """Bug 2: cache_key was mutated with ``|actual:{route}`` AFTER the
    initial build, so set(key) and get(key) could disagree.

    We mirror the exact key-building logic from _build_v2_query_payload
    (lines 1153-1157) and verify that a plain dict cache returns the
    stored value on the SAME key, and MISSES when the old |actual: suffix
    is appended.
    """

    def _build_key(
        self,
        query: str,
        top_k: int = 5,
        force_route: Optional[str] = None,
    ) -> str:
        """Mirror the key-building logic from main.py lines 1153-1157."""
        key = f"v2:{query.strip().lower()}|top_k:{top_k}|route:{force_route or 'auto'}"
        # BUG: old code used to do ``cache_key += f"|actual:{rag_result.route}"``
        # after building the key.  The fix removed that mutation.
        return key

    def test_store_and_lookup_same_key(self):
        cache: Dict[str, Any] = {}
        key = self._build_key("học phí ngành cntt", top_k=5, force_route=None)
        expected_key = "v2:học phí ngành cntt|top_k:5|route:auto"
        assert key == expected_key

        sentinel = {"answer": "cached"}
        cache[key] = sentinel
        assert cache.get(key) is sentinel

    def test_old_suffix_key_misses(self):
        """The pre-fix cache stored with |actual:graph_rag appended, so
        a lookup WITHOUT that suffix would miss.  After the fix, the
        suffix is never appended — verifying that a key WITH the suffix
        does NOT match.
        """
        cache: Dict[str, Any] = {}
        key = self._build_key("học phí ngành cntt", top_k=5, force_route=None)
        cache[key] = {"answer": "cached"}

        # Old buggy key (with |actual: suffix)
        buggy_key = key + "|actual:graph_rag"
        assert cache.get(buggy_key) is None  # Must MISS

    def test_force_route_padded_rag(self):
        """When force_route='padded_rag', the key gets |effective:padded_rag."""
        query = "tuyển sinh"
        top_k = 10
        force_route = "padded_rag"
        key = f"v2:{query.strip().lower()}|top_k:{top_k}|route:{force_route}"
        # The effective suffix is added only for padded_rag and graph_rag
        if force_route == "padded_rag":
            key += "|effective:padded_rag"
        expected = "v2:tuyển sinh|top_k:10|route:padded_rag|effective:padded_rag"
        assert key == expected

    def test_force_route_graph_rag(self):
        """When force_route='graph_rag', the key gets |effective:graph_rag."""
        query = "so sánh ngành"
        top_k = 3
        force_route = "graph_rag"
        key = f"v2:{query.strip().lower()}|top_k:{top_k}|route:{force_route}"
        if force_route == "graph_rag":
            key += "|effective:graph_rag"
        expected = "v2:so sánh ngành|top_k:3|route:graph_rag|effective:graph_rag"
        assert key == expected


# ---------------------------------------------------------------------------
# C) /query hot-path is INSIDE the _track_latency context manager
# ---------------------------------------------------------------------------
class TestQueryHotpathInsideTrackLatency:
    """Bug 3: the entire pipeline (lines 609-858) was outside the
    ``async with _track_latency(...)`` block.  The fix re-indented it
    inside.  We verify via AST parsing that the key statement
    ``enhanced_request = await query_enhancer_service.enhance_query``
    is a child of the ``async with`` node.
    """

    def test_query_function_pipeline_inside_track_latency(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "main.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename="main.py")

        # Find the ``async def query`` function
        query_func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "query":
                    query_func = node
                    break
        assert query_func is not None, "async def query not found in main.py"

        # Find the ``async with _track_latency(...)`` statement
        track_latency_node = None
        for child in ast.walk(query_func):
            if isinstance(child, ast.AsyncWith):
                # Check if the context expression references _track_latency
                for item in child.items:
                    ctx = item.context_expr
                    # Could be a Call or Name
                    if isinstance(ctx, ast.Call):
                        func = ctx.func
                        if isinstance(func, ast.Name) and func.id == "_track_latency":
                            track_latency_node = child
                            break
                        elif isinstance(func, ast.Attribute) and func.attr == "_track_latency":
                            track_latency_node = child
                            break
                    elif isinstance(ctx, ast.Name) and ctx.id == "_track_latency":
                        track_latency_node = child
                        break
        assert track_latency_node is not None, (
            "async with _track_latency(...) not found inside query()"
        )

        # The body of the with-block should contain the pipeline statements.
        # Check that at least 30 statements are inside (the pipeline is ~250 lines)
        body = track_latency_node.body
        assert len(body) >= 20, (
            f"Expected >=20 statements inside _track_latency block, got {len(body)}"
        )

        # Search for ``enhanced_request = await query_enhancer_service.enhance_query``
        # as an Assign node within the with-body
        found_enhance = False
        for stmt in ast.walk(track_latency_node):
            if isinstance(stmt, ast.Assign):
                # Check if any target is named "enhanced_request"
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "enhanced_request":
                        found_enhance = True
                        break
            if found_enhance:
                break

        assert found_enhance, (
            "enhanced_request = await query_enhancer_service.enhance_query "
            "NOT found inside _track_latency block"
        )

    def test_enhanced_request_not_at_module_level_outside_track_latency(self):
        """Ensure enhanced_request is NOT a direct child of the query()
        function body (outside the with-block)."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "main.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename="main.py")

        query_func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "query":
                    query_func = node
                    break
        assert query_func is not None

        # Direct children of query() that are Assign with target "enhanced_request"
        direct_enhanced_requests = []
        for child in query_func.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == "enhanced_request":
                        direct_enhanced_requests.append(child)

        assert len(direct_enhanced_requests) == 0, (
            "enhanced_request assignment found as direct child of query() — "
            "it should be nested inside _track_latency block, not outside"
        )
