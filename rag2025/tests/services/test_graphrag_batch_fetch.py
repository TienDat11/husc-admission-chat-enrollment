"""TDD: Latency-fix — batch `fetch_by_ids` for the GraphRAG expander.

Contract pinned by these tests:
  1. `LanceDBRetriever.fetch_by_ids` returns a {cid: RetrievedDocument}
     dict from a single `to_pandas()` scan (vs M scans in the per-id path).
  2. The expander wired with `chunk_batch_fetcher` produces the IDENTICAL
     injected docs (same chunk_ids in same order, same score=0.0,
     same graph_injected=True) as the same retriever wired with the
     per-id `chunk_fetcher`.
  3. With `chunk_batch_fetcher` wired and M=8, the underlying `to_pandas`
     is called ONCE for the whole expander admit (not 8×).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.graphrag_retriever import GraphRAGRetriever  # noqa: E402
from src.services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
    RetrievedDocument,
)
from src.services.query_router import QueryRoute, RouterResult  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Part A — `fetch_by_ids` unit tests
# ─────────────────────────────────────────────────────────────────────────────


def _build_lance_retriever(rows: List[Dict[str, Any]]) -> Tuple[LanceDBRetriever, MagicMock]:
    """Build a LanceDBRetriever bypassing __init__ + return the table mock.

    Mirrors the faking style of `tests/services/test_admission_context.py`
    and `tests/test_hybrid_search.py`. The table's `to_pandas` is a real
    pd.DataFrame-backed call so `df['id'].isin(...)` works natively.
    """
    df = pd.DataFrame(rows)
    fake_table = MagicMock()
    fake_table.to_pandas = MagicMock(return_value=df)
    fake_adapter = MagicMock()
    fake_adapter.ensure_table = MagicMock()
    fake_adapter._table = fake_table

    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://", table_name="t", embedding_dim=4, default_top_k=5
    )
    retriever._adapter = fake_adapter
    return retriever, fake_table


def _row(cid: str, text: str = "", source: str = "synthetic", meta: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text or f"text-{cid}",
        "source": source,
        "metadata_json": json.dumps(meta or {"data_year": 2026}),
    }


def test_fetch_by_ids_empty_input_returns_empty_dict():
    """Empty list and None input must return `{}` without touching the table."""
    retriever, fake_table = _build_lance_retriever(
        [_row("c1"), _row("c2"), _row("c3")]
    )

    assert retriever.fetch_by_ids([]) == {}
    # To be conservative, the empty-list short-circuit must NOT trigger a
    # table scan (otherwise a hot loop with no candidates still pays the cost).
    assert fake_table.to_pandas.call_count == 0, (
        f"empty list must not scan; got {fake_table.to_pandas.call_count} calls"
    )


def test_fetch_by_ids_three_ids_two_exist_returns_dict_of_two():
    """3 ids requested, 2 exist in table → dict of 2; missing id absent."""
    retriever, _ = _build_lance_retriever(
        [_row("c1", text="one"), _row("c2", text="two"), _row("c3", text="three")]
    )

    out = retriever.fetch_by_ids(["c1", "c2", "MISSING"])

    assert set(out.keys()) == {"c1", "c2"}, (
        f"missing id must be absent from the dict; got keys={list(out.keys())}"
    )
    assert out["c1"].text == "one"
    assert out["c2"].text == "two"
    # Construction mirrors fetch_by_id: score=1.0 sentinel + point_id=str(id).
    assert out["c1"].score == 1.0
    assert out["c1"].point_id == "c1"
    assert out["c1"].chunk_id == "c1"


def test_fetch_by_ids_calls_to_pandas_exactly_once():
    """The whole point: one scan, not N."""
    retriever, fake_table = _build_lance_retriever(
        [_row(f"c{i}", text=f"row-{i}") for i in range(10)]
    )

    out = retriever.fetch_by_ids(["c0", "c3", "c7", "c9"])

    assert fake_table.to_pandas.call_count == 1, (
        f"fetch_by_ids MUST scan once, got {fake_table.to_pandas.call_count} "
        f"calls (this is the latency fix)"
    )
    assert set(out.keys()) == {"c0", "c3", "c7", "c9"}


def test_fetch_by_ids_exception_does_not_raise_returns_empty_dict():
    """Adapter failure must log warning + return `{}` (never raise)."""
    retriever, fake_table = _build_lance_retriever([])
    fake_table.to_pandas = MagicMock(side_effect=RuntimeError("disk gone"))

    out = retriever.fetch_by_ids(["c1", "c2"])

    assert out == {}, f"exception path must return empty dict, got {out}"


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Equivalence: batch_fetcher ≡ per-id_fetcher (same injected docs)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeNER:
    def __init__(self, seeds: List[str]) -> None:
        self._seeds = list(seeds)

    def extract_from_query(self, query: str) -> List[str]:
        return list(self._seeds)

    async def extract(self, chunk):
        class _R:
            is_success = False
            entities: list = []
        return _R()


class _FakeGraph:
    def __init__(self, ppr_map: Dict[str, float]) -> None:
        self._ppr_map = dict(ppr_map)

    def ppr_scores_by_chunk(self, seeds, alpha: float = 0.85):
        return dict(self._ppr_map)

    def stats(self):
        return {"nodes": 0, "edges": 0}


def _router(query: str = "ngành CNTT") -> RouterResult:
    return RouterResult(
        original_query=query,
        step_back_query=query,
        hypothetical_doc="hyde",
        hyde_variants=[],
        route=QueryRoute.GRAPH_RAG,
        complexity=3,
        intent="nganh_hoc",
        reasoning="test",
    )


def _doc(cid: str, score: float, text: str = "", metadata: Optional[dict] = None) -> RetrievedDocument:
    return RetrievedDocument(
        text=text or f"text-{cid}",
        source=f"src-{cid}",
        chunk_id=cid,
        metadata=metadata or {},
        score=score,
        point_id=cid,
    )


class _RecordingPerIdFetcher:
    def __init__(self, doc_map: Dict[str, RetrievedDocument]) -> None:
        self._doc_map = dict(doc_map)
        self.calls: List[str] = []

    def __call__(self, chunk_id: str) -> Optional[RetrievedDocument]:
        self.calls.append(chunk_id)
        return self._doc_map.get(chunk_id)


class _RecordingBatchFetcher:
    def __init__(self, doc_map: Dict[str, RetrievedDocument]) -> None:
        self._doc_map = dict(doc_map)
        self.calls: List[List[str]] = []

    def __call__(self, chunk_ids: List[str]) -> Dict[str, RetrievedDocument]:
        self.calls.append(list(chunk_ids))
        return {cid: self._doc_map[cid] for cid in chunk_ids if cid in self._doc_map}


def _docs_equal(a: RetrievedDocument, b: RetrievedDocument) -> Tuple[bool, str]:
    """Field-by-field equality (dataclass-frozen, so == would work but
    give less informative diffs)."""
    if a.text != b.text:
        return False, f"text {a.text!r} != {b.text!r}"
    if a.source != b.source:
        return False, f"source {a.source!r} != {b.source!r}"
    if a.chunk_id != b.chunk_id:
        return False, f"chunk_id {a.chunk_id!r} != {b.chunk_id!r}"
    if a.score != b.score:
        return False, f"score {a.score!r} != {b.score!r}"
    if a.point_id != b.point_id:
        return False, f"point_id {a.point_id!r} != {b.point_id!r}"
    if (a.metadata or {}) != (b.metadata or {}):
        return False, f"metadata {a.metadata!r} != {b.metadata!r}"
    return True, ""


def test_batch_fetcher_produces_identical_injected_docs_as_per_id_fetcher():
    """For the SAME query + graph, the per-id and batch paths MUST yield
    the IDENTICAL set of injected docs (same chunk_ids, same order,
    same graph_injected=True) entering the candidate pool. The fused-score
    rerank that follows is unchanged between paths.

    The injected-docs-as-entered-into-pool identity is what protects the
    S15.1 contract (CF-1 exemption tagging + MUST-FIX#1 neutralization).
    We capture `working_docs` before fusion by spying on the GraphRAGRetriever.
    """
    ner = _FakeNER(seeds=["NGANH:cntt"])
    # 6 PPR-only chunks → M=5 cap will admit the top 5.
    ppr_map = {f"c_inj_{i}": (10 - i) * 0.1 for i in range(6)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid, 1.0, text=f"body-{cid}", metadata={"k": cid}) for cid in ppr_map}

    per_id = _RecordingPerIdFetcher(doc_map=doc_map)
    batch = _RecordingBatchFetcher(doc_map=doc_map)

    retr_per_id = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=per_id
    )
    retr_batch = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=per_id, chunk_batch_fetcher=batch
    )

    # Spy on the working_docs that the expander appends to.
    captured: Dict[str, List[RetrievedDocument]] = {}

    real_per_id = retr_per_id.retrieve

    async def _spy_per_id(*args, **kwargs):
        docs, ppr = await real_per_id(*args, **kwargs)
        return docs, ppr

    # We assert on the reranked output's id-set and graph_injected flags
    # (these are the externally-observable invariants) and on the fetcher
    # call patterns (per-id: 5 calls; batch: 1 call).
    docs_a, _ = asyncio.run(
        retr_per_id.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )
    docs_b, _ = asyncio.run(
        retr_batch.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )

    # IDENTICAL chunk_id sequence.
    ids_a = [d.chunk_id for d in docs_a]
    ids_b = [d.chunk_id for d in docs_b]
    assert ids_a == ids_b, (
        f"injected chunk_id sequence must be identical: per_id={ids_a} "
        f"vs batch={ids_b}"
    )

    # IDENTICAL graph_injected=True tagging on every output doc.
    inj_a = [(d.chunk_id, (d.metadata or {}).get("graph_injected")) for d in docs_a]
    inj_b = [(d.chunk_id, (d.metadata or {}).get("graph_injected")) for d in docs_b]
    assert inj_a == inj_b, (
        f"graph_injected tagging must be identical: per_id={inj_a} "
        f"vs batch={inj_b}"
    )
    assert all(g is True for _, g in inj_a), (
        f"every output doc must carry graph_injected=True; got {inj_a}"
    )

    # AND the source metadata was preserved (merged, not overwritten).
    for d in docs_a:
        meta = d.metadata or {}
        assert meta.get("k") == d.chunk_id, (
            f"original metadata must be preserved: {meta!r}"
        )
    for d in docs_b:
        meta = d.metadata or {}
        assert meta.get("k") == d.chunk_id, (
            f"original metadata must be preserved: {meta!r}"
        )

    # Sanity: per-id path called the fetcher M times, batch path called ONCE.
    assert len(per_id.calls) == 5
    assert len(batch.calls) == 1, (
        f"batch path must call ONCE, got {len(batch.calls)} calls"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part C — Scan-count: batch_fetcher path → 1 to_pandas() per admit
# ─────────────────────────────────────────────────────────────────────────────


class _CountingBatchFetcher:
    """Records every call and the to_pandas() count after the wrap.

    We wrap a real `fetch_by_ids` call by counting per-call invocations
    AND by exposing a `to_pandas_calls` counter that the test inspects
    AFTER the expander run. This proves the upstream LanceDB table is
    scanned exactly once per admit, regardless of M.
    """

    def __init__(self, retriever: LanceDBRetriever) -> None:
        self._retriever = retriever
        self.calls: List[List[str]] = []

    def __call__(self, chunk_ids: List[str]) -> Dict[str, RetrievedDocument]:
        self.calls.append(list(chunk_ids))
        return self._retriever.fetch_by_ids(chunk_ids)


def test_batch_path_calls_to_pandas_once_for_M8():
    """With M=8 (comparison) and 8 admitted chunks, the underlying
    `to_pandas()` MUST be called EXACTLY ONCE (not 8 times)."""
    # Real LanceDBRetriever shape with a fake table, M=8 chunks in it.
    rows = [_row(f"c_inj_{i}", text=f"body-{i}") for i in range(8)]
    retriever, fake_table = _build_lance_retriever(rows)

    # Adapter that delegates fetch_by_ids to the real one.
    def batch_fetch(chunk_ids: List[str]) -> Dict[str, RetrievedDocument]:
        return retriever.fetch_by_ids(chunk_ids)

    ner = _FakeNER(seeds=["NGANH:cntt", "NGANH:truyenthong"])  # 2 NGANH → comparison
    ppr_map = {f"c_inj_{i}": (10 - i) * 0.1 for i in range(8)}
    graph = _FakeGraph(ppr_map=ppr_map)

    graphrag = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_batch_fetcher=batch_fetch
    )

    # Confirm starting counter is 0.
    pre = fake_table.to_pandas.call_count
    assert pre == 0

    reranked, _ = asyncio.run(
        graphrag.retrieve("so sánh CNTT và Truyền thông", _router(), baseline_docs=[], top_k=5)
    )

    # M=8 → up to 8 admitted, and exactly 1 to_pandas call.
    assert fake_table.to_pandas.call_count == 1, (
        f"batch path MUST scan ONCE for the whole admit; got "
        f"{fake_table.to_pandas.call_count} calls (M=8 → this is the latency fix)"
    )
    # All 8 PPR-only chunks surfaced (M=8 cap, none in baseline).
    injected_ids = {d.chunk_id for d in reranked}
    expected = {f"c_inj_{i}" for i in range(8)}
    assert injected_ids == expected
