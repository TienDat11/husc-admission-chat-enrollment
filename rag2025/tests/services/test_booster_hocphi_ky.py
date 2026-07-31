"""TDD: BUG-2 (msg067) — học phí theo kỳ queries must surface the
"mỗi học kì" / "nộp học phí" chunk so the LLM has the per-kỳ payment
context.

Root cause: the relevant chunk
``chunked_24_chunk_315`` ("SV nộp học phí ở mỗi học kì theo đơn giá
tín chỉ...") used to be ranked inside the graph-retrieval top-k but
got pushed out after a graph rebuild. The booster is the safest
intervention: when a query combines "học phí" with a kỳ-specific
keyword, inject the relevant chunk. Mirrors BUG-1's pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# The chunk that contains "mỗi học kì" / "nộp học phí" / "tín chỉ"
# payment cadence. Defined in the graph (data/graph/entity_index.json)
# and re-emitted as a graphrag candidate.
PER_KY_CHUNK = "chunked_24_chunk_315"


QUERIES_TRIGGER = [
    "học phí đóng theo từng học kỳ thế nào",
    "SV nộp học phí ở mỗi học kì theo đơn giá tín chỉ",
    "học phí đóng theo tín chỉ thế nào",
    "mỗi học kì đóng học phí bao nhiêu",
    "học phí có đóng nhiều lần trong năm không",
    "học phí đóng mỗi kỳ bao nhiêu",
]


@pytest.mark.parametrize("query", QUERIES_TRIGGER)
def test_hocphi_perky_chunk_injected(query: str) -> None:
    """Booster must inject the per-kỳ payment chunk when the query
    combines 'học phí' with a kỳ-specific keyword (kỳ / kì / học kỳ
    / nhiều lần / theo tín chỉ)."""
    from services.aggregation_booster import detect_aggregation_chunks
    inject = detect_aggregation_chunks(query)
    assert PER_KY_CHUNK in inject, (
        f"BUG-2 regression: booster did not inject {PER_KY_CHUNK!r} "
        f"for query={query!r}; inject={inject!r}"
    )


def test_boost_with_aggregation_actually_injects_perky_chunk(monkeypatch) -> None:
    """End-to-end: ``boost_with_aggregation`` with a fake retriever
    must put the per-kỳ chunk into the returned docs."""
    from unittest.mock import MagicMock

    from services.aggregation_booster import boost_with_aggregation
    from services.lancedb_retrieval import RetrievedDocument

    chunk_text = (
        "SV nộp học phí ở mỗi học kì theo đơn giá tín chỉ của từng "
        "học phần đăng ký trong kỳ. HUSC không thu cả năm một lần."
    )

    def fetch_by_id(cid: str):
        if cid == PER_KY_CHUNK:
            return RetrievedDocument(
                text=chunk_text,
                source="fake_source",
                chunk_id=cid,
                metadata={"source_kind": "fake"},
                score=1.0,
                point_id=cid,
            )
        return None

    fake = MagicMock()
    fake.fetch_by_id.side_effect = fetch_by_id

    out = boost_with_aggregation(
        query="học phí đóng theo từng học kỳ thế nào",
        baseline_docs=[],
        lancedb_retriever=fake,
        top_k=5,
        max_inject=3,
    )
    injected_ids = [d.chunk_id for d in out]
    assert PER_KY_CHUNK in injected_ids, (
        f"BUG-2: boost_with_aggregation did not include "
        f"{PER_KY_CHUNK!r} in returned docs; ids={injected_ids!r}"
    )
    injected = next(d for d in out if d.chunk_id == PER_KY_CHUNK)
    assert (injected.metadata or {}).get("booster_injected") is True, (
        f"BUG-2: injected chunk missing booster_injected tag; "
        f"metadata={injected.metadata!r}"
    )