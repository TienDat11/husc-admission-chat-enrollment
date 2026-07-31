"""TDD: BUG-1 — IELTS / TOEFL / VSTEP / chứng chỉ ngoại ngữ queries
must surface the quy-đổi table chunk ``husc_tb65_quydoi_nn_2026``.

Root cause: the existing aggregation rule
``[r"chứng\s*chỉ\s*ngoại\s*ngữ", r"ielts", r"toefl", r"toeic", ...]``
injects ``husc_thongbao_id65_2026`` (a policy chunk), NOT the actual
quy-đổi table chunk. As a result the LLM has the policy text but no
table — and ``husc_tb65_quydoi_nn_2026`` (the official IELTS 6.5 = 9.5
mapping) does NOT surface in top-k.

Fix: add a dedicated booster rule in ``_AGGREGATION_RULES`` whose
patterns target "quy đổi" + the language-cert keywords, and inject
``husc_tb65_quydoi_nn_2026``. Mirrors the existing injection pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


IELTS_QUYDOI_TABLE_CHUNK = "husc_tb65_quydoi_nn_2026"


# Real candidate queries a high-school student would type.
QUERIES_TRIGGER = [
    "quy đổi IELTS 6.5 thành mấy điểm xét tuyển",
    "IELTS 6.5 được quy đổi bao nhiêu điểm",
    "chứng chỉ TOEFL quy đổi điểm như thế nào",
    "VSTEP quy đổi IELTS 6.5 được mấy điểm",
    "quy đổi chứng chỉ ngoại ngữ IELTS TOEIC",
]


@pytest.mark.parametrize("query", QUERIES_TRIGGER)
def test_ielts_quydoi_table_injected(query: str) -> None:
    """Booster must inject the official quy-đổi table chunk
    ``husc_tb65_quydoi_nn_2026`` for queries that combine 'quy đổi'
    with one of the language cert keywords (IELTS / TOEFL / VSTEP /
    'chứng chỉ ngoại ngữ' / TOEIC)."""
    from services.aggregation_booster import detect_aggregation_chunks
    inject = detect_aggregation_chunks(query)
    assert IELTS_QUYDOI_TABLE_CHUNK in inject, (
        f"BUG-1 regression: booster did not inject {IELTS_QUYDOI_TABLE_CHUNK!r} "
        f"for query={query!r}; inject={inject!r}"
    )


def test_boost_with_aggregation_actually_injects_table_chunk(monkeypatch) -> None:
    """End-to-end: ``boost_with_aggregation`` with a fake retriever
    must put the table chunk into the returned docs for an IELTS-quy-đổi
    query — proving the wired-in path works, not just the detector."""
    from unittest.mock import MagicMock

    from services.aggregation_booster import boost_with_aggregation
    from services.lancedb_retrieval import RetrievedDocument

    table_text = (
        "Bảng quy đổi điểm: IELTS 6.5 = 9.5 thang 10; TOEFL iBT 60 = 8.5; "
        "VSTEP B1 = 8.0; TOEIC 600 = 8.0. Áp dụng cho năm 2026."
    )

    def fetch_by_id(cid: str):
        if cid == IELTS_QUYDOI_TABLE_CHUNK:
            return RetrievedDocument(
                text=table_text,
                source="fake_source",
                chunk_id=cid,
                metadata={"source_kind": "fake"},
                score=1.0,
                point_id=cid,
            )
        return None

    fake = MagicMock()
    fake.fetch_by_id.side_effect = fetch_by_id

    baseline = []
    out = boost_with_aggregation(
        query="quy đổi IELTS 6.5 thành mấy điểm xét tuyển",
        baseline_docs=baseline,
        lancedb_retriever=fake,
        top_k=5,
        max_inject=3,
    )
    injected_ids = [d.chunk_id for d in out]
    assert IELTS_QUYDOI_TABLE_CHUNK in injected_ids, (
        f"BUG-1: boost_with_aggregation did not include "
        f"{IELTS_QUYDOI_TABLE_CHUNK!r} in returned docs; ids={injected_ids!r}"
    )
    # The injected chunk must carry the booster_injected tag so the
    # precision-cutoff / fusion exempt-block keeps it.
    injected = next(d for d in out if d.chunk_id == IELTS_QUYDOI_TABLE_CHUNK)
    assert (injected.metadata or {}).get("booster_injected") is True, (
        f"BUG-1: injected table chunk missing booster_injected tag; "
        f"metadata={injected.metadata!r}"
    )