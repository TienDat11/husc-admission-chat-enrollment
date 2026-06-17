"""TDD: when a detected aggregate chunk_id is ALREADY present in
`baseline_docs`, `boost_with_aggregation` must RE-TAG that existing
baseline doc with `metadata["booster_injected"]=True` (NOT skip it).
This protects the doc from the GraphRAG fusion top-k truncation that
otherwise drops it in favor of a generic `tuyensinh_overview`.

Live bug evidence: "HUSC đào tạo những ngành nào" (detector matches
the first rule) used to produce a generic answer because the
`liet_ke_28_nganh_2026_v2` doc was present in the vector baseline and
the booster silently `continue`-skipped it. Re-tagging fixes that.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


def _make_fake_retriever(chunk_id_to_text: dict):
    from services.lancedb_retrieval import RetrievedDocument
    from unittest.mock import MagicMock

    def fetch_by_id(cid: str):
        if cid not in chunk_id_to_text:
            return None
        return RetrievedDocument(
            text=chunk_id_to_text[cid],
            source="fake_source",
            chunk_id=cid,
            metadata={"source_kind": "fake"},
            score=1.0,
            point_id=cid,
        )

    fake = MagicMock()
    fake.fetch_by_id.side_effect = fetch_by_id
    return fake


def _baseline_doc(chunk_id: str, text: str = "baseline body"):
    from services.lancedb_retrieval import RetrievedDocument
    return RetrievedDocument(
        text=text,
        source="baseline",
        chunk_id=chunk_id,
        metadata={"source_kind": "vector"},
        score=0.42,
    )


def test_aggregate_chunk_already_in_baseline_gets_retagged_not_skipped():
    """When the baseline already contains the full 28-major listing chunk,
    the booster must re-tag it with `booster_injected=True` so the GraphRAG
    fusion exempt-block keeps it. The fake retriever must NOT be asked
    to re-fetch it (it's already there), and the returned doc list must
    not contain duplicates of that chunk_id.
    """
    from services.aggregation_booster import boost_with_aggregation

    chunk_id_to_text = {
        "liet_ke_28_nganh_2026_v2": "Liệt kê 28 ngành năm 2026.",
        "tuyensinh_overview_2026_v2": "Tổng quan tuyển sinh 2026.",
        "bang_xep_hang_chi_tieu_2026_v2": "Bảng xếp hạng chỉ tiêu 2026.",
    }
    fake_retriever = _make_fake_retriever(chunk_id_to_text)

    # The 28-major listing is ALREADY in the baseline (this is the bug
    # scenario — vector retrieval returned it because of a phrasal match).
    baseline = [
        _baseline_doc("liet_ke_28_nganh_2026_v2", "28 majors baseline body"),
        _baseline_doc("vector_chunk_a", "vector body a"),
        _baseline_doc("vector_chunk_b", "vector body b"),
        _baseline_doc("vector_chunk_c", "vector body c"),
        _baseline_doc("vector_chunk_d", "vector body d"),
    ]

    result = boost_with_aggregation(
        query="HUSC đào tạo những ngành nào",
        baseline_docs=baseline,
        lancedb_retriever=fake_retriever,
        top_k=5,
        max_inject=2,
    )

    # 1) The aggregate chunk must still appear exactly once (no dup).
    liet_ke = [d for d in result if d.chunk_id == "liet_ke_28_nganh_2026_v2"]
    assert len(liet_ke) == 1, (
        f"Expected exactly one liet_ke doc, got {len(liet_ke)}: "
        f"{[d.chunk_id for d in result]}"
    )

    # 2) The single remaining doc MUST be re-tagged.
    assert liet_ke[0].metadata.get("booster_injected") is True, (
        f"Baseline liet_ke doc was NOT re-tagged: metadata={liet_ke[0].metadata!r}"
    )

    # 3) The other baseline docs MUST NOT be re-tagged (only the aggregate
    #    chunk the detector targeted gets the flag).
    for d in result:
        if d.chunk_id in {"vector_chunk_a", "vector_chunk_b", "vector_chunk_c", "vector_chunk_d"}:
            assert "booster_injected" not in d.metadata, (
                f"Non-aggregate baseline doc {d.chunk_id} unexpectedly tagged: "
                f"{d.metadata!r}"
            )


def test_aggregate_chunk_already_in_baseline_does_not_consume_max_inject():
    """Re-tagging a baseline aggregate doc must NOT spend a max_inject slot.
    The booster should still be able to inject the OTHER aggregate chunks
    that aren't in baseline. This is the whole point of the fix: the
    generic overview that used to be the sole survivor should also appear.
    """
    from services.aggregation_booster import boost_with_aggregation

    chunk_id_to_text = {
        "tuyensinh_overview_2026_v2": "Tổng quan tuyển sinh 2026.",
        "bang_xep_hang_chi_tieu_2026_v2": "Bảng xếp hạng chỉ tiêu 2026.",
    }
    fake_retriever = _make_fake_retriever(chunk_id_to_text)

    baseline = [
        _baseline_doc("liet_ke_28_nganh_2026_v2", "28 majors baseline body"),
        _baseline_doc("vector_chunk_a"),
        _baseline_doc("vector_chunk_b"),
    ]

    result = boost_with_aggregation(
        query="HUSC đào tạo những ngành nào",
        baseline_docs=baseline,
        lancedb_retriever=fake_retriever,
        top_k=5,
        max_inject=2,
    )

    # The re-tag consumed zero inject slots → both remaining aggregate
    # chunks should have been freshly fetched and injected.
    cids = {d.chunk_id for d in result}
    assert "liet_ke_28_nganh_2026_v2" in cids
    assert "tuyensinh_overview_2026_v2" in cids, (
        f"Expected tuyensinh_overview to be freshly injected; got cids={cids}"
    )
    assert "bang_xep_hang_chi_tieu_2026_v2" in cids, (
        f"Expected bang_xep_hang to be freshly injected; got cids={cids}"
    )

    # Re-tagged baseline doc keeps its baseline score/identity
    # (only metadata changes; text/source/score/chunk_id preserved).
    liet_ke = next(d for d in result if d.chunk_id == "liet_ke_28_nganh_2026_v2")
    assert liet_ke.text == "28 majors baseline body"
    assert liet_ke.metadata.get("booster_injected") is True
