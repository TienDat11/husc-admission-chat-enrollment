# @spec(S15.5 / CF-2) test_booster_tag — aggregation_booster tags injected docs
"""TDD: CF-2 — every doc injected by `boost_with_aggregation` MUST carry
`metadata["booster_injected"] = True`. The flag is consumed by the
S15.5 precision-cutoff (a separate module, Lane A) to EXEMPT booster
chunks from the relative rerank-score drop.

Frozen-dataclass note: `RetrievedDocument` is `@dataclass(frozen=True)`, so
the booster cannot mutate `doc.metadata` in place — it reconstructs the
doc with merged metadata and preserves the score from `fetch_by_id`
(booster docs keep their injected priority; only EXPANDER chunks get
the 0.0 base-score treatment per ADR-A MUST-FIX #1).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_fake_retriever(chunk_id_to_text: dict):
    """Build a fake lancedb_retriever.

    `fetch_by_id(cid)` returns a `RetrievedDocument` with the given text
    and an empty metadata dict (the real one would carry `metadata_json`
    from the LanceDB row). The booster is responsible for layering on
    `booster_injected=True` itself.
    """
    from services.lancedb_retrieval import RetrievedDocument

    def fetch_by_id(cid: str):
        if cid not in chunk_id_to_text:
            return None
        return RetrievedDocument(
            text=chunk_id_to_text[cid],
            source="fake_source",
            chunk_id=cid,
            metadata={"source_kind": "fake"},
            score=1.0,  # mirrors real fetch_by_id injected priority
            point_id=cid,
        )

    fake = MagicMock()
    fake.fetch_by_id.side_effect = fetch_by_id
    return fake


def _baseline_doc(chunk_id: str, text: str = "baseline body") -> "RetrievedDocument":
    """A baseline retrieval result (vector/rerank). No `booster_injected` flag."""
    from services.lancedb_retrieval import RetrievedDocument
    return RetrievedDocument(
        text=text,
        source="baseline",
        chunk_id=chunk_id,
        metadata={"source_kind": "vector"},
        score=0.42,
    )


# ---------------------------------------------------------------------------
# Case 1 — query triggers detect_aggregation_chunks → injected doc carries flag
# ---------------------------------------------------------------------------

def test_booster_injected_docs_carry_booster_injected_metadata_flag():
    """A query that hits a booster rule (e.g. 'bao nhiêu ngành') causes
    `boost_with_aggregation` to inject known chunk_ids. Every injected doc
    MUST carry `metadata['booster_injected'] is True`."""
    from services.aggregation_booster import boost_with_aggregation

    # "bao nhiêu ngành" matches the very first rule → 3 candidate chunk_ids.
    # We only need to assert on whatever the booster ACTUALLY injects; pick
    # text for ALL three so the test stays robust if rule order changes.
    chunk_id_to_text = {
        "liet_ke_28_nganh_2026_v2": "Liệt kê 28 ngành năm 2026.",
        "tuyensinh_overview_2026_v2": "Tổng quan tuyển sinh 2026.",
        "bang_xep_hang_chi_tieu_2026_v2": "Bảng xếp hạng chỉ tiêu 2026.",
    }
    fake_retriever = _make_fake_retriever(chunk_id_to_text)

    # Baseline docs do NOT contain any of the booster chunk_ids.
    baseline = [
        _baseline_doc("vector_chunk_a", "vector body a"),
        _baseline_doc("vector_chunk_b", "vector body b"),
    ]

    result = boost_with_aggregation(
        query="Trường có bao nhiêu ngành đào tạo?",
        baseline_docs=baseline,
        lancedb_retriever=fake_retriever,
        top_k=5,
        max_inject=2,  # cap → first 2 chunk_ids from the rule
    )

    # The booster should have injected at least one doc.
    assert len(result) > len(baseline), (
        f"Expected booster to inject new docs; got {len(result)} (baseline={len(baseline)})"
    )

    # Every doc whose chunk_id came from the booster MUST carry the flag.
    injected_cids = set()
    for doc in result:
        # Booster-injected docs are placed at the front (prepended); the
        # first len(injected) entries are the injected ones.
        pass

    # Stronger assertion: the fake_retriever's fetch_by_id was called for
    # the rule's chunk_ids, and EVERY doc it returned was tagged.
    assert fake_retriever.fetch_by_id.called, "fetch_by_id should be invoked"
    called_cids = {call.args[0] for call in fake_retriever.fetch_by_id.call_args_list}
    assert called_cids, "fetch_by_id was called with no chunk_ids"

    # The returned list's leading docs are the injected ones (length
    # == min(max_inject, len(unique rule chunk_ids not in baseline))).
    # We don't hardcode that count; just assert the leading docs all
    # have the flag and were not in the baseline set.
    baseline_cids = {d.chunk_id for d in baseline}
    # Find the injected chunk_ids by intersecting returned chunk_ids with
    # the rule's candidate set.
    rule_candidates = {
        "liet_ke_28_nganh_2026_v2",
        "tuyensinh_overview_2026_v2",
        "bang_xep_hang_chi_tieu_2026_v2",
    }
    injected_in_result = [d for d in result if d.chunk_id in rule_candidates]
    assert injected_in_result, (
        "Expected the result to contain at least one rule-candidate chunk"
    )
    for doc in injected_in_result:
        assert doc.metadata.get("booster_injected") is True, (
            f"Injected doc {doc.chunk_id} missing booster_injected=True: "
            f"metadata={doc.metadata!r}"
        )

    # Non-injected baseline docs preserved in the result MUST NOT have the flag.
    for doc in result:
        if doc.chunk_id in baseline_cids:
            assert "booster_injected" not in doc.metadata, (
                f"Baseline doc {doc.chunk_id} should NOT carry booster_injected: "
                f"metadata={doc.metadata!r}"
            )


# ---------------------------------------------------------------------------
# Case 2 — query that does NOT match any rule → no flag is set on anyone
# ---------------------------------------------------------------------------

def test_no_aggregation_match_means_no_booster_injected_flag_anywhere():
    """When the query matches no booster rule, no doc is injected AND no
    doc gets the `booster_injected` flag (baseline is passed through)."""
    from services.aggregation_booster import boost_with_aggregation

    fake_retriever = _make_fake_retriever({})  # no fetches expected
    baseline = [
        _baseline_doc("vector_chunk_x"),
        _baseline_doc("vector_chunk_y"),
    ]

    result = boost_with_aggregation(
        query="Xin chào, hôm nay thời tiết thế nào?",  # no booster match
        baseline_docs=baseline,
        lancedb_retriever=fake_retriever,
        top_k=5,
    )

    assert not fake_retriever.fetch_by_id.called, (
        "fetch_by_id should NOT be called when no rule matches"
    )
    for doc in result:
        assert "booster_injected" not in doc.metadata, (
            f"Doc {doc.chunk_id} unexpectedly has booster_injected flag: "
            f"metadata={doc.metadata!r}"
        )
