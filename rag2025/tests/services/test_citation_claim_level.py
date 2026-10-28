"""Ticket 10 — claim-level citation contract.

@spec(10) Acceptance criteria measured at the module + generator seam:

1. Every claim about cutoff score / timeline / quota / tuition in an answer
   traces back to a source (criterion "Mọi claim ... truy được tới nguồn").
2. Citations carry the source document and page/section when available.
3. A missing page never breaks the response surface (optional on the model).
4. A multi-major comparison never collapses to one shared source_id: claims
   whose values live in different chunks get different chunk/source ids.

Plus the ticket-33 constraints the ticket body pins: the row label comes
from the attribution entry's ``identity`` (the anchor is never assumed to
live in the same chunk), and an unresolved row exposes its ``reason`` via
``note`` instead of letting the citation guess.

And the generator seam: ``LLMGenerator.generate_answer`` returns a
``citations`` key on BOTH the normal path and the T11-abstain path,
computed from the FINAL answer text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from services.citation import (  # noqa: E402
    ClaimCitation,
    build_citations,
    extract_claims,
    source_provenance,
)


def _lin_chunk(
    text: str,
    *,
    chunk_id: str,
    doc_id: str,
    page: int | None,
    record_id: str = "rec-1",
    version: str = "rec-1@v1",
    section: str | None = None,
    year: int = 2026,
    extra_meta: dict | None = None,
) -> dict:
    """Chunk shaped like production RetrievedDocument.to_dict(): lineage
    nested under ``metadata`` (lancedb_retrieval nests it there)."""
    md: dict = {
        "data_year": year,
        "record_id": record_id,
        "record_version_key": version,
        "source_document_id": doc_id,
        "source_page": page,
        "source_url": f"https://tuyensinh.husc.edu.vn/thongbao.php?id=99",
    }
    if section:
        md["section"] = section
    if extra_meta:
        md.update(extra_meta)
    return {"text": text, "summary": text[:80], "chunk_id": chunk_id, "source": doc_id, "metadata": md}


# ---------------------------------------------------------------------------
# Criterion 1: every high-impact claim type traces to a source
# ---------------------------------------------------------------------------

ANSWER = (
    "Điểm chuẩn ngành CNTT là 22,5. "
    "Học phí 16 triệu đồng/năm. "
    "Chỉ tiêu 1500 thí sinh. "
    "Nộp hồ sơ từ 15/8/2025."
)


def _four_value_chunks() -> list[dict]:
    return [
        _lin_chunk("Điểm chuẩn: 22,5", chunk_id="c-a", doc_id="doc-a", page=10),
        _lin_chunk("Học phí: 16.000.000 đồng/năm", chunk_id="c-b", doc_id="doc-b", page=3),
        _lin_chunk("chỉ tiêu: 1500 thí sinh", chunk_id="c-c", doc_id="doc-c", page=None),
        _lin_chunk("Nhận hồ sơ từ 15/8/2025", chunk_id="c-d", doc_id="doc-d", page=7),
    ]


def test_every_claim_type_is_grounded_to_a_source():
    cites = build_citations(ANSWER, _four_value_chunks())
    by_value = {c["value"]: c for c in cites}
    # The four claims the criterion names, each traced end to end.
    for value in ("22.5", "16000000", "1500", "15/8/2025"):
        assert value in by_value, f"claim {value} missing from citations"
        c = by_value[value]
        assert c["grounded"] is True, f"claim {value} not grounded: {c}"
        assert c["source_document_id"], f"claim {value} has no source document"
        assert c["chunk_id"], f"claim {value} has no chunk"
    assert {c["claim_type"] for c in cites} >= {"cutoff_score", "tuition", "quota", "timeline"}


def test_ungrounded_claim_is_surfaced_not_dropped():
    """A number in the answer that no chunk carries must appear with
    grounded=False (abstention consumes the flag; silence is not allowed)."""
    cites = build_citations("Điểm chuẩn ngành CNTT là 99,9.", _four_value_chunks())
    assert len(cites) == 1
    assert cites[0]["grounded"] is False
    assert cites[0]["source_document_id"] is None


# ---------------------------------------------------------------------------
# Criterion 2: source document + page/section when available
# ---------------------------------------------------------------------------

def test_citation_carries_document_page_and_section():
    cites = build_citations(
        ANSWER,
        [
            _lin_chunk("Điểm chuẩn: 22,5", chunk_id="c-a", doc_id="doc-a", page=10,
                       section="Phụ lục A", record_id="rec-9", version="rec-9@v3"),
        ] + _four_value_chunks()[1:],
    )
    cutoff = next(c for c in cites if c["claim_type"] == "cutoff_score")
    assert cutoff["source_document_id"] == "doc-a"
    assert cutoff["source_page"] == 10
    assert cutoff["section"] == "Phụ lục A"
    assert cutoff["record_id"] == "rec-9"
    assert cutoff["record_version_key"] == "rec-9@v3"


# ---------------------------------------------------------------------------
# Criterion 3: a missing page never breaks the surface
# ---------------------------------------------------------------------------

def test_missing_page_is_optional_not_fatal():
    cites = build_citations(ANSWER, _four_value_chunks())
    quota = next(c for c in cites if c["claim_type"] == "quota")
    assert quota["grounded"] is True
    assert quota["source_page"] is None  # the chunk simply has no page
    # The contract model validates the whole list with a None page present.
    models = [ClaimCitation(**c) for c in cites]
    assert models[2].source_page is None


def test_legacy_flat_lineage_and_garbage_page_tolerated():
    """Non-canonical chunks (flat metadata) resolve the same way; a
    non-numeric page degrades to None instead of raising."""
    flat = {"text": "Điểm chuẩn: 22,5", "chunk_id": "x", "metadata": {"source_document_id": "d1"}}
    assert source_provenance(flat)["source_document_id"] == "d1"
    bad = _lin_chunk("Học phí: 16.000.000 đồng/năm", chunk_id="y", doc_id="d2", page=None)
    bad["metadata"]["source_page"] = "không rõ"
    cites = build_citations("Học phí 16.000.000 đồng.", [bad])
    assert cites[0]["source_page"] is None
    assert cites[0]["grounded"] is True


# ---------------------------------------------------------------------------
# Criterion 4: per-cell provenance for a comparison table
# ---------------------------------------------------------------------------

def test_comparison_cells_do_not_share_one_source():
    answer = "Điểm chuẩn ngành CNTT 22,5; Kinh doanh 24,75."
    chunks = [
        _lin_chunk("CNTT | A00 | 22,5", chunk_id="c-it", doc_id="doc-it", page=4),
        _lin_chunk("Kinh doanh | D01 | 24,75", chunk_id="c-biz", doc_id="doc-biz", page=9),
    ]
    cites = build_citations(answer, chunks)
    cutoffs = [c for c in cites if c["claim_type"] == "cutoff_score"]
    assert len(cutoffs) == 2, "each comparison cell must be its own claim"
    assert {c["chunk_id"] for c in cutoffs} == {"c-it", "c-biz"}
    assert {c["source_document_id"] for c in cutoffs} == {"doc-it", "doc-biz"}
    assert len({c["source_document_id"] for c in cutoffs}) == 2, (
        "a comparison table must never collapse to one shared source_id"
    )


# ---------------------------------------------------------------------------
# Ticket 33 constraints: split table pieces, rowspan provenance
# ---------------------------------------------------------------------------

TABLE_PIECE = (
    "| Ngành | Mã | Điểm |\n"
    "|---|---|---|\n"
    "| | A00 | 22,5 |\n"
)


def test_leaning_row_label_comes_from_entry_identity():
    """The anchor row's label is NOT in this piece — the citation reads the
    attribution entry's identity (never assumes the anchor shares the chunk)."""
    chunk = _lin_chunk(
        TABLE_PIECE, chunk_id="c-t", doc_id="doc-t", page=2,
        extra_meta={
            "rowspan_attribution": [
                {"row": 3, "anchor_row": 1, "identity": "Khoa học máy tính", "cells": {"0": "Khoa học máy tính"}},
            ],
        },
    )
    cites = build_citations("Điểm chuẩn ngành CNTT là 22,5.", [chunk])
    assert cites[0]["grounded"] is True
    assert cites[0]["row_label"] == "Khoa học máy tính"
    assert cites[0]["note"] is None


def test_unresolved_row_exposes_reason_not_a_guess():
    chunk = _lin_chunk(
        TABLE_PIECE, chunk_id="c-u", doc_id="doc-u", page=2,
        extra_meta={
            "rowspan_unresolved": [
                {"row": 3, "columns": [0], "reason": "anchor outside table"},
            ],
        },
    )
    cites = build_citations("Điểm chuẩn ngành CNTT là 22,5.", [chunk])
    assert cites[0]["row_label"] is None, "must not infer an identity"
    assert cites[0]["note"] == "anchor outside table", "must read the reason verbatim"


# ---------------------------------------------------------------------------
# Generator seam: generate_answer returns citations on both paths
# ---------------------------------------------------------------------------

def _bare_generator(answer_text: str):
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "BAN LA TU VAN VIEN."
    gen.gen_model = "fake-model"
    gen.has_any_provider = True
    gen.unified_fallback_client = None
    gen.gen_fallback_model = "none"

    from unittest.mock import AsyncMock, MagicMock

    fake = MagicMock()
    fake._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "fake"
    resp.provider = "test"
    fake.chat = AsyncMock(return_value=resp)
    gen.unified_client = fake

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi
    return gen, fake


@pytest.mark.asyncio
async def test_generate_answer_returns_citations_on_normal_path(monkeypatch):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    from services.season import SeasonPhase

    gen, _ = _bare_generator("Học phí 2026 ngành CNTT là 16.000.000 đồng/năm.")
    gen._resolve_admission_context = lambda chunks: (2026, SeasonPhase.IN_SEASON, True)
    chunks = [
        _lin_chunk("Học phí năm 2026 ngành CNTT: 16.000.000 đồng", chunk_id="g1",
                   doc_id="doc-g", page=5, record_id="rec-g", version="rec-g@v2"),
    ]
    result = await gen.generate_answer(
        query="Học phí 2026 ngành CNTT bao nhiêu?", chunks=chunks, confidence=0.9,
    )
    cites = result["citations"]
    assert isinstance(cites, list) and cites, "normal path must return citations"
    tuition = next(c for c in cites if c["claim_type"] == "tuition")
    assert tuition["value"] == "16000000"
    assert tuition["grounded"] is True
    assert tuition["source_document_id"] == "doc-g"
    assert tuition["source_page"] == 5
    # JSON-serializable dicts (the /v2 payload embeds them directly).
    import json

    json.dumps(cites)


@pytest.mark.asyncio
async def test_generate_answer_abstain_path_still_traces_its_claims(monkeypatch):
    """Criterion 1 applies to the refusal too: every claim in the FINAL text
    (refusal included) must have a citation entry."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    from services.season import SeasonPhase
    from services.citation import extract_claims as _xc

    gen, fake = _bare_generator("unused — gate refuses before the model")
    gen._resolve_admission_context = lambda chunks: (2026, SeasonPhase.IN_SEASON, False)
    chunks = [
        _lin_chunk("Học phí năm 2025: 14.000.000 đồng", chunk_id="g2",
                   doc_id="doc-2025", page=8, year=2025),
    ]
    result = await gen.generate_answer(
        query="Học phí 2026 của trường là bao nhiêu?", chunks=chunks, confidence=0.5,
    )
    assert fake.chat.await_count == 0
    assert result["provider"].startswith("evidence-policy:")
    assert "citations" in result, "abstain path must also return the key"
    values = {c["value"] for c in result["citations"]}
    assert values == {c.value for c in _xc(result["answer"])}, (
        "citations must cover exactly the claims in the refusal text"
    )
    # A refusal that mentions a verified year traces it to that year's chunk.
    for c in result["citations"]:
        if c["value"] == "2025":
            assert c["grounded"] is True and c["source_document_id"] == "doc-2025"
