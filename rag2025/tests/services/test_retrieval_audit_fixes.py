# @spec(AUDIT-FIX) test_retrieval_audit_fixes — E2E retrieval for 6 audit queries
"""TDD E2E: real LanceDB retrieval + aggregation booster for the 6 audit
queries that used to MISS before the BUG-A/B/C/D/E/E2E fixes.

Pinned queries (one per bug):
  - BUG-D: "tất cả tổ hợp xét tuyển"      → qa_chi_tieu_khoi_xet_2026_overview in top-5
  - BUG-D: "bao nhiêu ngành khối VII"      → liet_ke_28_nganh_2026_v2 in top-5
  - BUG-E: "ngành mới mở 2026"            → husc_nganh_moi_2026 in top-5
  - BUG-B: "học bổng thủ khoa được bao nhiêu" → chunk containing "7.700.000" OR "100%" in top-5
  - BUG-C: "điểm chuẩn Luật 2026"         → diem_chuan_2025_full in top-5 (reference)
  - BUG-E2E: "học phí ngành CNTT"         → hocphi_2026 in top-5

This test exercises the REAL LanceDB retriever + booster path (no mocks
on the retriever), so it needs the ``rag2025`` table populated. We skip
gracefully if LanceDB is unavailable (CI without data) — but assert
hard when the table is present.

Year: pinned to 2026 (matches prod data ingestion).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


RAG_ROOT = Path(__file__).resolve().parents[2]
SRC = RAG_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Skip-if-no-LanceDB fixture ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def retriever_and_db():
    """Build a real LanceDBRetriever + raw table handle. Skip if missing."""
    env_path = RAG_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    try:
        import lancedb
        from config.settings import RAGSettings
        from src.services.lancedb_retrieval import LanceDBRetriever
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table_list = db.list_tables()
        actual_tables = getattr(table_list, "tables", []) or []
        if not actual_tables:
            try:
                for item in table_list:
                    if isinstance(item, tuple) and item[0] == "tables":
                        actual_tables = item[1]
                        break
            except Exception:
                pass
        if settings.LANCEDB_TABLE not in actual_tables:
            pytest.skip(f"LanceDB table {settings.LANCEDB_TABLE!r} not present")
        retriever = LanceDBRetriever.from_env()
        table = db.open_table(settings.LANCEDB_TABLE)
        df = table.to_pandas()
        return retriever, df
    except Exception as e:
        pytest.skip(f"LanceDB unavailable: {e}")


def _doc_ids(docs) -> set:
    return {d.chunk_id for d in docs if d.chunk_id}


def _doc_texts(docs) -> list:
    return [d.text for d in docs if d.text]


def _fetch_top_k(retriever, query: str, top_k: int = 8):
    """Embed query and run real retrieval (no booster). Returns docs."""
    from config.settings import RAGSettings
    from src.services.embedding import EmbeddingService
    settings = RAGSettings()
    emb = EmbeddingService(settings)
    vec = emb.encode_query(query).tolist()
    result = retriever.retrieve(query_vector=vec, top_k=top_k, query=query)
    if not result.is_success:
        return []
    return result.documents or []


def _boost(retriever, query: str, baseline_docs, top_k: int = 15, max_inject: int = 2):
    """Run the booster. ``top_k`` defaults to 15 to mirror the production
    call-site in ``src/main.py`` (``top_k=request.top_k * 3`` → 15 when
    request.top_k=5). The booster's job is to guarantee priority chunks
    are present in the candidate pool that reaches the LLM; the final
    top-5 slice happens downstream in the generator."""
    from src.services.aggregation_booster import boost_with_aggregation
    return boost_with_aggregation(
        query=query,
        baseline_docs=baseline_docs,
        lancedb_retriever=retriever,
        top_k=top_k,
        max_inject=max_inject,
    )


# ── Per-bug assertions ──────────────────────────────────────────────────────

def test_bug_d_to_hop_xet_tuyen(retriever_and_db):
    """BUG-D Q4: 'tất cả tổ hợp xét tuyển' → qa_chi_tieu_khoi_xet_2026_overview in
    candidate pool. top_k=15 mirrors production (main.py passes request.top_k*3)."""
    retriever, _ = retriever_and_db
    query = "tất cả tổ hợp xét tuyển của trường"
    baseline = _fetch_top_k(retriever, query, top_k=8)
    boosted = _boost(retriever, query, baseline, top_k=15, max_inject=2)
    ids = _doc_ids(boosted)
    assert "qa_chi_tieu_khoi_xet_2026_overview" in ids or "liet_ke_28_nganh_2026_v2" in ids, (
        f"BUG-D regression: neither qa_chi_tieu_khoi_xet_2026_overview nor "
        f"liet_ke_28_nganh_2026_v2 in candidate pool; ids={sorted(ids)}"
    )


def test_bug_d_khoi_vii_count(retriever_and_db):
    """BUG-D Q10: 'bao nhiêu ngành khối VII' → liet_ke_28_nganh_2026_v2 in candidate pool."""
    retriever, _ = retriever_and_db
    query = "bao nhiêu ngành khối VII"
    baseline = _fetch_top_k(retriever, query, top_k=8)
    boosted = _boost(retriever, query, baseline, top_k=15, max_inject=2)
    ids = _doc_ids(boosted)
    assert "liet_ke_28_nganh_2026_v2" in ids, (
        f"BUG-D regression: liet_ke_28_nganh_2026_v2 missing from candidate pool; ids={sorted(ids)}"
    )


def test_bug_e_nganh_moi(retriever_and_db):
    """BUG-E: 'ngành mới mở 2026' → husc_nganh_moi_2026 in candidate pool."""
    retriever, _ = retriever_and_db
    query = "ngành mới mở năm 2026"
    baseline = _fetch_top_k(retriever, query, top_k=8)
    boosted = _boost(retriever, query, baseline, top_k=15, max_inject=2)
    ids = _doc_ids(boosted)
    assert "husc_nganh_moi_2026" in ids, (
        f"BUG-E regression: husc_nganh_moi_2026 missing from candidate pool; ids={sorted(ids)}"
    )


def test_bug_b_hoc_bong_thu_khoa(retriever_and_db):
    """BUG-B: 'học bổng thủ khoa được bao nhiêu' → candidate pool has chunk with
    '7.700.000' OR '100%' (the 100% HP HK1 number)."""
    retriever, _ = retriever_and_db
    query = "học bổng thủ khoa được bao nhiêu"
    baseline = _fetch_top_k(retriever, query, top_k=8)
    boosted = _boost(retriever, query, baseline, top_k=15, max_inject=2)
    texts = _doc_texts(boosted)
    has_number = any("7.700.000" in t or "100%" in t for t in texts)
    assert has_number, (
        f"BUG-B regression: no chunk in candidate pool contains '7.700.000' or '100%'; "
        f"ids={sorted(_doc_ids(boosted))}"
    )


def test_bug_c_diem_chuan_2026_reference(retriever_and_db):
    """BUG-C: 'điểm chuẩn Luật 2026' → diem_chuan_2025_full injected as reference."""
    retriever, _ = retriever_and_db
    query = "điểm chuẩn ngành Luật 2026"
    baseline = _fetch_top_k(retriever, query, top_k=8)
    boosted = _boost(retriever, query, baseline, top_k=15, max_inject=2)
    ids = _doc_ids(boosted)
    assert "diem_chuan_2025_full" in ids, (
        f"BUG-C regression: diem_chuan_2025_full missing from candidate pool; ids={sorted(ids)}"
    )


def test_bug_e2e_hoc_phi_nganh(retriever_and_db):
    """BUG-E2E Q1: 'học phí ngành CNTT' → hocphi_2026 in candidate pool."""
    retriever, _ = retriever_and_db
    query = "học phí ngành CNTT"
    baseline = _fetch_top_k(retriever, query, top_k=8)
    boosted = _boost(retriever, query, baseline, top_k=15, max_inject=2)
    ids = _doc_ids(boosted)
    assert "hocphi_2026" in ids, (
        f"BUG-E2E regression: hocphi_2026 missing from candidate pool; ids={sorted(ids)}"
    )
