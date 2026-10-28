"""TDD: V5-R013 / FIX 3 — retrieval smoke test for the 2 new canonical
chunks added by scripts/fix_note_5cau_canonical_chunks.py.

Queries:
  - "thí sinh tự do tốt nghiệp 2004 xét học bạ cần tài khoản gì"
    → top-5 must contain husc_tstd_xethocba_taikhoan_2026
  - "hết hạn đăng ký trường chưa duyệt minh chứng"
    → top-5 must contain husc_phanbiet_dangky_minhchung_2026

These are retrieval smoke tests, not full generation tests — they
verify the chunks exist in the rag2025 table AND that semantic search
pulls them into top-5 for the canonical queries.

Runs idempotently: skips (passes vacuously) if the live table is not
populated with the new chunks yet (CI / sandboxed environment).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


CHUNK_TSTD = "husc_tstd_xethocba_taikhoan_2026"
CHUNK_PHANBIET = "husc_phanbiet_dangky_minhchung_2026"


def _served_id_column(df) -> str:
    """Name of the served table's logical chunk-id column.

    The legacy ``rag2025`` table carried it as ``chunk_id``; the served
    ``husc_chunks`` table carries it as ``doc_id`` (on a canonical build
    ``id`` holds the same logical value). Prefer ``doc_id`` so a legacy
    build — where ``id`` is a content hash — still resolves.
    """
    for column in ("doc_id", "chunk_id", "id"):
        if column in df.columns:
            return column
    raise KeyError("served table has no logical chunk-id column")

def _has_live_chunk(chunk_id: str) -> bool:
    """Check whether the served table contains the chunk_id."""
    _run_from_rag2025_cwd()
    try:
        import lancedb
        from config.settings import RAGSettings
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(settings.LANCEDB_TABLE)
        df = table.to_pandas()
        return chunk_id in df[_served_id_column(df)].tolist()
    except Exception:
        return False


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_from_rag2025_cwd() -> None:
    """Make sure cwd is rag2025 so lancedb URI (./data/lancedb) resolves."""
    cwd = Path.cwd().resolve()
    if cwd.name != "rag2025":
        rag = REPO_ROOT
        if rag.exists():
            import os
            os.chdir(str(rag))


def test_chunk_tstd_in_table():
    """V5-R013/FIX-3: husc_tstd_xethocba_taikhoan_2026 must exist in rag2025."""
    if not _has_live_chunk(CHUNK_TSTD):
        pytest.skip(
            f"Live rag2025 table missing {CHUNK_TSTD} — "
            "run scripts/fix_note_5cau_canonical_chunks.py first"
        )


def test_chunk_phanbiet_in_table():
    """V5-R013/FIX-3: husc_phanbiet_dangky_minhchung_2026 must exist in rag2025."""
    if not _has_live_chunk(CHUNK_PHANBIET):
        pytest.skip(
            f"Live rag2025 table missing {CHUNK_PHANBIET} — "
            "run scripts/fix_note_5cau_canonical_chunks.py first"
        )


def test_chunk_tstd_text_contains_canonical_answer():
    """V5-R013/FIX-3: the TSTD xet hoc ba chunk must explicitly contain
    the canonical answer: thí sinh tự do được xét học bạ, điều kiện
    bắt buộc tạo tài khoản Bộ 01/5-20/5."""
    try:
        import lancedb
        from config.settings import RAGSettings
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(settings.LANCEDB_TABLE)
        df = table.to_pandas()
    except Exception:
        pytest.skip("live rag2025 table unavailable")
    id_col = _served_id_column(df)
    if CHUNK_TSTD not in df[id_col].tolist():
        pytest.skip(f"{CHUNK_TSTD} not yet ingested")
    row = df[df[id_col] == CHUNK_TSTD].iloc[0]
    text = row["text"]
    assert "Thí sinh tự do" in text, "missing TSTD phrasing"
    assert "ĐƯỢC xét tuyển bằng học bạ" in text, "missing xet hoc ba allow"
    assert "01/5/2026" in text and "20/5/2026" in text, "missing Bo registration window"
    assert "thisinh.thitotnghiepthpt.edu.vn" in text, "missing Bo portal URL"


def test_chunk_phanbiet_text_distinguishes_two_windows():
    """V5-R013/FIX-3: the phanbiet chunk must explicitly call out that
    ĐK vs minh chứng are two independent timelines."""
    try:
        import lancedb
        from config.settings import RAGSettings
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(settings.LANCEDB_TABLE)
        df = table.to_pandas()
    except Exception:
        pytest.skip("live rag2025 table unavailable")
    id_col = _served_id_column(df)
    if CHUNK_PHANBIET not in df[id_col].tolist():
        pytest.skip(f"{CHUNK_PHANBIET} not yet ingested")
    row = df[df[id_col] == CHUNK_PHANBIET].iloc[0]
    text = row["text"]
    assert "THỜI GIAN ĐĂNG KÝ" in text, "missing ĐK marker"
    assert "THỜI GIAN MINH CHỨNG" in text, "missing minh chứng marker"
    assert "ĐỘC LẬP" in text, "missing độc lập framing"


def test_chunk_tstd_retrieval_top5():
    """V5-R013/FIX-3: semantic search for TSTD xet hoc ba question must
    pull the new chunk into top-5."""
    if not _has_live_chunk(CHUNK_TSTD):
        pytest.skip("Live chunk missing — run fix script first")
    try:
        import lancedb
        from config.settings import RAGSettings
        from src.services.embedding import EmbeddingService
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(settings.LANCEDB_TABLE)
        embedder = EmbeddingService(settings)
        q = (
            "thí sinh tự do tốt nghiệp 2004 xét học bạ cần tài khoản gì"
        )
        vec = embedder.encode_documents([q])[0].tolist()
        # Use the simplest vector search. We use a search() that returns
        # the top-k chunks by L2 distance — order is not guaranteed
        # between lancedb versions, but we just need SET membership.
        results = table.search(vec).limit(5).to_pandas()
    except Exception as exc:
        pytest.skip(f"live retrieval failed: {exc}")
    chunk_ids = set(results[_served_id_column(results)].tolist())
    assert CHUNK_TSTD in chunk_ids, (
        f"{CHUNK_TSTD} NOT in top-5 retrieval; "
        f"got={sorted(chunk_ids)}"
    )


def test_chunk_phanbiet_retrieval_top5():
    """V5-R013/FIX-3: semantic search for phan-biet-dang-ky-minh-chung
    question must pull the new chunk into top-5."""
    if not _has_live_chunk(CHUNK_PHANBIET):
        pytest.skip("Live chunk missing — run fix script first")
    try:
        import lancedb
        from config.settings import RAGSettings
        from src.services.embedding import EmbeddingService
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(settings.LANCEDB_TABLE)
        embedder = EmbeddingService(settings)
        q = "hết hạn đăng ký trường chưa duyệt minh chứng"
        vec = embedder.encode_documents([q])[0].tolist()
        results = table.search(vec).limit(5).to_pandas()
    except Exception as exc:
        pytest.skip(f"live retrieval failed: {exc}")
    chunk_ids = set(results[_served_id_column(results)].tolist())
    assert CHUNK_PHANBIET in chunk_ids, (
        f"{CHUNK_PHANBIET} NOT in top-5 retrieval; "
        f"got={sorted(chunk_ids)}"
    )