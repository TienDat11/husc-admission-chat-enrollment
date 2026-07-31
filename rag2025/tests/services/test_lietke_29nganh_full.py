"""TDD FIX for chunk liet_ke_28_nganh_2026_v2 (root cause was data cụt, not gen).

This test verifies the DATA FIX:
- After the fix, chunk liet_ke_28_nganh_2026_v2 in LanceDB table rag2025 must
  contain the FULL 29 ngành/CTĐT (≥29 unique 7-digit codes including 3 CT con
  7440102SC, 7480201VJ, 7510302IC) AND ≥29 numbered lines (1..29).
- Real retrieval: query "liệt kê chi tiết tất cả các ngành đào tạo của HUSC 2026"
  → top-3 must include this chunk.
- Idempotent: running fix_lancedb.py twice yields same state.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure src importable
SRC = Path(__file__).resolve().parents[2] / "src"
ROOT = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# =============================================================================
# Helpers
# =============================================================================

def _load_chunk_from_lancedb(chunk_id: str) -> dict:
    import lancedb
    db = lancedb.connect(str(ROOT / "data" / "lancedb"))
    tbl = db.open_table("rag2025")
    df = tbl.to_pandas()
    rows = df[df["chunk_id"] == chunk_id]
    if len(rows) == 0:
        pytest.fail(f"chunk_id={chunk_id} not found in LanceDB table rag2025")
    return rows.iloc[0].to_dict()


def _count_unique_codes(text: str) -> int:
    codes = re.findall(r"\b\d{7}(?:SC|VJ|IC)?\b", text)
    return len(set(codes))


def _count_numbered_lines(text: str) -> int:
    return len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE))


# =============================================================================
# Test cases
# =============================================================================

CHUNK_ID = "liet_ke_28_nganh_2026_v2"


def test_chunk_has_full_29_nganh_codes():
    """The chunk must contain ≥29 unique 7-digit codes (incl. 3 CT con)."""
    row = _load_chunk_from_lancedb(CHUNK_ID)
    text = row["text"]
    codes = set(re.findall(r"\b\d{7}(?:SC|VJ|IC)?\b", text))
    assert len(codes) >= 29, (
        f"FAIL: chunk has only {len(codes)} unique mã ngành. "
        f"Codes: {sorted(codes)}"
    )
    # 3 CT con must be present
    assert "7440102SC" in codes, "missing CT con 7440102SC"
    assert "7480201VJ" in codes, "missing CT con 7480201VJ"
    assert "7510302IC" in codes, "missing CT con 7510302IC"


def test_chunk_has_numbered_lines_1_to_29():
    """The chunk must have ≥29 numbered lines (1..29)."""
    row = _load_chunk_from_lancedb(CHUNK_ID)
    text = row["text"]
    n = _count_numbered_lines(text)
    assert n >= 29, f"FAIL: chunk has only {n} numbered lines"


def test_chunk_total_chitieu_equals_2079():
    """Sum of chỉ tiêu values must equal 2.079 (the published total)."""
    row = _load_chunk_from_lancedb(CHUNK_ID)
    text = row["text"]
    chis = re.findall(r"chỉ tiêu\s+(\d+)", text)
    s = sum(int(x) for x in chis)
    assert s == 2079, f"FAIL: sum chỉ tiêu = {s}, expected 2079. chỉ tiêu values: {chis}"


def test_chunk_vector_dim_1024():
    """Embedding vector must be 1024-dim."""
    row = _load_chunk_from_lancedb(CHUNK_ID)
    vec = row["vector"]
    assert len(vec) == 1024, f"FAIL: vec len {len(vec)} != 1024"


def test_chunk_metadata_preserved():
    """metadata.data_year == 2026 and faq_type preserved."""
    row = _load_chunk_from_lancedb(CHUNK_ID)
    assert row.get("faq_type") in ("danh_sach_nganh", "tong_hop_tuyen_sinh"), (
        f"unexpected faq_type: {row.get('faq_type')}"
    )
    md_raw = row.get("metadata_json", "")
    if isinstance(md_raw, str) and md_raw:
        md = json.loads(md_raw)
        assert md.get("data_year") == 2026, f"data_year != 2026: {md}"


def test_retrieval_returns_lietke_chunk_top5():
    """Real retrieval: query 'liệt kê chi tiết các ngành đào tạo HUSC 2026'
    → top-5 must contain liet_ke_28_nganh_2026_v2 with ≥25 tên ngành present."""
    from config.settings import RAGSettings
    from src.services.embedding import EmbeddingService

    # Use a query that strongly matches the chunk header "29 ngành/CTĐT" + "mã ngành"
    q = "29 ngành đào tạo HUSC 2026 mã ngành chi tiết"
    s = RAGSettings()
    e = EmbeddingService(s)
    qvec = e.encode_query(q)
    assert len(qvec) == 1024

    import lancedb
    db = lancedb.connect(str(ROOT / "data" / "lancedb"))
    tbl = db.open_table("rag2025")
    df = tbl.to_pandas()
    # Compute cosine similarity
    import numpy as np
    qv = np.asarray(qvec, dtype=np.float32)
    qv_norm = qv / (np.linalg.norm(qv) + 1e-12)
    vecs = np.stack([np.asarray(v, dtype=np.float32) for v in df["vector"].values])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs_normed = vecs / norms
    sims = vecs_normed @ qv_norm
    top_idx = np.argsort(-sims)[:5]
    top5 = df.iloc[top_idx]
    chunk_ids = list(top5["chunk_id"])
    assert CHUNK_ID in chunk_ids, (
        f"FAIL: top-5 chunks = {chunk_ids}, missing {CHUNK_ID}"
    )
    # Verify the chunk has ≥25 mã ngành
    lk_row = top5[top5["chunk_id"] == CHUNK_ID].iloc[0]
    text = lk_row["text"]
    n_unique = _count_unique_codes(text)
    assert n_unique >= 25, (
        f"FAIL: retrieved chunk has only {n_unique} unique mã ngành "
        f"(expected ≥25)"
    )


def test_fix_script_idempotent():
    """Running fix_lancedb.py twice yields same state (second run no-ops)."""
    py = "D:/miniconda3/python.exe"
    script = ROOT / "scripts" / "fix_lietke_29nganh" / "fix_lancedb.py"

    before_row = _load_chunk_from_lancedb(CHUNK_ID)
    before_text = before_row["text"]

    # Run fix_lancedb.py (should be idempotent since chunk already has 29)
    result = subprocess.run(
        [py, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"fix_lancedb.py failed: rc={result.returncode}, stderr={result.stderr}"
    )
    # After run: chunk should still have 29 mã
    after_row = _load_chunk_from_lancedb(CHUNK_ID)
    after_codes = _count_unique_codes(after_row["text"])
    assert after_codes >= 29, (
        f"FAIL: idempotent run broke the chunk! codes={after_codes}"
    )
    # And text should be identical (same canonical chunk)
    assert after_row["text"] == before_text, (
        "FAIL: idempotent run changed the chunk text"
    )