"""Tests for chunk_table_aware_v2 (T1.8 chunk size gates)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import tiktoken

from config.settings import settings

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunk_table_aware_v2.py"
ENC = tiktoken.get_encoding("cl100k_base")

HARD_MAX = 512
SOFT_MIN = 50


@pytest.fixture
def cta():
    spec = importlib.util.spec_from_file_location("chunk_table_aware_v2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _wide_table_lines(rows: int) -> list[str]:
    """Build an 8-col markdown table with `rows` data rows (wide bypass)."""
    header = "|" + "|".join([f"Col{i}" for i in range(8)]) + "|"
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [header, sep]
    for r in range(rows):
        lines.append("|" + "|".join([f"d{r}c{i}" for i in range(8)]) + "|")
    return lines


_BASE_META = {
    "source": "test",
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "thong_tin_chung",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
}


def test_chunk_table_hard_cap_splits_oversized_wide_table(cta):
    """T1.8: an oversized wide-table chunk is split into <=512-token pieces."""
    table = cta.MarkdownTable(_wide_table_lines(80))
    assert table.is_wide  # wide bypass path
    chunks = cta.chunk_table(table, _BASE_META, "doc_x", ENC, 0)

    assert len(chunks) > 1
    for c in chunks:
        assert cta.count_tokens(c["text"], ENC) <= HARD_MAX


def test_chunk_table_split_pieces_preserve_table_structure(cta):
    """T1.8: split table pieces keep header context (start with a table row)."""
    table = cta.MarkdownTable(_wide_table_lines(80))
    chunks = cta.chunk_table(table, _BASE_META, "doc_x", ENC, 0)

    for c in chunks:
        assert c["text"].strip().startswith("|")


def test_chunk_table_small_table_within_cap_unchanged(cta):
    """T1.8: a small table within the cap is returned as a single chunk."""
    table = cta.MarkdownTable(_wide_table_lines(3))
    chunks = cta.chunk_table(table, _BASE_META, "doc_x", ENC, 0)

    assert len(chunks) == 1
    assert cta.count_tokens(chunks[0]["text"], ENC) <= HARD_MAX


def test_enforce_size_gates_merges_undersized_into_neighbor(cta):
    """T1.8: a chunk below SOFT_MIN is merged into its previous neighbor."""
    big = {
        "id": "d_chunk_0", "chunk_id": 0, "text": "x" * 200,
        "text_plain": "x" * 200,
        "metadata": {"chunk_type": "wide_table", "token_count": 200},
        "breadcrumbs": [], "prev_chunk_id": "", "next_chunk_id": "",
        "sparse_terms": [],
    }
    tiny = {
        "id": "d_chunk_1", "chunk_id": 1, "text": "tiny",
        "text_plain": "tiny",
        "metadata": {"chunk_type": "text_section", "token_count": 1},
        "breadcrumbs": [], "prev_chunk_id": "", "next_chunk_id": "",
        "sparse_terms": [],
    }
    out = cta._enforce_size_gates([big, tiny], _BASE_META, "d", ENC)

    assert len(out) == 1
    assert cta.count_tokens(out[0]["text"], ENC) <= HARD_MAX


def test_enforce_size_gates_follows_settings_hard_max(cta, monkeypatch):
    """Settings-driven: HARD_MAX_TOKENS from config.settings controls the cap.

    The gate must read the knob at call time — patching the settings singleton
    must lower the split cap below the old hardcoded 512.
    """
    monkeypatch.setattr(settings, "HARD_MAX_TOKENS", 100)
    # Realistic token-rich text: repeated sentences, well above the patched cap.
    text = (
        "Đây là câu văn mẫu dùng để kiểm tra giới hạn cứng của việc tách chunk "
        "theo cấu hình HARD_MAX_TOKENS từ cài đặt chunking.\n" * 40
    )
    big = {
        "id": "d_chunk_0", "chunk_id": 0, "text": text,
        "text_plain": text,
        "metadata": {"chunk_type": "text_section", "token_count": 1000},
        "breadcrumbs": [], "prev_chunk_id": "", "next_chunk_id": "",
        "sparse_terms": [],
    }
    out = cta._enforce_size_gates([big], _BASE_META, "d", ENC)

    assert len(out) > 1
    for c in out:
        assert cta.count_tokens(c["text"], ENC) <= 100


def test_chunk_text_sections_follows_settings_budget(cta, monkeypatch):
    """Settings-driven: CHUNK_SIZE_TOKENS controls the text-section budget.

    The same document must yield a single chunk under a huge settings budget and
    many smaller chunks under a tiny settings budget — proving the text splitter
    no longer hardcodes its own 220-token constant.
    """
    text = "\n".join(
        "Đây là câu số %d trong đoạn văn mẫu dùng để kiểm tra việc tách chunk theo ngân sách token của cài đặt chunking." % i
        for i in range(60)
    )

    monkeypatch.setattr(settings, "CHUNK_SIZE_TOKENS", 5000)
    big = cta.chunk_text_sections(text, _BASE_META, "d", ENC, 0)

    monkeypatch.setattr(settings, "CHUNK_SIZE_TOKENS", 80)
    small = cta.chunk_text_sections(text, _BASE_META, "d", ENC, 0)

    assert len(big) == 1  # whole text fits comfortably in the huge budget
    assert len(small) > 1  # tiny budget forces many chunks
    for c in small:
        # 1.3x sentence-boundary overflow is the documented trim ceiling
        assert cta.count_tokens(c["text"], ENC) <= 80 * 1.3 + 20


# ══════════════════════════════════════════════════════════════════
# ST-6c structure guards — pipe-row atomicity (chunk_table_aware_v2.py)
# ══════════════════════════════════════════════════════════════════


def test_split_oversized_preserving_single_oversized_row_kept_whole(cta):
    """ST-6c: a single pipe row above the hard cap is kept whole, never cut
    mid-row.
    RED on current code: the per-table piece with the giant row fails the
    `all(pieces <= cap)` check and falls through to the balanced token split,
    which fragments the row.

    NOTE: the row must be genuinely token-oversized. A long run of 'x' merges
    into few cl100k tokens (``"x"*1500`` ≈ 190 tokens) and would stay under
    HARD_MAX — an accidental green. Vietnamese text ≈ 32 tokens/repeat, so
    ``phrase * 18`` ≈ 578 tokens, safely above the 512 cap.
    """
    phrase = "nội dung chi tiết về điều kiện xét tuyển đại học năm 2026 của trường "
    giant_row = "|%s|" % (phrase * 18)  # ~578 tokens, above HARD_MAX
    table = giant_row + "\n|small row|"

    pieces = cta._split_oversized_preserving(table, ENC, HARD_MAX)

    assert any(giant_row in p for p in pieces), "giant pipe row must be kept whole"
    assert any("|small row|" in p for p in pieces)


def test_split_oversized_preserving_pipe_rows_never_split(cta):
    """ST-6c (PIN, green on current code): an oversized table splits at row
    boundaries — every data row stays intact and every piece is within the cap.
    """
    rows = ["|Row %03d|data value %d|" % (i, i) for i in range(120)]
    table = "\n".join(rows)

    pieces = cta._split_oversized_preserving(table, ENC, HARD_MAX)

    assert len(pieces) > 1
    for p in pieces:
        assert cta.count_tokens(p, ENC) <= HARD_MAX
    for row in rows:
        assert any(row in p for p in pieces), f"row cut mid-row: {row!r}"


def test_split_oversized_preserving_keeps_giant_row_in_order(cta):
    """ST-6c (RED on current code): an oversized pipe row in the MIDDLE of
    pipe-table prose must not push lines that precede it AFTER it.

    Current code appends the oversized row to `pieces` without first flushing
    the pending `current` buffer, so the final flush emits the preceding lines
    after the giant row — corrupting sequential chunk order. Reproduced reorder:
    pieces == [GIANT, 'Giới thiệu:\n|rowA|  data\n|rowB|  data'].
    """
    phrase = "nội dung chi tiết về điều kiện xét tuyển đại học năm 2026 của trường "
    giant_row = "|%s|" % (phrase * 18)  # ~578 tokens, above HARD_MAX
    text = "Giới thiệu:\n|rowA|  data\n" + giant_row + "\n|rowB|  data"

    pieces = cta._split_oversized_preserving(text, ENC, HARD_MAX)

    assert any(giant_row in p for p in pieces), "giant pipe row must be kept whole"
    giant_idx = next(i for i, p in enumerate(pieces) if giant_row in p)
    earlier = "\n".join(pieces[:giant_idx])
    assert "Giới thiệu:" in earlier, "text before the giant row must stay ahead"
    assert "|rowA|" in earlier
