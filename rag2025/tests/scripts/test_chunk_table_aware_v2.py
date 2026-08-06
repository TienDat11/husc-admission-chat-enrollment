"""Tests for chunk_table_aware_v2 (T1.8 chunk size gates)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import tiktoken

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
