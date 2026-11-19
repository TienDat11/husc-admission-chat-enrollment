"""Chunk quality on the two official source PDFs (pypdf text, NFC).

Skipped when the PDFs are absent (``needs_source_pdf``).

TTS "mã+tổ hợp cùng chunk" (CHUNKING_RESEARCH_2026-10-28.md §2): for every
7-digit code ``\\b7\\d{6}\\b``, take the first line starting with ``1.`` within
300 characters after the code. The code counts as colocated when some chunk
holds both the code and the first 14 characters of that line. The denominator
is the number of codes that have such a line (111 on this text). This is the
document's heuristic, not a ground truth.

Measured with ``_pack_units_greedily`` at the spec's TTS target of 350: 92 of 111.
The research doc's 95/111 belongs to a semchunk split that cuts blocks, which
PL-6.5 forbids. Target 400 reaches 100/111; the spec keeps 350.
"""

from __future__ import annotations

import re
import statistics
import unicodedata

import pypdf
import pytest
import tiktoken

from src.domain.canonical.semantic_units import _pack_units_greedily

ENC = tiktoken.get_encoding("cl100k_base")
TT06 = "Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf"
TTS = "THONG_TIN_TUYEN_SINH_202611.pdf"

TTS_CODE = re.compile(r"\b7\d{6}\b")
TTS_COMBO_LINE = re.compile(r"^1\.", re.M)
DIEU_ONLY = re.compile(r"Điều\s+\d+")
# A chunk opens on a capital, a digit, a list marker, or the "a)" sub-item.
OPENING_OK = re.compile(r"^(?:[A-ZĐ0-9(\"“\-–•]|a\))")

TTS_DENOMINATOR = 111
TTS_COLOCATION_FLOOR = 92


def _nfc_text(path) -> str:
    reader = pypdf.PdfReader(str(path))
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    return unicodedata.normalize("NFC", raw)


def _pack(text: str):
    return _pack_units_greedily(
        text, enc=ENC, min_tokens=50, target_tokens=350, hard_max_tokens=512
    )


def _is_label_only(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) == 1 and bool(DIEU_ONLY.match(lines[0]))


def _short_chunks_end_their_dieu(chunks, toks) -> bool:
    for index, token_count in enumerate(toks):
        if token_count >= 50 or index + 1 >= len(chunks):
            continue
        if chunks[index + 1].label == chunks[index].label:
            return False
    return True


def _colocation(text: str, chunks) -> tuple[int, int]:
    """Return (colocated, denominator) for the TTS mã+tổ hợp metric."""
    texts = [chunk.text for chunk in chunks]
    colocated = 0
    denominator = 0
    for code_match in TTS_CODE.finditer(text):
        combo = TTS_COMBO_LINE.search(
            text, code_match.end(), code_match.end() + 300
        )
        if combo is None:
            continue
        denominator += 1
        line_start = text.rfind("\n", 0, combo.start()) + 1
        line_end = text.find("\n", combo.start())
        line = text[line_start: line_end if line_end != -1 else len(text)]
        key = line[:14]
        code = code_match.group(0)
        if any(code in t and key in t for t in texts):
            colocated += 1
    return colocated, denominator


@pytest.mark.needs_source_pdf
def test_tt06_meets_the_legal_budget(source_pdf):
    chunks = _pack(_nfc_text(source_pdf(TT06)))
    toks = [len(ENC.encode(chunk.text)) for chunk in chunks]

    assert 95 <= len(chunks) <= 135, len(chunks)
    assert 280 <= statistics.median(toks) <= 310, statistics.median(toks)
    assert max(toks) <= 512
    assert sum(1 for t in toks if t >= 30) / len(toks) >= 0.95
    assert not any(_is_label_only(chunk.text) for chunk in chunks)
    assert _short_chunks_end_their_dieu(chunks, toks)
    assert all(OPENING_OK.match(chunk.text.lstrip()) for chunk in chunks)


@pytest.mark.needs_source_pdf
def test_tts_stays_under_the_hard_cap_and_colocates_major_codes(source_pdf):
    text = _nfc_text(source_pdf(TTS))
    chunks = _pack(text)
    toks = [len(ENC.encode(chunk.text)) for chunk in chunks]

    assert max(toks) <= 512
    assert not any(_is_label_only(chunk.text) for chunk in chunks)

    colocated, denominator = _colocation(text, chunks)
    assert denominator == TTS_DENOMINATOR, denominator
    assert colocated >= TTS_COLOCATION_FLOOR, (colocated, denominator)
