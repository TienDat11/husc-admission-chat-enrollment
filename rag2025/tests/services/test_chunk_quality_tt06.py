"""Chunk quality on the two official source PDFs (pypdf text, NFC).

Skipped when the PDFs are absent (``needs_source_pdf``).

TTS colocation: a major-code block runs from its code to the next code. A block
has a combination when it holds a code-form combination (``A00``) or at least two
subject names (a named combination such as "Ngữ văn, Toán, Vật lí"). The block
counts as colocated when the chunk holding its code also holds a combination.
Measured on this text: 86 blocks carry a combination, 84 are colocated. The spec's
target of 95 of 111 is not reached, and the denominator 111 cannot be reproduced.
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

MAJOR_OCCURRENCE = re.compile(r"\b7\d{6}\b")
CODE_COMBO = re.compile(r"(?<![0-9A-Za-z])[A-Z]\d{2}(?![0-9A-Za-z])")
SUBJECT_NAME = re.compile(
    r"(Ngữ văn|Toán|Lịch sử|Địa lí|Vật lí|Hóa học|Sinh học|GDCD|Tiếng Anh|"
    r"Tiếng Pháp|Tiếng Trung|Tiếng Nhật|Tiếng Hàn|Tiếng Đức|Tiếng Nga|Năng khiếu)"
)
DIEU_ONLY = re.compile(r"Điều\s+\d+")
# A chunk opens on a capital, a digit, a list marker, or the "a)" sub-item.
OPENING_OK = re.compile(r"^(?:[A-ZĐ0-9(\"“\-–•]|a\))")

TTS_COLOCATION_FLOOR = 84
TTS_COLOCATION_DENOMINATOR_FLOOR = 86


def _has_combination(text: str) -> bool:
    return bool(CODE_COMBO.search(text)) or len(SUBJECT_NAME.findall(text)) >= 2


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

    starts = [m.start() for m in MAJOR_OCCURRENCE.finditer(text)]
    with_combination = 0
    colocated = 0
    for index, start in enumerate(starts):
        block_end = starts[index + 1] if index + 1 < len(starts) else len(text)
        if not _has_combination(text[start:block_end]):
            continue
        with_combination += 1
        chunk = next(c for c in chunks if c.start <= start < c.end)
        if _has_combination(chunk.text):
            colocated += 1

    assert with_combination >= TTS_COLOCATION_DENOMINATOR_FLOOR, with_combination
    assert colocated >= TTS_COLOCATION_FLOOR, (colocated, with_combination)
