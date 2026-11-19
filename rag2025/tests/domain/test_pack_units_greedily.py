"""Legal-boundary packer invariants on synthetic text (no source PDF needed).

Budgets under test: floor 50, target 350, hard cap 512, counted in cl100k_base
tokens with any Điều / major-code prefix included in the count.
"""

from __future__ import annotations

import re

import pytest
import tiktoken

from src.domain.canonical.semantic_units import _pack_units_greedily

ENC = tiktoken.get_encoding("cl100k_base")
MIN_TOKENS = 50
TARGET_TOKENS = 350
HARD_MAX_TOKENS = 512


def _pack(text: str):
    return _pack_units_greedily(
        text,
        enc=ENC,
        min_tokens=MIN_TOKENS,
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )


def _tokens(text: str) -> int:
    return len(ENC.encode(text))


def _sentences(count: int, start: int = 0) -> list[str]:
    return [
        f"Thí sinh phải nộp hồ sơ đúng thời hạn quy định tại khoản này, mục số {i}."
        for i in range(start, start + count)
    ]


def test_every_chunk_fits_the_hard_cap_with_its_prefix():
    text = "Điều 1. Phạm vi điều chỉnh\n" + "\n".join(_sentences(300))

    chunks = _pack(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _tokens(chunk.text) <= HARD_MAX_TOKENS


def test_a_long_plain_paragraph_splits_into_budgeted_chunks():
    text = " ".join(_sentences(60))

    chunks = _pack(text)

    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert MIN_TOKENS <= _tokens(chunk.text) <= HARD_MAX_TOKENS


def test_only_the_last_chunk_of_a_dieu_may_fall_below_the_floor():
    text = (
        "Điều 1. Phạm vi điều chỉnh\n" + "\n".join(_sentences(120))
        + "\nĐiều 2. Đối tượng áp dụng\n" + "\n".join(_sentences(3))
    )

    chunks = _pack(text)

    for index, chunk in enumerate(chunks):
        if _tokens(chunk.text) >= MIN_TOKENS:
            continue
        is_last = index == len(chunks) - 1
        next_same_dieu = not is_last and chunks[index + 1].label == chunk.label
        assert not next_same_dieu, f"short chunk {index} is not the end of its Điều"


def test_a_bare_dieu_heading_is_never_a_chunk_on_its_own():
    text = "Điều 1. Phạm vi điều chỉnh\n" + "\n".join(_sentences(40))

    chunks = _pack(text)

    for chunk in chunks:
        lines = [line for line in chunk.text.splitlines() if line.strip()]
        assert not (len(lines) == 1 and re.match(r"Điều\s+\d+", lines[0])), chunk.text


def test_an_undotted_dieu_heading_opens_a_new_chunk():
    text = (
        "Điều 4. Đối tượng tuyển sinh\n" + "\n".join(_sentences(40))
        + "\nĐiều 5 Quy định chung\n" + "\n".join(_sentences(40, start=40))
    )

    chunks = _pack(text)

    first_of_dieu_5 = next(chunk for chunk in chunks if chunk.label == "Điều 5")
    assert first_of_dieu_5.text.startswith("Điều 5. Quy định chung")


def test_a_chapter_heading_is_not_appended_to_the_dieu_body():
    text = (
        "Điều 1. Phạm vi điều chỉnh\n" + "\n".join(_sentences(5))
        + "\nChương II\n"
        + "Điều 2. Đối tượng áp dụng\n" + "\n".join(_sentences(5, start=5))
    )

    chunks = _pack(text)

    assert all("Chương II" not in chunk.text for chunk in chunks)


def test_no_chunk_starts_mid_sentence():
    text = "Điều 1. Phạm vi điều chỉnh\n" + "\n".join(_sentences(200))

    chunks = _pack(text)

    for chunk in chunks:
        first = chunk.text.lstrip()[0]
        assert first.isupper() or first.isdigit() or chunk.text.lstrip().startswith("a)"), (
            chunk.text[:60]
        )


@pytest.mark.parametrize("sentence_count", [1, 3])
def test_a_small_text_stays_one_chunk(sentence_count):
    text = " ".join(_sentences(sentence_count))

    chunks = _pack(text)

    assert len(chunks) == 1
    assert _tokens(chunks[0].text) <= HARD_MAX_TOKENS


@pytest.mark.parametrize("small_words", [5, 10, 20])
def test_a_short_mid_dieu_fragment_borrows_from_its_neighbour(small_words):
    """A short fragment between two large pieces must not stand alone under the floor."""
    big = " ".join(f"Thí sinh nộp hồ sơ đúng hạn số {i}." for i in range(120))
    small = " ".join(["ngắn"] * small_words)
    text = "Điều 3. Hồ sơ dự tuyển\n" + big + "\n" + small + "\n" + big
    chunks = _pack(text)
    for index, chunk in enumerate(chunks[:-1]):
        assert MIN_TOKENS <= _tokens(chunk.text) <= HARD_MAX_TOKENS, (index, _tokens(chunk.text))
