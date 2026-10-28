"""Contract of the Vietnamese analyzer that feeds the BM25 sparse index (ticket 07).

Every assertion here is a property the retriever depends on, not a description
of the implementation:

* a no-diacritics query must produce the same token stream as its accented form,
  otherwise the query can never match the accented chunk it is asking for;
* ``đ``/``Đ`` must fold, because Unicode gives them no canonical decomposition
  and a plain combining-mark strip silently leaves them alone;
* the original Vietnamese string must survive byte-identical, because the same
  string is served as source text and quoted as citation text;
* the exact codes (mã ngành, mã tổ hợp, số văn bản) must be recoverable from a
  query and from a chunk through the *same* function, since the retriever
  matches them by equality.
"""
from __future__ import annotations

import unicodedata

import pytest

from services.vietnamese_analyzer import (
    ANALYZER_VERSION,
    analyze,
    detect_codes,
    fold,
    tokenize,
)


class TestFold:
    """Diacritic folding of the search key."""

    @pytest.mark.parametrize(
        "accented, expected",
        [
            ("học phí", "hoc phi"),
            ("Đại học Khoa học Huế", "dai hoc khoa hoc hue"),
            ("tuyển sinh đại học", "tuyen sinh dai hoc"),
            ("điểm chuẩn", "diem chuan"),
            ("ngưỡng đảm bảo chất lượng", "nguong dam bao chat luong"),
            ("HỌC PHÍ", "hoc phi"),
            ("Tổ hợp xét tuyển", "to hop xet tuyen"),
        ],
    )
    def test_diacritics_and_case_are_folded(self, accented: str, expected: str) -> None:
        assert fold(accented) == expected

    def test_d_stroke_folds_to_d(self) -> None:
        """U+0111 has no canonical decomposition — NFD alone cannot fold it.

        This is the trap the ticket calls out: a normalizer that only strips
        combining marks leaves ``đ`` untouched, so every ``đ`` word (``điểm``,
        ``đại``, ``được``) stops matching its no-diacritics query.
        """
        assert unicodedata.normalize("NFD", "đ") == "đ"
        assert unicodedata.category("đ") != "Mn"
        assert fold("đ") == "d"
        assert fold("Đ") == "d"
        assert fold("đại") == "dai"

    def test_fold_is_invariant_to_input_decomposition(self) -> None:
        """NFC and NFD spellings of the same text fold identically."""
        nfc = unicodedata.normalize("NFC", "học phí đại học")
        nfd = unicodedata.normalize("NFD", "học phí đại học")
        assert nfc != nfd  # the two inputs really are different byte strings
        assert fold(nfc) == fold(nfd) == "hoc phi dai hoc"

    def test_fold_preserves_spacing_and_punctuation(self) -> None:
        """Folding is a character substitution, not a re-serialization."""
        assert fold("tổ hợp A00, A01") == "to hop a00, a01"
        assert fold("08/2022/TT-BGDĐT") == "08/2022/tt-bgddt"

    def test_fold_empty(self) -> None:
        assert fold("") == ""


class TestTokenize:
    """Tokens are taken from the folded key, punctuation-aware."""

    def test_code_is_not_glued_to_its_comma(self) -> None:
        """``"A00, A01"`` must yield ``a00``, not ``a00,``."""
        assert tokenize("tổ hợp A00, A01") == ["to", "hop", "a00", "a01"]

    def test_document_number_is_split_into_words(self) -> None:
        """A số văn bản is several tokens — that is exactly why the retriever
        needs the exact-code postings alongside the token stream."""
        assert "08" in tokenize("Thông tư số 08/2022/TT-BGDĐT")
        assert "bgddt" in tokenize("Thông tư số 08/2022/TT-BGDĐT")

    def test_accented_and_unaccented_queries_tokenize_identically(self) -> None:
        """The property the whole index rests on: both sides agree."""
        assert tokenize("học phí tín chỉ") == tokenize("hoc phi tin chi")

    def test_tokenize_empty(self) -> None:
        assert tokenize("") == []


class TestDetectCodes:
    """Exact codes named literally in the text."""

    def test_major_code(self) -> None:
        assert detect_codes("Ngành Công nghệ thông tin — Mã ngành: 7480201.") == ("7480201",)

    def test_to_hop_code(self) -> None:
        assert detect_codes("Xét tuyển tổ hợp A00, D01") == ("a00", "d01")

    def test_document_numbers(self) -> None:
        assert detect_codes("Thông tư số 08/2022/TT-BGDĐT") == ("08/2022/tt-bgddt",)
        assert detect_codes("Số 123/TB-ĐHH về việc xét tuyển") == ("123/tb-dhh",)

    def test_code_is_case_insensitive(self) -> None:
        """A query typed ``a00`` must key to the same code as ``A00``."""
        assert detect_codes("to hop a00") == detect_codes("tổ hợp A00") == ("a00",)

    def test_codes_are_deduplicated_in_first_seen_order(self) -> None:
        assert detect_codes("A00 rồi A01 rồi A00") == ("a00", "a01")

    def test_whitespace_inside_a_document_number_is_ignored(self) -> None:
        """``08 / 2022 / TT-BGDĐT`` is the same số văn bản."""
        assert detect_codes("08 / 2022 / TT-BGDĐT") == ("08/2022/tt-bgddt",)

    def test_plain_question_yields_no_codes(self) -> None:
        assert detect_codes("Học phí năm 2026 là bao nhiêu?") == ()
        assert detect_codes("") == ()


class TestAnalyze:
    """One pass over a text: original kept, keys derived."""

    @pytest.mark.parametrize(
        "text",
        [
            "Học phí tín chỉ ngành Công nghệ thông tin là 600.000 VNĐ",
            "Đại học Khoa học Huế — Điểm chuẩn 2026",
            "Thông tư số 08/2022/TT-BGDĐT ngày 06 tháng 6 năm 2022",
            unicodedata.normalize("NFD", "Ngưỡng đảm bảo chất lượng đầu vào"),
            "Xét tuyển tổ hợp A00, A01, D01",
        ],
    )
    def test_original_text_survives_byte_identical(self, text: str) -> None:
        """Source text and citation text must not be deformed by normalization."""
        result = analyze(text)
        assert result.original == text
        assert result.original.encode("utf-8") == text.encode("utf-8")

    def test_derived_keys_are_the_folded_forms(self) -> None:
        result = analyze("Học phí ngành 7480201")
        assert result.folded == "hoc phi nganh 7480201"
        assert result.tokens == ["hoc", "phi", "nganh", "7480201"]
        assert result.codes == ("7480201",)

    def test_no_diacritics_query_tokens_are_a_subset_of_the_chunk_tokens(self) -> None:
        """What makes a no-diacritics query find the accented chunk."""
        chunk = "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026"
        query = "hoc phi tin chi nganh cong nghe thong tin"
        assert set(tokenize(query)) <= set(tokenize(chunk))

    def test_empty_text(self) -> None:
        result = analyze("")
        assert result.original == ""
        assert result.folded == ""
        assert result.tokens == []
        assert result.codes == ()

    def test_version_is_declared(self) -> None:
        """The analyzer version is part of the corpus identity — see
        ``HybridSearchService.corpus_version``; an empty string would make two
        differently-analyzed corpora indistinguishable."""
        assert ANALYZER_VERSION.strip()
