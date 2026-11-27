"""Tests for src.utils.text: Vietnamese folding rules and the stroke letter."""
from src.utils.text import fold_diacritics, fold_diacritics_nfkd


def test_fold_strips_marks_and_maps_stroke_d():
    assert fold_diacritics("Đại Học Huế") == "dai hoc hue"


def test_fold_uppercase_stroke_d_and_marks():
    assert fold_diacritics("ĐÀ NẴNG") == "da nang"


def test_fold_keep_d_preserves_stroke_letter():
    assert fold_diacritics("Đại học", keep_d=True) == "đai hoc"


def test_fold_preserves_punctuation_and_spacing():
    assert fold_diacritics("Điểm chuẩn:  25,5") == "diem chuan:  25,5"


def test_fold_empty_string():
    assert fold_diacritics("") == ""


def test_nfkd_variant_keeps_stroke_d():
    assert fold_diacritics_nfkd("Đại học") == "đai hoc"


def test_nfkd_variant_strips_marks_and_lowercases():
    assert fold_diacritics_nfkd("Hệ THỐNG") == "he thong"


def test_combining_class_flag_differs_from_mn_rule_on_some_marks():
    import unicodedata

    from src.utils.text import fold_diacritics as f

    ch = next(
        chr(c)
        for c in range(0x80, 0x3000)
        if (unicodedata.combining(chr(c)) != 0) != (unicodedata.category(chr(c)) == "Mn")
    )
    kept = "a" + ch
    assert f(kept) != f(kept, use_combining_class=True)
    expected = "a" if unicodedata.combining(ch) else kept
    assert f(kept, use_combining_class=True) == expected
