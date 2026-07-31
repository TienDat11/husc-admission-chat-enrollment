"""Core tests for Vietnamese supersession detector."""
from __future__ import annotations

import sys
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.release import TransitionKind  # noqa: E402
from src.services.supersession_detector import (  # noqa: E402
    _is_negative_context,
    detect,
    extract_effective_date,
    normalize,
)


def test_normalize_nfc_nbsp_whitespace() -> None:
    text = "Căn cu Thông tu  số 08;\n\n\nNội dung"
    result = normalize(text)
    assert unicodedata.is_normalized("NFC", result)
    assert " " not in result
    assert "  " not in result
    assert "\n\n\n" not in result


def test_negative_context_cancu() -> None:
    text = "Căn cứ Thông tư số 08/2022/TT-BGDĐT ngày 06/6/2022 của Bộ GDĐT;"
    start = text.index("Thông tư")
    assert _is_negative_context(text, (start, start + len("Thông tư"))) is True


def test_negative_context_false_on_replaces() -> None:
    text = "Thông tư này thay thế Thông tư số 09/2020/TT-BGDĐT ngày 07/5/2020."
    start = text.index("thay thế")
    assert _is_negative_context(text, (start, start + len("thay thế"))) is False


def test_effective_date_parse() -> None:
    text = "Thông tư này có hiệu lực thi hành từ ngày 22 tháng 7 năm 2022."
    assert extract_effective_date(text) == date(2022, 7, 22)


def test_detect_replaces() -> None:
    text = "Thông tư này thay thế Thông tư số 09/2020/TT-BGDĐT ngày 07/5/2020."
    candidates = detect(text, source_doc_id="source-1")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.kind == TransitionKind.REPLACES
    assert candidate.confidence >= 0.85
    assert "thay thế Thông tư số 09/2020/TT-BGDĐT" in candidate.evidence_span


def test_detect_repeals() -> None:
    text = "Bãi bỏ toàn bộ Quyết định số 999/QĐ-ĐHKH ngày 01/01/2020."
    candidates = detect(text, source_doc_id="source-2")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.kind == TransitionKind.REPEALS
    assert candidate.confidence >= 0.85


def test_detect_amends() -> None:
    text = "Thông tư này sửa đổi, bổ sung một số điều của Quy chế tuyển sinh ban hành theo Thông tư 08/2022."
    candidates = detect(text, source_doc_id="t")
    assert len(candidates) >= 1
    assert any(candidate.kind == TransitionKind.AMENDS for candidate in candidates)


def test_detect_corrects() -> None:
    text = "Quyết định này đính chính nội dung tại Điều 5 của Quyết định số 1234/QĐ-ĐHKH."
    candidates = detect(text, source_doc_id="t")
    assert len(candidates) >= 1
    assert any(candidate.kind == TransitionKind.CORRECTS for candidate in candidates)


def test_no_false_positive_on_cancu() -> None:
    text = "Căn cứ Thông tư số 08/2022/TT-BGDĐT ngày 06/6/2022 của Bộ GDĐT;"
    result = detect(text, source_doc_id="t")
    assert len(result) == 0
