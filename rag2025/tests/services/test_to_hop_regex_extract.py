"""Table-driven: _regex_extract / extract_from_query match every ALL_TO_HOP_CODES.

Minimal proof that regex-based TO_HOP extraction (zero LLM) works for
C01, A00, A01 and every other code in the canonical list.
"""
from __future__ import annotations

import pytest

from src.data.to_hop_codes import ALL_TO_HOP_CODES, TO_HOP_PATTERN_RE
from src.services.ner_service import _regex_extract


@pytest.mark.parametrize("code", ALL_TO_HOP_CODES)
def test_to_hop_pattern_matches_every_code(code: str) -> None:
    text = f"xét tuyển tổ hợp {code}"
    matches = [m.group(0).upper() for m in TO_HOP_PATTERN_RE.finditer(text)]
    assert code in matches, f"TO_HOP_PATTERN_RE failed to match {code}"


def test_extract_from_query_to_hop_seeds() -> None:
    from src.services.ner_service import NERService

    svc = NERService(llm=None)
    svc._llm = None  # prevent auto-init of UnifiedLLMClient

    seeds = svc.extract_from_query("ngành CNTT xét tuyển C01, A00, A01")
    tohop = sorted(s for s in seeds if s.startswith("TO_HOP:"))

    assert tohop == ["TO_HOP:a00", "TO_HOP:a01", "TO_HOP:c01"]


@pytest.mark.parametrize("code", ["C01", "A00", "A01"])
def test_regex_extract_to_hop(code: str) -> None:
    result = _regex_extract(f"tổ hợp {code}")
    tohop = [e for e in result["entities"] if e["type"] == "TO_HOP"]
    assert len(tohop) >= 1
    assert tohop[0]["normalized"] == code.lower()
