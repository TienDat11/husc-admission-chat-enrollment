"""Vietnamese analyzer for BM25 sparse retrieval (ticket 07).

ONE analyzer, applied identically to the INDEX side and the QUERY side — a BM25
index is only meaningful when both sides produce the same token stream.

Three jobs:

1. :func:`fold` — build the diacritic-folded SEARCH KEY. The original Vietnamese
   text is never rewritten: folding produces an *additional* key, so source text
   and citation text stay byte-identical end to end.
2. :func:`tokenize` — punctuation-aware tokens over the folded key, so
   ``"tổ hợp A00, A01"`` yields ``a00`` / ``a01`` instead of ``a00,`` / ``a01,``
   (whitespace ``split()`` keeps the comma attached, so the code never matches).
3. :func:`detect_codes` — the exact codes a text names literally (mã ngành, mã
   tổ hợp, số văn bản). The retriever boosts chunks that DECLARE that code,
   instead of hoping the code survives tokenization among the other terms.

Vietnamese specifics
--------------------
* Diacritics are combining marks (category ``Mn``) once decomposed to NFD, so
  dropping ``Mn`` folds ``học phí`` -> ``hoc phi``.
* ``đ`` / ``Đ`` are **not** ``d`` + a combining mark: Unicode defines no
  canonical decomposition for U+0111 / U+0110, so ``normalize("NFD", "đ")``
  returns ``"đ"`` unchanged and a bare ``Mn``-strip leaves it alone. They must
  be mapped explicitly, or every ``đ`` word (``điểm``, ``đại``, ``được``) stops
  matching its no-diacritics query.
* ``ă`` / ``â`` / ``ê`` / ``ô`` / ``ơ`` / ``ư`` **do** decompose (``ơ`` is
  ``o`` + U+031B, ``ư`` is ``u`` + U+031B), so the ``Mn``-strip handles them.

The code patterns are the repo's canonical ones, imported rather than forked:
``services.major_code_validator``, ``data.to_hop_codes`` and
``services.vn_legal_structure``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

from services.major_code_validator import _CODE_IN_TEXT_RX
from services.vn_legal_structure import RX_DOC_CENTRAL, RX_DOC_LOCAL
from src.data.to_hop_codes import TO_HOP_PATTERN_RE

# Bump when the token stream or the folding rules change: this string is part of
# the BM25 corpus identity (`HybridSearchService.corpus_version`), because two
# differently-analyzed corpora are not comparable indexes.
ANALYZER_VERSION = "vi-bm25-1"

# Word characters after folding. Punctuation is the separator, so an
# adjacent comma / slash never rides along on a token.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

# The canonical code shapes, in the repo's own definitions:
#   mã ngành   7-digit base + optional 2-letter variant suffix (7480201VJ)
#   mã tổ hợp  letter + 2 digits (A00, D01, X26) — the canonical HUSC list
#   số văn bản 08/2022/TT-BGDĐT (central) and 123/TB-ĐHH (internal)
_CODE_PATTERNS: Tuple[re.Pattern, ...] = (
    _CODE_IN_TEXT_RX,
    TO_HOP_PATTERN_RE,
    RX_DOC_CENTRAL,
    RX_DOC_LOCAL,
)


def fold(text: str) -> str:
    """Return the diacritic-folded search key for ``text``.

    Case-folded and diacritic-stripped; every other character (including
    whitespace and punctuation) is preserved, so the mapping is a pure
    character substitution and is cheap to reason about. ``text`` itself is
    never mutated — callers keep the original string.
    """
    if not text:
        return ""
    # str.lower() maps "Đ" -> "đ", so the explicit đ mapping below covers both.
    out: List[str] = []
    for ch in unicodedata.normalize("NFD", text.lower()):
        if ch == "đ":
            out.append("d")
        elif ord(ch) < 128:
            # Fast path: ASCII is never a combining mark.
            out.append(ch)
        elif unicodedata.category(ch) != "Mn":
            out.append(ch)
    return "".join(out)


def tokenize(text: str) -> List[str]:
    """Token stream for BM25: word characters of the folded search key.

    No stopword filtering: BM25's IDF already discounts corpus-wide terms, and
    dropping tokens would change the scoring surface the frozen retrieval
    baseline was measured on for no gain in recall.
    """
    return _TOKEN_RE.findall(fold(text))


def detect_codes(text: str) -> Tuple[str, ...]:
    """Exact codes named in ``text``, canonicalized and de-duplicated.

    Canonical form = folded + whitespace-stripped, so ``"08 / 2022 / TT-BGDĐT"``
    and ``"08/2022/TT-BGDĐT"`` produce the same key, and the same function run
    over a chunk produces the same key for the same code. Detection order is
    document order within each pattern, and pattern order is fixed, so the
    result is deterministic.
    """
    if not text:
        return ()
    found: List[str] = []
    seen: set = set()
    for rx in _CODE_PATTERNS:
        for match in rx.finditer(text):
            raw = match.group(1) if match.lastindex else match.group(0)
            code = _WHITESPACE_RE.sub("", fold(raw))
            if code and code not in seen:
                seen.add(code)
                found.append(code)
    return tuple(found)


@dataclass(frozen=True)
class AnalyzedText:
    """One text seen through the analyzer.

    ``original`` is the untouched input (byte-identical); ``folded`` and
    ``tokens`` are the derived search keys; ``codes`` are the exact codes the
    text names.
    """

    original: str
    folded: str
    tokens: List[str] = field(default_factory=list)
    codes: Tuple[str, ...] = ()


def analyze(text: str) -> AnalyzedText:
    """Analyze ``text`` once, sharing the folding work across all three jobs."""
    if not text:
        return AnalyzedText(original=text or "", folded="", tokens=[], codes=())
    folded = fold(text)
    return AnalyzedText(
        original=text,
        folded=folded,
        tokens=_TOKEN_RE.findall(folded),
        codes=detect_codes(text),
    )
