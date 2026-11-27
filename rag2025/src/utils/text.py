"""Vietnamese text folding shared by search, guard and parser code.

Two rules exist on purpose; callers pick the one whose results they already
depend on:

- :func:`fold_diacritics` lower-cases first, then NFD-decomposes and drops
  combining marks; đ maps to d unless ``keep_d``.
- :func:`fold_diacritics_nfkd` NFKD-decomposes first, drops combining marks,
  then lower-cases; đ (U+0111, no decomposition) is kept.
"""
from __future__ import annotations

import unicodedata


def fold_diacritics(text: str, *, keep_d: bool = False, use_combining_class: bool = False) -> str:
    """Lower-case ``text`` and strip Vietnamese diacritics, character by character.

    Every character other than a combining mark is preserved, so whitespace and
    punctuation pass through unchanged. Đ/đ become d/d unless ``keep_d`` is
    true. ``use_combining_class`` selects the mark test (see ``_is_mark``).
    """
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", text.lower()):
        if ch == "đ":
            out.append(ch if keep_d else "d")
        elif _is_mark(ch, use_combining_class):
            continue
        else:
            out.append(ch)
    return "".join(out)


def _is_mark(ch: str, use_combining_class: bool) -> bool:
    # Mn (nonspacing mark) is the analyzer/admin rule. The source-document header
    # rule historically dropped anything with a nonzero combining class instead;
    # the two differ on about a thousand code points, so each caller keeps its own.
    if use_combining_class:
        return unicodedata.combining(ch) != 0
    return unicodedata.category(ch) == "Mn"


def fold_diacritics_nfkd(text: str) -> str:
    """NFKD-decompose, drop combining marks, then lower-case.

    Used by the guard matchers. The stroke letter đ survives this rule, so
    the guards keep their own đ handling.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
