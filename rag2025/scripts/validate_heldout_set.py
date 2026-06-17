"""Validator for the HUSC held-out eval set (anti-overfitting evaluation).

The held-out JSON file is the canonical artifact; this module exposes a
``validate_set(items)`` function that re-checks the composition invariants
independently of pytest, so it can be used by CI hooks, notebooks, or manual
audits.

Composition contract (enforced):
  * ≥40 items, unique ``ho###`` ids
  * per-category counts (with tolerance ±1):
      paraphrase=10, multi_aspect=8, comparison=6, temporal=6,
      out_of_scope=5, edge_case=5
  * every item has id/question/gold_route/expected_behavior/category/notes
  * ``gold_route`` ∈ {"padded", "hybrid", "graph"}
  * ``expected_behavior`` ∈ {"answer", "abstain", "refuse"}
  * ``category`` ∈ {paraphrase, multi_aspect, comparison, temporal,
    out_of_scope, edge_case}
  * ``out_of_scope`` items must be ``refuse``
  * every item's question must be a non-empty string
  * no question is meaningfully duplicate of any 86Q or 60Q-adversarial
    question — enforced by bidirectional 15-char substring containment
    AND ≥5-gram content-shingle overlap (after stopword removal +
    diacritic folding). Edge cases shorter than 4 content tokens are
    exempt from the shingle check (an ultra-short edge-case question
    is intentionally different from a long old one).

Run as a script:
    python rag2025/scripts/validate_heldout_set.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

# ---------- Composition contract ----------

MIN_ITEMS = 40
EXPECTED_COUNTS: Dict[str, int] = {
    "paraphrase": 10,
    "multi_aspect": 8,
    "comparison": 6,
    "temporal": 6,
    "out_of_scope": 5,
    "edge_case": 5,
}
CATEGORY_TOLERANCE = 1
TOTAL_PLANNED = sum(EXPECTED_COUNTS.values())  # 40

VALID_ROUTES = {"padded", "hybrid", "graph"}
VALID_BEHAVIORS = {"answer", "abstain", "refuse"}
VALID_CATEGORIES = set(EXPECTED_COUNTS)

ID_RX = re.compile(r"^ho\d{3}$")
# Substring containment threshold for whole-string check (chars, post-norm).
DUP_SUBSTR_LEN = 15
# Contiguous content-shingle (n-gram) length used for semantic duplication.
SHINGLE_N = 5
# Skip the shingle disjointness check when the new question has fewer than
# this many content tokens (so edge-case ultra-short questions are exempt).
MIN_CONTENT_TOKENS_FOR_SHINGLE = 4

# ---------- Stopwords (generic + domain) ----------

# These are tokens we ignore when building content-shingles.  They cover
# Vietnamese particles, pronouns, generic verbs, and very common HUSC
# boilerplate so that the disjointness check doesn't trip on shared
# "của HUSC là gì v ad"-style tails.
STOPWORDS: Set[str] = {
    # Pronouns
    "em", "toi", "minh", "ban", "anh", "chi", "ong", "ba", "chung",
    "ta", "ho", "no", "ai", "gi", "j", "z",
    # Particles / question markers
    "v", "ad", "a", "o", "e", "the", "nao", "tn", "uh", "nhe", "nha",
    "ạ",
    # Question content words
    "k", "khong", "dc", "đc", "sao", "bao", "nhieu", "may", "khi",
    "nao", "gi", "the", "nao", "tn",
    # Verbs / auxiliaries
    "co", "la", "thi", "thì", "duoc", "phai", "can", "nen", "se", "da",
    "dang", "roi", "het", "tiep", "lam", "cho", "giup", "hoi",
    "cung", "cang", "cung", "lam", "làm",
    # Connectors
    "va", "hoac", "hay", "nhung", "cac", "mot", "hai", "ba", "bon",
    "sau", "bay", "tam", "chin", "muoi",
    "tai", "tu", "den", "trong", "ngoai", "giua",
    "tren", "duoi", "neu", "vi", "boi", "do", "qua",
    "hon", "nua", "them", "thì", "thi", "ma",
    "ra", "vao", "len", "xuong", "xuong",
    "nhu", "cung", "cang", "rat",
    # Prepositions
    "cua", "voi", "o", "tai", "tu", "trong", "ngoai", "giua",
    "tren", "duoi", "de", "ra", "vao", "bang",
    # Time
    "nam", "ngay", "thang", "tuan", "gio", "h", "phut", "giay",
    "truoc", "sau", "nay", "kia", "qua", "roi",
    # Domain-generic school terms
    "truong", "dh", "husc", "cho", "toi", "theo", "cap", "moi",
    "cu", "cung", "cau", "noi", "thuoc", "chinh",
    # School verbs
    "hoc", "lam", "xet", "tuyen", "thi", "sinh", "vien", "khoa",
    # Score / grade
    "diem", "lop", "sv", "gv", "hs", "hp", "xt", "dk", "đk",
    "mon",
    # Common abbreviations that are also words
    "so", "ma", "ten", "hinh", "cach", "tiet",
    # Generic filler
    "a", "ạ",
    "tn", "tnh",
    # Disambiguation: short generic tokens we always drop
    "tb", "sv",
}

# ---------- Paths (rag2025/) ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "eval" / "husc_heldout_v1.json"
EXISTING_86Q_PATH = REPO_ROOT / "data" / "eval" / "husc_thi_sinh_thuc.json"
EXISTING_ADV_PATH = REPO_ROOT / "data" / "eval" / "husc_adversarial.json"


# ---------- Normalization ----------

_PUNCT_RX = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RX = re.compile(r"\s+", re.UNICODE)
_TOKEN_RX = re.compile(r"\w+", re.UNICODE)


def _normalize_text(q: str) -> str:
    """Lowercase, strip diacritics (NFKD + drop combining), collapse whitespace.

    The output is what we use for whole-string containment comparisons.
    Diacritics-stripped form so 'Học phí' matches 'hoc phi'.
    """
    if not isinstance(q, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", q)
    folded = "".join(c for c in nfkd if not unicodedata.combining(c))
    folded = folded.lower()
    folded = _PUNCT_RX.sub(" ", folded)
    folded = _WS_RX.sub(" ", folded).strip()
    return folded


def _tokens(q: str) -> List[str]:
    """Word tokens of the diacritic-folded text."""
    return _TOKEN_RX.findall(_normalize_text(q))


def _content_tokens(q: str) -> List[str]:
    """Tokens after dropping stopwords and 1-char tokens.

    Used for shingle construction. Returns a list (preserves order) so
    callers can build n-grams deterministically.
    """
    return [t for t in _tokens(q) if t not in STOPWORDS and len(t) > 1]


def _shingles(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    """Return set of n-gram tuples from a token sequence."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ---------- Public API ----------

def validate_set(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate composition of the held-out set.

    Loads 86Q + adversarial on its own; tests can call
    ``validate_set_with_corpus`` to inject a custom corpus.
    """
    existing_86q = _load_questions(EXISTING_86Q_PATH)
    existing_adv = _load_questions(EXISTING_ADV_PATH)
    return validate_set_with_corpus(items, existing_86q, existing_adv)


def validate_set_with_corpus(
    items: List[Dict[str, Any]],
    existing_86q: List[str],
    existing_adv: List[str],
) -> Dict[str, Any]:
    """Validate held-out set composition; return a report dict.

    Schema:
        {"ok": bool, "errors": [str, ...], "stats": {...}}
    """
    errors: List[str] = []

    if not isinstance(items, list):
        return {
            "ok": False,
            "errors": [f"items must be a list, got {type(items).__name__}"],
            "stats": {},
        }

    # 1. total
    if len(items) < MIN_ITEMS:
        errors.append(f"total: want ≥{MIN_ITEMS}, got {len(items)}")

    # 2. ids unique + well-formed
    ids = [it.get("id", "") for it in items]
    if len(set(ids)) != len(ids):
        dups = sorted(i for i, c in Counter(ids).items() if c > 1)
        errors.append(f"duplicate ids: {dups}")
    for iid in ids:
        if not ID_RX.fullmatch(iid or ""):
            errors.append(f"bad id format: {iid!r} (expected ho###)")

    # 3. required fields on every item
    REQUIRED = ("id", "question", "gold_route", "expected_behavior", "category", "notes")
    for idx, it in enumerate(items):
        for fld in REQUIRED:
            if fld not in it:
                errors.append(f"item[{idx}] id={it.get('id')!r}: missing field {fld!r}")
        q = it.get("question")
        if not isinstance(q, str) or not q.strip():
            errors.append(
                f"item {it.get('id')!r}: question must be non-empty string, got {q!r}"
            )

    # 4. gold_route validity
    for it in items:
        r = it.get("gold_route")
        if r not in VALID_ROUTES:
            errors.append(
                f"item {it.get('id')!r}: bad gold_route {r!r} "
                f"(want one of {sorted(VALID_ROUTES)})"
            )

    # 5. expected_behavior validity
    for it in items:
        eb = it.get("expected_behavior")
        if eb not in VALID_BEHAVIORS:
            errors.append(
                f"item {it.get('id')!r}: bad expected_behavior {eb!r} "
                f"(want one of {sorted(VALID_BEHAVIORS)})"
            )

    # 6. category validity + counts
    cat_counts = Counter(it.get("category") for it in items)
    for cat, want in EXPECTED_COUNTS.items():
        got = cat_counts.get(cat, 0)
        if abs(got - want) > CATEGORY_TOLERANCE:
            errors.append(
                f"category {cat}: want {want} (±{CATEGORY_TOLERANCE}), got {got}"
            )
    extras = set(cat_counts) - VALID_CATEGORIES
    if extras:
        errors.append(f"unexpected categories: {sorted(extras)}")

    # 7. out_of_scope ⇒ refuse
    for it in items:
        if it.get("category") == "out_of_scope" and it.get("expected_behavior") != "refuse":
            errors.append(
                f"item {it.get('id')!r} (out_of_scope) must be 'refuse', "
                f"got {it.get('expected_behavior')!r}"
            )

    # 8. substring disjointness vs 86Q + adversarial
    norm_86q = [_normalize_text(q) for q in existing_86q]
    norm_adv = [_normalize_text(q) for q in existing_adv]
    for it in items:
        q = it.get("question", "")
        qn = _normalize_text(q)
        if not qn:
            continue
        new_content = _content_tokens(q)
        new_shingles: Set[Tuple[str, ...]] = (
            _shingles(new_content, SHINGLE_N)
            if len(new_content) >= MIN_CONTENT_TOKENS_FOR_SHINGLE
            else set()
        )
        # Walk both corpora; report at most one error per (item, corpus) pair,
        # then stop so the report stays readable.
        for src_norm, src_raw, src_label in (
            (norm_86q, existing_86q, "86Q"),
            (norm_adv, existing_adv, "adversarial"),
        ):
            reported = False
            for prior_norm, prior_raw in zip(src_norm, src_raw):
                if not prior_norm:
                    continue
                # 8a. whole-string containment (bidirectional, ≥ DUP_SUBSTR_LEN).
                if (
                    len(qn) >= DUP_SUBSTR_LEN
                    and len(prior_norm) >= DUP_SUBSTR_LEN
                ):
                    if qn in prior_norm:
                        errors.append(
                            f"item {it.get('id')!r}: question is a substring of an "
                            f"existing {src_label} question "
                            f"({_truncate(prior_norm, 60)!r})"
                        )
                        reported = True
                        break
                    if prior_norm in qn:
                        errors.append(
                            f"item {it.get('id')!r}: an existing {src_label} question "
                            f"is a substring of this item "
                            f"({_truncate(prior_norm, 60)!r})"
                        )
                        reported = True
                        break
                # 8b. SHINGLE_N-gram content-shingle overlap.
                if new_shingles:
                    prior_shingles = _shingles(_content_tokens(prior_raw), SHINGLE_N)
                    overlap = new_shingles & prior_shingles
                    if overlap:
                        errors.append(
                            f"item {it.get('id')!r}: question shares a "
                            f"{SHINGLE_N}-gram content-shingle with an existing "
                            f"{src_label} question: {list(overlap)[0]!r} "
                            f"(prior: {_truncate(prior_norm, 60)!r})"
                        )
                        reported = True
                        break
            if reported:
                continue

    stats = {
        "total": len(items),
        "min_required": MIN_ITEMS,
        "by_category": dict(cat_counts),
        "by_route": dict(Counter(it.get("gold_route") for it in items)),
        "by_behavior": dict(Counter(it.get("expected_behavior") for it in items)),
        "corpus_86q_size": len(existing_86q),
        "corpus_adv_size": len(existing_adv),
    }
    return {"ok": not errors, "errors": errors, "stats": stats}


# ---------- Helpers (used by tests) ----------

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _load_questions(path: pathlib.Path) -> List[str]:
    """Read existing eval JSON and return its question strings.

    Missing file → empty list (callers decide whether that's an error).
    """
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [it.get("question", "") for it in data if isinstance(it, dict)]


def load_data(path: pathlib.Path = DEFAULT_DATA_PATH) -> List[Dict[str, Any]]:
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path",
        type=pathlib.Path,
        default=DEFAULT_DATA_PATH,
        help="Path to husc_heldout_v1.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any validation error.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    items = load_data(args.path)
    report = validate_set(items)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["ok"]:
        return 2 if args.strict else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
