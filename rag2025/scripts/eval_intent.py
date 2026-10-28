"""Intent taxonomy + deterministic classifier for the per-intent eval harness.

Ticket 08. This module is the SINGLE SOURCE OF TRUTH for the question

    "which intent bucket does this eval question belong to?"

It is deliberately dependency-light (stdlib + the router's own regexes) so both
the runner (``eval_run_instrumented.py``) and the report
(``eval_per_intent_report.py``) can import it without side effects.

Taxonomy
--------
``simple`` | ``multihop`` | ``comparative``

Where intent comes from (priority order)
---------------------------------------
1. **dataset label** — the questions file's ``category`` (or ``intent``) field,
   used ONLY when it names a taxonomy class:

       simple                     -> simple
       multihop | multi_hop       -> multihop
       comparison | comparative   -> comparative
       so_sanh                    -> comparative

   Labels that are not taxonomy classes (``temporal``, ``paraphrase``,
   ``multi_aspect``, ``out_of_scope``, ``edge_case``, ...) are NOT used as the
   bucket — they fall through to the derived rule. The manifest records both the
   raw label and the bucket, so nothing is hidden.

   Which datasets carry a label (measured, not assumed):
     * ``data/eval/husc_86q_test.json``  — 303 questions, ``category``
       ∈ {simple:119, multihop:119, temporal:65}
     * ``data/eval/husc_heldout_v1.json`` — 40 questions, ``category``
       ∈ {paraphrase, multi_aspect, comparison, temporal, out_of_scope, edge_case}
     * ``data/eval/husc_thi_sinh_thuc.json`` — 86 questions, **no intent label
       at all** (only ``id`` + ``question``) → derived rule only
     * ``data/eval/router_model_benchmark_gold.json`` — route labels
       (padded/hybrid/graph), NOT intents → derived rule only

2. **derived rule** — deterministic text rule, documented in
   :func:`derive_intent`. Used for every question with no taxonomy label.

The router's own ``intent`` field (``so_sanh`` / ``liet_ke`` / ``general``) is
NOT used: it comes from an LLM classifier that is dead in this environment
(both providers 403/401), so every baseline record falls back to
``intent="general"`` and the field carries no signal. The manifest reports that
distribution as ``intent.diagnostics.router_intent`` so the claim is auditable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Bootstrap the same import paths the runner uses, so `services.query_router`
# resolves whether this module is imported from scripts/, tests/, or a CLI.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent  # rag2025/
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the router's own deterministic pattern sets — do NOT fork a second
# definition of "what is a comparison / enumeration question".
from services.query_router import (  # noqa: E402
    _COMPARISON_PATTERNS,
    _ENUMERATION_PATTERNS,
)

INTENT_TAXONOMY: Tuple[str, ...] = ("simple", "multihop", "comparative")

# Dataset label -> taxonomy bucket. Only labels that NAME a taxonomy class.
LABEL_TO_INTENT: Dict[str, str] = {
    "simple": "simple",
    "multihop": "multihop",
    "multi_hop": "multihop",
    "comparison": "comparative",
    "comparative": "comparative",
    "so_sanh": "comparative",
}

# ---------------------------------------------------------------------------
# Derived rule — the non-overlapping attribute lexicon.
#
# Overlaps are collapsed on purpose: "học phí tín chỉ" is ONE attribute (tuition
# per credit), and generic words like "ngành"/"tuyển sinh" are entities, not
# attributes, so they are excluded. A question that names >=2 DISTINCT
# attributes has more than one information need -> multihop.
# ---------------------------------------------------------------------------
ATTRIBUTE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("hoc_phi", r"học\s+phí|tín\s+chỉ"),
    ("diem_chuan", r"điểm\s+chuẩn|điểm\s+trúng\s+tuyển|điểm\s+xét\s+tuyển"),
    ("chi_tieu", r"chỉ\s+tiêu"),
    ("to_hop", r"tổ\s+hợp|khối\s+thi"),
    ("phuong_thuc", r"phương\s+thức"),
    ("hoc_ba", r"học\s+bạ"),
    ("ho_so", r"hồ\s+sơ"),
    ("dang_ky", r"đăng\s+ký"),
    ("hoc_bong", r"học\s+bổng"),
    ("ky_tuc_xa", r"ký\s+túc\s+xá"),
    ("dia_chi", r"địa\s+chỉ"),
    ("lien_he", r"hotline|số\s+điện\s+thoại|tổng\s+đài|email"),
    ("ma_truong", r"mã\s+trường|mã\s+ngành"),
    ("thoi_gian", r"thời\s+gian"),
    ("le_phi", r"lệ\s+phí"),
    ("uu_tien", r"ưu\s+tiên"),
)

_ATTRIBUTE_RES = tuple(
    (key, re.compile(pattern, re.IGNORECASE)) for key, pattern in ATTRIBUTE_PATTERNS
)

# Aggregate / superlative asks need a roll-up over many entities -> multihop.
_AGGREGATION_PATTERNS = re.compile(
    r"tổng\s+(?:số|cộng|chỉ\s+tiêu|học\s+phí|điểm)|"
    r"cao\s+nhất|thấp\s+nhất|nhiều\s+nhất|ít\s+nhất|trung\s+bình|"
    r"bao\s+nhiêu\s+ngành",
    re.IGNORECASE,
)

# Year-over-year movement -> comparative (needs both years' facts).
_DELTA_PATTERNS = re.compile(r"tăng|giảm|thay\s+đổi|chênh", re.IGNORECASE)

_YEAR_PATTERN = re.compile(r"\b20\d{2}\b")

INTENT_RULE_TEXT = (
    "priority 1: dataset label mapped to the taxonomy when it names a class "
    "(simple|multihop|multi_hop|comparison|comparative|so_sanh); "
    "priority 2 (derived rule): comparative <- services.query_router._COMPARISON_PATTERNS "
    "OR (>=2 distinct years AND a delta verb tang|giam|thay doi|chenh); "
    "multihop <- services.query_router._ENUMERATION_PATTERNS OR an aggregation marker "
    "(tong ..., cao nhat, thap nhat, nhieu nhat, it nhat, trung binh, bao nhieu nganh) "
    "OR >=2 distinct attributes (ATTRIBUTE_PATTERNS) OR >=2 distinct years; "
    "simple <- otherwise."
)


def normalize_label(label: Optional[str]) -> Optional[str]:
    """Fold a raw dataset label to its lookup key (lowercase, `_`-joined)."""
    if not label or not isinstance(label, str):
        return None
    return re.sub(r"[\s\-]+", "_", label.strip().lower()) or None


def derive_intent(question: str) -> str:
    """Apply the stated text rule; first match wins (comparative > multihop > simple)."""
    text = question or ""

    if _COMPARISON_PATTERNS.search(text):
        return "comparative"
    years = set(_YEAR_PATTERN.findall(text))
    if len(years) >= 2 and _DELTA_PATTERNS.search(text):
        return "comparative"

    if _ENUMERATION_PATTERNS.search(text) or _AGGREGATION_PATTERNS.search(text):
        return "multihop"

    attributes = {
        key for key, pattern in _ATTRIBUTE_RES if pattern.search(text)
    }
    if len(attributes) >= 2:
        return "multihop"
    if len(years) >= 2:
        return "multihop"

    return "simple"


def classify_intent(
    question: str, label: Optional[str] = None
) -> Tuple[str, str]:
    """Return ``(intent, source)``.

    ``source`` is ``"dataset_label"`` when the dataset named a taxonomy class,
    else ``"derived_rule"``.
    """
    key = normalize_label(label)
    if key in LABEL_TO_INTENT:
        return LABEL_TO_INTENT[key], "dataset_label"
    return derive_intent(question), "derived_rule"


def _taxonomy_label(item: Dict[str, Any]) -> Optional[str]:
    """The item's OWN dataset label, if it carries one.

    Only ``category`` counts: every labelled dataset in ``data/eval/`` uses
    that field, while EvalRecords carry an ``intent`` field in the ROUTER's
    vocabulary (``general``/``so_sanh``/``liet_ke``) which must NOT be mistaken
    for a dataset label (``so_sanh`` would otherwise silently bucket as
    ``comparative``).
    """
    value = item.get("category")
    if value and normalize_label(str(value)) in LABEL_TO_INTENT:
        return str(value)
    return None


def select_ci_subset(
    questions: Sequence[Dict[str, Any]],
    n: int,
    label_lookup: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Deterministic, intent-stratified subset for CI runs.

    Round-robins over the taxonomy buckets (``simple``, ``multihop``,
    ``comparative``) in that order, preserving each bucket's dataset order, so
    the subset always spans every non-empty bucket when ``n`` >= the number of
    non-empty buckets. Buckets that run dry are skipped; the result has
    ``min(n, len(questions))`` questions and is a pure function of the input
    order (same input -> same subset).

    Bucketing uses the SAME intent source as the report: an item's own
    ``category``/``intent`` field, else ``label_lookup[id]`` (the dataset's
    labels, for records that do not carry them), else the derived text rule.
    """
    if n is None or n <= 0:
        return list(questions)

    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in INTENT_TAXONOMY}
    for q in questions:
        label = _taxonomy_label(q)
        if label is None and label_lookup:
            label = label_lookup.get(str(q.get("id")))
        intent, _ = classify_intent(q.get("question", ""), label)
        buckets[intent].append(q)

    cursors: Dict[str, int] = {k: 0 for k in INTENT_TAXONOMY}
    out: List[Dict[str, Any]] = []
    while len(out) < n:
        progressed = False
        for intent in INTENT_TAXONOMY:
            cursor = cursors[intent]
            if cursor < len(buckets[intent]) and len(out) < n:
                out.append(buckets[intent][cursor])
                cursors[intent] = cursor + 1
                progressed = True
        if not progressed:
            break
    return out


def intent_counts(questions: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Bucket counts for a question list (used by CI-subset logging/tests)."""
    counts = {k: 0 for k in INTENT_TAXONOMY}
    for q in questions:
        intent, _ = classify_intent(
            q.get("question", ""), q.get("category") or q.get("intent")
        )
        counts[intent] += 1
    return counts
