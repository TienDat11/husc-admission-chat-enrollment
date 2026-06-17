"""Adversarial eval set generator / validator (S4 — hallucination pressure test).

The adversarial JSON file is the canonical artifact; this module exposes a
``validate_set(items)`` function that re-checks the composition invariants
independently of pytest, so it can be used by CI hooks, notebooks, or manual
audits.

Composition contract (enforced):
  * exactly 60 items, unique ``adv###`` ids
  * per-category counts: temporal=15, fabricated_entity=15, out_of_scope=10,
    multi_hop=10, numeric_precision=10
  * ``expected_behavior`` ∈ {"answer", "abstain", "refuse"}
  * every out_of_scope item must be ``refuse``
  * every fabricated_entity item must carry a ``trap_code`` that FAILS
    ``is_valid_2026_major`` (C11)
  * every temporal item's question must contain a 4-digit year (regex
    ``\\b20\\d{2}\\b``); and there must be ≥5 minimal year-pairs
  * every numeric_precision item must have a non-empty ``trap_value``

Run as a script:
    python rag2025/scripts/gen_adversarial_set.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

# ---------- Composition contract ----------

EXPECTED_COUNTS: Dict[str, int] = {
    "temporal": 15,
    "fabricated_entity": 15,
    "out_of_scope": 10,
    "multi_hop": 10,
    "numeric_precision": 10,
}
TOTAL = sum(EXPECTED_COUNTS.values())  # 60
VALID_BEHAVIORS = {"answer", "abstain", "refuse"}
YEAR_RX = re.compile(r"\b20\d{2}\b")
ID_RX = re.compile(r"^adv\d{3}$")

# Path to the JSON artifact (sibling of scripts/ at rag2025/data/eval/)
DEFAULT_DATA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "eval"
    / "husc_adversarial.json"
)

# Path to the validator module (rag2025/src/services/major_code_validator.py)
_VALIDATOR_SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"


def _import_validator():
    """Lazy import the C11 validator without hard-failing the module."""
    sys.path.insert(0, str(_VALIDATOR_SRC_DIR))
    from services.major_code_validator import is_valid_2026_major  # type: ignore

    return is_valid_2026_major


# ---------- Public API ----------

def validate_set(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate composition of the adversarial set. Returns a report dict:

        {"ok": bool, "errors": [str, ...], "stats": {...}}

    The script-side intent is fast pre-flight; the pytest tests give
    fine-grained per-rule enforcement.
    """
    errors: List[str] = []

    if not isinstance(items, list):
        return {"ok": False, "errors": [f"items must be a list, got {type(items).__name__}"], "stats": {}}

    # 1. total
    if len(items) != TOTAL:
        errors.append(f"total: want {TOTAL}, got {len(items)}")

    # 2. ids unique + well-formed
    ids = [it.get("id", "") for it in items]
    if len(set(ids)) != len(ids):
        dups = sorted(i for i, c in Counter(ids).items() if c > 1)
        errors.append(f"duplicate ids: {dups}")
    for i in ids:
        if not ID_RX.fullmatch(i):
            errors.append(f"bad id format: {i!r} (expected adv###)")

    # 3. category counts
    cat_counts = Counter(it.get("category") for it in items)
    for cat, want in EXPECTED_COUNTS.items():
        got = cat_counts.get(cat, 0)
        if got != want:
            errors.append(f"category {cat}: want {want}, got {got}")
    extras = set(cat_counts) - set(EXPECTED_COUNTS)
    if extras:
        errors.append(f"unexpected categories: {sorted(extras)}")

    # 4. expected_behavior
    for it in items:
        eb = it.get("expected_behavior")
        if eb not in VALID_BEHAVIORS:
            errors.append(
                f"item {it.get('id')!r}: bad expected_behavior {eb!r}"
            )

    # 5. out_of_scope ⇒ refuse
    for it in items:
        if it.get("category") == "out_of_scope" and it.get("expected_behavior") != "refuse":
            errors.append(
                f"item {it.get('id')!r} (out_of_scope) must be 'refuse', "
                f"got {it.get('expected_behavior')!r}"
            )

    # 6. fabricated_entity trap_code must fail is_valid_2026_major
    is_valid_2026_major = _import_validator()
    for it in items:
        if it.get("category") == "fabricated_entity":
            tc = it.get("trap_code")
            if tc is None:
                errors.append(
                    f"item {it.get('id')!r} (fabricated_entity) missing trap_code"
                )
                continue
            if is_valid_2026_major(tc):
                errors.append(
                    f"item {it.get('id')!r}: trap_code {tc!r} IS a valid 2026 code"
                )

    # 7. temporal: 4-digit year + ≥5 minimal pairs
    temps = [it for it in items if it.get("category") == "temporal"]
    for it in temps:
        q = it.get("question", "")
        if not YEAR_RX.search(q):
            errors.append(
                f"item {it.get('id')!r} (temporal) has no 4-digit year: {q!r}"
            )
    normalized = Counter(YEAR_RX.sub("YYYY", it.get("question", "")) for it in temps)
    pairs = [q for q, n in normalized.items() if n >= 2]
    if len(pairs) < 5:
        errors.append(
            f"temporal: need ≥5 minimal year-pairs, found {len(pairs)}"
        )

    # 8. numeric_precision trap_value
    for it in items:
        if it.get("category") == "numeric_precision":
            tv = it.get("trap_value")
            if not (isinstance(tv, str) and tv.strip()):
                errors.append(
                    f"item {it.get('id')!r} (numeric_precision) needs non-empty trap_value, got {tv!r}"
                )

    stats = {
        "total": len(items),
        "by_category": dict(cat_counts),
        "by_behavior": dict(Counter(it.get("expected_behavior") for it in items)),
        "temporal_year_pairs": len(pairs),
    }
    return {"ok": not errors, "errors": errors, "stats": stats}


def load_data(path: pathlib.Path = DEFAULT_DATA_PATH) -> List[Dict[str, Any]]:
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path",
        type=pathlib.Path,
        default=DEFAULT_DATA_PATH,
        help="Path to husc_adversarial.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any validation error.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    items = load_data(args.path)
    report = validate_set(items)

    # Print compact report
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["ok"]:
        return 2 if args.strict else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
