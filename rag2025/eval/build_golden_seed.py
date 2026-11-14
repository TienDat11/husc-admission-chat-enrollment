"""Build the U1 golden seed deterministically (plan §PL-6.6, unit U1).

Input: the legacy golden records file (gitignored, read-only):
    data/eval/husc86q_golden_records.json
Output (committed, derived data):
    tests/eval/fixtures/golden_seed.jsonl
    tests/eval/fixtures/legacy_id_mapping.json

Mapping rule (legacy record id -> coordinate_key):
    legacy  = RECORD_TYPE:scope:subject:year:source_tag[:more...]
    new     = RECORD_TYPE:scope:subject:year:none:methods_hash
  - ``admission_round`` is null in the source, so the round slot is the fixed
    token ``none`` (no token is defined by §PL-6.6 for a missing value);
  - the legacy 5th segment is a source tag, not a round: it is kept in the
    mapping file as ``legacy_source_tag`` and is NOT part of the key;
  - ``subject`` keeps its program suffix only when the 7-digit major code has
    more than one subject variant in the source (e.g. 7480201 and
    7480201-ky-su-viet-nhat); otherwise the bare code is used;
  - ``methods_hash`` defaults to ``"all"`` (the legacy file has no method
    restriction).

Each default used is listed in the mapping file's ``defaults``.

Run: ``python eval/build_golden_seed.py``. Output is byte-identical across runs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "eval" / "husc86q_golden_records.json"
DEFAULT_OUT = ROOT / "tests" / "eval" / "fixtures" / "golden_seed.jsonl"
DEFAULT_MAPPING = ROOT / "tests" / "eval" / "fixtures" / "legacy_id_mapping.json"

METHODS_HASH_ALL = "all"
ROUND_NONE = "none"
_SUBJECT_CODE = re.compile(r"^\d{7}")

# Hand-written out-of-scope questions (plan U1: >=10, five groups). These must
# be refused: expect_abstain=true.
OUT_OF_SCOPE: list[dict[str, str]] = [
    {"group": "truong_khac", "question": "Học phí ngành Y khoa tại Đại học Y Hà Nội năm 2026 là bao nhiêu?"},
    {"group": "truong_khac", "question": "Điểm chuẩn ngành Luật của Đại học Kinh tế Quốc dân năm 2026 là bao nhiêu?"},
    {"group": "tin_tuc_thoi_tiet", "question": "Thời tiết Huế hôm nay thế nào?"},
    {"group": "tin_tuc_thoi_tiet", "question": "Tin tức bóng đá hôm nay có gì mới?"},
    {"group": "tu_van_ngoai_tuyen_sinh", "question": "Nên học tiếng Anh bằng cách nào để giao tiếp tốt?"},
    {"group": "tu_van_ngoai_tuyen_sinh", "question": "Cho tôi công thức nấu phở bò."},
    {"group": "nam_chua_cong_bo", "question": "Điểm chuẩn ngành Công nghệ thông tin năm 2030 là bao nhiêu?"},
    {"group": "nam_chua_cong_bo", "question": "Học phí năm 2031 của trường HUSC là bao nhiêu?"},
    {"group": "chao_hoi", "question": "Xin chào, bạn là ai?"},
    {"group": "chao_hoi", "question": "Cảm ơn bạn nhiều nhé!"},
]


def _parse_legacy(legacy_id: str) -> tuple[str, str, str, str, str]:
    parts = legacy_id.split(":")
    if len(parts) < 5:
        raise ValueError(f"legacy id has fewer than 5 segments: {legacy_id!r}")
    record_type, scope, subject, year = parts[:4]
    source_tag = ":".join(parts[4:])
    if not _SUBJECT_CODE.match(subject):
        raise ValueError(f"subject does not start with a 7-digit major code: {legacy_id!r}")
    return record_type, scope, subject, year, source_tag


def _program_variants(legacy_ids: list[str]) -> dict[str, set[str]]:
    """Distinct raw subjects per 7-digit major code across the whole source."""
    variants: dict[str, set[str]] = defaultdict(set)
    for legacy in legacy_ids:
        _, _, subject, _, _ = _parse_legacy(legacy)
        variants[subject[:7]].add(subject)
    return variants


def legacy_to_fact_and_key(
    legacy_id: str, variants: dict[str, set[str]], defaults: set[str]
) -> tuple[tuple[str, str, str, str], str, str]:
    """Return (fact identity, coordinate_key, legacy_source_tag) for one legacy id."""
    record_type, scope, subject, year, source_tag = _parse_legacy(legacy_id)
    code = subject[:7]
    if len(variants[code]) > 1:
        kept = subject
        defaults.add("subject_suffix_kept")
    else:
        kept = code
    defaults.update({"admission_round_none", "methods_hash_all"})
    fact = (record_type, scope, kept, year)
    key = f"{record_type}:{scope}:{kept}:{year}:{ROUND_NONE}:{METHODS_HASH_ALL}"
    return fact, key, source_tag


def build(source: dict) -> tuple[list[dict], dict]:
    defaults: set[str] = set()
    all_legacy = sorted(
        {rid for q in source["questions"] for rid in (q.get("expected_record_ids") or [])}
    )
    variants = _program_variants(all_legacy)
    mapping: dict[str, dict[str, str]] = {}
    fact_of_key: dict[str, tuple[str, str, str, str]] = {}
    rows: list[dict] = []
    for question in source["questions"]:
        legacy_ids = sorted(question.get("expected_record_ids") or [])
        if not legacy_ids:
            continue
        keys = []
        for legacy in legacy_ids:
            if legacy not in mapping:
                fact, key, tag = legacy_to_fact_and_key(legacy, variants, defaults)
                # Uniqueness guarantee: one key must identify one fact identity.
                # Source copies of the same fact (different source tags) share a key.
                if fact_of_key.setdefault(key, fact) != fact:
                    raise ValueError(f"coordinate_key collision between programs: {key}")
                mapping[legacy] = {"coordinate_key": key, "legacy_source_tag": tag}
            keys.append(mapping[legacy]["coordinate_key"])
        rows.append(
            {
                "id": question["id"],
                "group": "in_scope",
                "question": question["question"],
                "legacy_record_ids": legacy_ids,
                "expected_coordinate_keys": sorted(set(keys)),
                "expect_abstain": False,
                "expected_pages": None,
            }
        )

    for index, item in enumerate(OUT_OF_SCOPE, start=1):
        rows.append(
            {
                "id": f"oos{index:03d}",
                "group": "out_of_scope",
                "out_of_scope_group": item["group"],
                "question": item["question"],
                "legacy_record_ids": [],
                "expected_coordinate_keys": [],
                "expect_abstain": True,
                "expected_pages": None,
            }
        )

    rows.sort(key=lambda r: r["id"])
    meta = {
        "source_artifact_version": source.get("artifact_version"),
        "mapping_rule": "RECORD_TYPE:scope:subject:year:none:methods_hash (plan §PL-6.6; admission_round null)",
        "defaults": sorted(defaults | {"expected_pages_null"}),
        "in_scope_count": sum(1 for r in rows if r["group"] == "in_scope"),
        "out_of_scope_count": sum(1 for r in rows if r["group"] == "out_of_scope"),
        "mapping": dict(sorted(mapping.items())),
    }
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the U1 golden seed (plan §PL-6.6).")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mapping-out", type=Path, default=DEFAULT_MAPPING)
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text(encoding="utf-8"))
    rows, meta = build(source)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.mapping_out.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"defaults used: {meta['defaults']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
