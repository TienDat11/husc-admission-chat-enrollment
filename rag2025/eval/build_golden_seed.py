"""Build the U1 golden seed deterministically (plan §PL-6.6 / §PL-6.7, unit U1).

Rows (one JSON object per line, sorted by id):
  in_scope      >=40  questions of data/eval/husc86q_golden_records.json with a
                      non-empty expected_record_ids; each record id maps to a
                      coordinate_key (§PL-6.6: admission_round "none", methods "all").
  out_of_scope  >=10  hand-written, five groups, expect_abstain=true.
  supersession  26    §PL-6.7 matrix. Legal text comes from
                      tests/fixtures/legal_vn/golden_40.jsonl (a sentence's expect_kind
                      picks its matrix family). Old/new coordinate keys are synthetic per
                      case; the expected outcome is the matrix relation, not the output
                      of supersession_detector.

Side output: tests/eval/fixtures/legacy_id_mapping.json (legacy record id -> key).
Usage: python eval/build_golden_seed.py --out tests/eval/fixtures/golden_seed.jsonl
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
LEGAL_FIXTURE = ROOT / "tests" / "fixtures" / "legal_vn" / "golden_40.jsonl"

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

# §PL-6.7 matrix: (matrix_row, relation of each case in that row). Counts sum to 26.
SUPERSESSION_MATRIX: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1to1_same_id", ("REPLACES",) * 3),
    ("1to1_diff_id", ("REPLACES",) * 3),
    ("amends_dieu", ("AMENDS",) * 5),
    ("corrects", ("CORRECTS",) * 2),
    ("repeals_expires", ("REPEALS", "REPEALS", "EXPIRES")),
    ("split_1_to_n", ("SPLIT",) * 2),
    ("merge_n_to_1", ("MERGE",) * 3),
    ("chain_a_b_c", ("CHAIN",) * 2),
    ("temporal_handoff", ("HANDOFF",)),
    ("coexist_revoke", ("COEXIST", "REVOKE_SUPERSESSION")),
)
# golden_40 expect_kind whose sentences supply the legal text of each matrix row.
_ROW_TEXT_KIND: dict[str, str] = {
    "1to1_same_id": "REPLACES",
    "1to1_diff_id": "REPLACES",
    "amends_dieu": "AMENDS",
    "corrects": "CORRECTS",
    "repeals_expires": "REPEALS",
    "split_1_to_n": "REPLACES",
    "merge_n_to_1": "REPLACES",
    "chain_a_b_c": "REPLACES",
    "temporal_handoff": "REPLACES",
    "coexist_revoke": "REPLACES",
}
LEGAL_RECORD_TYPE = "DIEU_KIEN"  # legal units are DIEU_KIEN (plan U4, record_type rule)
LEGAL_SCOPE = "phap_quy"  # scope token of the synthetic legal instrument
OLD_YEAR = 2025
NEW_YEAR = 2026
EARLIER_YEAR = 2024  # first link of the A->B->C chain


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


def _legal_key(article: int, year: int) -> str:
    return f"{LEGAL_RECORD_TYPE}:{LEGAL_SCOPE}:dieu_{article}:{year}:{ROUND_NONE}:{METHODS_HASH_ALL}"


def _supersession_keys(row_name: str, base: int) -> tuple[list[str], list[str]]:
    """Return (old keys, new keys) for one case; article numbers start at ``base``."""
    if row_name in ("1to1_same_id", "amends_dieu"):
        same = _legal_key(base, NEW_YEAR)
        return [same], [same]
    if row_name in ("1to1_diff_id", "corrects", "temporal_handoff", "coexist_revoke"):
        return [_legal_key(base, OLD_YEAR)], [_legal_key(base, NEW_YEAR)]
    if row_name == "repeals_expires":
        return [_legal_key(base, OLD_YEAR)], []
    if row_name == "split_1_to_n":
        return [_legal_key(base, OLD_YEAR)], [_legal_key(base + 1, NEW_YEAR), _legal_key(base + 2, NEW_YEAR)]
    if row_name == "merge_n_to_1":
        old = [_legal_key(base + k, OLD_YEAR) for k in range(3)]
        return old, [_legal_key(base + 3, NEW_YEAR)]
    if row_name == "chain_a_b_c":
        return [_legal_key(base, EARLIER_YEAR), _legal_key(base, OLD_YEAR)], [_legal_key(base, NEW_YEAR)]
    raise ValueError(f"unknown supersession matrix row: {row_name!r}")


def build_supersession(legal_rows: list[dict]) -> list[dict]:
    """26 supersession cases of §PL-6.7, ordered by matrix row then case."""
    pools: dict[str, list[str]] = {}
    for row in legal_rows:
        if row["expect_positive"]:
            pools.setdefault(row["expect_kind"], []).append(row["text"])

    rows: list[dict] = []
    index = 0
    for row_name, relations in SUPERSESSION_MATRIX:
        pool = pools[_ROW_TEXT_KIND[row_name]]
        for position, relation in enumerate(relations):
            index += 1
            base = 100 + 10 * index  # disjoint article numbers per case
            old, new = _supersession_keys(row_name, base)
            rows.append(
                {
                    "id": f"ss{index:03d}",
                    "group": "supersession",
                    "matrix_row": row_name,
                    "relation": relation,
                    "question": pool[position % len(pool)],
                    "legacy_record_ids": [],
                    "old_coordinate_keys": old,
                    "new_coordinate_keys": new,
                    "expected_coordinate_keys": sorted(set(old + new)),
                    "expect_abstain": False,
                    "expected_pages": None,
                }
            )
    return rows


def _load_legal(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(source: dict, legal_rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    if legal_rows is None:
        legal_rows = _load_legal(LEGAL_FIXTURE)
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

    rows.extend(build_supersession(legal_rows))
    rows.sort(key=lambda r: r["id"])
    matrix_counts = {name: len(rel) for name, rel in SUPERSESSION_MATRIX}
    meta = {
        "source_artifact_version": source.get("artifact_version"),
        "mapping_rule": "RECORD_TYPE:scope:subject:year:none:methods_hash (plan §PL-6.6; admission_round null)",
        "supersession_source": str(LEGAL_FIXTURE.relative_to(ROOT)).replace("\\", "/"),
        "defaults": sorted(defaults | {"expected_pages_null"}),
        "in_scope_count": sum(1 for r in rows if r["group"] == "in_scope"),
        "out_of_scope_count": sum(1 for r in rows if r["group"] == "out_of_scope"),
        "supersession_count": sum(1 for r in rows if r["group"] == "supersession"),
        "supersession_matrix": matrix_counts,
        "mapping": dict(sorted(mapping.items())),
    }
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the U1 golden seed (plan §PL-6.6 / §PL-6.7).")
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
