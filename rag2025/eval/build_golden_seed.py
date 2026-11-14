"""Build the U1 golden seed deterministically (plan §PL-6.6, unit U1).

Input: the legacy golden records file (gitignored, read-only):
    data/eval/husc86q_golden_records.json
Output (committed, derived data):
    tests/eval/fixtures/golden_seed.jsonl
    tests/eval/fixtures/legacy_id_mapping.json

Mapping rule (legacy record id -> coordinate_key):
    legacy  = RECORD_TYPE:scope:subject:year:round[:source...]
    new     = RECORD_TYPE:scope:subject7:year:round:methods_hash
  - the trailing ``source`` segment is dropped (not a coordinate component);
  - ``subject`` is cut to its 7-digit major code when it carries a suffix;
  - ``methods_hash`` defaults to ``"all"`` (the legacy file has no method
    restriction). Each default used is listed in the mapping file's ``defaults``.

Run: ``python eval/build_golden_seed.py``. Output is byte-identical across runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "eval" / "husc86q_golden_records.json"
DEFAULT_OUT = ROOT / "tests" / "eval" / "fixtures" / "golden_seed.jsonl"
DEFAULT_MAPPING = ROOT / "tests" / "eval" / "fixtures" / "legacy_id_mapping.json"

METHODS_HASH_ALL = "all"
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


def legacy_to_coordinate_key(legacy_id: str, defaults: set[str]) -> str:
    parts = legacy_id.split(":")
    if len(parts) < 5:
        raise ValueError(f"legacy id has fewer than 5 segments: {legacy_id!r}")
    record_type, scope, subject, year, round_ = parts[:5]
    # Subjects carrying a program suffix (e.g. "7480201-ky-su-viet-nhat") are
    # kept whole: cutting them to the 7-digit code collides with the plain
    # major code and merges two distinct programs. This deviates from the
    # literal "mã ngành 7 số" wording of §PL-6.6 and is reported.
    if not _SUBJECT_CODE.match(subject):
        raise ValueError(f"subject is not a 7-digit major code: {legacy_id!r}")
    if subject != subject[:7]:
        defaults.add("subject_suffix_kept")
    defaults.add("methods_hash_all")
    return f"{record_type}:{scope}:{subject}:{year}:{round_}:{METHODS_HASH_ALL}"


def build(source: dict) -> tuple[list[dict], dict]:
    defaults: set[str] = set()
    mapping: dict[str, str] = {}
    rows: list[dict] = []
    for question in source["questions"]:
        legacy_ids = sorted(question.get("expected_record_ids") or [])
        if not legacy_ids:
            continue
        keys = []
        for legacy in legacy_ids:
            if legacy not in mapping:
                mapping[legacy] = legacy_to_coordinate_key(legacy, defaults)
            keys.append(mapping[legacy])
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
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("legacy -> coordinate_key mapping is not injective")

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
        "mapping_rule": "RECORD_TYPE:scope:subject7:year:round:methods_hash (plan §PL-6.6)",
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
