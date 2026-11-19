"""Supersession tiers (plan §PL-6.7) checked against the golden supersession matrix."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.identity import parse_coordinate_key  # noqa: E402
from src.domain.release import TransitionKind  # noqa: E402
from src.services.supersession_detector import (  # noqa: E402
    SemanticRecord,
    apply_revocations,
    detect,
    link_keys,
    resolve_head,
    semantic_edge,
    semantic_scores,
)

FIXTURES = ROOT / "tests" / "eval" / "fixtures"
ROWS = [
    json.loads(line)
    for line in (FIXTURES / "golden_seed.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ROWS = [row for row in ROWS if row["group"] == "supersession"]
ROW_IDS = [row["id"] for row in ROWS]


def _row(row_id: str) -> dict:
    return next(row for row in ROWS if row["id"] == row_id)


def _key(subject: str, year: int) -> str:
    return f"DIEU_KIEN:phap_quy:{subject}:{year}:none:all"


def test_golden_matrix_counts_match_legacy_mapping() -> None:
    mapping = json.loads((FIXTURES / "legacy_id_mapping.json").read_text(encoding="utf-8"))
    assert dict(Counter(row["matrix_row"] for row in ROWS)) == mapping["supersession_matrix"]
    assert len(ROWS) == 26


@__import__("pytest").mark.parametrize("row_id", ROW_IDS)
def test_golden_row_links_cover_expected_keys(row_id: str) -> None:
    row = _row(row_id)
    edges = link_keys(row["old_coordinate_keys"], row["new_coordinate_keys"], relation=row["relation"])
    covered = {key for edge in edges for key in (edge.old_key, edge.new_key) if key}
    assert covered == set(row["expected_coordinate_keys"])
    assert all(edge.relation == row["relation"] for edge in edges)


def test_chain_resolves_transitively_to_newest_head() -> None:
    row = _row("ss022")
    edges = link_keys(row["old_coordinate_keys"], row["new_coordinate_keys"], relation="CHAIN")
    assert resolve_head(edges, row["old_coordinate_keys"][0]) == row["new_coordinate_keys"][0]


def test_coexist_and_handoff_are_reviewer_decisions_not_succession() -> None:
    coexist = link_keys([_key("dieu_350", 2025)], [_key("dieu_350", 2026)], relation="COEXIST")
    assert {edge.tier for edge in coexist} == {"decision"}
    assert resolve_head(coexist, _key("dieu_350", 2025)) == _key("dieu_350", 2025)


def test_revocation_cancels_a_replace_edge() -> None:
    old, new = _key("dieu_360", 2025), _key("dieu_360", 2026)
    succession = link_keys([old], [new], relation="REPLACES")
    revoke = link_keys([old], [new], relation="REVOKE_SUPERSESSION")
    assert resolve_head(succession, old) == new
    assert resolve_head(apply_revocations([*succession, *revoke]), old) == old


def test_tier1_amends_names_the_whole_article_record() -> None:
    text = (
        "Điều 170 khoản 2 của Quy chế tuyển sinh được sửa đổi, bổ sung "
        "theo Quyết định 10/QĐ-BGD."
    )
    amends = [c for c in detect(text, source_doc_id="doc") if c.kind == TransitionKind.AMENDS]
    assert amends and "Điều 170 khoản 2" in amends[0].affected_articles
    key = _key("dieu_170", 2026)
    edge = link_keys([key], [key], relation="AMENDS")[0]
    assert edge.tier == "explicit_ref"
    assert parse_coordinate_key(edge.old_key).subject == "dieu_170"


OLD_TEXT = (
    "Thí sinh đăng ký xét tuyển theo tổ hợp môn thi với điểm sàn 18 điểm "
    "trên thang điểm 30 cho các ngành đào tạo của trường."
)
NEW_TEXT = OLD_TEXT.replace("18", "19")


def _record(key: str, text: str, record_type: str = "DIEU_KIEN") -> SemanticRecord:
    return SemanticRecord(key, record_type, frozenset({"7140201"}), text)


def test_tier3_proposes_near_identical_same_type_records() -> None:
    edge = semantic_edge(
        _record(_key("dieu_10", 2025), OLD_TEXT), _record(_key("dieu_11", 2026), NEW_TEXT)
    )
    assert edge is not None and edge.tier == "semantic"


def test_tier3_rejects_different_record_type() -> None:
    assert (
        semantic_edge(
            _record(_key("dieu_10", 2025), OLD_TEXT),
            _record(_key("dieu_11", 2026), NEW_TEXT, record_type="NGANH"),
        )
        is None
    )


def test_tier3_pair_scoring_is_under_five_milliseconds() -> None:
    pairs = 200
    start = time.perf_counter()
    for _ in range(pairs):
        semantic_scores(OLD_TEXT, NEW_TEXT)
    per_pair = (time.perf_counter() - start) / pairs
    assert per_pair < 0.005, f"{per_pair * 1000:.2f} ms per pair"
