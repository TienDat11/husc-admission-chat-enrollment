"""TDD tests for the Opus-subagent judge-batch preparer (Slice S5a).

This module is a PURE data transformer — it does NOT score anything. It takes
S2 EvalRecord JSONL + the GT JSON, and emits fixed-size batch JSONs that an
Opus subagent will read + score with the embedded rubric.

Contract under test:
  * JUDGE_RUBRIC: a non-empty markdown string naming 4 metrics +
    the claims decomposition contract.
  * prepare_batches(records, gt_by_id, *, batch_size, run_id) -> list[dict]
      - each batch has: batch_id, run_id, rubric (== JUDGE_RUBRIC), items[]
      - each item: id, question, answer, contexts (FULL chunk text), and
        (from gt_by_id if present) reference_answer, expected_behavior,
        critical_facts.
      - All input records appear exactly once across batches.
      - Deterministic: same input -> byte-identical output.
  * write_batches(batches, out_dir) -> writes batch_NNN.json (zero-padded).
  * main() arg-parses --run-id, --records, --gt, --batch-size.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from typing import Any, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "eval_judge_prepare.py"


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def jb_module():
    """Import eval_judge_prepare.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("eval_judge_prepare", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rubric(jb_module) -> str:
    return jb_module.JUDGE_RUBRIC


# ---------- Helpers ----------

def _fake_record(
    rid: str,
    q: str = "Câu hỏi mẫu?",
    answer: str = "Câu trả lời mẫu.",
    chunk_texts: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a fake S2 EvalRecord-like dict with FULL-text contexts."""
    if chunk_texts is None:
        chunk_texts = [
            f"Đoạn văn số 1 cho {rid} — nội dung giả định.",
            f"Đoạn văn số 2 cho {rid} — thông tin bổ sung.",
        ]
    return {
        "id": rid,
        "question": q,
        "answer": answer,
        "retrieved_chunks": [
            {
                "chunk_id": f"chunk_{rid}_{i}",
                "source": "canonical:demo",
                "score": 0.9 - 0.1 * i,
                "text": t,
            }
            for i, t in enumerate(chunk_texts)
        ],
        "expected_behavior": "answer",
    }


def _fake_gt(qid_to_facts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a fake gt_by_id keyed by id."""
    out: Dict[str, Dict[str, Any]] = {}
    for qid, payload in qid_to_facts.items():
        out[qid] = {
            "id": qid,
            "reference_answer": payload.get("reference_answer", f"GT answer for {qid}"),
            "expected_behavior": payload.get("expected_behavior", "answer"),
            "critical_facts": payload.get(
                "critical_facts",
                [{"value": f"fact_{qid}", "fact_kind": "demo"}],
            ),
        }
    return out


# ---------- Tests ----------

def test_rubric_is_nonempty_and_names_all_required_concepts(rubric: str) -> None:
    """JUDGE_RUBRIC is a non-empty markdown string naming 4 metrics + claims."""
    assert isinstance(rubric, str)
    assert rubric.strip(), "JUDGE_RUBRIC must be non-empty"
    # Length sanity: rubric should be substantial (catches accidental empty/short).
    assert len(rubric) > 200, f"rubric suspiciously short: {len(rubric)} chars"
    for needle in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "claims",
        "supported",
        "null",  # N/A instruction
    ):
        assert needle in rubric, f"JUDGE_RUBRIC missing required concept: {needle!r}"


def test_rubric_explains_faithfulness_decomposition(rubric: str) -> None:
    """Faithfulness metric must instruct the subagent to decompose into atomic claims."""
    low = rubric.lower()
    # Decomposition language
    assert "decompose" in low or "atomic" in low, (
        "rubric must instruct the subagent to decompose the answer into atomic claims"
    )
    # Null/N-A language for auto-answer (no contexts)
    assert re.search(r"\bnull\b", low), "rubric must mention 'null' for N/A scoring"
    # Fraction-by-supported-claims formula
    assert "supported_claims" in low or "supported" in low, (
        "rubric must reference the supported-claims denominator for faithfulness"
    )


def test_rubric_mentions_writeback_contract(rubric: str) -> None:
    """Rubric must instruct the subagent on the verdict-JSON writeback format."""
    low = rubric.lower()
    # The subagent must emit claims + rationale
    assert "rationale" in low, "rubric must require a rationale field"
    # cite field for each claim (chunk_id | null)
    assert "cite" in low, "rubric must require a cite field per claim"


def test_prepare_batches_covers_all_records_with_expected_split(
    jb_module,
) -> None:
    """5 records × batch_size=2 → 3 batches (2+2+1), every id appears once."""
    records = [_fake_record(f"msg{i:03d}") for i in range(5)]
    gt_by_id: Dict[str, Dict[str, Any]] = {}
    batches = jb_module.prepare_batches(
        records, gt_by_id, batch_size=2, run_id="r1"
    )
    assert len(batches) == 3
    sizes = [len(b["items"]) for b in batches]
    assert sizes == [2, 2, 1]
    seen: List[str] = []
    for b in batches:
        for it in b["items"]:
            seen.append(it["id"])
    assert seen == [f"msg{i:03d}" for i in range(5)]
    # No duplicates
    assert len(seen) == len(set(seen))


def test_item_contexts_contain_full_chunk_text_not_ids(jb_module) -> None:
    """`contexts` must be FULL text of each retrieved chunk, not chunk ids."""
    chunk_a = (
        "Đoạn văn A: Học phí khối V là 600.000 VNĐ/tín chỉ. "
        "Mã ngành áp dụng: 7480201, 7220104."
    )
    chunk_b = "Đoạn văn B: Tổ hợp xét tuyển gồm A00, A01, D01."
    rec = _fake_record("msg007", chunk_texts=[chunk_a, chunk_b])
    batches = jb_module.prepare_batches([rec], {}, batch_size=5, run_id="r2")
    assert len(batches) == 1
    item = batches[0]["items"][0]
    # contexts must be a list of strings, equal to FULL chunk text
    assert isinstance(item["contexts"], list)
    assert all(isinstance(c, str) for c in item["contexts"])
    assert item["contexts"] == [chunk_a, chunk_b]
    # Specifically: full text preserved byte-equal
    assert chunk_a in item["contexts"][0]
    assert chunk_b in item["contexts"][1]
    # NOT chunk ids
    for c in item["contexts"]:
        assert "chunk_msg007_" not in c


def test_gt_join_pulls_reference_answer_and_behavior(jb_module) -> None:
    """Items in gt_by_id get reference_answer + expected_behavior + critical_facts."""
    gt_by_id = _fake_gt(
        {
            "msg010": {
                "reference_answer": "Đáp án tham chiếu cho msg010.",
                "expected_behavior": "answer",
                "critical_facts": [
                    {"value": "600000", "fact_kind": "hoc_phi"},
                    {"value": "7480201", "fact_kind": "ma_nganh"},
                ],
            },
            "msg011": {
                "reference_answer": "Tôi không có dữ liệu.",
                "expected_behavior": "abstain",
                "critical_facts": [],
            },
        }
    )
    records = [
        _fake_record("msg010"),
        _fake_record("msg011"),
        _fake_record("msg012"),  # NOT in gt
    ]
    batches = jb_module.prepare_batches(
        records, gt_by_id, batch_size=10, run_id="r3"
    )
    assert len(batches) == 1
    items = {it["id"]: it for it in batches[0]["items"]}

    # msg010 — has full GT payload
    assert items["msg010"]["reference_answer"] == "Đáp án tham chiếu cho msg010."
    assert items["msg010"]["expected_behavior"] == "answer"
    assert items["msg010"]["critical_facts"] == [
        {"value": "600000", "fact_kind": "hoc_phi"},
        {"value": "7480201", "fact_kind": "ma_nganh"},
    ]

    # msg011 — abstain behavior, empty critical_facts
    assert items["msg011"]["reference_answer"] == "Tôi không có dữ liệu."
    assert items["msg011"]["expected_behavior"] == "abstain"
    assert items["msg011"]["critical_facts"] == []

    # msg012 — NOT in gt → nulls + empty list
    assert items["msg012"]["reference_answer"] is None
    assert items["msg012"]["expected_behavior"] is None
    assert items["msg012"]["critical_facts"] == []


def test_determinism_same_input_byte_identical_batches(jb_module) -> None:
    """Two calls with same input → byte-identical JSON."""
    records = [_fake_record(f"msg{i:03d}") for i in range(7)]
    gt_by_id = _fake_gt(
        {
            "msg001": {
                "reference_answer": "ref-1",
                "expected_behavior": "answer",
                "critical_facts": [{"value": "x", "fact_kind": "demo"}],
            },
            "msg003": {
                "reference_answer": "ref-3",
                "expected_behavior": "refuse",
                "critical_facts": [],
            },
        }
    )
    b1 = jb_module.prepare_batches(
        records, gt_by_id, batch_size=3, run_id="runX"
    )
    b2 = jb_module.prepare_batches(
        records, gt_by_id, batch_size=3, run_id="runX"
    )
    s1 = json.dumps(b1, sort_keys=True, ensure_ascii=False)
    s2 = json.dumps(b2, sort_keys=True, ensure_ascii=False)
    assert s1 == s2, "prepare_batches must be deterministic (byte-identical)"


def test_rubric_is_embedded_in_every_batch(jb_module, rubric: str) -> None:
    """Every batch dict has rubric == JUDGE_RUBRIC (string-equal)."""
    records = [_fake_record(f"msg{i:03d}") for i in range(4)]
    batches = jb_module.prepare_batches(
        records, {}, batch_size=2, run_id="r4"
    )
    assert len(batches) == 2
    for b in batches:
        assert b["rubric"] == rubric
        # All batches carry the SAME rubric string (no per-batch variation).
    assert batches[0]["rubric"] == batches[1]["rubric"]


def test_batch_ids_are_zero_padded_and_unique(jb_module) -> None:
    """batch_id is 'batch_NNN' zero-padded; covers all records exactly once."""
    records = [_fake_record(f"msg{i:03d}") for i in range(6)]
    batches = jb_module.prepare_batches(
        records, {}, batch_size=2, run_id="r5"
    )
    ids = [b["batch_id"] for b in batches]
    assert ids == ["batch_000", "batch_001", "batch_002"]
    # run_id is propagated
    assert all(b["run_id"] == "r5" for b in batches)


def test_write_batches_creates_zero_padded_files(
    jb_module, tmp_path: pathlib.Path
) -> None:
    """write_batches writes out_dir/batch_NNN.json for every batch."""
    records = [_fake_record(f"msg{i:03d}") for i in range(3)]
    batches = jb_module.prepare_batches(
        records, {}, batch_size=2, run_id="r6"
    )
    out_dir = tmp_path / "judge_batches" / "r6"
    jb_module.write_batches(batches, out_dir)
    files = sorted(out_dir.glob("batch_*.json"))
    assert [f.name for f in files] == ["batch_000.json", "batch_001.json"]
    # Round-trip: each file is valid JSON with the expected shape
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert "batch_id" in data
        assert "rubric" in data
        assert "items" in data
        assert isinstance(data["items"], list)
