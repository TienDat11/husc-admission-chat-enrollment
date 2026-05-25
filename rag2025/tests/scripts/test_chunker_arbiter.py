"""Tests for chunker_arbiter (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunker_arbiter.py"


@pytest.fixture
def arb():
    spec = importlib.util.spec_from_file_location("chunker_arbiter", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_iou_identical_returns_one(arb):
    spans = [(0, 100)]
    assert arb.compute_iou(spans, spans) == 1.0


def test_compute_iou_disjoint_returns_zero(arb):
    assert arb.compute_iou([(0, 50)], [(100, 150)]) == 0.0


def test_compute_iou_partial_overlap(arb):
    iou = arb.compute_iou([(0, 100)], [(50, 150)])
    # intersect = 50, union = 150, iou = 50/150 = 1/3
    assert abs(iou - (50 / 150)) < 1e-9


def test_compute_iou_empty_returns_zero(arb):
    assert arb.compute_iou([], [(0, 10)]) == 0.0
    assert arb.compute_iou([(0, 10)], []) == 0.0


def test_spans_with_cursor_uses_explicit_offsets(arb):
    chunks = [
        {"text": "abc", "start_offset": 0, "end_offset": 10},
        {"text": "def", "start_offset": 10, "end_offset": 25},
    ]
    spans = arb._spans_with_cursor(chunks)
    assert spans == [(0, 10), (10, 25)]


def test_spans_with_cursor_falls_back_to_text_length(arb):
    chunks = [{"text": "abc"}, {"text": "defgh"}]
    spans = arb._spans_with_cursor(chunks)
    assert spans == [(0, 3), (3, 8)]


def test_winner_defaults_to_system_v2(arb):
    """When IoU is high (chunkers agree), system_v2 keeps the default win."""
    chunks = [{"text": "exactly the same chunk text"}]
    decision = arb.arbitrate_one(
        nid=74,
        source_html="any",
        system_chunks=chunks,
        haiku_chunks=chunks,
        claude_chunks=chunks,
        judge_runner=lambda prompt: 0.9,
    )
    assert decision["winner"] == "system_v2"
    assert decision["escalated"] is False
    assert decision["iou_pairs"]["system_vs_haiku"] == 1.0


def test_ensemble_wins_when_iou_low_and_faithfulness_gap_large(arb):
    """When system disagrees with ensemble AND ensemble is more faithful, ensemble wins."""
    sys_chunks = [{"text": "a"}]                          # span 0..1
    hai_chunks = [{"text": "FAITHFUL_CHUNK" + "z" * 86}] # span 0..100 — different span
    cla_chunks = [{"text": "c" * 100}]                   # span 0..100 — different span
    # Judge triggers only on haiku chunk text; source_html is neutral so it
    # doesn't accidentally match the trigger pattern inside the SOURCE section.
    judge = lambda prompt: 0.95 if "FAITHFUL_CHUNK" in prompt else 0.5

    decision = arb.arbitrate_one(
        nid=74,
        source_html="neutral source",
        system_chunks=sys_chunks,
        haiku_chunks=hai_chunks,
        claude_chunks=cla_chunks,
        judge_runner=judge,
    )
    assert decision["winner"] == "haiku_v1"
    assert "Ensemble override" in decision["rationale"]


def test_no_override_when_faithfulness_gap_small(arb):
    """If IoU is low but faithfulness gap is below 0.1, system_v2 still wins."""
    sys_chunks = [{"text": "aaa"}]
    hai_chunks = [{"text": "bbbbbbbbbb"}]
    cla_chunks = [{"text": "ccc"}]
    decision = arb.arbitrate_one(
        nid=74,
        source_html="any",
        system_chunks=sys_chunks,
        haiku_chunks=hai_chunks,
        claude_chunks=cla_chunks,
        judge_runner=lambda prompt: 0.85,  # equal faithfulness
    )
    assert decision["winner"] == "system_v2"


def test_escalation_after_three_consecutive_low_iou(arb):
    """Hard rule: 3 consecutive docs with IoU(sys,haiku) < 0.5 → winner=None."""
    state = arb._ConsecutiveEscalation()
    sys_chunks = [{"text": "aaa"}]
    hai_chunks = [{"text": "bbbbbbbbbb"}]
    cla_chunks = [{"text": "ccc"}]
    judge = lambda prompt: 0.5

    # First 2 docs: not yet escalated.
    for nid in (1, 2):
        d = arb.arbitrate_one(
            nid=nid,
            source_html="x",
            system_chunks=sys_chunks,
            haiku_chunks=hai_chunks,
            claude_chunks=cla_chunks,
            judge_runner=judge,
            escalation_state=state,
        )
        assert d["escalated"] is False

    # Third doc: escalation fires.
    d3 = arb.arbitrate_one(
        nid=3,
        source_html="x",
        system_chunks=sys_chunks,
        haiku_chunks=hai_chunks,
        claude_chunks=cla_chunks,
        judge_runner=judge,
        escalation_state=state,
    )
    assert d3["escalated"] is True
    assert d3["winner"] is None
    assert "consecutive" in d3["rationale"]


def test_escalation_resets_on_high_iou_doc(arb):
    """High-IoU doc breaks the consecutive streak."""
    state = arb._ConsecutiveEscalation()
    sys_chunks = [{"text": "aaa"}]
    hai_chunks_low = [{"text": "bbbbbbbbbb"}]
    hai_chunks_high = sys_chunks  # identical → IoU=1.0
    judge = lambda prompt: 0.5

    arb.arbitrate_one(nid=1, source_html="x", system_chunks=sys_chunks, haiku_chunks=hai_chunks_low, claude_chunks=sys_chunks, judge_runner=judge, escalation_state=state)
    arb.arbitrate_one(nid=2, source_html="x", system_chunks=sys_chunks, haiku_chunks=hai_chunks_low, claude_chunks=sys_chunks, judge_runner=judge, escalation_state=state)
    # High-IoU doc — resets streak.
    arb.arbitrate_one(nid=3, source_html="x", system_chunks=sys_chunks, haiku_chunks=hai_chunks_high, claude_chunks=sys_chunks, judge_runner=judge, escalation_state=state)
    assert state.streak == 0
