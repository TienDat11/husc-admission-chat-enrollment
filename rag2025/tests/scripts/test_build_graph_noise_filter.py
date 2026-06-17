"""TDD: build_graph noise filter (S18 — keep stale/discontinued chunks out of graph_ner).

Context (2026-06-09): rebuilding the knowledge graph must NOT ingest chunks
that are stale or describe discontinued majors, otherwise the entity graph
seeds PPR toward dead facts (e.g. the discontinued major ``7480107TD``, or any
chunk explicitly flagged ``is_superseded``). This is the "Vừa phải" filter
level: drop superseded + explicit stale-id chunks, but KEEP short chunks
(many are valid Q&A).

Contract (locked here):
  * ``_is_noise_chunk(data)`` returns True when:
      - ``data['metadata']['is_superseded']`` is truthy, OR
      - the chunk id/chunk_id is in ``STALE_CHUNK_IDS``, OR
      - the id contains a discontinued-major marker (``7480107TD``).
  * It returns False for normal chunks (including short ones — we do NOT
    drop by length at this level).
  * ``STALE_CHUNK_IDS`` includes the discontinued-major page.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_graph.py"


@pytest.fixture(scope="module")
def bg():
    spec = importlib.util.spec_from_file_location("build_graph", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_helper_and_constant_exist(bg):
    assert hasattr(bg, "_is_noise_chunk"), "build_graph must expose _is_noise_chunk()"
    assert hasattr(bg, "STALE_CHUNK_IDS"), "build_graph must expose STALE_CHUNK_IDS"


def test_superseded_chunk_is_noise(bg):
    data = {"id": "some_chunk", "text": "x", "metadata": {"is_superseded": True}}
    assert bg._is_noise_chunk(data) is True


def test_superseded_string_true_is_noise(bg):
    # LanceDB-style serialized metadata may carry the string "true".
    data = {"id": "c", "text": "x", "metadata": {"is_superseded": "true"}}
    assert bg._is_noise_chunk(data) is True


def test_discontinued_major_id_is_noise(bg):
    data = {"id": "husc_nganh_7480107TD_2025", "text": "Quản trị và phân tích dữ liệu", "metadata": {}}
    assert bg._is_noise_chunk(data) is True


def test_explicit_stale_id_is_noise(bg):
    # Any id placed in STALE_CHUNK_IDS is filtered.
    for sid in bg.STALE_CHUNK_IDS:
        data = {"id": sid, "text": "x", "metadata": {}}
        assert bg._is_noise_chunk(data) is True, f"{sid} should be noise"


def test_normal_chunk_is_not_noise(bg):
    data = {"id": "husc_nganh_7480201_2026", "text": "CNTT 400 chỉ tiêu", "metadata": {"data_year": 2026}}
    assert bg._is_noise_chunk(data) is False


def test_short_chunk_is_NOT_filtered_at_this_level(bg):
    # "Vừa phải" level keeps short chunks (valid Q&A). No length drop.
    data = {"id": "qa_short_2026", "text": "Có. KTX còn chỗ.", "metadata": {}}
    assert bg._is_noise_chunk(data) is False


def test_discontinued_major_in_stale_ids(bg):
    assert "husc_nganh_7480107TD_2025" in bg.STALE_CHUNK_IDS


def test_missing_metadata_is_safe(bg):
    # No metadata key at all → not noise (don't crash).
    assert bg._is_noise_chunk({"id": "c", "text": "x"}) is False
