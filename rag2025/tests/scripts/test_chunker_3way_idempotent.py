"""Idempotency gate — chunker_3way produces byte-equal outputs across 2 runs (TDD V5-R030)."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunker_3way.py"


@pytest.fixture
def chunker3():
    spec = importlib.util.spec_from_file_location("chunker_3way", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_runs_produce_byte_equal_jsonl_outputs(chunker3, tmp_path):
    """Run chunker_3way twice with identical inputs + deterministic stubs.

    Assert all 3 lane outputs are byte-equal across runs.
    """
    deterministic_chunks = [
        {"info_type": "hoc_phi", "semantic_topic": "fee", "suggested_title": "Học phí 2026", "text": "Học phí 600.000 VNĐ/tín chỉ"},
        {"info_type": "diem_chuan", "semantic_topic": "score", "suggested_title": "Điểm chuẩn 2026", "text": "Điểm chuẩn ngành CNTT: 24.5"},
    ]

    runners = {
        "haiku_runner": lambda html, prompt: deterministic_chunks,
        "claude_runner": lambda html, prompt: deterministic_chunks,
        "system_runner": lambda html: deterministic_chunks,
    }

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    chunker3.run_3way(74, "<html>same input</html>", out1, "PROMPT_HAIKU", "PROMPT_CLAUDE", api_key="k", **runners)
    chunker3.run_3way(74, "<html>same input</html>", out2, "PROMPT_HAIKU", "PROMPT_CLAUDE", api_key="k", **runners)

    for lane in ("system_v2.jsonl", "haiku_v1.jsonl", "claude_v1.jsonl"):
        p1 = out1 / "74" / lane
        p2 = out2 / "74" / lane
        assert p1.exists() and p2.exists(), f"Missing output: {lane}"
        assert p1.read_bytes() == p2.read_bytes(), f"Byte mismatch in {lane}"


def test_two_runs_produce_byte_equal_manifests(chunker3, tmp_path):
    """Manifest SHA256 lines must be identical across runs."""
    deterministic_chunks = [{"info_type": "qa", "text": "Q: A?"}]
    runners = {
        "haiku_runner": lambda html, prompt: deterministic_chunks,
        "claude_runner": lambda html, prompt: deterministic_chunks,
        "system_runner": lambda html: deterministic_chunks,
    }

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    chunker3.run_3way(59, "<html/>", out1, "P1", "P2", api_key="k", **runners)
    chunker3.run_3way(59, "<html/>", out2, "P1", "P2", api_key="k", **runners)

    m1 = (out1 / "59" / "manifest.sha256").read_text(encoding="utf-8")
    m2 = (out2 / "59" / "manifest.sha256").read_text(encoding="utf-8")
    assert m1 == m2


def test_outputs_have_stable_sha256_across_runs(chunker3, tmp_path):
    """Each lane output's SHA256 hash must match between runs (binary identity)."""
    chunks = [{"info_type": "phuongthuc", "text": "PT01: Xét tuyển thẳng"}]
    runners = {
        "haiku_runner": lambda html, prompt: chunks,
        "claude_runner": lambda html, prompt: chunks,
        "system_runner": lambda html: chunks,
    }

    sha_run1: dict[str, str] = {}
    sha_run2: dict[str, str] = {}

    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    chunker3.run_3way(63, "<html/>", out1, "P", "P", api_key="k", **runners)
    chunker3.run_3way(63, "<html/>", out2, "P", "P", api_key="k", **runners)

    for lane in ("system_v2.jsonl", "haiku_v1.jsonl", "claude_v1.jsonl"):
        sha_run1[lane] = _file_sha256(out1 / "63" / lane)
        sha_run2[lane] = _file_sha256(out2 / "63" / lane)

    assert sha_run1 == sha_run2


def test_different_inputs_produce_different_manifests(chunker3, tmp_path):
    """Sanity: when chunks differ, manifests must differ — proves we're not just hashing constants."""
    chunks_a = [{"info_type": "qa", "text": "Q1"}]
    chunks_b = [{"info_type": "qa", "text": "Q2 (different)"}]

    runners_a = {
        "haiku_runner": lambda html, prompt: chunks_a,
        "claude_runner": lambda html, prompt: chunks_a,
        "system_runner": lambda html: chunks_a,
    }
    runners_b = {
        "haiku_runner": lambda html, prompt: chunks_b,
        "claude_runner": lambda html, prompt: chunks_b,
        "system_runner": lambda html: chunks_b,
    }

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    chunker3.run_3way(75, "<html/>", out_a, "P", "P", api_key="k", **runners_a)
    chunker3.run_3way(75, "<html/>", out_b, "P", "P", api_key="k", **runners_b)

    m_a = (out_a / "75" / "manifest.sha256").read_text(encoding="utf-8")
    m_b = (out_b / "75" / "manifest.sha256").read_text(encoding="utf-8")
    assert m_a != m_b
