"""Tests for check_no_raw_metadata_source (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_no_raw_metadata_source.py"


@pytest.fixture
def gate_module():
    spec = importlib.util.spec_from_file_location("check_no_raw_metadata_source", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_directory_returns_no_violations(gate_module, tmp_path):
    (tmp_path / "clean.py").write_text(
        "from services._metadata_helpers import get_source_label\n"
        "label = get_source_label(chunk)\n",
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert violations == []


def test_detects_metadata_get_source(gate_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'val = metadata.get("source")\n',
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 1
    path, line_no, line = violations[0]
    assert path.name == "bad.py"
    assert line_no == 1
    assert "metadata.get" in line


def test_detects_metadata_subscript_double_quote(gate_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'val = metadata["source"]\n',
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 1


def test_detects_metadata_subscript_single_quote(gate_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        "val = metadata['source']\n",
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 1


def test_detects_metadata_get_source_single_quote(gate_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        "val = metadata.get('source')\n",
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 1


def test_allowlist_skips_helpers_file(gate_module, tmp_path):
    (tmp_path / "_metadata_helpers.py").write_text(
        'val = metadata.get("source")\n',
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert violations == []


def test_allowlist_skips_helpers_test(gate_module, tmp_path):
    (tmp_path / "test_metadata_helpers.py").write_text(
        'val = metadata.get("source")\n',
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert violations == []


def test_walks_subdirectories(gate_module, tmp_path):
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "bad.py").write_text('metadata["source"]\n', encoding="utf-8")
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 1


def test_real_codebase_has_no_violations_after_p1_2(gate_module):
    """Integration: rag2025/src/ should be clean after P1-2 dual-read adapter migration."""
    repo_root = Path(__file__).resolve().parents[3]
    src = repo_root / "rag2025" / "src"
    assert src.exists(), f"Expected {src} to exist"
    violations = gate_module.find_violations(src)
    if violations:
        msg = "\n".join(f"{p}:{ln}: {line}" for p, ln, line in violations)
        pytest.fail(f"Unexpected raw metadata['source'] access in src:\n{msg}")


def test_returns_correct_line_number(gate_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        "# line 1\n# line 2\nval = metadata.get('source')\n",
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 1
    _, line_no, _ = violations[0]
    assert line_no == 3


def test_returns_multiple_violations_per_file(gate_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'a = metadata["source"]\nb = metadata.get("source")\n',
        encoding="utf-8",
    )
    violations = gate_module.find_violations(tmp_path)
    assert len(violations) == 2
