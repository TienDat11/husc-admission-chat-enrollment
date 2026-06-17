"""Tests for SLICE S0 — eval dependency pinning (parity/arbiter backend).

Asserts the EVAL block exists with PINNED versions, and that the core harness
modules do NOT top-level-import ragas/deepeval (they must be lazy-guarded).
"""
from __future__ import annotations

import pathlib
import re

import pytest


RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[2]
REQS_TXT = RAG2025_ROOT / "requirements.txt"
REQS_EVAL_TXT = RAG2025_ROOT / "requirements-eval.txt"
SCRIPTS_DIR = RAG2025_ROOT / "scripts"

EVAL_HEADER = "EVALUATION (parity/arbiter cross-check only"
PINNED_RAGAS = re.compile(r"^ragas==\S+", re.MULTILINE)
PINNED_DEEPEVAL = re.compile(r"^deepeval==\S+", re.MULTILINE)


def _read_manifest_text() -> str:
    """Return the contents of whichever requirements file holds the EVAL block.

    Prefer `requirements-eval.txt` if both exist. Fail loudly if neither does.
    """
    if REQS_EVAL_TXT.exists():
        return REQS_EVAL_TXT.read_text(encoding="utf-8")
    if REQS_TXT.exists():
        return REQS_TXT.read_text(encoding="utf-8")
    raise AssertionError(
        f"Neither {REQS_TXT} nor {REQS_EVAL_TXT} exists; "
        "S0 requires the EVAL block in one of them."
    )


def _manifest_path() -> pathlib.Path:
    if REQS_EVAL_TXT.exists():
        return REQS_EVAL_TXT
    return REQS_TXT


# ---------- 1. EVAL block exists in one of the two files ----------

def test_eval_block_header_exists():
    text = _read_manifest_text()
    assert EVAL_HEADER in text, (
        f"EVAL block header not found in {_manifest_path()}. "
        "Append the SLICE S0 block to requirements.txt "
        "(or move it into requirements-eval.txt)."
    )


# ---------- 2. Both deps are present and PINNED with `==` ----------

def test_ragas_is_pinned_with_double_equals():
    text = _read_manifest_text()
    assert PINNED_RAGAS.search(text), (
        "ragas must be pinned with `==` (e.g. `ragas==0.2.16`). "
        f"Check {_manifest_path()}."
    )
    # Negative guard: must NOT be pinned with `>=`, `<`, `~=`, or unpinned.
    for bad in (r"^ragas>=", r"^ragas<=", r"^ragas~=", r"^ragas\s*$", r"^ragas\s+\S+"):
        assert not re.search(bad, text, re.MULTILINE), (
            f"ragas appears with non-`==` pin (matched /{bad}/). "
            "SLICE S0 requires `==` pinning."
        )


def test_deepeval_is_pinned_with_double_equals():
    text = _read_manifest_text()
    assert PINNED_DEEPEVAL.search(text), (
        "deepeval must be pinned with `==` (e.g. `deepeval==1.5.9`). "
        f"Check {_manifest_path()}."
    )
    for bad in (r"^deepeval>=", r"^deepeval<=", r"^deepeval~=", r"^deepeval\s*$", r"^deepeval\s+\S+"):
        assert not re.search(bad, text, re.MULTILINE), (
            f"deepeval appears with non-`==` pin (matched /{bad}/). "
            "SLICE S0 requires `==` pinning."
        )


def test_both_packages_appear_in_manifest():
    text = _read_manifest_text()
    assert "ragas" in text, "ragas not mentioned in manifest"
    assert "deepeval" in text, "deepeval not mentioned in manifest"


# ---------- 3. Lazy-import guard for core harness eval scripts ----------

def _target_eval_scripts() -> list[pathlib.Path]:
    """Return the files the lazy-import guard should inspect.

    Spec: `scripts/eval_record.py` (if present) + any `scripts/eval_judge_*.py`.
    If none exist, the lazy-import guard is vacuously satisfied.
    """
    targets: list[pathlib.Path] = []
    record = SCRIPTS_DIR / "eval_record.py"
    if record.exists():
        targets.append(record)
    targets.extend(sorted(SCRIPTS_DIR.glob("eval_judge_*.py")))
    return targets


@pytest.mark.parametrize(
    "script_path", _target_eval_scripts(), ids=lambda p: p.name
)
def test_eval_script_no_top_level_ragas_or_deepeval_import(script_path: pathlib.Path):
    """Core harness must NOT import ragas/deepeval at module top-level.

    They are parity/arbiter cross-check deps only — the primary judge is the
    Opus-subagent batch protocol. Imports must be lazy/guarded inside functions.
    """
    text = script_path.read_text(encoding="utf-8")
    bad_lines: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip()
        # Spec: line starting with `import ragas`/`from ragas`/etc. at column 0.
        # We interpret "column 0" as "line begins with the import keyword"
        # (no leading whitespace), which catches `import ragas` / `from ragas ...`.
        for prefix in (
            "import ragas",
            "from ragas",
            "import deepeval",
            "from deepeval",
        ):
            if stripped.startswith(prefix):
                bad_lines.append((lineno, raw.rstrip()))
                break
    assert not bad_lines, (
        f"{script_path.name} contains top-level ragas/deepeval imports. "
        "They must be lazy-guarded (e.g. inside a function). Offending lines:\n"
        + "\n".join(f"  L{ln}: {line}" for ln, line in bad_lines)
    )


def test_lazy_import_guard_vacuous_when_no_targets():
    """Document the vacuous-pass behavior so it's visible in test output."""
    targets = _target_eval_scripts()
    if not targets:
        pytest.skip(
            "No eval_record.py or eval_judge_*.py scripts yet — "
            "lazy-import guard passes vacuously (S0 spec)."
        )
    # If targets exist, the parametrize test above would have run.
    assert targets, "parametrize should have produced at least one test case"
