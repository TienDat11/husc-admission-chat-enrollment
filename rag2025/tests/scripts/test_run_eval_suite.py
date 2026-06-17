"""TDD tests for Slice S10 — eval suite orchestrator.

The orchestrator runs the eval harness stages in order with one run_id+seed,
fail-fast on missing config, and PAUSES at the Opus-subagent judging gate.

Stages (in order):
  s2_instrumented, s5a_prepare, [judging_gate], s5b_aggregate,
  s5c_ramclouds, s6_halltype, s7_report

All tests use INJECTED deps (spies) — no real LLM, no network, no disk.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Any, Callable, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "run_eval_suite.py"


# ---------- Module loader ----------

@pytest.fixture(scope="module")
def suite():
    """Import run_eval_suite.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("run_eval_suite", str(SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Spy factory ----------

def _make_spy_deps(order_log: List[str], verdicts_exist: bool = True) -> Dict[str, Any]:
    """Build a deps dict of stage-spy callables.

    Each spy:
      - appends its stage name to order_log
      - records the kwargs it was called with (callable(**kwargs))
      - returns a sentinel result

    The judging gate check is also a spy: it records the run_id and returns
    the `verdicts_exist` flag.
    """
    calls: Dict[str, List[Dict[str, Any]]] = {}

    def make_spy(name: str) -> Callable[..., Dict[str, Any]]:
        def spy(*, run_id, seed, **kwargs):
            calls.setdefault(name, []).append({"run_id": run_id, "seed": seed, "kwargs": kwargs})
            order_log.append(name)
            return {"stage": name, "ok": True, "run_id": run_id, "seed": seed}
        # Lazily-resolved view: reads calls.get(name, []) at access time,
        # so the test's `assert stage not in calls` accurately reflects
        # whether the stage was actually invoked.
        spy.calls = property(lambda self: calls.get(name, []))  # type: ignore[attr-defined]
        return spy

    def judging_gate_check(*, run_id, **_kwargs):
        calls.setdefault("judging_gate", []).append({"run_id": run_id})
        order_log.append("judging_gate")
        return verdicts_exist

    deps: Dict[str, Any] = {
        "s2_instrumented": make_spy("s2_instrumented"),
        "s5a_prepare": make_spy("s5a_prepare"),
        "judging_gate_check": judging_gate_check,
        "s5b_aggregate": make_spy("s5b_aggregate"),
        "s5c_ramclouds": make_spy("s5c_ramclouds"),
        "s6_halltype": make_spy("s6_halltype"),
        "s7_report": make_spy("s7_report"),
    }
    deps["_calls"] = calls
    return deps


# =============================================================================
# 1. check_config: fail-fast on missing/empty RAMCLOUDS_API_KEY
# =============================================================================

def test_check_config_missing_key_raises(suite):
    """check_config({}) → raises (SystemExit | RuntimeError)."""
    with pytest.raises((SystemExit, RuntimeError)):
        suite.check_config({})


def test_check_config_empty_key_raises(suite):
    """check_config({"RAMCLOUDS_API_KEY": ""}) → raises (SystemExit | RuntimeError)."""
    with pytest.raises((SystemExit, RuntimeError)):
        suite.check_config({"RAMCLOUDS_API_KEY": ""})


def test_check_config_present_does_not_raise(suite):
    """check_config({"RAMCLOUDS_API_KEY": "x"}) → does not raise."""
    # Should NOT raise — any exception here fails the test.
    suite.check_config({"RAMCLOUDS_API_KEY": "x"})


# =============================================================================
# 2. run_id + seed propagation to every stage
# =============================================================================

def test_run_id_and_seed_propagated_to_every_stage(suite):
    """EVERY automatable stage spy must receive run_id=='t1' and seed==42."""
    order: List[str] = []
    deps = _make_spy_deps(order, verdicts_exist=True)

    result = suite.run_suite(run_id="t1", seed=42, deps=deps)

    assert result["status"] == "complete", result

    calls = deps["_calls"]
    automatable = (
        "s2_instrumented",
        "s5a_prepare",
        "s5b_aggregate",
        "s5c_ramclouds",
        "s6_halltype",
        "s7_report",
    )
    for stage in automatable:
        assert stage in calls, f"stage {stage} not called"
        assert len(calls[stage]) == 1, (
            f"stage {stage} called {len(calls[stage])} times, expected 1"
        )
        c = calls[stage][0]
        assert c["run_id"] == "t1", f"{stage}: run_id={c['run_id']!r}"
        assert c["seed"] == 42, f"{stage}: seed={c['seed']!r}"


# =============================================================================
# 3. stage ordering
# =============================================================================

def test_stage_ordering(suite):
    """Stages run in [s2, s5a, judging_gate, s5b, s5c, s6, s7] order."""
    order: List[str] = []
    deps = _make_spy_deps(order, verdicts_exist=True)

    result = suite.run_suite(run_id="t1", seed=42, deps=deps)

    assert result["status"] == "complete", result
    expected = [
        "s2_instrumented",
        "s5a_prepare",
        "judging_gate",
        "s5b_aggregate",
        "s5c_ramclouds",
        "s6_halltype",
        "s7_report",
    ]
    assert order == expected, f"got order={order}, expected={expected}"


# =============================================================================
# 4. judging gate pause: verdicts missing → S5b..S7 NOT called
# =============================================================================

def test_judging_gate_pause_when_verdicts_missing(suite):
    """verdicts_exist=False → return paused, downstream stages NOT called."""
    order: List[str] = []
    deps = _make_spy_deps(order, verdicts_exist=False)

    result = suite.run_suite(run_id="t1", seed=42, deps=deps)

    assert result["status"] == "paused", result
    assert result["paused_at"] == "opus_judging", result
    assert isinstance(result.get("batch_dir"), str) and result["batch_dir"], result
    assert isinstance(result.get("verdict_dir"), str) and result["verdict_dir"], result
    assert isinstance(result.get("instructions"), str) and result["instructions"], result

    calls = deps["_calls"]
    # Upstream stages SHOULD have run.
    for stage in ("s2_instrumented", "s5a_prepare"):
        assert stage in calls, f"upstream {stage} should have run"
        assert stage in order, f"upstream {stage} should be in order log"

    # Downstream stages MUST NOT have run.
    for stage in ("s5b_aggregate", "s5c_ramclouds", "s6_halltype", "s7_report"):
        assert stage not in calls, (
            f"downstream {stage} should NOT be called when verdicts missing"
        )
        assert stage not in order, (
            f"downstream {stage} should NOT appear in order log when verdicts missing"
        )

    # The judging gate itself SHOULD have been checked.
    assert "judging_gate" in calls
    assert "judging_gate" in order


# =============================================================================
# 5. judging gate proceed: verdicts exist → S5b..S7 ARE called
# =============================================================================

def test_judging_gate_proceed_when_verdicts_exist(suite):
    """verdicts_exist=True → S5b..S7 all called, status='complete'."""
    order: List[str] = []
    deps = _make_spy_deps(order, verdicts_exist=True)

    result = suite.run_suite(run_id="t1", seed=42, deps=deps)

    assert result["status"] == "complete", result
    calls = deps["_calls"]
    for stage in ("s5b_aggregate", "s5c_ramclouds", "s6_halltype", "s7_report"):
        assert stage in calls, f"stage {stage} should run when verdicts exist"
        assert len(calls[stage]) == 1, f"stage {stage} should run exactly once"


# =============================================================================
# 6. smoke flag passed to S2
# =============================================================================

def test_smoke_flag_passed_to_s2(suite):
    """run_suite(smoke=True) → S2 spy receives smoke=True."""
    order: List[str] = []
    deps = _make_spy_deps(order, verdicts_exist=True)

    suite.run_suite(run_id="t1", seed=42, smoke=True, deps=deps)

    s2_calls = deps["_calls"]["s2_instrumented"]
    assert len(s2_calls) == 1, s2_calls
    call = s2_calls[0]
    # The orchestrator must propagate smoke to S2. We accept it under
    # any of the conventional locations: kwargs["smoke"], kwargs["subset"],
    # or a top-level "smoke" key in the call record.
    sm = call["kwargs"].get("smoke")
    subset = call["kwargs"].get("subset")
    assert sm is True or subset is True, (
        f"smoke not propagated to s2: kwargs={call['kwargs']}"
    )


# =============================================================================
# 7. module docstring documents env vars and CI-bounded semantics
# =============================================================================

def test_module_docstring_documents_env_vars(suite):
    """The module docstring must document the env vars + CI-bounded claim.

    The LLM is NOT reproducible — the doc must NOT claim the LLM output
    is reproducible. We assert the substring "reproducible LLM" is absent.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    # Extract the first triple-quoted docstring (module-level).
    if '"""' not in src:
        pytest.fail("module docstring not found")
    parts = src.split('"""')
    if len(parts) < 3:
        pytest.fail("module docstring not properly delimited")
    doc = parts[1]

    # Required env vars / claims
    assert "RAMCLOUDS_API_KEY" in doc, "docstring must mention RAMCLOUDS_API_KEY"
    assert "EVAL_BOOTSTRAP_N" in doc, "docstring must mention EVAL_BOOTSTRAP_N"
    assert "EVAL_JUDGE_RUNS" in doc, "docstring must mention EVAL_JUDGE_RUNS"
    assert "CI-bounded" in doc, (
        "docstring must explicitly call out the CI-bounded nature of the harness"
    )

    # The LLM is NOT reproducible — assert we do NOT claim it is.
    assert "reproducible LLM" not in doc, (
        "docstring MUST NOT claim the LLM output is reproducible; "
        "we are CI-bounded under sampling noise, not bit-reproducible"
    )
