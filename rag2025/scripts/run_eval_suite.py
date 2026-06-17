"""Slice S10 — Eval suite orchestrator (single entrypoint for the RAGAS-grade harness).

This is the SINGLE entrypoint that runs the eval harness stages in order with one
``run_id + seed``, fail-fast on missing config, and PAUSES at the Opus-subagent
judging gate (which is an ORCHESTRATION step, NOT a python call — the orchestrator
prints the batch + verdict directories and instructions, and stops).

Stages (and which are automatable vs orchestration-gated):

  1. S2  ``eval_run_instrumented``       — records JSONL                       [automatable]
  2. S5a ``eval_judge_prepare``          — judge batches                       [automatable]
  3. **Opus-subagent judging**          — verdict files                       [ORCHESTRATION GATE]
  4. S5b ``eval_judge_aggregate``        — aggregate (needs verdicts)           [automatable, gated]
  5. S5c ``eval_judge_ramclouds``        — κ agreement (real gateway)           [automatable]
  6. S6  ``eval_judge_halltype``         — hallucination classification        [automatable]
  7. S7  ``eval_report_metrics``         — final report (needs 4+5+6)          [automatable]

Environment variables (documented here so reviewers can audit the contract):

  * ``RAMCLOUDS_API_KEY`` (REQUIRED) — the ramclouds gateway API key. The
    orchestrator refuses to start without it.
  * ``RAMCLOUDS_STREAM=false`` (default ``false``) — the ramclouds judge MUST
    run in non-streaming mode for judge determinism (one JSON object per
    request, not a token stream). Set explicitly to ``true`` only for manual
    debugging — never in CI.
  * ``EVAL_BOOTSTRAP_N=1000`` — number of bootstrap resamples for the
    aggregator's 95% CI. Higher → tighter CI, slower aggregation.
  * ``EVAL_JUDGE_RUNS=3`` — number of independent judge passes per item.
    We aggregate via per-id mean, then bootstrap CI over the per-id means.
  * ``EVAL_POSITION_SWAP_PCT=0.2`` — fraction of batches where the items
    list is reversed before judging, to control for position bias. Set to
    ``0.0`` to disable.
  * ``EVAL_BATCH_SIZE=10`` — number of items per judge batch.

Reproducibility contract (read carefully — this is NOT a bit-reproducible run):

  The ``seed`` argument governs ONLY the bootstrap resampling determinism in
  the aggregator (S5b). It does NOT govern LLM output. LLM output is NOT
  reproducible across runs (network latency, gateway batching, server-side
  sampling noise). We compensate by:

    1. running the judge ``EVAL_JUDGE_RUNS=3`` times per item, and
    2. reporting a 95% bootstrap CI around the per-id mean, and
    3. gating on the CI LO bound, not the point estimate.

  This is "CI-bounded under sampling noise", NOT bit-reproducible.
  Bit-exact re-runs are NOT a goal of this harness; the LLM output cannot
  be re-derived from the seed alone.

Public API
----------

  * ``check_config(env: dict) -> None`` — fail-fast on missing/empty
    ``RAMCLOUDS_API_KEY``. Raises ``RuntimeError`` with a clear message.
  * ``run_suite(*, run_id, seed=42, stages=None, smoke=False, deps=None) -> dict``
    — orchestrates the seven stages. ``deps`` is a dict of injected stage
    callables (for testing: spies; for production: real wiring via
    ``default_deps()``). At the judging gate, if verdict files do not exist
    for the run, returns a ``paused`` status with batch_dir / verdict_dir /
    instructions and STOPS (does not run S5b..S7).
  * ``default_deps() -> dict`` — production wiring of stage callables
    (imports the other scripts' functions; import errors are surfaced
    loudly at first use, NOT at module import).
  * ``main(argv=None) -> int`` — argparse CLI: ``--run-id --seed 42 --smoke --stages a,b,c``.
    Calls ``check_config(os.environ)`` FIRST, then ``run_suite``.

Test injection points
---------------------

The ``deps`` parameter is the seam that makes the orchestrator TDD-friendly
without invoking real LLMs / network / disk. Each stage is a callable
taking keyword arguments ``(*, run_id, seed, **extra)`` and returning a
result dict. ``judging_gate_check`` takes ``(*, run_id, **_)`` and returns
``True`` iff verdict files exist for that ``run_id``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


# ---------- Paths ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
JUDGE_BATCHES_ROOT = RAG2025_ROOT / "results" / "eval_harness" / "judge_batches"
JUDGE_VERDICTS_ROOT = RAG2025_ROOT / "results" / "eval_harness" / "judge_verdicts"
RECORDS_ROOT = RAG2025_ROOT / "results" / "eval_harness" / "records"


# ---------- Constants ----------

# Default smoke subset size — when --smoke is on, S2 runs only this many questions.
DEFAULT_SMOKE_N: int = 3

# The set of orchestrator-known stage keys (used to validate --stages filter).
ALL_STAGE_KEYS: tuple = (
    "s2_instrumented",
    "s5a_prepare",
    "s5b_aggregate",
    "s5c_ramclouds",
    "s6_halltype",
    "s7_report",
)

# A short, copy-paste-able instruction block for the Opus subagent judging step.
JUDGING_INSTRUCTIONS: str = (
    "Opus-subagent judging is an ORCHESTRATION GATE, not a python call. "
    "To unblock this run:\n"
    "  1. Read each judge batch JSON from the batch_dir below.\n"
    "  2. Use an Opus subagent (e.g. via the Claude Code Task tool) to score\n"
    "     each item with the rubric embedded in the batch. The rubric is the\n"
    "     fixed JUDGE_RUBRIC contract from eval_judge_prepare.py.\n"
    "  3. Write ONE verdict JSON per batch to the verdict_dir below, named\n"
    "     verdict_<run_id>_batch_NNN.json, matching the S5a writeback contract.\n"
    "  4. Re-run this orchestrator with the SAME --run-id. It will detect the\n"
    "     verdict files and proceed to S5b..S7.\n"
    "Note: the seed governs ONLY bootstrap resampling; LLM output is NOT\n"
    "reproducible — we are CI-bounded under sampling noise (3 runs + 95% CI)."
)


# =============================================================================
# 1. check_config — fail-fast
# =============================================================================

def check_config(env: Mapping[str, str]) -> None:
    """Fail-fast validator for the orchestrator's required env vars.

    Args:
      env: a mapping (typically ``os.environ``) to validate.

    Raises:
      RuntimeError — with a clear, actionable message — iff
        ``RAMCLOUDS_API_KEY`` is missing or empty.

    Notes:
      Other env vars (``EVAL_BOOTSTRAP_N`` etc.) are documented but have
      defensible defaults; we do NOT fail-fast on them here. A bad
      RAMCLOUDS_API_KEY is the only thing that absolutely cannot be
      worked around at runtime.
    """
    key = env.get("RAMCLOUDS_API_KEY") if env is not None else None
    if not key or not str(key).strip():
        raise RuntimeError(
            "RAMCLOUDS_API_KEY is missing or empty. "
            "Set it in the environment (or in rag2025/.env) before running "
            "the eval suite. The ramclouds gateway is required for S5c "
            "(Cohen's κ agreement) and the judge runs."
        )


# =============================================================================
# 2. default_deps — production wiring
# =============================================================================

def _import_module(name: str):
    """Lazy import by dotted name; return None on ImportError (NOT on other errors)."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
    except Exception as exc:  # pragma: no cover - defensive
        # Surface the error so a misconfigured env is not silently masked.
        raise RuntimeError(f"failed to import stage module {name!r}: {exc}") from exc


def _stub_stage(name: str) -> Callable[..., Dict[str, Any]]:
    """Build a placeholder stage callable for when a real module is unavailable.

    This keeps the orchestrator import-clean even if a stage's script has
    not been built yet. Real wiring (importing the actual ``main()`` /
    functions) is left to the per-stage callables below; the stub is a
    safety net that never silently no-ops in production — it raises.
    """

    def _callable(*, run_id, seed, **_: Any) -> Dict[str, Any]:
        raise RuntimeError(
            f"stage {name!r} has no production wiring. Import the corresponding "
            f"script (eval_run_instrumented.py / eval_judge_prepare.py / etc.) "
            f"and wire it into default_deps() in run_eval_suite.py."
        )

    _callable.__name__ = f"_stub_{name}"
    return _callable


def _wrap_main(script_path: pathlib.Path, *, stage_name: str, argv_extra: Optional[Sequence[str]] = None) -> Callable[..., Dict[str, Any]]:
    """Build a stage callable that invokes a script's ``main(argv)`` via ``runpy``.

    The callable captures ``script_path`` and a static ``argv_extra`` (e.g.
    ``["--records", "..."]``). The orchestrator appends ``--run-id`` and
    ``--seed`` at call time, so the downstream script's argparse can read
    them. Returns a dict indicating success.
    """
    def _callable(*, run_id, seed, **_: Any) -> Dict[str, Any]:
        import runpy
        argv = [str(script_path), *(argv_extra or []), "--run-id", str(run_id), "--seed", str(seed)]
        old_argv = sys.argv
        try:
            sys.argv = argv
            runpy.run_path(str(script_path), run_name="__main__")
        finally:
            sys.argv = old_argv
        return {"stage": stage_name, "ok": True, "run_id": run_id, "seed": seed}
    _callable.__name__ = f"_wrap_{stage_name}"
    return _callable


def _default_verdicts_exist_check(*, run_id: str, **_kwargs: Any) -> bool:
    """Production judging-gate check: True iff verdict_dir/<run_id> has files."""
    verdict_dir = JUDGE_VERDICTS_ROOT / str(run_id)
    if not verdict_dir.is_dir():
        return False
    try:
        next(verdict_dir.iterdir())
    except StopIteration:
        return False
    return True


def _default_s2(*, run_id: str, seed: int, smoke: bool = False, **_: Any) -> Dict[str, Any]:
    """Production S2 wiring: invoke eval_run_instrumented.py via runpy.

    Honors the ``smoke`` flag by passing ``--smoke`` through to the
    instrumented runner (which selects a 3-question subset).
    """
    import runpy
    script = RAG2025_ROOT / "scripts" / "eval_run_instrumented.py"
    argv = [str(script), "--run-id", str(run_id), "--seed", str(seed)]
    if smoke:
        argv.append("--smoke")
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
    return {"stage": "s2_instrumented", "ok": True, "run_id": run_id, "seed": seed, "smoke": smoke}


def default_deps() -> Dict[str, Callable[..., Any]]:
    """Build the production ``deps`` dict.

    Each stage callable is a thin wrapper around the corresponding script's
    ``main()`` (invoked via ``runpy`` with a synthesized ``sys.argv``). This
    keeps the orchestrator agnostic to the scripts' internals — the wiring
    is a contract, not a re-implementation.

    The judging-gate check inspects the verdict dir for the run; it does
    NOT call any LLM.

    NOTE: if a script is missing or its main() signature changes, the
    corresponding stage callable will raise at first use (NOT at import).
    """
    scripts = RAG2025_ROOT / "scripts"

    def _wrap(name: str, *extra: str) -> Callable[..., Any]:
        return _wrap_main(scripts / f"{name}.py", stage_name=name, argv_extra=list(extra) if extra else None)

    return {
        "s2_instrumented": _default_s2,
        "s5a_prepare": _wrap("eval_judge_prepare"),
        "judging_gate_check": _default_verdicts_exist_check,
        "s5b_aggregate": _wrap("eval_judge_aggregate"),
        "s5c_ramclouds": _wrap("eval_judge_ramclouds"),
        "s6_halltype": _wrap("eval_judge_halltype"),
        "s7_report": _wrap("eval_report_metrics"),
    }


# =============================================================================
# 3. run_suite — orchestrator
# =============================================================================

def _batch_dir(run_id: str) -> pathlib.Path:
    return JUDGE_BATCHES_ROOT / str(run_id)


def _verdict_dir(run_id: str) -> pathlib.Path:
    return JUDGE_VERDICTS_ROOT / str(run_id)


def _filter_deps(deps: Mapping[str, Callable[..., Any]], stages: Optional[Sequence[str]]) -> Dict[str, Callable[..., Any]]:
    """Return a deps dict filtered to the requested stage keys (None = all).

    The judging gate is mandatory and cannot be filtered out — it is the
    coordination point between automatable stages and the Opus subagent.
    """
    if not stages:
        return dict(deps)
    wanted = {s.strip() for s in stages if s and s.strip()}
    unknown = wanted - set(ALL_STAGE_KEYS)
    if unknown:
        raise ValueError(
            f"unknown --stages value(s): {sorted(unknown)}. "
            f"Valid: {list(ALL_STAGE_KEYS)}"
        )
    out: Dict[str, Callable[..., Any]] = {}
    for k, v in deps.items():
        if k == "judging_gate_check":
            out[k] = v  # always keep
        elif k in wanted:
            out[k] = v
    return out


def _ensure_required_deps(deps: Mapping[str, Callable[..., Any]]) -> None:
    """Sanity-check that all required stage keys are present in the deps dict."""
    required = (
        "s2_instrumented",
        "s5a_prepare",
        "judging_gate_check",
        "s5b_aggregate",
        "s5c_ramclouds",
        "s6_halltype",
        "s7_report",
    )
    missing = [k for k in required if k not in deps]
    if missing:
        raise RuntimeError(
            f"deps dict is missing required stage key(s): {missing}. "
            f"Use default_deps() or inject a full spy set."
        )


def run_suite(
    *,
    run_id: str,
    seed: int = 42,
    stages: Optional[Sequence[str]] = None,
    smoke: bool = False,
    deps: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> Dict[str, Any]:
    """Orchestrate the seven eval harness stages for one ``run_id + seed``.

    Args:
      run_id: a unique identifier for the run (also used as a directory name).
      seed: an int propagated to every stage. NOTE: governs ONLY bootstrap
        resampling determinism (S5b). LLM output is NOT reproducible; this
        is a CI-bounded harness, not a bit-reproducible one.
      stages: optional sequence of stage keys to run (e.g. ``["s5b", "s7"]``
        to re-run just aggregation + report). ``None`` means all stages.
      smoke: if ``True``, the S2 stage runs a 3-question subset (a smoke
        run). All other stages are unaffected.
      deps: optional mapping of stage callables. For testing, inject spies.
        For production, use ``default_deps()``.

    Returns:
      A status dict. Two shapes:

      * On success: ``{"status": "complete", "run_id", "seed", "stages": {...}}``
        where ``stages`` is a dict of ``stage_name -> result_dict``.
      * On judging-gate pause: ``{"status": "paused", "paused_at":
        "opus_judging", "run_id", "seed", "batch_dir", "verdict_dir",
        "instructions", "stages": {<upstream>}}``.

    Raises:
      RuntimeError — if ``deps`` is missing required keys, or if a stage
        callable itself raises. The orchestrator does NOT swallow stage
        errors; it propagates them so a partial run is loud.
    """
    if not run_id or not str(run_id).strip():
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")

    active_deps = dict(deps) if deps is not None else default_deps()
    active_deps = _filter_deps(active_deps, stages)
    _ensure_required_deps(active_deps)

    stages_run: Dict[str, Any] = {}

    # ---- S2: instrumented runner ----
    stages_run["s2_instrumented"] = active_deps["s2_instrumented"](
        run_id=run_id, seed=seed, smoke=smoke
    )

    # ---- S5a: judge batch preparer ----
    stages_run["s5a_prepare"] = active_deps["s5a_prepare"](
        run_id=run_id, seed=seed
    )

    # ---- Stage 3: ORCHESTRATION GATE (Opus subagent judging) ----
    # This is NOT a python call. It is a human / orchestrator step.
    verdicts_ready = bool(
        active_deps["judging_gate_check"](run_id=run_id)
    )
    if not verdicts_ready:
        return {
            "status": "paused",
            "paused_at": "opus_judging",
            "run_id": run_id,
            "seed": seed,
            "smoke": smoke,
            "batch_dir": str(_batch_dir(run_id)),
            "verdict_dir": str(_verdict_dir(run_id)),
            "instructions": JUDGING_INSTRUCTIONS,
            "stages": stages_run,
        }

    # ---- S5b: aggregator ----
    stages_run["s5b_aggregate"] = active_deps["s5b_aggregate"](
        run_id=run_id, seed=seed
    )

    # ---- S5c: ramclouds κ agreement ----
    stages_run["s5c_ramclouds"] = active_deps["s5c_ramclouds"](
        run_id=run_id, seed=seed
    )

    # ---- S6: hallucination type classifier ----
    stages_run["s6_halltype"] = active_deps["s6_halltype"](
        run_id=run_id, seed=seed
    )

    # ---- S7: final metrics report ----
    stages_run["s7_report"] = active_deps["s7_report"](
        run_id=run_id, seed=seed
    )

    return {
        "status": "complete",
        "run_id": run_id,
        "seed": seed,
        "smoke": smoke,
        "stages": stages_run,
    }


# =============================================================================
# 4. main — CLI
# =============================================================================

def _parse_stages_arg(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or not raw.strip():
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. ``check_config(os.environ)`` runs FIRST (fail-fast)."""
    parser = argparse.ArgumentParser(
        prog="run_eval_suite",
        description=(
            "Slice S10 — single-entrypoint orchestrator for the RAGAS-grade "
            "eval harness. Runs S2, S5a, [judging gate], S5b, S5c, S6, S7 "
            "in order with one run_id+seed. Pauses at the Opus-subagent "
            "judging gate with instructions for the orchestrator."
        ),
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier (also a directory name).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap seed (default 42). Governs ONLY bootstrap resampling; "
             "LLM output is NOT bit-reproducible — this is a CI-bounded harness.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke run: S2 uses a 3-question subset. All other stages unchanged.",
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help=(
            "Comma-separated subset of stage keys to run "
            f"(valid: {','.join(ALL_STAGE_KEYS)}). Default: all. "
            "The judging gate is mandatory and cannot be filtered."
        ),
    )
    args = parser.parse_args(argv)

    # FAIL-FAST: env validation BEFORE any work.
    check_config(os.environ)

    stages = _parse_stages_arg(args.stages)
    result = run_suite(
        run_id=args.run_id,
        seed=args.seed,
        stages=stages,
        smoke=args.smoke,
    )

    # Print a terse summary to stdout (action-first; this is a CLI).
    if result["status"] == "paused":
        print(f"[run_eval_suite] PAUSED at opus_judging for run_id={result['run_id']}")
        print(f"  batch_dir:   {result['batch_dir']}")
        print(f"  verdict_dir: {result['verdict_dir']}")
        print(f"  instructions:")
        for line in result["instructions"].splitlines():
            print(f"    {line}")
        return 0  # exit 0 — pause is a normal flow, not an error

    print(f"[run_eval_suite] COMPLETE run_id={result['run_id']} seed={result['seed']}")
    for k, v in (result.get("stages") or {}).items():
        ok = (v or {}).get("ok", True)
        print(f"  {k}: ok={ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
