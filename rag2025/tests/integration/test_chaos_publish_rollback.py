"""Ticket 29: the chaos publish/rollback drill, as a repeatable test.

Runs the REAL drill (:file:`rag2025/scripts/chaos_publish_rollback.py`) against
a throwaway tmp_path store: the real ``src/main.py`` app, the real admin
router, the real retriever/booster/router/fusion, and a publish
(``/admin/reload-table``) plus a rollback (``/admin/rollback``) injected
mid-load. Only the generator is stubbed (the LLM provider is down), so the
latencies measured here are routing + retrieval + fusion + payload build, NOT
end-to-end LLM latency.

Three claims, one test each:

1. A clean drill verdicts PASS with ZERO mixed-release answers, and the
   booster's second read (``fetch_by_id``) actually ran — otherwise "0 mixed"
   would be the vacuous absence of the read that can mix, not its consistency.
2. Replaying the same scenario reproduces the same verdict and the same
   request/error/injection tallies (latency is timing, not identity, and is
   excluded).
3. The detector is mutation-proven: forcing the *other* release's row into one
   booster read makes it fire on exactly that answer. A gate that cannot be
   shown to close is not a gate.

Run: ``cd rag2025 && ../.venv/Scripts/python.exe -m pytest
tests/integration/test_chaos_publish_rollback.py -v``
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]  # rag2025/
SCRIPT = ROOT / "scripts" / "chaos_publish_rollback.py"

#: A smaller DEFAULT_SCENARIO: 4 workers, 20 requests over 4s, publish at 1.5s,
#: rollback at 3.0s — both swaps land inside the load, like the live drill.
SMALL_SCENARIO: Dict[str, Any] = {
    "workers": 4,
    "duration_s": 4.0,
    "rate_rps": 5.0,
    "publish_at_s": 1.5,
    "rollback_at_s": 3.0,
    "query_delay_s": 0.0,
    "seed": 29,
}


def _load_drill():
    """Import the drill script as a module (it self-installs sys.path + chdir)."""
    spec = importlib.util.spec_from_file_location("chaos_drill_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: @dataclass resolves cls.__module__ via sys.modules,
    # so an unregistered module makes every dataclass in the script raise
    # AttributeError: 'NoneType' object has no attribute '__dict__'.
    sys.modules["chaos_drill_script"] = module
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("chaos_drill_script", None)
        raise
    return module


@pytest.fixture(scope="module")
def drill():
    """The drill module, with the process environment restored afterwards.

    The script deliberately rewrites os.environ (LANCEDB_URI, ADMIN_IDENTITIES,
    ...) before ``src/main.py`` is imported; restoring wholesale keeps the rest
    of the suite from inheriting a chaos-test config.
    """
    saved = dict(os.environ)
    module = _load_drill()
    yield module
    os.environ.clear()
    os.environ.update(saved)


def _run(drill_module: Any, workdir: Path) -> Any:
    drill_module._configure_env(workdir)
    return asyncio.run(
        drill_module._run_drill(dict(SMALL_SCENARIO), workdir)
    )


def _replay_snapshot(result: Any) -> Tuple:
    """The parts of a drill run that a replay must reproduce.

    Latency, the in/out-of-window split and the fetch_by_id call count are
    wall-clock artifacts of where the swap happened to land; the verdict, the
    status tallies and the injection outcomes are the drill's identity.
    """
    return (
        result.counts["requests"],
        result.counts["http_200"],
        result.counts["http_non_200"],
        result.counts["mixed_release_answers"],
        result.counts["router_llm_calls"],
        result.counts["booster_chunk_miss"],
        tuple(sorted(result.errors_by_status.items())),
        tuple((i["label"], i["status"]) for i in result.injections),
        drill_verdict(result),
    )


def drill_verdict(result: Any) -> str:
    from chaos_publish_rollback import verdict_line

    return verdict_line(result)


def test_clean_drill_verdicts_pass_with_zero_mixed_answers(drill, tmp_path):
    result = _run(drill, tmp_path)

    # Both injections must have really happened: a drill where the rollback
    # 409s would report "0 mixed" without ever rolling anything back.
    labels = {i["label"]: i["status"] for i in result.injections}
    assert labels == {"publish": 200, "rollback": 200}, result.injections

    assert result.counts["requests"] == 20
    assert result.counts["http_200"] == 20
    assert result.errors_by_status == {}
    assert result.counts["mixed_release_answers"] == 0
    assert result.mixed == []

    # Evidence the mixing seam actually ran (design §5/§Remaining-work 2): the
    # booster's second read of the serving retriever was called, and every
    # chunk it looked up existed (zero misses).
    assert result.counts["fetch_by_id_calls"] > 0
    assert result.counts["booster_chunk_miss"] == 0

    assert drill.verdict_ok(result) is True
    assert "PASS" in drill_verdict(result)


def test_replay_reproduces_the_same_verdict(drill, tmp_path):
    first = _run(drill, tmp_path / "run1")
    second = _run(drill, tmp_path / "run2")
    assert _replay_snapshot(first) == _replay_snapshot(second)


def test_mutation_proof_detector_bites(drill, tmp_path):
    """The detector must FIRE on a deliberately mixed response.

    Without this, criterion "no mixed-release answers" is unfalsifiable: a
    detector that never matches anything also reports zero.
    """
    result = asyncio.run(drill._run_mutation_proof(tmp_path))

    assert result.counts["mixed_release_answers"] == 1
    assert result.counts["poisoned_reads"] == 1
    assert len(result.mixed) == 1
    assert sorted(result.mixed[0]["releases"]) == sorted(
        [drill.RELEASE_A, drill.RELEASE_B]
    )
    assert drill.verdict_ok(result) is True
    assert "detector fired" in drill_verdict(result)
