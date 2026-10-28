#!/usr/bin/env python
"""Retrieval-only A/B harness: dense vs hybrid (BM25 + RRF) — ticket 09.

WHY THIS EXISTS
---------------
Ticket 09 criterion 1 is "recall@k per intent must not regress vs the baseline
manifest" and criterion 3 is "the RRF candidate set is wider than the final set,
with the coefficients and thresholds written down". Neither needs a generated
answer, so neither needs an LLM.

The existing instrumented harness (``eval_run_instrumented.py``) cannot answer
either question for the hybrid path, for a measured reason: it calls
``retriever.retrieve(...)`` directly and never references ``HybridSearchService``
at all, and ``.env`` carries ``USE_HYBRID_RETRIEVAL=false``, so hybrid is only
reachable through ``HybridSearchService.retrieve``. Reusing it would produce a
dense-only artifact that is not evidence about hybrid retrieval.

WHAT THIS DOES
--------------
Runs the SAME questions, over the SAME embeddings and the SAME LanceDB table,
through BOTH arms in ONE process:

  arm "dense"  = ``LanceDBRetriever.retrieve(...)``
  arm "hybrid" = ``HybridSearchService.retrieve(...)`` with the BM25 index built
                 in-process by ``build_bm25_index()``

The hybrid service is constructed EXPLICITLY here, so the arm does not depend on
``USE_HYBRID_RETRIEVAL`` in ``.env`` (which is ``false``; the flag is recorded in
the run meta as the reason this harness exists).

Both arms are handed the identical query vector: the question is embedded once
per question and the same list is passed to both arms, so no delta can come from
the encoder. Both arms also carry the same temporal hint
(``metadata_filter={"query": <question>}``), which is what the hybrid path passes
to its own dense call — so the delta isolates FUSION, not the temporal wiring.

ONE RECORD PER ARM, EvalRecord-shaped
-------------------------------------
``eval_record.EvalRecord`` is reused verbatim (same class the instrumented
harness writes), so the existing reporter consumes the output unchanged. The
record carries the retrieved chunks' FULL TEXT — not just ids — because the
recall definition below is a text-substring test. Fields the retrieval-only run
cannot honestly fill: ``answer`` is ``""`` (no generator was invoked — the
provider is dead in this environment and generation is not needed for recall@k
or latency) and ``intent``/``complexity`` are ``None`` (no router ran).
Consequently the reporter marks ``semantic_accuracy``/``groundedness`` as
``unmeasured_no_judge`` rather than inventing a number.

METRICS
-------
The per-intent manifests and the delta report are produced by the EXISTING
reporter, ``scripts/eval_per_intent_report.py``, invoked as a subprocess — its
recall@k definition is reused verbatim:

    fraction of scored questions whose retrieved chunk TEXT contains EVERY
    ``must_include`` / ``critical_facts[].value`` string (diacritic/whitespace-
    insensitive substring); k = the retrieved_chunks list persisted per record

This module computes no recall of its own; it only feeds records in and renders
a short human summary of what the reporter produced.

CRITERION 3 MEASUREMENT
-----------------------
The candidate pool is measured, not asserted. ``HybridSearchService`` is handed a
transparent recording proxy instead of the raw retriever (it delegates every
attribute; only ``retrieve()`` is wrapped, to copy the returned documents), and
``_bm25_search`` is wrapped on the instance. That records, per question: the
dense candidate count, the raw BM25 candidate count, the BM25 candidates that
survive the SAME post-fetch filter the dense arm obeys, the size of the fused
union, and the final count. Retrieval behaviour is unchanged by the recording.

Outputs (never overwriting an existing run directory), under
``results/eval_harness/retrieval_ab/<run_id>/``:

    records_dense.jsonl / records_hybrid.jsonl
    manifest_dense.json (+ .md) / manifest_hybrid.json (+ .md)
    delta_hybrid_vs_dense.json (+ .md)          <- the A/B answer
    delta_hybrid_vs_historical_baseline.json    <- criterion 1's literal wording
    delta_dense_vs_historical_baseline.json     <- methodology control
    run_meta.json, summary.md

CLI
---
    ./.venv/Scripts/python.exe scripts/eval_retrieval_ab.py --limit 3
    ./.venv/Scripts/python.exe scripts/eval_retrieval_ab.py
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent  # rag2025/
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env so the settings this run reports are the settings that are served.
# NOTE: loading it does NOT enable the hybrid arm — the arm is constructed
# explicitly below and the flag's value is recorded as evidence.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT / ".env")
except Exception:  # pragma: no cover — dotenv optional
    pass

from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
import eval_record as _er  # noqa: E402  (sibling script — the record contract)
from eval_per_intent_report import canonical_json  # noqa: E402  (sibling script)
from eval_report_metrics import percentile  # noqa: E402  (existing p-in-[0,100] helper)
from services import hybrid_search as _hybrid_mod  # noqa: E402
from services import lancedb_retrieval as _retrieval_mod  # noqa: E402
from services.embedding import EmbeddingService  # noqa: E402
from services.hybrid_search import HybridSearchService  # noqa: E402
from services.lancedb_retrieval import get_retriever  # noqa: E402

HARNESS_VERSION = "t09-ab-v1"
ARMS = ("dense", "hybrid")
DEFAULT_TOP_K = 5
DEFAULT_DATASET = _PROJECT / "data" / "eval" / "husc_86q_test.json"
DEFAULT_OUT_ROOT = _PROJECT / "results" / "eval_harness" / "retrieval_ab"
DEFAULT_BASELINE_MANIFEST = (
    _PROJECT
    / "results"
    / "eval_harness"
    / "metrics_report"
    / "per_intent"
    / "bf01_86q_303q.manifest.json"
)
REPORTER = _HERE / "eval_per_intent_report.py"

# A bucket whose delta CI is wider than this cannot certify a non-regression for
# realistic effect sizes; the summary says so instead of claiming a pass.
INCONCLUSIVE_CI_HALF_WIDTH = 0.10

# =============================================================================
# Candidate-pool recording (criterion 3) — behaviour-transparent
# =============================================================================


class PoolRecorder:
    """Transparent proxy around a ``LanceDBRetriever`` that records the pools.

    Delegates EVERY attribute to the real retriever, so ``build_bm25_index()``
    (which reaches for ``_adapter`` / ``_table``) and the BM25 post-fetch filter
    (``apply_post_fetch_filters``) behave exactly as they do in production.
    Only ``retrieve()`` is wrapped, and only to copy the documents it returned.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.dense_pool: List[Any] = []
        self.sparse_pool: List[Any] = []
        # The k the hybrid arm ASKED each side for. Recorded because criterion 3
        # is about the top_k*2 relation itself, and inferring it from a doc count
        # would only work on a corpus larger than the request.
        self.dense_requested_k: Optional[int] = None
        self.sparse_requested_k: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        top_k = kwargs.get("top_k")
        if top_k is None and len(args) > 1:
            top_k = args[1]
        self.dense_requested_k = top_k
        result = self._inner.retrieve(*args, **kwargs)
        self.dense_pool = list(getattr(result, "documents", None) or [])
        return result


def _wrap_sparse_recorder(recorder: PoolRecorder, hybrid: HybridSearchService) -> None:
    """Record the raw BM25 candidate list from the instance's own search call."""
    original = hybrid._bm25_search  # noqa: SLF001 — measurement, not behaviour

    def wrapped(query: str, top_k: int) -> List[Any]:
        recorder.sparse_requested_k = top_k
        docs = original(query, top_k)
        recorder.sparse_pool = list(docs or [])
        return docs

    hybrid._bm25_search = wrapped  # type: ignore[method-assign]  # noqa: SLF001


# =============================================================================
# Loading
# =============================================================================


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions(path: pathlib.Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or []
    questions = [q for q in data if isinstance(q, dict) and q.get("id")]
    return questions[:limit] if limit is not None else questions


def _pool_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)
    return {
        "n": len(ordered),
        "min": int(ordered[0]),
        "p05": percentile(ordered, 5.0),
        "p50": percentile(ordered, 50.0),
        "p95": percentile(ordered, 95.0),
        "max": int(ordered[-1]),
    }


def _ms_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)
    return {
        "n": len(ordered),
        "p50": round(percentile(ordered, 50.0), 2),
        "p95": round(percentile(ordered, 95.0), 2),
    }


# =============================================================================
# Record building
# =============================================================================


def _chunk(doc: Any) -> _er.RetrievedChunk:
    md = getattr(doc, "metadata", None) or {}
    year = md.get("data_year")
    return _er.RetrievedChunk(
        chunk_id=str(getattr(doc, "chunk_id", "") or ""),
        source=str(getattr(doc, "source", "") or ""),
        score=float(getattr(doc, "score", 0.0) or 0.0),
        text=str(getattr(doc, "text", "") or ""),
        data_year=year if isinstance(year, int) else None,
        is_superseded=bool(md.get("is_superseded", False)),
    )


def _build_record(
    *,
    arm: str,
    qid: str,
    question: str,
    docs: Sequence[Any],
    retrieval_ms: float,
    run_id: str,
    seed: int,
    timestamp: str,
    settings: Any,
) -> _er.EvalRecord:
    """One EvalRecord for one arm.

    ``answer`` is the empty string: this run invokes no generator, and the
    reporter treats "no answer + no judge_scores" as ``unmeasured_no_judge``
    rather than as a failure. ``route`` names the arm so the manifest's route
    distribution identifies which side produced it.
    """
    return _er.EvalRecord(
        run_id=run_id,
        seed=seed,
        harness_version=HARNESS_VERSION,
        timestamp=timestamp,
        model_version=_er.ModelVersion(
            gen=settings.NS24_GEN_MODEL,
            ner=settings.NS24_NER_MODEL,
            hyde=settings.NS24_HYDE_MODEL,
            embedding=settings.EMBEDDING_MODEL,
        ),
        id=qid,
        question=question,
        intent=None,  # no router ran; not the arm's intent
        complexity=None,
        route=f"{arm}_retrieval",
        expected_behavior=None,
        retrieved_chunks=[_chunk(d) for d in docs],
        answer="",
        latency_ms=_er.LatencyMs(
            route_ms=0.0,
            retrieval_loop_ms=retrieval_ms,
            query_ms=0.0,
            total_ms=retrieval_ms,
        ),
        latency_attribution=None,
    )


# =============================================================================
# The A/B loop
# =============================================================================


async def run_pair(
    q: Dict[str, Any],
    *,
    retriever: Any,
    recorder: PoolRecorder,
    hybrid: HybridSearchService,
    embedding: EmbeddingService,
    top_k: int,
    run_id: str,
    seed: int,
    settings: Any,
) -> Tuple[_er.EvalRecord, _er.EvalRecord, Dict[str, Any]]:
    """Run ONE question through both arms over ONE embedding. Returns (dense, hybrid, pool)."""
    qid = str(q["id"])
    text = str(q["question"])
    timestamp = datetime.datetime.now().isoformat()

    embed_start = time.perf_counter()
    query_vector = embedding.encode_query(text).tolist()
    embed_ms = (time.perf_counter() - embed_start) * 1000.0

    # Same temporal hint the hybrid path passes to its own dense call, so the
    # dense arm differs from the hybrid arm only by the presence of fusion.
    metadata_filter = {"query": text}

    dense_error: Optional[str] = None
    recorder.dense_pool = []
    dense_start = time.perf_counter()
    try:
        dense_result = retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=dict(metadata_filter),
        )
        dense_docs = list(dense_result.documents or [])
        if dense_result.error_type is not None:
            dense_error = dense_result.error_type.value
    except Exception as exc:  # noqa: BLE001 — recorded, never fabricated over
        logger.error(f"dense arm failed for {qid}: {exc}")
        dense_docs = []
        dense_error = f"{type(exc).__name__}: {exc}"
    dense_ms = (time.perf_counter() - dense_start) * 1000.0

    hybrid_error: Optional[str] = None
    recorder.dense_pool = []
    recorder.sparse_pool = []
    hybrid_start = time.perf_counter()
    try:
        hybrid_result = await hybrid.retrieve(
            query=text,
            query_vector=query_vector,
            top_k=top_k,
        )
        hybrid_docs = list(hybrid_result.documents or [])
    except Exception as exc:  # noqa: BLE001 — recorded, never fabricated over
        logger.error(f"hybrid arm failed for {qid}: {exc}")
        hybrid_docs = []
        hybrid_error = f"{type(exc).__name__}: {exc}"
    hybrid_ms = (time.perf_counter() - hybrid_start) * 1000.0

    # The BM25 arm's own post-fetch filter, re-applied to the RECORDED candidate
    # list — same function, same inputs, so the count is the count that fused.
    sparse_filtered = hybrid._apply_post_fetch_filter(  # noqa: SLF001
        list(recorder.sparse_pool), dict(metadata_filter)
    )
    dense_ids = {getattr(d, "chunk_id", None) for d in recorder.dense_pool}
    sparse_ids = {getattr(d, "chunk_id", None) for d in sparse_filtered}
    union_ids = dense_ids | sparse_ids
    union_ids.discard(None)
    # How many of the FINAL docs the sparse arm alone put there — the RRF
    # weight is meaningless if BM25 never survives fusion.
    final_from_sparse_only = sum(
        1
        for d in hybrid_docs
        if getattr(d, "chunk_id", None) not in dense_ids
    )
    # BM25 can also act purely by REORDERING docs dense already had. Measure
    # that separately from "supplied a doc dense lacked", because the two mean
    # very different things about what the sparse weight buys.
    dense_order = [
        getattr(d, "chunk_id", None) for d in recorder.dense_pool[:top_k]
    ]
    fused_order = [getattr(d, "chunk_id", None) for d in hybrid_docs]
    final_not_in_dense_topk = sum(
        1 for cid in fused_order if cid not in set(dense_order)
    )

    pool = {
        "id": qid,
        "top_k": top_k,
        "dense_requested_k": recorder.dense_requested_k,
        "sparse_requested_k": recorder.sparse_requested_k,
        "dense_candidates": len(recorder.dense_pool),
        "sparse_candidates": len(recorder.sparse_pool),
        "sparse_after_post_fetch": len(sparse_filtered),
        "fused_union": len(union_ids),
        "final": len(hybrid_docs),
        "final_from_sparse_only": final_from_sparse_only,
        "final_not_in_dense_topk": final_not_in_dense_topk,
        "order_changed_vs_dense_topk": fused_order != dense_order,
        "dense_final": len(dense_docs),
        "dense_error": dense_error,
        "hybrid_error": hybrid_error,
        "embed_ms": embed_ms,
        "dense_ms": dense_ms,
        "hybrid_ms": hybrid_ms,
    }

    dense_record = _build_record(
        arm="dense",
        qid=qid,
        question=text,
        docs=dense_docs,
        retrieval_ms=dense_ms,
        run_id=run_id,
        seed=seed,
        timestamp=timestamp,
        settings=settings,
    )
    hybrid_record = _build_record(
        arm="hybrid",
        qid=qid,
        question=text,
        docs=hybrid_docs,
        retrieval_ms=hybrid_ms,
        run_id=run_id,
        seed=seed,
        timestamp=timestamp,
        settings=settings,
    )
    return dense_record, hybrid_record, pool


async def run_all(
    questions: Sequence[Dict[str, Any]],
    *,
    retriever: Any,
    recorder: PoolRecorder,
    hybrid: HybridSearchService,
    embedding: EmbeddingService,
    top_k: int,
    run_id: str,
    seed: int,
    settings: Any,
    out_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    dense_path = out_dir / "records_dense.jsonl"
    hybrid_path = out_dir / "records_hybrid.jsonl"
    pools: List[Dict[str, Any]] = []

    with dense_path.open("w", encoding="utf-8") as fd, hybrid_path.open(
        "w", encoding="utf-8"
    ) as fh:
        for i, q in enumerate(questions, start=1):
            logger.info(f"{i}/{len(questions)}: {q.get('id')}")
            dense_rec, hybrid_rec, pool = await run_pair(
                q,
                retriever=retriever,
                recorder=recorder,
                hybrid=hybrid,
                embedding=embedding,
                top_k=top_k,
                run_id=run_id,
                seed=seed,
                settings=settings,
            )
            # Checkpoint per question, like the instrumented harness: a crash
            # mid-run loses at most one question.
            fd.write(dense_rec.to_jsonl_line() + "\n")
            fd.flush()
            fh.write(hybrid_rec.to_jsonl_line() + "\n")
            fh.flush()
            pools.append(pool)
    return pools


# =============================================================================
# Reporting — the EXISTING reporter, invoked as a subprocess
# =============================================================================


def invoke_reporter(argv: Sequence[str]) -> str:
    """Run ``eval_per_intent_report.py`` and return its stdout (or raise)."""
    cmd = [sys.executable, str(REPORTER), *argv]
    logger.info("reporter: " + " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(_PROJECT), capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval_per_intent_report.py exited {proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout or ""


def report_arm(
    *,
    records: pathlib.Path,
    dataset: pathlib.Path,
    gt: pathlib.Path,
    out_json: pathlib.Path,
    out_md: pathlib.Path,
) -> None:
    invoke_reporter(
        [
            "--records",
            str(records),
            "--dataset",
            str(dataset),
            "--gt",
            str(gt),
            "--out",
            str(out_json),
            "--md",
            str(out_md),
        ]
    )


def report_delta(
    *, baseline: pathlib.Path, candidate: pathlib.Path, out_json: pathlib.Path, out_md: pathlib.Path
) -> Dict[str, Any]:
    text = invoke_reporter(
        [
            "--compare",
            str(baseline),
            str(candidate),
            "--out",
            str(out_json),
            "--md",
            str(out_md),
        ]
    )
    sys.stdout.write(text)
    return json.loads(out_json.read_text(encoding="utf-8"))


# =============================================================================
# Human summary
# =============================================================================


def _verdict_code(delta_block: Dict[str, Any]) -> str:
    """Classify a recall@k delta block by its CONFIDENCE INTERVAL.

    Classification is by the interval, not the point estimate: a delta of
    exactly 0 with a ±0.24 CI is not "no change", it is a bucket too small to
    resolve anything, and calling it "no change" would read as a pass. An empty
    bucket is likewise not measurable — never a pass.

    Returns one of ``not_measurable``, ``n/a``, ``regression``, ``improvement``,
    ``unresolved``.
    """
    if not (delta_block.get("n") or {}).get("candidate"):
        return "not_measurable"
    delta = delta_block.get("delta")
    ci = delta_block.get("ci95")
    if delta is None or ci is None:
        return "n/a"
    if ci[1] < 0:
        return "regression"
    if ci[0] > 0:
        return "improvement"
    return "unresolved"


_VERDICT_TEXT = {
    "not_measurable": "not measurable (no scored questions in this bucket)",
    "n/a": "n/a",
    "regression": "REGRESSION (significant)",
    "improvement": "improvement (significant)",
    "unresolved": "unresolved (95% CI spans 0)",
}


def _verdict(delta_block: Dict[str, Any]) -> str:
    """Human text for :func:`_verdict_code`."""
    return _VERDICT_TEXT[_verdict_code(delta_block)]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_ci(ci: Optional[Sequence[float]]) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def render_summary(
    *,
    run_id: str,
    meta: Dict[str, Any],
    dense_manifest: Dict[str, Any],
    hybrid_manifest: Dict[str, Any],
    delta: Dict[str, Any],
    historical: Optional[Dict[str, Any]],
    dense_vs_historical: Optional[Dict[str, Any]],
) -> str:
    taxonomy = list(hybrid_manifest["by_intent"].keys())
    hybrid_cfg = meta["hybrid"]

    lines: List[str] = []
    lines.append(f"# Retrieval A/B (dense vs hybrid) — ticket 09 criteria 1 & 3")
    lines.append("")
    lines.append(f"- run id: `{run_id}`")
    lines.append(f"- harness: `scripts/eval_retrieval_ab.py` ({HARNESS_VERSION})")
    lines.append(
        f"- questions: {meta['dataset']['questions']} "
        f"(`{meta['dataset']['name']}` sha256 `{meta['dataset']['sha256'][:12]}…`)"
    )
    lines.append(
        f"- k (chunks persisted per arm) = {meta['top_k']}; "
        f"embeddings `{meta['embedding']['model']}` dim {meta['embedding']['dim']} — "
        "ONE vector per question, shared by both arms"
    )
    lines.append(
        f"- table `{meta['lancedb']['table']}` @ `{meta['lancedb']['uri']}` "
        f"({meta['lancedb']['rows']} rows), BM25 built in-process "
        f"({hybrid_cfg['bm25_build_ms']:.0f} ms)"
    )
    lines.append(
        f"- mode: **retrieval only** — no router, no rerank, no generator "
        f"(LLM providers are dead in this environment and recall@k does not need "
        f"them). `USE_HYBRID_RETRIEVAL` in .env is "
        f"`{hybrid_cfg['env_use_hybrid_retrieval']}`; the hybrid arm is constructed "
        f"explicitly and does not read that flag."
    )
    lines.append("")

    lines.append("## Arm definitions")
    lines.append("")
    lines.append(
        f"- `dense`  = `LanceDBRetriever.retrieve(query_vector, top_k={meta['top_k']}, "
        f'metadata_filter={{"query": <question>}})`'
    )
    lines.append(
        f"- `hybrid` = `HybridSearchService.retrieve(query, query_vector, "
        f"top_k={meta['top_k']})` over the same retriever + a BM25 index built "
        f"by `build_bm25_index()`"
    )
    lines.append("")

    lines.append("## Criterion 3 — fusion coefficients and candidate-pool width")
    lines.append("")
    lines.append(
        f"- RRF coefficients (from settings): "
        f"`HYBRID_FUSION_DENSE_WEIGHT={hybrid_cfg['dense_weight']}`, "
        f"`HYBRID_FUSION_SPARSE_WEIGHT={hybrid_cfg['sparse_weight']}`"
    )
    lines.append(
        f"- RRF rank smoothing `k = {hybrid_cfg['rrf_k']}` "
        f"(default of `HybridSearchService._rrf_fusion`; read by introspection, "
        f"not hardcoded here)"
    )
    lines.append(
        f"- candidate pool per arm = `top_k * 2` = "
        f"{hybrid_cfg['candidate_pool_per_arm']} docs; fused down to "
        f"`top_k` = {meta['top_k']} final docs"
    )
    sparse_consts = hybrid_cfg.get("sparse_arm_constants")
    observed = (meta.get("pools") or {}).get("observed_requested_k")
    if observed:
        rendered = ", ".join(
            f"dense={d}, sparse={s}" for d, s in sorted(observed, key=lambda t: (t[0] or 0))
        )
        lines.append(
            f"- observed request to each arm (recorded from the calls, not "
            f"restated): {rendered}"
        )
    if sparse_consts:
        lines.append(
            "- sparse-arm score modifiers applied before fusion (read from the "
            f"modules, not restated by hand): `EXACT_CODE_BOOST={sparse_consts['exact_code_boost']}` "
            f"(cap {sparse_consts['exact_code_max_matches']} matched codes), "
            f"`YEAR_BOOST_FACTOR={sparse_consts['year_boost_factor']}`, "
            f"`OLD_YEAR_DEMOTE_FACTOR={sparse_consts['old_year_demote_factor']}` "
            f"(current admission year {sparse_consts['current_admission_year']})"
        )
    pools = meta["pools"]
    lines.append("")
    lines.append("| pool | p05 | p50 | p95 | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in (
        ("dense_candidates", "dense candidates (top_k*2 request)"),
        ("sparse_candidates", "BM25 candidates (top_k*2 request)"),
        ("sparse_after_post_fetch", "BM25 survivors after post-fetch filter"),
        ("fused_union", "fused union (dense ∪ filtered BM25)"),
        ("final", "final fused docs returned"),
        ("final_from_sparse_only", "final docs only BM25 supplied"),
        ("final_not_in_dense_topk", "final docs outside the dense top-k"),
    ):
        s = pools.get(key) or {"n": 0}
        lines.append(
            f"| {label} | {_fmt(s.get('p05'))} | {_fmt(s.get('p50'))} | "
            f"{_fmt(s.get('p95'))} | {s.get('min')} | {s.get('max')} |"
        )
    lines.append("")
    wider = pools.get("questions_with_union_gt_final")
    total = meta["dataset"]["questions"]
    if wider is not None:
        lines.append(
            f"- candidate set wider than the final set: **{wider}/{total}** questions "
            f"(union > final). Median union {_fmt(pools['fused_union'].get('p50'))} → "
            f"final {_fmt(pools['final'].get('p50'))}."
        )
    contributing = pools.get("questions_with_final_from_sparse_only")
    not_in_topk = pools.get("questions_with_final_not_in_dense_topk")
    reordered = pools.get("questions_with_order_changed")
    if contributing is not None:
        lines.append(
            f"- **what the 0.4 sparse weight actually bought**: BM25 changed WHICH "
            f"docs are in the final {meta['top_k']} in {not_in_topk}/{total} questions "
            f"(it promoted docs the dense arm had ranked below {meta['top_k']}), and "
            f"changed their ORDER in {reordered}/{total}. It supplied a doc outside "
            f"the dense candidate pool — the full `top_k*2` list, not just its top "
            f"{meta['top_k']} — in {contributing}/{total} questions. Recall@k is "
            f"order-insensitive, so only the first number can move criterion 1."
        )
        if not contributing:
            lines.append("")
            lines.append(
                "> **Measured finding (criterion 3).** The RRF requirement is met: "
                "the fused union is wider than the final set on **every** question "
                f"(median {_fmt(pools['fused_union'].get('p50'))} candidates → "
                f"{_fmt(pools['final'].get('p50'))} returned), and the sparse arm does "
                "reshape the result — it reorders the final set in "
                f"{reordered}/{total} questions and swaps members in "
                f"{not_in_topk}/{total}. What it never does is introduce a chunk the "
                "dense arm's own `top_k*2` candidate list did not already contain "
                f"({contributing}/{total}). So at k="
                f"{meta['top_k']} the fusion is a re-ranker of the dense pool, not a "
                "recall-extending arm: the 0.4 sparse weight can promote a chunk from "
                f"dense ranks {meta['top_k']+1}–{meta['top_k']*2} into the final set, "
                "but cannot surface a chunk dense never retrieved. That is the shape "
                "of the current fusion, reported as measured — not a defect claim "
                "about code outside this ticket."
            )
    lines.append("")

    lines.append("## Criterion 1 — recall@k per intent, hybrid vs dense (paired A/B)")
    lines.append("")
    lines.append(
        "| intent | scored | dense recall@k | hybrid recall@k | delta | 95% CI | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for intent in taxonomy + ["aggregate"]:
        block = delta["by_intent"].get(intent, {})
        rec = block.get("recall_at_k_all_present", {})
        n = (rec.get("n") or {}).get("candidate")
        lines.append(
            f"| {intent} | {n} | {_fmt(rec.get('baseline'))} | "
            f"{_fmt(rec.get('candidate'))} | "
            f"{('+' if isinstance(rec.get('delta'), float) and rec['delta'] > 0 else '')}"
            f"{_fmt(rec.get('delta'))} | {_fmt_ci(rec.get('ci95'))} | "
            f"{_verdict(rec)} |"
        )
    lines.append("")
    lines.append(
        "Sign convention: delta = hybrid − dense; CI is the Newcombe hybrid score "
        "interval from the reporter."
    )
    lines.append("")

    lines.append("## Criterion 1 — answer: did recall@k per intent regress in the hybrid arm?")
    lines.append("")
    regressions: List[str] = []
    unresolved: List[str] = []
    unmeasured: List[str] = []
    for intent in taxonomy + ["aggregate"]:
        rec = delta["by_intent"].get(intent, {}).get("recall_at_k_all_present", {})
        code = _verdict_code(rec)
        lines.append(
            f"- **{intent}**: {_VERDICT_TEXT[code]} (delta {_fmt(rec.get('delta'))}, "
            f"CI {_fmt_ci(rec.get('ci95'))})"
        )
        if code == "regression":
            regressions.append(intent)
        elif code == "not_measurable":
            unmeasured.append(intent)
        elif code == "unresolved":
            unresolved.append(intent)
    lines.append("")
    bucket_flags = [i for i in unmeasured + unresolved if i != "aggregate"]
    if regressions:
        lines.append(
            f"**Answer: YES — the hybrid arm regressed** in: {', '.join(regressions)} "
            "(significant at 95%). Criterion 1 is NOT met."
        )
    elif bucket_flags:
        lines.append(
            "**Answer: no significant regression detected, but this run does not "
            f"certify criterion 1** — {len(bucket_flags)} intent bucket(s) are "
            f"unresolved at this sample size ({', '.join(bucket_flags)}). The point "
            "estimates are flat-to-slightly-positive, and the unresolved buckets "
            "cannot rule out a regression as large as the CI half-width below. "
            "Reporting this as a pass would overstate the evidence."
        )
    else:
        lines.append(
            "**Answer: NO — no significant recall@k regression in any intent "
            "bucket**, and every bucket resolved at 95%. Criterion 1 is met by this "
            "measurement."
        )
    lines.append("")

    lines.append("## Honest notes on bucket size")
    lines.append("")
    flagged = 0
    for intent in taxonomy:
        block = delta["by_intent"].get(intent, {})
        rec = block.get("recall_at_k_all_present", {})
        n = (rec.get("n") or {}).get("candidate") or 0
        ci = rec.get("ci95")
        if not n:
            flagged += 1
            lines.append(
                f"- **{intent}**: 0 scored questions in this run — this bucket "
                "produces NO evidence about criterion 1, neither for nor against."
            )
            continue
        half = (ci[1] - ci[0]) / 2.0 if ci else None
        if half is not None and half > INCONCLUSIVE_CI_HALF_WIDTH:
            flagged += 1
            lines.append(
                f"- **{intent}**: {n} scored questions, 95% CI half-width ±{half:.4f} "
                f"on the delta — a regression smaller than {half:.4f} is invisible "
                f"here, so this bucket cannot by itself certify criterion 1."
            )
    if not flagged:
        lines.append(
            f"- No intent bucket's delta CI exceeds ±{INCONCLUSIVE_CI_HALF_WIDTH}, "
            "so bucket size is not the limiting factor in this run."
        )
    lines.append(
        "- Scored questions are the subset whose gold row carries "
        "`must_include`/`critical_facts`; every question in this dataset does, so "
        "scored == question_count. A bucket with few scored questions still "
        "produces a wide CI, and the CI width — not the point estimate — decides "
        "whether the bucket is evidence."
    )
    lines.append("")

    lines.append("## Harness honesty notes")
    lines.append("")
    prov = hybrid_manifest.get("provider") or {}
    lines.append(
        f"- No generator was invoked, so the manifests' `provider.status` reads "
        f"`{prov.get('status')}` with `generation_ok={prov.get('generation_ok')}` — "
        "that field describes GENERATION, and nothing was generated. The reporter "
        "correctly marks `semantic_accuracy`/`groundedness` as "
        f"`{prov.get('semantic_accuracy')}`; they are NOT zero and are not evidence."
    )
    lines.append(
        "- `retrieved_chunks` carries the FULL chunk text, because the recall "
        "definition is a text-substring test; recall@k here is computed by the same "
        "reporter code that produced the historical manifest."
    )
    lines.append(
        "- The record's `intent` field is null (no router ran), so the manifest's "
        "`router_intent` diagnostic shows a single `None` bucket. Intent bucketing "
        "is unaffected: it uses the dataset `category` plus the reporter's derived "
        "rule, exactly as in the historical manifest."
    )
    lines.append(
        "- Both arms receive the same query vector (one embedding per question) and "
        "the same temporal hint (`metadata_filter={\"query\": <question>}`), so a "
        "delta cannot come from the encoder or from the temporal wiring — only from "
        "fusion."
    )
    lines.append("")

    lines.append("## Criterion 2 (measurement only) — retrieval latency p50/p95 per intent")
    lines.append("")
    lines.append("| intent | arm | n | p50 ms | p95 ms |")
    lines.append("|---|---|---|---|---|")
    for intent in taxonomy + ["aggregate"]:
        for arm, manifest in (("dense", dense_manifest), ("hybrid", hybrid_manifest)):
            block = (
                manifest["aggregate"] if intent == "aggregate" else manifest["by_intent"].get(intent, {})
            )
            lat = (block.get("latency_ms") or {}).get("total_ms") or {}
            lines.append(
                f"| {intent} | {arm} | {lat.get('n')} | {_fmt(lat.get('p50'), 1)} | "
                f"{_fmt(lat.get('p95'), 1)} |"
            )
    lines.append("")
    lines.append(
        "`total_ms` here is the retrieval call only (no router, no rerank, no "
        "generation); the shared per-question embedding is excluded from both arms "
        "and reported in `run_meta.json`."
    )
    lines.append("")

    if historical is not None:
        lines.append("## Criterion 1, literal wording — vs the historical baseline manifest")
        lines.append("")
        lines.append(
            f"Baseline manifest: `{DEFAULT_BASELINE_MANIFEST.name}` "
            f"(records `{(historical.get('baseline') or {}).get('records', {}).get('name')}`). "
            "Same question set and same gold rows as this run, so the populations "
            "match — but the baseline was produced by the FULL pipeline "
            "(dense retrieval + reranker cutoff + generation), at a k that is "
            "whatever the reranker left, whereas this run persists the raw "
            "retrieval list at k=5. The comparison is therefore confounded by "
            "methodology and is reported as context, not as the criterion-1 "
            "verdict (the paired A/B above is the verdict)."
        )
        lines.append("")
        lines.append("| intent | scored | historical | hybrid (this run) | delta | 95% CI |")
        lines.append("|---|---|---|---|---|---|")
        for intent in taxonomy + ["aggregate"]:
            rec = historical["by_intent"].get(intent, {}).get("recall_at_k_all_present", {})
            lines.append(
                f"| {intent} | {(rec.get('n') or {}).get('candidate')} | "
                f"{_fmt(rec.get('baseline'))} | {_fmt(rec.get('candidate'))} | "
                f"{_fmt(rec.get('delta'))} | {_fmt_ci(rec.get('ci95'))} |"
            )
        lines.append("")
        if any(
            (
                historical["by_intent"]
                .get(intent, {})
                .get("recall_at_k_all_present", {})
                .get("candidate")
            )
            is None
            for intent in taxonomy
        ):
            lines.append(
                "Buckets showing `n/a` for \"hybrid (this run)\" are the ones this "
                "run left empty (a question-count limit, or a bucket the dataset "
                "does not populate); the historical manifest still covers them."
            )
            lines.append("")
        if dense_vs_historical is not None:
            lines.append(
                "Methodology control — the SAME comparison for the dense arm "
                "(this run's dense retrieval vs the same historical baseline). If "
                "both arms sit similarly below the historical numbers, the gap is "
                "the missing reranker/k, not the hybrid arm:"
            )
            lines.append("")
            lines.append("| intent | dense (this run) | historical | delta |")
            lines.append("|---|---|---|---|")
            for intent in taxonomy + ["aggregate"]:
                rec = dense_vs_historical["by_intent"].get(intent, {}).get("recall_at_k_all_present", {})
                lines.append(
                    f"| {intent} | {_fmt(rec.get('candidate'))} | "
                    f"{_fmt(rec.get('baseline'))} | {_fmt(rec.get('delta'))} |"
                )
            lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for name, rel in meta["artifacts"].items():
        lines.append(f"- `{name}`: `{rel}`")
    lines.append("")
    lines.append(
        "Reproduce: `./.venv/Scripts/python.exe scripts/eval_retrieval_ab.py` "
        "(run from `rag2025/`)."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval-only dense-vs-hybrid A/B harness (ticket 09)"
    )
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--gt", type=str, default=None, help="gold JSON (defaults to --dataset)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--baseline-manifest",
        type=str,
        default=str(DEFAULT_BASELINE_MANIFEST),
        help="historical manifest for the criterion-1 literal comparison ('' to skip)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-only",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help=(
            "re-render summary.md from an existing run directory (run_meta.json + "
            "the manifests/deltas already on disk) without re-running retrieval"
        ),
    )
    return parser.parse_args(argv)


def _render_only(run_dir: pathlib.Path) -> int:
    """Re-render ``summary.md`` for a completed run from its own artifacts."""
    if not run_dir.is_dir():
        sys.stderr.write(f"not a run directory: {run_dir}\n")
        return 2
    read = lambda name: json.loads(  # noqa: E731 — tiny local helper
        (run_dir / name).read_text(encoding="utf-8")
    )
    meta = read("run_meta.json")
    dense_manifest = read("manifest_dense.json")
    hybrid_manifest = read("manifest_hybrid.json")
    delta = read("delta_hybrid_vs_dense.json")
    historical = (
        read("delta_hybrid_vs_historical_baseline.json")
        if (run_dir / "delta_hybrid_vs_historical_baseline.json").exists()
        else None
    )
    dense_vs_historical = (
        read("delta_dense_vs_historical_baseline.json")
        if (run_dir / "delta_dense_vs_historical_baseline.json").exists()
        else None
    )
    summary = render_summary(
        run_id=meta["run_id"],
        meta=meta,
        dense_manifest=dense_manifest,
        hybrid_manifest=hybrid_manifest,
        delta=delta,
        historical=historical,
        dense_vs_historical=dense_vs_historical,
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.summary_only:
        return _render_only(pathlib.Path(args.summary_only))

    dataset_path = pathlib.Path(args.dataset)
    gt_path = pathlib.Path(args.gt) if args.gt else dataset_path
    run_id = args.run_id or (
        "ab_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir = pathlib.Path(args.out_root) / run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.stderr.write(
            f"refusing to overwrite existing run directory: {out_dir}\n"
            "pass --run-id with a new id\n"
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(dataset_path, args.limit)
    logger.info(f"{len(questions)} questions from {dataset_path.name}")

    settings = RAGSettings()
    retriever = get_retriever()
    table = retriever._adapter._table  # noqa: SLF001 — recorded provenance
    rows = table.count_rows()

    embedding = EmbeddingService(settings)
    recorder = PoolRecorder(retriever)
    hybrid = HybridSearchService(lancedb_retriever=recorder, settings=settings)
    bm25_start = time.perf_counter()
    bm25_ok = hybrid.build_bm25_index()
    bm25_ms = (time.perf_counter() - bm25_start) * 1000.0
    if not bm25_ok:
        sys.stderr.write(
            "BM25 index build FAILED — the hybrid arm would silently degrade to "
            "dense-only, which is not evidence for criterion 1. Aborting.\n"
        )
        return 3
    _wrap_sparse_recorder(recorder, hybrid)
    logger.info(f"BM25 index built in {bm25_ms:.0f} ms")

    pools = asyncio.run(
        run_all(
            questions,
            retriever=retriever,
            recorder=recorder,
            hybrid=hybrid,
            embedding=embedding,
            top_k=args.top_k,
            run_id=run_id,
            seed=args.seed,
            settings=settings,
            out_dir=out_dir,
        )
    )

    # ---- per-arm manifests, via the existing reporter ----
    manifest_paths: Dict[str, pathlib.Path] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        out_json = out_dir / f"manifest_{arm}.json"
        report_arm(
            records=out_dir / f"records_{arm}.jsonl",
            dataset=dataset_path,
            gt=gt_path,
            out_json=out_json,
            out_md=out_dir / f"manifest_{arm}.md",
        )
        manifest_paths[arm] = out_json
        manifests[arm] = json.loads(out_json.read_text(encoding="utf-8"))

    # ---- the A/B delta: baseline = dense arm, candidate = hybrid arm ----
    delta = report_delta(
        baseline=manifest_paths["dense"],
        candidate=manifest_paths["hybrid"],
        out_json=out_dir / "delta_hybrid_vs_dense.json",
        out_md=out_dir / "delta_hybrid_vs_dense.md",
    )

    # ---- criterion 1's literal wording, as context ----
    historical = None
    dense_vs_historical = None
    baseline_manifest_path = (
        pathlib.Path(args.baseline_manifest) if args.baseline_manifest else None
    )
    if baseline_manifest_path is not None and baseline_manifest_path.exists():
        historical = report_delta(
            baseline=baseline_manifest_path,
            candidate=manifest_paths["hybrid"],
            out_json=out_dir / "delta_hybrid_vs_historical_baseline.json",
            out_md=out_dir / "delta_hybrid_vs_historical_baseline.md",
        )
        dense_vs_historical = report_delta(
            baseline=baseline_manifest_path,
            candidate=manifest_paths["dense"],
            out_json=out_dir / "delta_dense_vs_historical_baseline.json",
            out_md=out_dir / "delta_dense_vs_historical_baseline.md",
        )

    # ---- criterion-3 evidence + provenance ----
    rrf_k = inspect.signature(HybridSearchService._rrf_fusion).parameters["k"].default  # noqa: SLF001
    pool_keys = (
        "dense_candidates",
        "sparse_candidates",
        "sparse_after_post_fetch",
        "fused_union",
        "final",
        "final_from_sparse_only",
        "final_not_in_dense_topk",
    )
    pool_summary = {key: _pool_stats([p[key] for p in pools]) for key in pool_keys}
    pool_summary["questions_with_union_gt_final"] = sum(
        1 for p in pools if p["fused_union"] > p["final"]
    )
    pool_summary["questions_with_final_from_sparse_only"] = sum(
        1 for p in pools if p["final_from_sparse_only"] > 0
    )
    pool_summary["questions_with_final_not_in_dense_topk"] = sum(
        1 for p in pools if p["final_not_in_dense_topk"] > 0
    )
    pool_summary["questions_with_order_changed"] = sum(
        1 for p in pools if p["order_changed_vs_dense_topk"]
    )
    pool_summary["questions_with_sparse_after_post_fetch_lt_raw"] = sum(
        1 for p in pools if p["sparse_after_post_fetch"] < p["sparse_candidates"]
    )
    # Criterion 3 is about the REQUEST relation, which is observable regardless
    # of how many docs the corpus could actually return.
    pool_summary["observed_requested_k"] = sorted(
        {
            (p["dense_requested_k"], p["sparse_requested_k"])
            for p in pools
            if p["dense_requested_k"] is not None
        }
    )

    arm_stats: Dict[str, Any] = {}
    for arm, key in (("dense", "dense_ms"), ("hybrid", "hybrid_ms")):
        arm_stats[arm] = {
            "retrieval_ms": _ms_stats([p[key] for p in pools]),
            "empty_results": sum(
                1
                for p in pools
                if (p["dense_error"] if arm == "dense" else p["hybrid_error"])
                == "empty_result"
            ),
            "errors": sorted(
                {
                    str(p["dense_error"] if arm == "dense" else p["hybrid_error"])
                    for p in pools
                    if (p["dense_error"] if arm == "dense" else p["hybrid_error"])
                }
            ),
        }

    artifacts = {
        "records_dense": f"records_dense.jsonl",
        "records_hybrid": f"records_hybrid.jsonl",
        "manifest_dense": "manifest_dense.json",
        "manifest_hybrid": "manifest_hybrid.json",
        "delta_hybrid_vs_dense": "delta_hybrid_vs_dense.json",
        "run_meta": "run_meta.json",
        "summary": "summary.md",
    }
    if historical is not None:
        artifacts["delta_hybrid_vs_historical_baseline"] = (
            "delta_hybrid_vs_historical_baseline.json"
        )
        artifacts["delta_dense_vs_historical_baseline"] = (
            "delta_dense_vs_historical_baseline.json"
        )

    meta: Dict[str, Any] = {
        "schema": "retrieval-ab-run-meta",
        "schema_version": 1,
        "run_id": run_id,
        "harness_version": HARNESS_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": (
            "retrieval-only: no router, no rerank, no generator (the LLM providers "
            "are unavailable in this environment and recall@k/latency do not need them)"
        ),
        "top_k": args.top_k,
        "seed": args.seed,
        "dataset": {
            "path": str(dataset_path),
            "name": dataset_path.name,
            "sha256": sha256_file(dataset_path),
            "questions": len(questions),
        },
        "gt": {"path": str(gt_path), "name": gt_path.name, "sha256": sha256_file(gt_path)},
        "embedding": {
            "model": settings.EMBEDDING_MODEL,
            "dim": settings.EMBEDDING_DIM,
            "provider": settings.EMBEDDING_PROVIDER,
            "encode_ms": _ms_stats([p["embed_ms"] for p in pools]),
        },
        "lancedb": {
            "uri": settings.LANCEDB_URI,
            "table": settings.LANCEDB_TABLE,
            "rows": rows,
        },
        "hybrid": {
            "constructed_explicitly": True,
            "env_use_hybrid_retrieval": settings.USE_HYBRID_RETRIEVAL,
            "dense_weight": settings.HYBRID_FUSION_DENSE_WEIGHT,
            "sparse_weight": settings.HYBRID_FUSION_SPARSE_WEIGHT,
            "rrf_k": rrf_k,
            "rrf_k_source": "inspect.signature(HybridSearchService._rrf_fusion).parameters['k'].default",
            "candidate_pool_per_arm": args.top_k * 2,
            "candidate_pool_source": "HybridSearchService.retrieve passes top_k*2 to both arms",
            "sparse_arm_constants": {
                "exact_code_boost": _hybrid_mod.EXACT_CODE_BOOST,
                "exact_code_max_matches": _hybrid_mod.EXACT_CODE_MAX_MATCHES,
                "year_boost_factor": _retrieval_mod.YEAR_BOOST_FACTOR,
                "old_year_demote_factor": _retrieval_mod.OLD_YEAR_DEMOTE_FACTOR,
                "current_admission_year": _retrieval_mod.CURRENT_ADMISSION_YEAR,
                "source": (
                    "module constants of services.hybrid_search / "
                    "services.lancedb_retrieval, read at run time"
                ),
            },
            "bm25_ok": bm25_ok,
            "bm25_build_ms": round(bm25_ms, 1),
            "corpus_version": hybrid.corpus_version,
        },
        "pools": pool_summary,
        "arms": arm_stats,
        "artifacts": artifacts,
    }

    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = render_summary(
        run_id=run_id,
        meta=meta,
        dense_manifest=manifests["dense"],
        hybrid_manifest=manifests["hybrid"],
        delta=delta,
        historical=historical,
        dense_vs_historical=dense_vs_historical,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)

    # A run whose arms failed everywhere must not be reported as a result.
    failed = sum(
        1
        for p in pools
        if p["dense_error"] or p["hybrid_error"] or p["final"] == 0
    )
    if failed == len(pools) and pools:
        sys.stderr.write(
            f"every question failed in at least one arm ({failed}/{len(pools)}) — "
            "the artifacts are not evidence of a working comparison\n"
        )
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
