"""Instrumented 86Q eval runner (Slice S2 — full evidence + 3 latency seams).

Mirrors the legacy `eval_thi_sinh_thuc.py` flow:
  router.route() → encode+retrieve loop → dedupe → boost → pipeline.query() →
  generator.generate_answer(), with the auto-answer short-circuit.

Differs from legacy in two material ways:
  1. Persists FULL evidence per question (EvalRecord) — chunk TEXT, full
     answer, 3 latency seams + total, model versions, expected_behavior.
     Legacy persisted only chunk_ids[:5] and a single latency_s.
  2. Times THREE honest latency seams (`route`, `retrieval_loop`, `query`)
     + total. The timing wraps the calls in this runner — NEVER edits the
     pipeline. Optional fine-grained attribution is applied via a
     defensive runtime monkeypatch (try/except — never breaks the run).

Read-only on the pipeline. The legacy `eval_thi_sinh_thuc.py` is untouched.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Bootstrap path so we can import config + src.services + the eval_record
# sibling. The legacy runner uses the same sys.path trick.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent  # rag2025/
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env so NS24_* model overrides (gen/hyde/ner) are picked up —
# the legacy runner does this too (eval_thi_sinh_thuc.py:17-18). Without it the
# runner falls back to the hardcoded defaults in llm_generator.py.
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT / ".env")
except Exception:  # pragma: no cover — dotenv optional
    pass

from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
import eval_record as _er  # noqa: E402  (sibling script)
from eval_intent import intent_counts, select_ci_subset  # noqa: E402  (sibling script)
from src.services.aggregation_booster import boost_with_aggregation  # noqa: E402


HARNESS_VERSION = "s2-v1"

# Module-level flags — the harness/report reads these to print the
# "cannot isolate HyDE / gen" caveat. Defaults to False; flipped to True the
# first time `_attribution_setup` successfully wraps a target method.
HYDE_ISOLATABLE: bool = False
GEN_ISOLATABLE: bool = False


# =============================================================================
# Best-effort attribution — runtime monkeypatch, never breaks the run.
# =============================================================================


def _install_timing(
    obj: Any, attr: str, accumulator: Dict[str, float], key: str
) -> Optional[Callable]:
    """Monkeypatch `obj.<attr>` to accumulate elapsed ms into `accumulator[key]`.

    Returns the ORIGINAL callable for restore, or None if not installable.
    Defensive: any exception returns None.
    """
    try:
        original = getattr(obj, attr, None)
        if not callable(original):
            return None
    except Exception:
        return None

    if _is_async_callable(original):
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await original(*args, **kwargs)
            finally:
                accumulator[key] = accumulator.get(key, 0.0) + (
                    time.perf_counter() - t0
                ) * 1000.0
    else:
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                accumulator[key] = accumulator.get(key, 0.0) + (
                    time.perf_counter() - t0
                ) * 1000.0

    try:
        setattr(obj, attr, wrapper)
    except Exception:
        return None
    return original


def _is_async_callable(fn: Callable) -> bool:
    try:
        import asyncio as _aio
        return _aio.iscoroutinefunction(fn)
    except Exception:
        return False


def _attribution_setup(
    pipeline: Any, generator: Any
) -> Tuple[List[Tuple[Any, str, Any]], Dict[str, float]]:
    """Install timing monkeypatches; return (cleanups, accumulator)."""
    cleanups: List[Tuple[Any, str, Any]] = []
    accumulator: Dict[str, float] = {}

    router = getattr(pipeline, "_router", None)
    if router is not None:
        for attr, key in (
            ("_generate_hyde_doc", "hyde_ms"),
            ("_step_back", "router_inner_ms"),
            ("_classify", "router_inner_ms"),
        ):
            original = _install_timing(router, attr, accumulator, key)
            if original is not None:
                cleanups.append((router, attr, original))

    if generator is not None:
        original = _install_timing(generator, "generate_answer", accumulator, "gen_ms")
        if original is not None:
            cleanups.append((generator, "generate_answer", original))

    return cleanups, accumulator


def _attribution_restore(cleanups: List[Tuple[Any, str, Any]]) -> None:
    for obj, attr, original in cleanups:
        try:
            setattr(obj, attr, original)
        except Exception:
            pass


def _maybe_attribution(accumulator: Dict[str, float]) -> Optional[Any]:
    """Return a LatencyAttribution iff at least one field was populated; else None."""
    if not accumulator:
        return None
    return _er.LatencyAttribution(
        hyde_ms=accumulator.get("hyde_ms"),
        router_inner_ms=accumulator.get("router_inner_ms"),
        gen_ms=accumulator.get("gen_ms"),
        rerank_ms=accumulator.get("rerank_ms"),
    )


# =============================================================================
# Record builder
# =============================================================================


def _build_record(
    *,
    qid: str,
    query: str,
    answer: str,
    route: str,
    intent: Optional[str],
    complexity: Optional[int],
    top_docs: List[Any],
    run_id: str,
    seed: int,
    timestamp: str,
    settings: Any,
    route_ms: float,
    retrieval_loop_ms: float,
    query_ms: float,
    total_ms: float,
    latency_attribution: Optional[Any],
    gt_lookup: Optional[Dict[str, str]],
    llm_token_calls: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Materialize a fully-populated EvalRecord from extracted data."""
    retrieved = []
    for doc in top_docs or []:
        md = getattr(doc, "metadata", {}) or {}
        retrieved.append(
            _er.RetrievedChunk(
                chunk_id=getattr(doc, "chunk_id", "") or "",
                source=getattr(doc, "source", "") or "",
                score=float(getattr(doc, "score", 0.0) or 0.0),
                text=getattr(doc, "text", "") or "",
                data_year=md.get("data_year"),
                is_superseded=bool(md.get("is_superseded", False)),
            )
        )

    mv = _er.ModelVersion(
        gen=settings.NS24_GEN_MODEL,
        ner=settings.NS24_NER_MODEL,
        hyde=settings.NS24_HYDE_MODEL,
        embedding=settings.EMBEDDING_MODEL,
    )
    expected = None
    if gt_lookup:
        expected = gt_lookup.get(qid)

    # T12: convert token call data to LLMTokens models + totals.
    calls: List[Any] = []
    total_prompt = 0
    total_completion = 0
    for c in (llm_token_calls or []):
        p = int(c.get("prompt_tokens", 0) or 0)
        comp = int(c.get("completion_tokens", 0) or 0)
        calls.append(_er.LLMTokens(
            prompt_tokens=p,
            completion_tokens=comp,
            provider=str(c.get("provider", "")),
            model=str(c.get("model", "")),
        ))
        total_prompt += p
        total_completion += comp
    llm_tokens_total = _er.LLMTokensTotal(
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
        call_count=len(calls),
    ) if calls else None

    return _er.EvalRecord(
        run_id=run_id,
        seed=seed,
        harness_version=HARNESS_VERSION,
        timestamp=timestamp,
        model_version=mv,
        id=qid,
        question=query,
        intent=intent,
        complexity=complexity,
        route=route,
        expected_behavior=expected,
        retrieved_chunks=retrieved,
        answer=answer,
        latency_ms=_er.LatencyMs(
            route_ms=route_ms,
            retrieval_loop_ms=retrieval_loop_ms,
            query_ms=query_ms,
            total_ms=total_ms,
        ),
        latency_attribution=latency_attribution,
        llm_tokens=llm_tokens_total,
        llm_token_calls=calls,
    )


# =============================================================================
# Core runner
# =============================================================================
def _snapshot_all_token_ledgers(pipeline: Any, generator: Any) -> List[Dict[str, Any]]:
    """T12: collect token-usage dicts from every LLM client we can reach.

    Defensive: any client that lacks ``token_ledger`` (e.g. test fakes,
    older versions) is skipped silently.
    """
    calls: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    candidates: List[Any] = []
    for obj in (pipeline, generator):
        if obj is None:
            continue
        # Generator LLM clients (primary + fallback + hyde)
        for attr in ("unified_client", "unified_fallback_client",
                    "_hyde_client"):
            c = getattr(obj, attr, None)
            if c is not None and id(c) not in seen_ids:
                seen_ids.add(id(c))
                candidates.append(c)
        # Pipeline: router LLM client
        router = getattr(obj, "_router", None)
        if router is not None:
            c = getattr(router, "_llm", None)
            if c is not None and id(c) not in seen_ids:
                seen_ids.add(id(c))
                candidates.append(c)
        # Pipeline: NER service clients (primary + fallback)
        ner = getattr(obj, "_ner", None)
        if ner is not None:
            for attr in ("_llm", "_fallback_llm"):
                c = getattr(ner, attr, None)
                if c is not None and id(c) not in seen_ids:
                    seen_ids.add(id(c))
                    candidates.append(c)
        # Pipeline: top-level _llm (router_llm)
        c = getattr(obj, "_llm", None)
        if c is not None and id(c) not in seen_ids:
            seen_ids.add(id(c))
            candidates.append(c)
    for client in candidates:
        ledger = getattr(client, "token_ledger", None)
        if ledger is None:
            continue
        for call in ledger.snapshot():
            calls.append({
                "provider": call.provider,
                "model": call.model,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
            })
    return calls


def _drain_token_ledgers(pipeline: Any, generator: Any) -> List[Dict[str, Any]]:
    """T12: collect AND reset all token ledgers, returning this question's calls.

    Defensive: clients without ``token_ledger`` (test fakes, older versions)
    are skipped; ``reset()`` failures are swallowed so a broken ledger never
    crashes the run.
    """
    calls = _snapshot_all_token_ledgers(pipeline, generator)
    # Reset ledgers so the next question starts fresh
    for obj in (pipeline, generator):
        if obj is None:
            continue
        for attr in ("unified_client", "unified_fallback_client",
                     "_hyde_client"):
            c = getattr(obj, attr, None)
            ledger = getattr(c, "token_ledger", None)
            if ledger is not None:
                try:
                    ledger.reset()
                except Exception:
                    pass
        router = getattr(obj, "_router", None)
        if router is not None:
            c = getattr(router, "_llm", None)
            ledger = getattr(c, "token_ledger", None)
            if ledger is not None:
                try:
                    ledger.reset()
                except Exception:
                    pass
        ner = getattr(obj, "_ner", None)
        if ner is not None:
            for attr in ("_llm", "_fallback_llm"):
                c = getattr(ner, attr, None)
                ledger = getattr(c, "token_ledger", None)
                if ledger is not None:
                    try:
                        ledger.reset()
                    except Exception:
                        pass
        c = getattr(obj, "_llm", None)
        ledger = getattr(c, "token_ledger", None)
        if ledger is not None:
            try:
                ledger.reset()
            except Exception:
                pass
    return calls



async def run_one_instrumented(
    q: Dict,
    *,
    pipeline: Any,
    retriever: Any,
    embedding: Any,
    generator: Any,
    run_id: str,
    seed: int,
    settings: Any,
    gt_lookup: Optional[Dict[str, str]] = None,
) -> Any:
    """Run one question, persist full evidence to an EvalRecord.

    Dependency-injected — the unit tests pass fakes; main() passes the real
    pipeline/retriever/embedding/generator built from settings.
    """
    query = q["question"]
    qid = q["id"]
    timestamp = datetime.now().isoformat()

    attribution: Dict[str, float] = {}
    cleanups: List[Tuple[Any, str, Any]] = []

    try:
        # ---- Best-effort attribution setup (defensive — never breaks run) ----
        try:
            cleanups, attribution = _attribution_setup(pipeline, generator)
        except Exception as e:  # pragma: no cover — truly defensive
            logger.warning(f"Attribution setup failed for {qid}: {e}")
            cleanups, attribution = [], {}

        # Flip module-level flags so the report can print the caveat.
        global HYDE_ISOLATABLE, GEN_ISOLATABLE
        try:
            HYDE_ISOLATABLE = HYDE_ISOLATABLE or any(
                c[1] == "_generate_hyde_doc" for c in cleanups
            )
            GEN_ISOLATABLE = GEN_ISOLATABLE or any(
                c[0] is generator and c[1] == "generate_answer" for c in cleanups
            )
        except Exception:
            pass

        total_start = time.perf_counter()

        # T12: drain token ledgers at the START so any calls made during
        # this question are isolated. The drain resets ledgers; we call it
        # again at the end to capture this question's calls.
        _drain_token_ledgers(pipeline, generator)

        # ---- Seam 1: route (includes HyDE; fused inside router.route()) ----
        route_start = time.perf_counter()
        router_result: Any = None
        try:
            router_result = await pipeline._router.route(query)
        except Exception as e:
            logger.warning(f"Router failed for {qid}: {e}")
        route_ms = (time.perf_counter() - route_start) * 1000.0

        # ---- Auto-answer short-circuit (skip retrieval + gen) ----
        if (
            router_result is not None
            and getattr(router_result, "skip_retrieval", False)
            and getattr(router_result, "auto_answer", None)
        ):
            total_ms = (time.perf_counter() - total_start) * 1000.0
            return _build_record(
                qid=qid,
                query=query,
                answer=router_result.auto_answer,
                route="hyde_auto_answer",
                intent=getattr(router_result, "intent", None),
                complexity=getattr(router_result, "complexity", None),
                top_docs=[],
                run_id=run_id,
                seed=seed,
                timestamp=timestamp,
                settings=settings,
                route_ms=route_ms,
                retrieval_loop_ms=0.0,
                query_ms=0.0,
                total_ms=total_ms,
                latency_attribution=_maybe_attribution(attribution),
                gt_lookup=gt_lookup,
                llm_token_calls=_drain_token_ledgers(pipeline, generator),
            )

        # ---- Seam 2: retrieval loop (embed + retrieve + dedupe + boost) ----
        retrieval_start = time.perf_counter()
        baseline_docs: List = []
        try:
            all_q = getattr(router_result, "all_queries_for_embedding", None) or [query]
            results: List = []
            for variant in all_q[:4]:
                qv = embedding.encode_single(variant).tolist()
                # Pass the query TEXT so the retriever's temporal wiring
                # (S14.3) can classify intent + apply the year hard-filter.
                # Without query=, classification silently no-ops (the whole
                # temporal fix would be invisible in this eval path).
                rr = retriever.retrieve(qv, top_k=10, query=variant)
                if getattr(rr, "is_success", False):
                    results.extend(rr.documents)
            seen = set()
            unique: List = []
            for d in results:
                cid = getattr(d, "chunk_id", None) or ""
                if cid not in seen:
                    seen.add(cid)
                    unique.append(d)
            baseline_docs = unique[:15]
            try:
                baseline_docs = boost_with_aggregation(
                    query, baseline_docs, retriever, top_k=15, max_inject=2
                )
            except Exception as e:
                logger.warning(f"Boost failed for {qid}: {e}")
        except Exception as e:
            logger.warning(f"Retrieval failed for {qid}: {e}")
            baseline_docs = []
        retrieval_loop_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # ---- Seam 3: pipeline.query + gen (rerank + gen, fused) ----
        query_start = time.perf_counter()
        route: str = "error"
        top_docs: List = []
        intent: Optional[str] = None
        complexity: Optional[int] = None
        try:
            rag_result = await pipeline.query(
                user_query=query,
                baseline_docs=baseline_docs,
                top_k=5,
                router_result=router_result,
            )
            route = rag_result.route
            top_docs = rag_result.documents
            # S15.5 / ADR-E: `pipeline.query()` already applied the relative
            # rerank-score cutoff (`apply_rerank_cutoff` with env
            # RERANK_KEEP_RATIO / RERANK_MIN_KEEP, exempting both
            # booster_injected and graph_injected docs). `top_docs` here is
            # already the cut list — DO NOT re-cut, or precision recovery
            # gets compounded.
            inner = getattr(rag_result, "router_result", None)
            if inner is not None:
                intent = getattr(inner, "intent", None)
                complexity = getattr(inner, "complexity", None)
        except Exception as e:
            logger.warning(f"Pipeline.query failed for {qid}: {e}")
            route = "error"
            top_docs = baseline_docs[:5]

        # Build chunks_for_gen (mirrors legacy shape).
        chunks_for_gen = []
        for doc in top_docs:
            md = getattr(doc, "metadata", {}) or {}
            chunks_for_gen.append(
                {
                    "text": getattr(doc, "text", ""),
                    "summary": md.get("summary", ""),
                    "metadata": md,
                }
            )

        is_enum = any(
            k in query.lower() for k in ["bao nhiêu", "danh sách", "liệt kê", "tất cả"]
        )

        answer: str = ""
        try:
            gr = await generator.generate_answer(
                query=query,
                chunks=chunks_for_gen,
                confidence=0.5,
                is_program_list_query=is_enum,
            )
            answer = (gr or {}).get("answer", "") or ""
        except Exception as e:
            logger.warning(f"Generator failed for {qid}: {e}")
            answer = f"ERROR: {e}"
        query_ms = (time.perf_counter() - query_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return _build_record(
            qid=qid,
            query=query,
            answer=answer,
            route=route,
            intent=intent,
            complexity=complexity,
            top_docs=top_docs,
            run_id=run_id,
            seed=seed,
            timestamp=timestamp,
            settings=settings,
            route_ms=route_ms,
            retrieval_loop_ms=retrieval_loop_ms,
            query_ms=query_ms,
            total_ms=total_ms,
            latency_attribution=_maybe_attribution(attribution),
            gt_lookup=gt_lookup,
            llm_token_calls=_drain_token_ledgers(pipeline, generator),
        )
    finally:
        _attribution_restore(cleanups)


# =============================================================================
# Provider-error capture (ticket 08) — verbatim evidence for the report.
#
# LLMGenerator swallows the provider exception (llm_generator.py:785-790 logs it
# and returns a canned fallback), so the raw error never reaches the record. The
# only honest place to get it verbatim is the pipeline's own log. This sink
# captures the provider-failure lines during the run and flushes them to a
# sidecar next to the JSONL, which `eval_per_intent_report.py` reads to mark
# semantic_accuracy / groundedness as blocked_by_provider.
# =============================================================================

_PROVIDER_ERROR_PATTERNS = re.compile(
    r"(LLM generation failed:|All LLM providers failed|LLM \[[^\]]+\] failed:)"
)


class _ProviderErrorCapture:
    """Collect verbatim provider-failure log messages (loguru sink)."""

    def __init__(self) -> None:
        self._handler_id: Optional[int] = None
        self._order: List[str] = []
        self._counts: Dict[str, int] = {}

    def __enter__(self) -> "_ProviderErrorCapture":
        try:
            self._handler_id = logger.add(self._sink, level="WARNING")
        except Exception:  # pragma: no cover — defensive, never breaks a run
            self._handler_id = None
        return self

    def _sink(self, message: Any) -> None:
        try:
            text = str(message.record["message"])
        except Exception:  # pragma: no cover
            return
        if _PROVIDER_ERROR_PATTERNS.search(text):
            if text not in self._counts:
                self._order.append(text)
            self._counts[text] = self._counts.get(text, 0) + 1

    def __exit__(self, *exc: Any) -> None:
        if self._handler_id is not None:
            try:
                logger.remove(self._handler_id)
            except Exception:  # pragma: no cover
                pass
            self._handler_id = None

    def entries(self) -> List[Dict[str, Any]]:
        """Deduped, order-preserving verbatim errors with occurrence counts."""
        return [
            {"text": text, "count": self._counts[text], "source": "run_meta_log"}
            for text in self._order
        ]


def run_meta_path(out_path: Any) -> Path:
    """Sidecar path for a records JSONL: ``<stem>.run-meta.json``."""
    return Path(out_path).with_suffix(".run-meta.json")


def _write_run_meta(
    out_path: Any, entries: List[Dict[str, Any]], completed: int
) -> Path:
    meta_path = run_meta_path(out_path)
    meta_path.write_text(
        json.dumps(
            {
                "schema": "eval-run-meta",
                "schema_version": 1,
                "records_file": Path(out_path).name,
                "completed": completed,
                "provider_errors": entries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return meta_path


# Bulk runner + checkpointing
# =============================================================================


async def run_all(
    questions: List[Dict],
    *,
    pipeline: Any,
    retriever: Any,
    embedding: Any,
    generator: Any,
    run_id: str,
    seed: int,
    settings: Any,
    out_path: str,
    gt_lookup: Optional[Dict[str, str]] = None,
) -> int:
    """Run all questions; append one EvalRecord JSONL line per question.

    Mirrors legacy's checkpoint-on-each-question pattern: every successful
    question is flushed to disk before moving on, so a crash mid-run loses
    at most one record. Also writes the provider-error sidecar (ticket 08):
    ``<out>.run-meta.json``.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    capture = _ProviderErrorCapture()
    with capture, out_file.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            logger.info(f"{i+1}/{len(questions)}: {q['id']}")
            try:
                rec = await run_one_instrumented(
                    q,
                    pipeline=pipeline,
                    retriever=retriever,
                    embedding=embedding,
                    generator=generator,
                    run_id=run_id,
                    seed=seed,
                    settings=settings,
                    gt_lookup=gt_lookup,
                )
                f.write(rec.to_jsonl_line() + "\n")
                f.flush()
                completed += 1
            except Exception as e:
                logger.error(f"  -> ERROR: {q['id']}: {e}")
    _write_run_meta(out_path, capture.entries(), completed)
    return completed


# =============================================================================
# main() — for the REAL run; do NOT invoke from unit tests.
# =============================================================================


def main() -> None:
    """Construct the real pipeline and run all 86Q questions.

    Mirrors legacy `eval_thi_sinh_thuc.py:154-198` for pipeline construction.
    NEVER called from unit tests — they pass fakes through `run_one_instrumented`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Instrumented 86Q RAG eval runner (S2)"
    )
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke mode: only run the first N questions (e.g. --limit 3).",
    )
    parser.add_argument(
        "--ci-subset",
        type=int,
        default=None,
        help=(
            "CI mode: run only a deterministic, intent-stratified subset of N "
            "questions (spans simple/multihop/comparative). Same input -> same "
            "subset. Applied after --limit."
        ),
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(_PROJECT / "data" / "eval" / "husc_thi_sinh_thuc.json"),
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=str(_PROJECT / "data" / "eval" / "husc_thi_sinh_thuc_gt.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output JSONL path. Default: "
            "rag2025/results/eval_harness/86q_records_<run_id>.jsonl"
        ),
    )
    args = parser.parse_args()

    if args.out is None:
        out_dir = _PROJECT / "results" / "eval_harness"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / f"86q_records_{args.run_id}.jsonl")

    with open(args.questions, "r", encoding="utf-8") as fh:
        questions = json.load(fh)
    if args.limit is not None:
        questions = questions[: args.limit]
        logger.info(f"Smoke mode: limited to first {args.limit} questions")
    if args.ci_subset is not None:
        questions = select_ci_subset(questions, args.ci_subset)
        logger.info(
            f"CI subset: {len(questions)} questions, intent counts="
            f"{intent_counts(questions)}"
        )
    logger.info(f"Loaded {len(questions)} questions")

    gt_lookup: Optional[Dict[str, str]] = None
    gt_path = Path(args.gt)
    if gt_path.exists():
        with gt_path.open("r", encoding="utf-8") as fh:
            gt_list = json.load(fh)
        gt_lookup = {
            item["id"]: item.get("expected_behavior")
            for item in gt_list
            if "id" in item
        }
        logger.info(f"Loaded GT lookup for {len(gt_lookup)} questions")
    else:
        logger.info(
            f"No GT file at {gt_path}; proceeding without expected_behavior"
        )

    # Late imports — never at module top (keeps the runner importable for tests
    # without instantiating the real pipeline / LLM clients).
    from services.graphrag_retriever import UnifiedRAGPipeline
    from services.lancedb_retrieval import LanceDBRetriever
    from services.embedding import EmbeddingService
    from services.llm_generator import LLMGenerator

    settings = RAGSettings()
    pipeline = UnifiedRAGPipeline.from_disk()
    retriever = LanceDBRetriever.from_env()
    embedding = EmbeddingService(settings)
    generator = LLMGenerator()

    asyncio.run(
        run_all(
            questions,
            pipeline=pipeline,
            retriever=retriever,
            embedding=embedding,
            generator=generator,
            run_id=args.run_id,
            seed=args.seed,
            settings=settings,
            out_path=args.out,
            gt_lookup=gt_lookup,
        )
    )


if __name__ == "__main__":
    main()
