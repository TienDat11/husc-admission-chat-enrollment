"""Per-intent eval manifest + two-manifest delta report (ticket 08).

Extends the existing instrumented harness (``eval_run_instrumented.py``, whose
JSONL EvalRecords are the input) with the reporting the ticket asks for:

  * recall@k, semantic accuracy and latency BROKEN OUT by intent
    (simple / multihop / comparative), not just aggregate;
  * latency p50 AND p95 per intent;
  * a two-manifest compare mode that prints the delta WITH confidence intervals;
  * machine-readable output that is reproducible (same input -> same bytes);
  * a small CI subset (``--ci-subset N``), intent-stratified.

It is a REPORTER: it never runs retrieval/generation and never judges answers.
Pure arithmetic + string rendering over records already on disk.

Intent source / classifier
--------------------------
See ``eval_intent.py`` (the single source of truth). In short: the dataset's own
``category`` label when it names a taxonomy class, else a deterministic text
rule built on the router's own comparison/enumeration regexes.

Dead-provider honesty
---------------------
``semantic_accuracy`` and ``groundedness`` need a generated answer + a judge.
When generation was blocked by the provider, both are reported as

    {"value": null, "status": "blocked_by_provider", "why": ..., "provider_errors": [...]}

— never 0, never a bare null. Retrieval metrics (recall@k, latency) are still
produced. ``groundedness`` is measured from ``judge_scores.groundedness`` (or
``faithfulness``) and ``semantic_accuracy`` from ``judge_scores.semantic_accuracy``
when a judge has run; otherwise blocked.

Confidence intervals
--------------------
  * rates        — Wilson score interval (two-sided, z=1.96);
  * rate deltas  — Newcombe hybrid score interval (closed form, no sampling);
  * percentiles  — seeded bootstrap percentile interval (deterministic: the same
                   input always produces the same interval);
  * p50/p95 deltas — independent two-sample seeded bootstrap of the difference.
Bootstrap parameters are pinned (``BOOTSTRAP_SEED`` / ``BOOTSTRAP_RESAMPLES``) so
re-runs are byte-identical.

CLI
---
    # normal mode
    <python> scripts/eval_per_intent_report.py \
        --records results/eval_harness/86q_records_bf01_86q_20260912T152147Z.jsonl \
        --dataset data/eval/husc_86q_test.json \
        --gt      data/eval/husc_86q_test.json \
        --out     results/eval_harness/metrics_report/per_intent/bf01_86q.json

    # compare mode (baseline manifest vs candidate manifest)
    <python> scripts/eval_per_intent_report.py \
        --compare <baseline_manifest.json> <candidate_manifest.json> \
        --out     results/eval_harness/metrics_report/per_intent/delta.json

Reproducibility: the written manifest carries NO wall-clock field by default, so
the same inputs produce byte-identical files. Pass ``--with-timestamp`` to embed
``provenance.generated_at`` (that field is excluded from ``content_hash``).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import random
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent  # rag2025/
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_intent import (  # noqa: E402
    INTENT_RULE_TEXT,
    INTENT_TAXONOMY,
    classify_intent,
    select_ci_subset,
)
from eval_report_metrics import percentile  # noqa: E402  (existing S7 helper, p in [0,100])

MANIFEST_SCHEMA = "per-intent-eval-manifest"
MANIFEST_SCHEMA_VERSION = 1
DIFF_SCHEMA = "per-intent-eval-manifest-diff"

Z_95 = 1.96
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000

# The canned LLMGenerator fallback (llm_generator.py:787-790). Presence of this
# marker — or the runner's "ERROR: <exc>" prefix — means generation was blocked.
GENERATION_FALLBACK_MARKERS: Tuple[str, ...] = (
    "dịch vụ tạo câu trả lời đang tạm thời không khả dụng",
)

RECALL_DEFINITION = (
    "fraction of scored questions whose retrieved chunk TEXT contains EVERY "
    "`must_include` / `critical_facts[].value` string (diacritic/whitespace-"
    "insensitive substring); k = the retrieved_chunks list persisted per record"
)

CI_METHODS = {
    "rate": "wilson_score_interval_z1.96",
    "rate_delta": "newcombe_hybrid_score_interval",
    "percentile": f"bootstrap_percentile_{BOOTSTRAP_RESAMPLES}_seed{BOOTSTRAP_SEED}",
    "percentile_delta": (
        f"independent_bootstrap_difference_{BOOTSTRAP_RESAMPLES}_seed{BOOTSTRAP_SEED}"
    ),
}


# =============================================================================
# Text + statistics helpers
# =============================================================================


def norm(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics + collapse whitespace."""
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round for stable serialization (None-safe)."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def wilson_interval(
    successes: int, n: int, z: float = Z_95
) -> Optional[List[float]]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [_r(max(0.0, centre - margin)), _r(min(1.0, centre + margin))]


def newcombe_diff_interval(
    k1: int, n1: int, k2: int, n2: int, z: float = Z_95
) -> Optional[List[float]]:
    """Newcombe hybrid score interval for ``p1 - p2`` (closed form, deterministic).

    SIGN CONVENTION: the interval is for ``p1 - p2`` (``p1 = k1/n1``), so callers
    that report ``candidate - baseline`` must pass the candidate as ``(k1, n1)``.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    i1 = wilson_interval(k1, n1, z)
    i2 = wilson_interval(k2, n2, z)
    if i1 is None or i2 is None:
        return None
    p1, p2 = k1 / n1, k2 / n2
    delta = p1 - p2
    lower = delta - math.sqrt((p1 - i1[0]) ** 2 + (i2[1] - p2) ** 2)
    upper = delta + math.sqrt((i1[1] - p1) ** 2 + (p2 - i2[0]) ** 2)
    return [_r(max(-1.0, lower)), _r(min(1.0, upper))]


def _bootstrap_index(rng: random.Random, n: int) -> int:
    """Index drawn from the pinned Mersenne-Twister stream.

    Uses only ``Random.random()`` (stable across CPython versions) so the
    bootstrap is reproducible byte-for-byte.
    """
    return int(rng.random() * n)


def bootstrap_percentile_interval(
    values: Sequence[float],
    pct: float,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Optional[List[float]]:
    """Seeded bootstrap percentile interval for ``percentile(values, pct)``."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats: List[float] = []
    for _ in range(resamples):
        sample = sorted(values[_bootstrap_index(rng, n)] for _ in range(n))
        stats.append(percentile(sample, pct))
    stats.sort()
    lo = stats[int(0.025 * resamples)]
    hi = stats[min(resamples - 1, int(0.975 * resamples))]
    return [_r(lo, 2), _r(hi, 2)]


def bootstrap_percentile_diff_interval(
    base_values: Sequence[float],
    cand_values: Sequence[float],
    pct: float,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Optional[List[float]]:
    """Seeded independent two-sample bootstrap interval for a percentile delta."""
    if len(base_values) < 2 or len(cand_values) < 2:
        return None
    rng = random.Random(seed)
    nb, nc = len(base_values), len(cand_values)
    stats: List[float] = []
    for _ in range(resamples):
        sb = sorted(base_values[_bootstrap_index(rng, nb)] for _ in range(nb))
        sc = sorted(cand_values[_bootstrap_index(rng, nc)] for _ in range(nc))
        stats.append(percentile(sc, pct) - percentile(sb, pct))
    stats.sort()
    lo = stats[int(0.025 * resamples)]
    hi = stats[min(resamples - 1, int(0.975 * resamples))]
    return [_r(lo, 2), _r(hi, 2)]


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Optional[List[float]]:
    """Seeded bootstrap interval for the mean (used for judge-score means)."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats: List[float] = []
    for _ in range(resamples):
        stats.append(sum(values[_bootstrap_index(rng, n)] for _ in range(n)) / n)
    stats.sort()
    return [
        _r(stats[int(0.025 * resamples)], 4),
        _r(stats[min(resamples - 1, int(0.975 * resamples))], 4),
    ]


# =============================================================================
# Input loading
# =============================================================================


def load_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_gt_lookup(path: Optional[pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    return {e["id"]: e for e in data if isinstance(e, dict) and "id" in e}


def load_label_lookup(path: Optional[pathlib.Path]) -> Dict[str, str]:
    """Per-question dataset intent label (``category``/``intent``), by id."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or []
    out: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        label = entry.get("category") or entry.get("intent")
        if entry.get("id") and label:
            out[entry["id"]] = label
    return out


def gold_fact_strings(gt_entry: Optional[Dict[str, Any]]) -> List[str]:
    """Required fact strings for one question (both gold schemas)."""
    if not gt_entry:
        return []
    facts: List[str] = [str(x) for x in (gt_entry.get("must_include") or [])]
    for fact in gt_entry.get("critical_facts") or []:
        if isinstance(fact, dict) and fact.get("value"):
            facts.append(str(fact["value"]))
    return [f for f in facts if f.strip()]


def generation_block_evidence(answer: str) -> Optional[Tuple[str, str]]:
    """Return ``(verbatim_error, source)`` when this answer proves a blocked generation.

    Sources, in order:
      * ``record_exception`` — the runner wrote ``ERROR: <exc>`` (verbatim text);
      * ``answer_fallback_marker`` — the canned LLMGenerator fallback string.
    """
    text = answer or ""
    stripped = text.lstrip()
    if stripped[:7].upper() == "ERROR: " or stripped[:6].upper() == "ERROR:":
        return stripped.split(":", 1)[1].strip(), "record_exception"
    low = text.lower()
    for marker in GENERATION_FALLBACK_MARKERS:
        if marker in low:
            return marker, "answer_fallback_marker"
    return None


def run_meta_path(records_path: pathlib.Path) -> pathlib.Path:
    """Sidecar the runner writes next to the JSONL (see ``eval_run_instrumented``)."""
    return records_path.with_suffix(".run-meta.json")


def load_provider_errors(path: Optional[pathlib.Path]) -> List[Dict[str, Any]]:
    """Verbatim provider errors captured during the run (loguru sidecar)."""
    if path is None or not path.exists():
        return []
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = meta.get("provider_errors") or []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"text": item, "count": 1, "source": "run_meta_log"})
        elif isinstance(item, dict) and item.get("text"):
            out.append(
                {
                    "text": str(item["text"]),
                    "count": int(item.get("count", 1)),
                    "source": str(item.get("source", "run_meta_log")),
                }
            )
    return out


# =============================================================================
# Per-question rows
# =============================================================================


def _judge_scores(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    scores = rec.get("judge_scores")
    return scores if isinstance(scores, dict) else None


def build_rows(
    records: Sequence[Dict[str, Any]],
    *,
    gt_lookup: Dict[str, Dict[str, Any]],
    label_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Flatten each record into the row the metrics consume."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        qid = rec.get("id") or ""
        question = rec.get("question") or ""
        intent, source = classify_intent(question, label_lookup.get(qid))

        facts = [norm(f) for f in gold_fact_strings(gt_lookup.get(qid))]
        facts = [f for f in facts if f]
        chunks = rec.get("retrieved_chunks") or []
        context = norm(" ".join((c.get("text") or "") for c in chunks))

        if facts:
            hits = sum(1 for f in facts if f in context)
            recall_hit: Optional[int] = 1 if hits == len(facts) else 0
            recall_any: Optional[int] = 1 if hits else 0
        else:
            recall_hit = None
            recall_any = None

        latency = rec.get("latency_ms") or {}
        evidence = generation_block_evidence(rec.get("answer") or "")

        rows.append(
            {
                "id": qid,
                "intent": intent,
                "intent_source": source,
                "intent_label": label_lookup.get(qid),
                "router_intent": rec.get("intent"),
                "route": rec.get("route") or "unknown",
                "recall_needed": bool(facts),
                "recall_hit": recall_hit,
                "recall_any": recall_any,
                "total_ms": (
                    float(latency["total_ms"])
                    if isinstance(latency.get("total_ms"), (int, float))
                    else None
                ),
                "route_ms": latency.get("route_ms"),
                "retrieval_loop_ms": latency.get("retrieval_loop_ms"),
                "query_ms": latency.get("query_ms"),
                "blocked": evidence is not None,
                "block_evidence": evidence,
                "judge": _judge_scores(rec),
            }
        )
    return rows


# =============================================================================
# Metrics
# =============================================================================


def _rate_block(hits: int, n: int) -> Dict[str, Any]:
    return {
        "hits": hits,
        "scored_questions": n,
        "rate": _r(hits / n) if n else None,
        "ci95": wilson_interval(hits, n),
        "ci95_method": CI_METHODS["rate"],
    }


def _recall_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r["recall_needed"]]
    all_hits = sum(1 for r in scored if r["recall_hit"] == 1)
    any_hits = sum(1 for r in scored if r["recall_any"] == 1)
    return {
        "definition": RECALL_DEFINITION,
        "scored_questions": len(scored),
        "coverage_note": (
            "scored_questions can be far below question_count: gold rows without "
            "`must_include`/`critical_facts` contribute no retrieval signal."
        ),
        "all_present": _rate_block(all_hits, len(scored)),
        "any_present": _rate_block(any_hits, len(scored)),
    }


def _latency_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, seam in (
        ("total_ms", "total_ms"),
        ("route_ms", "route_ms"),
        ("retrieval_loop_ms", "retrieval_loop_ms"),
        ("query_ms", "query_ms"),
    ):
        values = [
            float(r[seam])
            for r in rows
            if isinstance(r.get(seam), (int, float))
        ]
        block: Dict[str, Any] = {
            "n": len(values),
            "p50": _r(percentile(values, 50.0), 1) if values else None,
            "p95": _r(percentile(values, 95.0), 1) if values else None,
        }
        if key == "total_ms":
            block["mean"] = _r(sum(values) / len(values), 1) if values else None
            block["ci95_p50"] = bootstrap_percentile_interval(values, 50.0)
            block["ci95_p95"] = bootstrap_percentile_interval(values, 95.0)
            block["ci95_method"] = CI_METHODS["percentile"]
            # Raw values are kept so a two-manifest diff can re-bootstrap the
            # delta exactly, without re-reading the records.
            block["values"] = [_r(v, 2) for v in values]
        out[key] = block
    return out


def _judge_block(
    rows: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    *,
    label: str,
    blocked_errors: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    values: List[float] = []
    for r in rows:
        judge = r["judge"] or {}
        for key in keys:
            value = judge.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
                break
    if values:
        return {
            "value": _r(sum(values) / len(values)),
            "status": "measured",
            "n": len(values),
            "ci95": bootstrap_mean_interval(values),
            "ci95_method": f"bootstrap_mean_{BOOTSTRAP_RESAMPLES}_seed{BOOTSTRAP_SEED}",
            "source": f"judge_scores.{'/'.join(keys)}",
        }
    blocked = [r for r in rows if r["blocked"]]
    if not blocked:
        return {
            "value": None,
            "status": "unmeasured_no_judge",
            "why": (
                f"{label} needs a judge score; no `judge_scores` were persisted in "
                f"the records and generation was not blocked (0/{len(rows)}), so the "
                f"metric is simply unmeasured — NOT zero."
            ),
            "n": len(rows),
            "generation_blocked": 0,
            "provider_errors": [],
        }
    errors = list(blocked_errors)
    if not errors:
        evidence = sorted(
            {
                r["block_evidence"][0]
                for r in blocked
                if r.get("block_evidence")
            }
        )
        errors = [
            {"text": text, "count": 1, "source": "answer_fallback_marker"}
            for text in evidence
        ]
    return {
        "value": None,
        "status": "blocked_by_provider",
        "why": (
            f"{label} needs a generated answer plus a judge; generation was "
            f"blocked by the LLM provider(s) on {len(blocked)}/{len(rows)} "
            f"question(s) in this bucket and no judge_scores were persisted."
        ),
        "n": len(rows),
        "generation_blocked": len(blocked),
        "provider_errors": errors,
    }


def _route_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["route"]] = counts.get(r["route"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def metrics_for_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    provider_errors: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """All per-bucket metrics for one set of rows."""
    sources: Dict[str, int] = {}
    for r in rows:
        sources[r["intent_source"]] = sources.get(r["intent_source"], 0) + 1
    return {
        "question_count": len(rows),
        "intent_source": dict(sorted(sources.items())),
        "generation": {
            "ok": sum(1 for r in rows if not r["blocked"]),
            "blocked": sum(1 for r in rows if r["blocked"]),
        },
        "recall_at_k": _recall_block(rows),
        "semantic_accuracy": _judge_block(
            rows,
            ("semantic_accuracy",),
            label="semantic_accuracy",
            blocked_errors=provider_errors,
        ),
        "groundedness": _judge_block(
            rows,
            ("groundedness", "faithfulness"),
            label="groundedness",
            blocked_errors=provider_errors,
        ),
        "latency_ms": _latency_block(rows),
        "route_distribution": _route_distribution(rows),
    }


# =============================================================================
# Manifest
# =============================================================================


def _sha256_file(path: Optional[pathlib.Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(path: Optional[pathlib.Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return {"name": path.name, "sha256": _sha256_file(path)}


def _question_fingerprint(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Identity of the QUESTION SET (sorted ids), independent of the run.

    Two manifests whose fingerprints match cover the same questions, so a delta
    between them is a system delta; a mismatch means the deltas compare
    different populations and ``build_diff`` says so.
    """
    ids = sorted(str(r.get("id", f"#{i}")) for i, r in enumerate(records))
    return {
        "count": len(ids),
        "sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }


def canonical_json(obj: Any) -> str:
    """Canonical serialization: sorted keys, fixed separators, trailing newline."""
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, separators=(",", ": "))
        + "\n"
    )


def _content_hash(manifest: Dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k not in ("content_hash", "provenance")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_manifest(
    records: Sequence[Dict[str, Any]],
    *,
    records_path: pathlib.Path,
    gt_lookup: Dict[str, Dict[str, Any]],
    label_lookup: Dict[str, str],
    gt_path: Optional[pathlib.Path] = None,
    dataset_path: Optional[pathlib.Path] = None,
    provider_errors: Optional[Sequence[Dict[str, Any]]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the machine-readable per-intent manifest.

    ``provider_errors`` defaults to the run-meta sidecar next to the records
    (``<records>.run-meta.json``) when the caller passes ``None`` — the same
    convention the CLI uses.
    """
    if provider_errors is None:
        provider_errors = load_provider_errors(run_meta_path(records_path))
    else:
        provider_errors = list(provider_errors)
    rows = build_rows(records, gt_lookup=gt_lookup, label_lookup=label_lookup)

    by_intent: Dict[str, Any] = {}
    for intent in INTENT_TAXONOMY:
        bucket = [r for r in rows if r["intent"] == intent]
        by_intent[intent] = metrics_for_rows(bucket, provider_errors=provider_errors)

    blocked = sum(1 for r in rows if r["blocked"])
    provider_status = (
        "blocked" if rows and blocked == len(rows) else ("partial" if blocked else "ok")
    )
    aggregate = metrics_for_rows(rows, provider_errors=provider_errors)
    router_intents: Dict[str, int] = {}
    for r in rows:
        key = str(r["router_intent"])
        router_intents[key] = router_intents.get(key, 0) + 1

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "intent": {
            "taxonomy": list(INTENT_TAXONOMY),
            "rule": INTENT_RULE_TEXT,
            "classifier": "scripts/eval_intent.classify_intent",
            "dataset_labels_used": sorted(
                {r["intent_label"] for r in rows if r["intent_label"]}
            ),
            "dataset_labels_not_in_taxonomy": sorted(
                {
                    str(r["intent_label"])
                    for r in rows
                    if r["intent_label"]
                    and classify_intent("", r["intent_label"])[1] != "dataset_label"
                }
            ),
            "diagnostics": {
                "router_intent": dict(sorted(router_intents.items())),
                "router_intent_note": (
                    "the pipeline's own `intent` field is not used for bucketing: "
                    "the LLM classify path collapses to 'general' when the "
                    "providers are down, and the deterministic regex fast-path "
                    "labels (so_sanh/liet_ke) only restate the same comparison/"
                    "enumeration regexes the derived rule already applies."
                ),
            },
        },
        "inputs": {
            "records": _file_ref(records_path),
            "gt": _file_ref(gt_path),
            "dataset": _file_ref(dataset_path),
            "questions": _question_fingerprint(records),
        },
        "provider": {
            "status": provider_status,
            "generation_ok": len(rows) - blocked,
            "generation_blocked": blocked,
            "errors": list(provider_errors),
            "semantic_accuracy": aggregate["semantic_accuracy"]["status"],
            "groundedness": aggregate["groundedness"]["status"],
        },
        "aggregate": aggregate,
        "by_intent": by_intent,
        "ci_methods": CI_METHODS,
    }
    if generated_at is not None:
        manifest["provenance"] = {"generated_at": generated_at}
    manifest["content_hash"] = _content_hash(manifest)
    return manifest


def render_manifest_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        f"# Per-intent eval manifest (schema v{manifest['schema_version']})",
        "",
        f"records: `{(manifest['inputs']['records'] or {}).get('name')}`",
        f"provider status: **{manifest['provider']['status']}** "
        f"(ok={manifest['provider']['generation_ok']}, "
        f"blocked={manifest['provider']['generation_blocked']})",
    ]
    not_in_taxonomy = (manifest.get("intent") or {}).get(
        "dataset_labels_not_in_taxonomy"
    ) or []
    derived_count = sum(
        (blk.get("intent_source") or {}).get("derived_rule", 0)
        for blk in (manifest.get("by_intent") or {}).values()
    )
    if derived_count or not_in_taxonomy:
        label_note = (
            "intent labels: the dataset `category` wins where it names a taxonomy "
            f"class; {derived_count}/{manifest['aggregate']['question_count']} "
            "question(s) bucketed by the derived text rule"
        )
        if not_in_taxonomy:
            label_note += (
                f" (dataset label(s) outside the taxonomy: {not_in_taxonomy})"
            )
        lines.append(label_note + ".")
    lines += [
        "",
        "| intent | n | recall@k all-present (hits/n) | 95% CI | p50 ms | p95 ms | sem.acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in list(INTENT_TAXONOMY) + ["aggregate"]:
        block = manifest["by_intent"].get(name) if name != "aggregate" else manifest["aggregate"]
        recall = block["recall_at_k"]["all_present"]
        lat = block["latency_ms"]["total_ms"]
        acc = block["semantic_accuracy"]
        acc_text = (
            f"{acc['value']:.3f}" if acc.get("value") is not None else acc["status"]
        )
        ci = recall["ci95"]
        if recall["rate"] is None:
            recall_text = "n/a"
        else:
            recall_text = (
                f"{recall['rate']:.4f} "
                f"({recall.get('hits')}/{recall.get('scored_questions')})"
            )
        lines.append(
            f"| {name} | {block['question_count']} | "
            f"{recall_text} | "
            f"{'[' + ', '.join(f'{x:.3f}' for x in ci) + ']' if ci else 'n/a'} | "
            f"{lat['p50'] if lat['p50'] is not None else 'n/a'} | "
            f"{lat['p95'] if lat['p95'] is not None else 'n/a'} | {acc_text} |"
        )
    if manifest["provider"]["errors"]:
        lines += ["", "## Provider errors (verbatim)", ""]
        for err in manifest["provider"]["errors"]:
            lines.append(f"- `{err['text']}` (x{err.get('count', 1)}, {err.get('source')})")
    return "\n".join(lines) + "\n"


# =============================================================================
# Two-manifest delta
# =============================================================================


def _rate_delta(base: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    """Delta + Newcombe CI for one rate block (``hits``/``scored_questions``/``rate``)."""
    b_rate, c_rate = base.get("rate"), cand.get("rate")
    bn = int(base.get("scored_questions", 0))
    cn = int(cand.get("scored_questions", 0))
    if b_rate is None or c_rate is None:
        return {
            "baseline": b_rate,
            "candidate": c_rate,
            "delta": None,
            "ci95": None,
            "n": {"baseline": bn, "candidate": cn},
            "why": "metric is not defined for one of the two manifests (no scored questions)",
        }
    # CI for `candidate - baseline`, matching the reported delta sign.
    ci = newcombe_diff_interval(
        int(cand.get("hits", 0)), cn, int(base.get("hits", 0)), bn
    )
    return {
        "baseline": b_rate,
        "candidate": c_rate,
        "delta": _r(c_rate - b_rate),
        "ci95": ci,
        "ci95_method": CI_METHODS["rate_delta"],
        "n": {"baseline": bn, "candidate": cn},
        "significant": None if ci is None else (ci[0] > 0 or ci[1] < 0),
    }


def _latency_delta(
    base_latency: Dict[str, Any], cand_latency: Dict[str, Any], stat: str
) -> Dict[str, Any]:
    b = (base_latency or {}).get("total_ms") or {}
    c = (cand_latency or {}).get("total_ms") or {}
    b_val, c_val = b.get(stat), c.get(stat)
    b_raw, c_raw = b.get("values") or [], c.get("values") or []
    if b_val is None or c_val is None:
        return {"baseline": b_val, "candidate": c_val, "delta": None, "ci95": None,
                "why": "latency unavailable in one manifest"}
    return {
        "baseline": b_val,
        "candidate": c_val,
        "delta": _r(c_val - b_val, 1),
        "ci95": bootstrap_percentile_diff_interval(
            b_raw, c_raw, 50.0 if stat == "p50" else 95.0
        ),
        "ci95_method": CI_METHODS["percentile_delta"],
        "n": {"baseline": b.get("n"), "candidate": c.get("n")},
    }


def _judge_delta(base: Dict[str, Any], cand: Dict[str, Any], key: str) -> Dict[str, Any]:
    b = base.get(key) or {}
    c = cand.get(key) or {}
    if b.get("value") is None or c.get("value") is None:
        errors: List[Dict[str, Any]] = []
        for block in (b, c):
            for err in block.get("provider_errors") or []:
                if err not in errors:
                    errors.append(err)
        return {
            "baseline": b.get("value"),
            "candidate": c.get("value"),
            "delta": None,
            "ci95": None,
            "status": "blocked_by_provider",
            "why": (
                "cannot diff a metric that requires generation: "
                f"baseline={b.get('status')}, candidate={c.get('status')}"
            ),
            "provider_errors": errors,
        }
    return {
        "baseline": b.get("value"),
        "candidate": c.get("value"),
        "delta": _r(c["value"] - b["value"]),
        "ci95": None,
        "status": "measured",
        "why": "delta CI not computed (both manifests measured; use per-manifest ci95)",
    }


def build_diff(base: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    """Per-intent + aggregate delta with confidence intervals."""
    by_intent: Dict[str, Any] = {}
    for intent in list(INTENT_TAXONOMY) + ["aggregate"]:
        b = base["aggregate"] if intent == "aggregate" else base["by_intent"].get(intent, {})
        c = cand["aggregate"] if intent == "aggregate" else cand["by_intent"].get(intent, {})
        b_recall = (b.get("recall_at_k") or {})
        c_recall = (c.get("recall_at_k") or {})
        by_intent[intent] = {
            "question_count": {
                "baseline": b.get("question_count"),
                "candidate": c.get("question_count"),
            },
            "recall_at_k_all_present": _rate_delta(
                b_recall.get("all_present") or {}, c_recall.get("all_present") or {}
            ),
            "recall_at_k_any_present": _rate_delta(
                b_recall.get("any_present") or {}, c_recall.get("any_present") or {}
            ),
            "latency_p50_ms": _latency_delta(b.get("latency_ms"), c.get("latency_ms"), "p50"),
            "latency_p95_ms": _latency_delta(b.get("latency_ms"), c.get("latency_ms"), "p95"),
            "semantic_accuracy": _judge_delta(b, c, "semantic_accuracy"),
            "groundedness": _judge_delta(b, c, "groundedness"),
        }

    warnings = [
        "recall deltas use the Newcombe hybrid score interval (independent samples).",
        "latency deltas use an independent two-sample bootstrap of the difference; "
        "manifests are not paired even when they cover the same questions.",
        f"bootstrap is pinned to seed={BOOTSTRAP_SEED}, resamples={BOOTSTRAP_RESAMPLES} "
        "so the diff is reproducible.",
    ]
    blocked_errors: List[Dict[str, Any]] = []
    for manifest in (base, cand):
        for err in (manifest.get("provider") or {}).get("errors") or []:
            if err not in blocked_errors:
                blocked_errors.append(err)
    if blocked_errors:
        warnings.append(
            "generation-dependent metrics are blocked_by_provider in this diff: "
            + "; ".join(e["text"] for e in blocked_errors[:3])
        )

    # A delta is only interpretable when both sides cover the same questions.
    b_inputs = base.get("inputs") or {}
    c_inputs = cand.get("inputs") or {}
    b_questions = b_inputs.get("questions") or {}
    c_questions = c_inputs.get("questions") or {}
    if (
        b_questions.get("sha256")
        and c_questions.get("sha256")
        and b_questions["sha256"] != c_questions["sha256"]
    ):
        warnings.append(
            "baseline and candidate cover DIFFERENT question sets "
            f"(baseline n={b_questions.get('count')}, "
            f"candidate n={c_questions.get('count')}): the deltas compare "
            "different populations, not just different systems."
        )
    b_dataset = (b_inputs.get("dataset") or {}).get("name")
    c_dataset = (c_inputs.get("dataset") or {}).get("name")
    if b_dataset != c_dataset:
        warnings.append(
            f"intent labels come from different datasets ({b_dataset} vs {c_dataset})."
        )

    return {
        "schema": DIFF_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "baseline": {
            "records": (base.get("inputs") or {}).get("records"),
            "content_hash": base.get("content_hash"),
            "provider_status": (base.get("provider") or {}).get("status"),
        },
        "candidate": {
            "records": (cand.get("inputs") or {}).get("records"),
            "content_hash": cand.get("content_hash"),
            "provider_status": (cand.get("provider") or {}).get("status"),
        },
        "intent": base.get("intent") or cand.get("intent"),
        "by_intent": by_intent,
        "warnings": warnings,
        "provider_errors": blocked_errors,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_ci(ci: Optional[Sequence[float]], digits: int = 4) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def render_diff_table(diff: Dict[str, Any]) -> str:
    lines = [
        "intent       metric                       baseline   candidate  delta      ci95",
        "-----------  ---------------------------  ---------  ---------  ---------  ---------------------------",
    ]
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    for intent in list(INTENT_TAXONOMY) + ["aggregate"]:
        block = diff["by_intent"][intent]
        rows.append((intent, "recall@k all-present", block["recall_at_k_all_present"]))
        rows.append((intent, "recall@k any-present", block["recall_at_k_any_present"]))
        rows.append((intent, "latency p50 ms", block["latency_p50_ms"]))
        rows.append((intent, "latency p95 ms", block["latency_p95_ms"]))
        rows.append((intent, "semantic_accuracy", block["semantic_accuracy"]))
        rows.append((intent, "groundedness", block["groundedness"]))
    for intent, metric, block in rows:
        delta = block.get("delta")
        delta_text = f"{delta:+.4f}" if isinstance(delta, float) else "n/a"
        lines.append(
            f"{intent:<11}  {metric:<27}  {_fmt(block.get('baseline')):<9}  "
            f"{_fmt(block.get('candidate')):<9}  {delta_text:<9}  {_fmt_ci(block.get('ci95'))}"
        )
    lines.append("")
    for warning in diff.get("warnings") or []:
        lines.append(f"note: {warning}")
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-intent eval manifest + two-manifest delta report (ticket 08)"
    )
    parser.add_argument("--records", type=str, default=None, help="EvalRecord JSONL from eval_run_instrumented.py")
    parser.add_argument("--gt", type=str, default=None, help="gold JSON (must_include / critical_facts)")
    parser.add_argument("--dataset", type=str, default=None, help="questions JSON (provides `category` intent labels)")
    parser.add_argument("--run-meta", type=str, default=None, help="provider-error sidecar written by the runner")
    parser.add_argument("--ci-subset", type=int, default=None, help="report only a deterministic intent-stratified subset of N questions")
    parser.add_argument("--limit", type=int, default=None, help="report only the first N records")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("BASELINE", "CANDIDATE"), default=None)
    parser.add_argument("--out", type=str, default=None, help="machine-readable JSON output path")
    parser.add_argument("--md", type=str, default=None, help="optional Markdown sidecar")
    parser.add_argument("--with-timestamp", action="store_true", help="embed provenance.generated_at (breaks byte-identical re-runs)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.compare:
        base_path, cand_path = (pathlib.Path(p) for p in args.compare)
        base = json.loads(base_path.read_text(encoding="utf-8"))
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        diff = build_diff(base, cand)
        text = render_diff_table(diff)
        sys.stdout.write(text)
        if args.out:
            out_path = pathlib.Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(canonical_json(diff), encoding="utf-8")
        if args.md:
            md_path = pathlib.Path(args.md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(text, encoding="utf-8")
        return 0

    if not args.records:
        sys.stderr.write("--records is required (or use --compare)\n")
        return 2

    records_path = pathlib.Path(args.records)
    records = load_records(records_path)
    gt_path = pathlib.Path(args.gt) if args.gt else None
    dataset_path = pathlib.Path(args.dataset) if args.dataset else None
    label_lookup = load_label_lookup(dataset_path)
    # None -> build_manifest auto-detects the runner's sidecar next to the records.
    provider_errors = (
        load_provider_errors(pathlib.Path(args.run_meta)) if args.run_meta else None
    )

    if args.limit is not None:
        records = records[: args.limit]
    if args.ci_subset is not None:
        records = select_ci_subset(records, args.ci_subset, label_lookup=label_lookup)

    manifest = build_manifest(
        records,
        records_path=records_path,
        gt_lookup=load_gt_lookup(gt_path),
        label_lookup=label_lookup,
        gt_path=gt_path,
        dataset_path=dataset_path,
        provider_errors=provider_errors,
        generated_at=datetime.datetime.now().isoformat()
        if args.with_timestamp
        else None,
    )

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(canonical_json(manifest), encoding="utf-8")
    if args.md:
        md_path = pathlib.Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_manifest_markdown(manifest), encoding="utf-8")
    if not args.out:
        sys.stdout.write(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
