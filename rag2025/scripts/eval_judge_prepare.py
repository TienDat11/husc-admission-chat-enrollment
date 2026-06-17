"""Slice S5a — Judge-batch preparer + fixed rubric.

PURE data transformer. Takes the S2 EvalRecord JSONL (full evidence) + the
ground-truth JSON, and emits fixed-size batch JSONs that an Opus subagent
will read + score with the embedded rubric.

This module does NOT call any LLM and does NOT score anything. It only
shapes the input contract for the offline judge.

Public API:
  * JUDGE_RUBRIC: str — the fixed markdown rubric embedded in every batch.
  * prepare_batches(records, gt_by_id, *, batch_size, run_id) -> list[dict]
  * write_batches(batches, out_dir) -> writes batch_NNN.json
  * main() — argparse: --run-id, --records, --gt, --batch-size

Determinism:
  * Stable input ordering preserved; no timestamps/random/uuid.
  * Pure function of (records, gt_by_id, batch_size, run_id) — same inputs
    produce byte-identical output.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------- Constants ----------

RAG2025_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = RAG2025_ROOT / "results" / "eval_harness" / "judge_batches"

# Default writeback location for the Opus subagent's verdict JSON.
VERDICT_FORMAT_NOTE = (
    "verdict_<run_id>_batch_NNN.json — one verdict JSON per input batch, "
    "with the same batch_id echoed back."
)


# ---------- JUDGE_RUBRIC (fixed) ----------

JUDGE_RUBRIC: str = """\
# Judge Rubric — HUSC Admissions RAG (Slice S5a)

You are an Opus subagent scoring a small batch of RAG Q&A items. You will
read each `items[]` entry, decompose the answer, and emit ONE verdict JSON
per batch. This rubric is the FIXED scoring contract — do not improvise.

For EACH item, you must compute FOUR metrics, EACH in the closed interval
[0.0, 1.0] (or `null` where the metric is not applicable, per the rules
below). You must also emit a `claims` decomposition and a `rationale`.

## Metric 1 — `faithfulness` (0..1 or null)

Definition: decompose the `answer` into atomic factual CLAIMS (one assertion
per claim, no compound statements). For each claim, mark
`supported: true` ONLY if the claim is entailed by the provided `contexts`.
`score = supported_claims / total_claims`.

Special cases:
* If `contexts` is empty (auto-answer / no retrieval), emit
  `faithfulness: null` — not applicable, do NOT guess.
* If the answer contains no atomic claims (e.g. a one-word refusal), emit
  `faithfulness: 1.0` (vacuously supported).

## Metric 2 — `answer_relevancy` (0..1)

Definition: how directly the answer addresses the question. Ignore
correctness. A perfect answer that misses the question's intent scores low;
a partial answer that hits the exact question's intent scores high.

Anchor points:
* 1.0 — answers the question directly and completely (in intent).
* 0.5 — partially addresses the question, or addresses a neighbouring one.
* 0.0 — unrelated to the question.

## Metric 3 — `context_precision` (0..1)

Definition: fraction of provided `contexts` that are RELEVANT to answering
the question. A context is "relevant" if it contains information that could
plausibly help answer the question (or a part of it).

`score = relevant_context_count / total_context_count`.
If `contexts` is empty, emit `context_precision: null`.

## Metric 4 — `context_recall` (0..1 or null)

Definition: fraction of the `reference_answer`'s facts that are present in
the union of `contexts`. Used to measure retrieval coverage.

Special cases:
* If `reference_answer` is null or `expected_behavior != "answer"`, emit
  `context_recall: null` (no reference to measure recall against).
* If `reference_answer` is non-null but contains no extractable facts, emit
  `context_recall: 1.0` (nothing to recall).

## Per-item claims decomposition

For EACH item, emit:
```
"claims": [
  {"claim": "<atomic factual statement>",
   "supported": true | false,
   "cite": "<chunk_id that supports the claim>" | null}
]
```

Rules:
* `cite` MUST be a `chunk_id` from the `contexts` list (string-equal), or
  `null` if the claim is unsupported.
* Do NOT invent chunk_ids. Do NOT cite by index; cite by id.
* One claim per line; do not combine two assertions into one claim.

## Per-item rationale

For each item, emit a short `rationale` (1-3 sentences) explaining the
non-obvious calls — especially why any claim is unsupported, why
`context_recall` is null, and any edge case in the answer (refusal, hedge,
auto-answer).

## Verdict JSON writeback (CONTRACT)

For a batch with `batch_id = "batch_003"` and `run_id = "<run_id>"`, write
ONE verdict JSON to:

  rag2025/results/eval_harness/judge_verdicts/<run_id>/verdict_<run_id>_batch_003.json

The verdict file must have this exact shape:

```
{
  "batch_id": "batch_003",
  "run_id": "<run_id>",
  "rubric_version": "S5a",
  "judge": "opus",
  "items": [
    {
      "id": "<item id>",
      "scores": {
        "faithfulness": <float 0..1 | null>,
        "answer_relevancy": <float 0..1>,
        "context_precision": <float 0..1 | null>,
        "context_recall": <float 0..1 | null>
      },
      "claims": [
        {"claim": "...", "supported": true | false, "cite": "<chunk_id>" | null}
      ],
      "rationale": "1-3 sentences"
    }
  ]
}
```

Failure to follow the schema will be rejected by the aggregator.

## What you must NOT do

* Do NOT modify the rubric.
* Do NOT re-order `items` — preserve input order so the aggregator can
  join by `id`.
* Do NOT score from your own knowledge — only from the `contexts` provided.
* Do NOT guess a `cite` — if a claim is unsupported, set `cite: null`.
* Do NOT collapse multiple atomic claims into one.
"""


# ---------- Core functions ----------

def _coerce_record(rec: Any) -> Dict[str, Any]:
    """Accept either an EvalRecord (Pydantic) or a dict-like record."""
    if hasattr(rec, "model_dump"):
        return rec.model_dump()
    if isinstance(rec, dict):
        return rec
    raise TypeError(f"unsupported record type: {type(rec).__name__}")


def _extract_contexts(rec: Dict[str, Any]) -> List[str]:
    """Pull the FULL `text` of each retrieved chunk, in input order."""
    chunks = rec.get("retrieved_chunks") or []
    out: List[str] = []
    for c in chunks:
        if isinstance(c, dict):
            txt = c.get("text", "")
        else:  # Pydantic RetrievedChunk
            txt = getattr(c, "text", "")
        out.append(txt if isinstance(txt, str) else "")
    return out


def _gt_payload(gt_by_id: Dict[str, Dict[str, Any]], rid: str) -> Dict[str, Any]:
    """Return the GT-derived fields for an item. Deterministic on missing GT."""
    gt = gt_by_id.get(rid)
    if not gt:
        return {
            "reference_answer": None,
            "expected_behavior": None,
            "critical_facts": [],
        }
    ref = gt.get("reference_answer")
    # Treat empty string as null (matches the GT file convention).
    if isinstance(ref, str) and not ref.strip():
        ref = None
    return {
        "reference_answer": ref,
        "expected_behavior": gt.get("expected_behavior"),
        "critical_facts": list(gt.get("critical_facts") or []),
    }


def prepare_batches(
    records: Sequence[Any],
    gt_by_id: Dict[str, Dict[str, Any]],
    *,
    batch_size: int = 10,
    run_id: str = "run",
) -> List[Dict[str, Any]]:
    """Chunk records into fixed-size batches with the rubric + GT joined in.

    Deterministic: stable input ordering, no timestamps, no randomness.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    n = len(records)
    out: List[Dict[str, Any]] = []
    # Iterate by index; slicing list[0:n] preserves order.
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        items: List[Dict[str, Any]] = []
        for rec in records[start:end]:
            d = _coerce_record(rec)
            rid = str(d.get("id", ""))
            gt = _gt_payload(gt_by_id, rid)
            items.append(
                {
                    "id": rid,
                    "question": d.get("question", ""),
                    "answer": d.get("answer", ""),
                    "contexts": _extract_contexts(d),
                    "reference_answer": gt["reference_answer"],
                    "expected_behavior": gt["expected_behavior"],
                    "critical_facts": gt["critical_facts"],
                }
            )
        batch_index = start // batch_size
        out.append(
            {
                "batch_id": f"batch_{batch_index:03d}",
                "run_id": str(run_id),
                "rubric": JUDGE_RUBRIC,
                "items": items,
            }
        )
    return out


def write_batches(
    batches: List[Dict[str, Any]],
    out_dir: str | pathlib.Path,
) -> List[pathlib.Path]:
    """Write each batch to out_dir/batch_NNN.json (zero-padded). Returns paths."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: List[pathlib.Path] = []
    for b in batches:
        bid = b["batch_id"]
        # Sanity: enforce the zero-padded id shape so filenames align with batch_id.
        if not bid.startswith("batch_"):
            raise ValueError(f"unexpected batch_id shape: {bid!r}")
        target = out_path / f"{bid}.json"
        target.write_text(
            json.dumps(b, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(target)
    return written


# ---------- CLI ----------

def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_gt_by_id(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Read a GT JSON (array of entries) into {id: entry}."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"GT file must be a JSON array, got {type(data).__name__}")
    return {e["id"]: e for e in data if isinstance(e, dict) and "id" in e}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Opus-judge batch JSONs from S2 EvalRecords + GT."
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument(
        "--records",
        required=True,
        type=pathlib.Path,
        help="Path to S2 EvalRecord JSONL.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        type=pathlib.Path,
        help="Path to GT JSON (array of {id, reference_answer, ...}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Records per batch (default 10).",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Override output dir (default: rag2025/results/eval_harness/judge_batches/<run_id>/).",
    )
    args = parser.parse_args(argv)

    records = _read_jsonl(args.records)
    gt_by_id = _read_gt_by_id(args.gt)
    batches = prepare_batches(
        records,
        gt_by_id,
        batch_size=args.batch_size,
        run_id=args.run_id,
    )
    out_dir = args.out_dir or (RESULTS_ROOT / args.run_id)
    written = write_batches(batches, out_dir)
    print(
        f"[eval_judge_prepare] run_id={args.run_id} "
        f"records={len(records)} batches={len(batches)} "
        f"out_dir={out_dir}"
    )
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
