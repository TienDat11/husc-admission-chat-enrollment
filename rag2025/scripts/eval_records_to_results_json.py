"""Converter: EvalRecord JSONL -> legacy report-consumable `results.json`.

The fixed-pipeline eval (scripts/eval_run_instrumented.py) now writes the full-
evidence EvalRecord schema to `results/eval_harness/86q_records_<run_id>.jsonl`.
The two report generators (gen_thi_sinh_pdf.py, gen_thi_sinh_reports.py) still
read a legacy schema from `results/86q_thi_sinh_results.json`. This script
bridges the two — any EvalRecord run can be re-shaped into the report's
expected layout without re-running the pipeline.

Schema mapping (per record):
    EvalRecord (source)                      legacy result entry
    ──────────────────                       ──────────────────
    id                        ──>           id
    question                  ──>           question
    answer                    ──>           answer
    not _is_fallback(answer)  ──>           has_answer
    route                     ──>           route
    complexity                ──>           complexity (default 1)
    intent                    ──>           intent (default "general")
    model_version.gen         ──>           provider (best-effort)
    latency_ms.total_ms /1000 ──>           latency_s (rounded 1dp)
    retrieved_chunks[].chunk_id──>          chunk_ids
    retrieved_chunks[].source ──>           chunk_sources

The has_answer rule mirrors gen_thi_sinh_pdf._has_fallback EXACTLY so the
report's pass/fail counts stay consistent across both pipelines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the project config for the canonical results dir if the caller does
# not override --out. Avoids hard-coding the layout in two places.
try:
    from config.settings import RESULTS_DIR  # type: ignore
except Exception:  # pragma: no cover - config optional
    RESULTS_DIR = ROOT / "results"


# ─────────────────────── has_answer rule ───────────────────────


def _is_fallback(ans: str) -> bool:
    """Replicate gen_thi_sinh_pdf._has_fallback EXACTLY.

    A short answer that opens with an explicit "not found" sentinel is a
    fallback. Naive substring scans for "không đủ" false-positive on legit
    answers ("không đủ điểm/điều kiện") so the legacy rule stays strict.
    """
    a = (ans or "").strip().lower()
    return (
        not a
        or len(a) < 50
        or a.startswith("tôi không tìm thấy")
        or a.startswith("không tìm thấy")
        or "không tìm thấy thông tin này trong tài liệu" in a
        or a.startswith("tôi không cung cấp")
    )


# ─────────────────────── convert() ───────────────────────


def convert(
    records: list[dict],
    *,
    gen_model: str,
    hyde_model: str,
    ner_model: str,
    embedding_model: str,
) -> dict:
    """Pure: EvalRecord dicts -> legacy report dict (metadata + results)."""
    if not records:
        return {
            "metadata": {
                "ner_model": ner_model,
                "hyde_model": hyde_model,
                "gen_model": gen_model,
                "embedding_model": embedding_model,
                "total": 0,
                "completed": 0,
            },
            "results": [],
        }

    # Pull the model versions from the FIRST record so the report header
    # reflects the real models that produced the run (the caller-supplied
    # gen_model/hyde_model/... kwargs are fallbacks if the record lacks them).
    first_mv = records[0].get("model_version") or {}
    resolved_gen = first_mv.get("gen") or gen_model
    resolved_hyde = first_mv.get("hyde") or hyde_model
    resolved_ner = first_mv.get("ner") or ner_model
    resolved_emb = first_mv.get("embedding") or embedding_model

    metadata = {
        "ner_model": resolved_ner,
        "hyde_model": resolved_hyde,
        "gen_model": resolved_gen,
        "embedding_model": resolved_emb,
        "total": len(records),
        "completed": len(records),
    }

    results = []
    for rec in records:
        chunks = rec.get("retrieved_chunks") or []
        answer = rec.get("answer", "")
        results.append(
            {
                "id": rec.get("id"),
                "question": rec.get("question"),
                "answer": answer,
                "has_answer": not _is_fallback(answer),
                "route": rec.get("route"),
                "complexity": rec.get("complexity") if rec.get("complexity") is not None else 1,
                "intent": rec.get("intent") if rec.get("intent") is not None else "general",
                "provider": (rec.get("model_version") or {}).get("gen") or resolved_gen,
                "latency_s": round(((rec.get("latency_ms") or {}).get("total_ms") or 0) / 1000.0, 1),
                "chunk_ids": [c.get("chunk_id") for c in chunks],
                "chunk_sources": [c.get("source") for c in chunks],
            }
        )

    return {"metadata": metadata, "results": results}


# ─────────────────────── I/O helpers ───────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file — one JSON object per line, blank lines ignored."""
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _derive_model_defaults(records: list[dict]) -> tuple[str, str, str, str]:
    """Best-effort model names from the first record (used as argparse defaults)."""
    if not records:
        return ("", "", "", "")
    mv = records[0].get("model_version") or {}
    return (
        mv.get("gen", ""),
        mv.get("hyde", ""),
        mv.get("ner", ""),
        mv.get("embedding", ""),
    )


# ─────────────────────── main() ───────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an EvalRecord JSONL run into the legacy "
            "results/86q_thi_sinh_results.json schema consumed by the PDF + "
            "Markdown report generators."
        )
    )
    parser.add_argument(
        "--records",
        required=True,
        type=Path,
        help="Path to EvalRecord JSONL (e.g. results/eval_harness/86q_records_s15.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "86q_thi_sinh_results.json",
        help="Output JSON path. Default: results/86q_thi_sinh_results.json",
    )
    parser.add_argument(
        "--gen-model",
        default=None,
        help="Optional override for metadata.gen_model (else pulled from record 0).",
    )
    parser.add_argument(
        "--hyde-model",
        default=None,
        help="Optional override for metadata.hyde_model (else pulled from record 0).",
    )
    parser.add_argument(
        "--ner-model",
        default=None,
        help="Optional override for metadata.ner_model (else pulled from record 0).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional override for metadata.embedding_model (else pulled from record 0).",
    )
    args = parser.parse_args(argv)

    records = _read_jsonl(args.records)

    # If caller didn't override the model kwargs, derive them from record 0
    # so the convert() signature stays keyword-only-stable for tests.
    dflt_gen, dflt_hyde, dflt_ner, dflt_emb = _derive_model_defaults(records)
    out = convert(
        records,
        gen_model=args.gen_model if args.gen_model is not None else dflt_gen,
        hyde_model=args.hyde_model if args.hyde_model is not None else dflt_hyde,
        ner_model=args.ner_model if args.ner_model is not None else dflt_ner,
        embedding_model=args.embedding_model if args.embedding_model is not None else dflt_emb,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    n = len(out["results"])
    gen = out["metadata"].get("gen_model", "?")
    print(f"Wrote {args.out} — {n} records (gen={gen})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
