"""A1 baseline measurement (plan U1, Verification).

Read-only. Writes eval/baseline_A1.json. Fields that cannot be computed are
written as ``"not_computed: <reason>"`` and are never filled with a guess.

Measured here:
  - token_drift: cl100k_base (tiktoken) vs the Qwen3-Embedding tokenizer on
    the first N non-empty chunks of the ``husc_chunks`` LanceDB table
    (read-only). If a tokenizer cannot load, the field is null with a reason.

Not measured (see reasons written to the JSON):
  - recall_at_5, recall_at_10, citation_precision, abstention_rate: the seed has
    no ground truth that maps to LanceDB chunk ids (expected_pages is null and
    expected_coordinate_keys are record keys, not chunk ids).
  - chunk_stats (tt06, tts): no reproducible entry point that produces the
    production chunks of a PDF without a canonical store.

Run: ``HF_HOME=<cache> HF_HUB_OFFLINE=1 python eval/run_baseline.py --out eval/baseline_A1.json``
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"  # config/settings.py:66
SAMPLE_SIZE = 200
NOT_COMPUTED_RECALL = "not_computed: seed has no ground truth mapping to LanceDB chunk ids (expected_pages null)"
NOT_COMPUTED_CHUNKS = "not_computed: no reproducible entry point for production chunks of a PDF without canonical store"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _sample_chunks(lancedb_dir: Path) -> list[str]:
    import lancedb  # local import: the script stays importable without the DB stack

    db = lancedb.connect(str(lancedb_dir))
    table = db.open_table("husc_chunks").to_arrow().to_pylist()
    texts = [r["text"] for r in sorted(table, key=lambda r: r["id"]) if r.get("text")]
    return texts[:SAMPLE_SIZE]


def _token_drift(texts: list[str]) -> dict:
    try:
        import tiktoken

        cl100k = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - reported, not failed
        return {"reason": f"cl100k_base unavailable: {type(exc).__name__}"}
    try:
        from transformers import AutoTokenizer

        qwen = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    except Exception as exc:  # noqa: BLE001 - reported, not failed
        return {"reason": f"{QWEN_TOKENIZER} unavailable offline: {type(exc).__name__}"}

    ratios = []
    for text in texts:
        a = len(cl100k.encode(text))
        b = len(qwen.encode(text, add_special_tokens=False))
        if a:
            ratios.append(b / a)
    ratios.sort()

    def pct(p: float) -> float:
        return ratios[min(len(ratios) - 1, int(round(p * (len(ratios) - 1))))]

    return {
        "definition": "ratio = qwen_tokens / cl100k_tokens per chunk",
        "sample_size": len(ratios),
        "cl100k_vs_qwen": round(statistics.fmean(ratios), 4),
        "p50": round(pct(0.5), 4),
        "p95": round(pct(0.95), 4),
        "max": round(ratios[-1], 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 baseline (plan U1).")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "baseline_A1.json")
    parser.add_argument(
        "--lancedb-dir",
        type=Path,
        default=ROOT / "data" / "lancedb",
        help="read-only LanceDB directory (main checkout data)",
    )
    args = parser.parse_args(argv)

    try:
        texts = _sample_chunks(args.lancedb_dir)
        drift = _token_drift(texts)
    except Exception as exc:  # noqa: BLE001 - reported, not failed
        drift = {"reason": f"lancedb read failed: {type(exc).__name__}"}

    result = {
        "recall_at_5": NOT_COMPUTED_RECALL,
        "recall_at_10": NOT_COMPUTED_RECALL,
        "citation_precision": NOT_COMPUTED_RECALL,
        "abstention_rate": NOT_COMPUTED_RECALL,
        "chunk_stats": {"tt06": NOT_COMPUTED_CHUNKS, "tts": NOT_COMPUTED_CHUNKS},
        "token_drift": drift if "cl100k_vs_qwen" in drift else {"value": None, **drift},
        "git_sha": _git_sha(),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": {"provider": "qwen", "model": QWEN_TOKENIZER, "offline": os.environ.get("HF_HUB_OFFLINE") == "1"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    print(json.dumps(result["token_drift"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
