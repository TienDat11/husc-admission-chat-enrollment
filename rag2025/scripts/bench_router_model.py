"""Router-model benchmark runner (S-bench).

Drives ONE candidate model through the 3 router roles on the gold set
(`data/eval/router_model_benchmark_gold.json`) and scores it:

  1. classify  → route accuracy vs gold (+ JSON-validity hard gate)
  2. step-back → entity preservation (deterministic substring check)
  3. HyDE      → domain-vocab presence + no-invented-URL (deterministic)

Quality-only — latency is NOT scored (per user). Writes a raw+scored JSON to
`results/eval_harness/router_bench/<model>.json`.

Usage:
  python scripts/bench_router_model.py --model gpt-5.4
  python scripts/bench_router_model.py --model gemini-3.5-flash-low

The model is forced via `get_llm_client(force_model=...)`; the router is
constructed with that client so _classify/_step_back/_generate_hyde_doc all
use the candidate. Read-only on the pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT / ".env")
except Exception:  # pragma: no cover
    pass

_GOLD = _PROJECT / "data" / "eval" / "router_model_benchmark_gold.json"
_URL_RX = re.compile(r"https?://\S+")
_VALID_ROUTES = {"padded", "hybrid", "graph"}


def score_classify(gold_case: Dict[str, Any], result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one classify case. JSON-validity is a hard gate (0 if invalid)."""
    if not isinstance(result, dict) or result.get("route") not in _VALID_ROUTES:
        return {"id": gold_case["id"], "json_valid": False, "route": None,
                "gold": gold_case["gold_route"], "score": 0.0}
    route = result["route"]
    gold = gold_case["gold_route"]
    acceptable = set(gold_case.get("acceptable_routes", [gold]))
    if route == gold:
        score = 1.0
    elif route in acceptable:
        score = 0.5
    else:
        score = 0.0
    return {"id": gold_case["id"], "json_valid": True, "route": route,
            "gold": gold, "score": score}


def score_stepback(gold_case: Dict[str, Any], rewrite: str) -> Dict[str, Any]:
    """Entity-preservation: every token in must_preserve AND at least one of
    must_preserve_any must appear in the rewrite (case-insensitive)."""
    r = (rewrite or "").lower()
    must = gold_case.get("must_preserve", [])
    must_any = gold_case.get("must_preserve_any", [])
    ok_must = all(tok.lower() in r for tok in must)
    ok_any = (not must_any) or any(tok.lower() in r for tok in must_any)
    passed = ok_must and ok_any
    return {"id": gold_case["id"], "preserved": passed,
            "missing": [t for t in must if t.lower() not in r],
            "score": 1.0 if passed else 0.0}


def score_hyde(gold_case: Dict[str, Any], doc: str) -> Dict[str, Any]:
    """Domain-vocab present + NO invented URL + non-empty single passage."""
    d = (doc or "")
    dl = d.lower()
    has_domain = any(t.lower() in dl for t in gold_case.get("domain_terms_any", []))
    has_url = bool(_URL_RX.search(d))
    non_empty = len(d.strip()) >= 10
    # leaked reasoning heuristic: chain-of-thought markers
    leaked = any(m in dl for m in ["chúng ta cần", "let me", "first, i", "bước 1:", "tôi sẽ viết"])
    passed = has_domain and not has_url and non_empty and not leaked
    return {"id": gold_case["id"], "has_domain": has_domain, "has_url": has_url,
            "non_empty": non_empty, "leaked_reasoning": leaked,
            "score": 1.0 if passed else 0.0}


async def run_model(model: str) -> Dict[str, Any]:
    from services.query_router import SmartQueryRouter
    from services.llm_client import get_llm_client

    gold = json.loads(_GOLD.read_text(encoding="utf-8"))
    client = get_llm_client(force_model=model)
    router = SmartQueryRouter(llm=client)

    classify_results: List[Dict[str, Any]] = []
    for c in gold["classify_cases"]:
        try:
            res = await router._classify(c["query"])
        except Exception as e:
            res = None
            classify_results.append({**score_classify(c, None), "error": str(e)[:80]})
            continue
        classify_results.append(score_classify(c, res))

    stepback_results: List[Dict[str, Any]] = []
    for s in gold["stepback_cases"]:
        try:
            rw = await router._step_back(s["query"])
        except Exception as e:
            rw = ""
            stepback_results.append({**score_stepback(s, ""), "error": str(e)[:80]})
            continue
        stepback_results.append(score_stepback(s, rw or ""))

    hyde_results: List[Dict[str, Any]] = []
    for h in gold["hyde_cases"]:
        try:
            doc = await router._generate_hyde_doc(h["query"], h["query"])
        except Exception as e:
            doc = ""
            hyde_results.append({**score_hyde(h, ""), "error": str(e)[:80]})
            continue
        hyde_results.append(score_hyde(h, doc or ""))

    def mean(rs: List[Dict[str, Any]]) -> float:
        return round(sum(r["score"] for r in rs) / len(rs), 4) if rs else 0.0

    json_valid_rate = round(
        sum(1 for r in classify_results if r.get("json_valid")) / len(classify_results), 4
    ) if classify_results else 0.0

    return {
        "model": model,
        "summary": {
            "classify_route_accuracy": mean(classify_results),
            "classify_json_valid_rate": json_valid_rate,
            "stepback_entity_preservation": mean(stepback_results),
            "hyde_domain_fidelity": mean(hyde_results),
            "overall_quality": round(
                (mean(classify_results) * 0.5
                 + mean(stepback_results) * 0.25
                 + mean(hyde_results) * 0.25), 4
            ),
        },
        "classify": classify_results,
        "stepback": stepback_results,
        "hyde": hyde_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Router-model quality benchmark (S-bench)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = asyncio.run(run_model(args.model))
    out_dir = _PROJECT / "results" / "eval_harness" / "router_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    out = Path(args.out) if args.out else out_dir / f"{safe}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    s = report["summary"]
    print(f"[{args.model}] route_acc={s['classify_route_accuracy']} "
          f"json_valid={s['classify_json_valid_rate']} "
          f"stepback={s['stepback_entity_preservation']} "
          f"hyde={s['hyde_domain_fidelity']} overall={s['overall_quality']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
