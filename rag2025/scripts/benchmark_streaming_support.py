"""
benchmark_streaming_support.py — Phase 0 Verification (sprint-0-verification)

Wraps: rag2025/.omc/plans/latency_reduction_plan_v2.md Section 0.2 (G3)
Spec: docs/rag-chatbot-husc-spec.md S7 (latency reduction)

Verifies streaming support across 3 LLM providers and measures TTFT (Time-To-First-Token).
Decision input for B4 streaming Sprint 3: if all providers support streaming with TTFT < 3s,
SSE endpoint /query/stream is viable. If TTFT > 5s, reconsider streaming priority.

Usage:
    python rag2025/scripts/benchmark_streaming_support.py

Required env vars (from .env):
    RAMCLOUDS_KEY + RAMCLOUDS_GEN_MODEL
    OPENAI_KEY (optional)
    GOOGLE_KEY (optional)

Output:
    - results/phase0_streaming_support_<UTC-timestamp>.json
    - markdown table to stdout
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 60
MAX_TOKENS = 600
TEMPERATURE = 0.1
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PROMPTS = {
    "short": "List 3 admission requirements for HUSC in one sentence.",
    "medium": (
        "Explain the undergraduate admission process at Hue University of Sciences. "
        "Cover: application timeline, required documents, entrance exam subjects, "
        "selection criteria, and appeal process. Be concise but thorough."
    ),
    "long": (
        "Provide a comprehensive guide to university admissions at Hue University of Sciences (HUSC). "
        "Include the following sections:\n"
        "1. Overview of HUSC and its faculties\n"
        "2. Admission methods (high-school transcript, entrance exam, direct admission)\n"
        "3. Detailed application timeline from announcement to enrollment\n"
        "4. Required documents for each admission method\n"
        "5. Entrance exam structure: subjects, scoring, bonus points\n"
        "6. Selection and ranking criteria with example score calculations\n"
        "7. Tuition fees and scholarship opportunities\n"
        "8. International student admission process\n"
        "9. Dormitory and housing information\n"
        "10. Contact information and important dates for the upcoming year\n"
        "Write approximately 500 tokens."
    ),
}

PROVIDERS: dict[str, dict[str, str | None]] = {
    "ramclouds": {
        "api_key_env": "RAMCLOUDS_KEY",
        "model_env": "RAMCLOUDS_GEN_MODEL",
        "base_url_env": "RAMCLOUDS_BASE_URL",
        "base_url_default": "https://ramclouds.me/v1",
    },
    "openai": {
        "api_key_env": "OPENAI_KEY",
        "model_env": "OPENAI_MODEL",
        "base_url_env": None,
        "base_url_default": None,
    },
    "google": {
        "api_key_env": "GOOGLE_KEY",
        "model_env": "GOOGLE_MODEL",
        "base_url_env": None,
        "base_url_default": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def _run_stream(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    label: str,
) -> dict:
    """Run a single streaming request and collect metrics."""
    ttft_ms: float | None = None
    token_deltas: list[float] = []
    total_tokens = 0
    first_delta_chunks = 0
    errors = 0
    stream_supported = False
    real_streaming = True
    t_start = time.perf_counter()

    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            ),
            timeout=TIMEOUT_S,
        )

        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content is None:
                continue

            t_now = time.perf_counter()
            if ttft_ms is None:
                ttft_ms = (t_now - t_start) * 1000
                stream_supported = True
                first_delta_chunks = 1
            else:
                first_delta_chunks += 1

            token_count = len(delta.content.split())
            total_tokens += max(token_count, 1)
            token_deltas.append(t_now)

    except asyncio.TimeoutError:
        errors += 1
    except Exception as exc:
        errors += 1
        if "does not support streaming" in str(exc).lower():
            stream_supported = False
            real_streaming = False
        return {
            "label": label,
            "stream_supported": False,
            "real_streaming": False,
            "ttft_ms": None,
            "tokens_per_sec": None,
            "total_time_s": None,
            "errors": errors,
            "error_detail": str(exc)[:200],
        }

    t_end = time.perf_counter()
    total_time_s = t_end - t_start

    # Detect fake streaming: single large chunk with all content
    if first_delta_chunks == 1 and total_tokens > 20:
        real_streaming = False

    tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0

    return {
        "label": label,
        "stream_supported": stream_supported,
        "real_streaming": real_streaming,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "total_time_s": round(total_time_s, 2),
        "errors": errors,
    }


async def _benchmark_provider(name: str, cfg: dict) -> dict:
    """Benchmark one provider across all prompt lengths."""
    api_key = os.environ.get(cfg["api_key_env"])
    model = os.environ.get(cfg["model_env"] or "", "")

    if not api_key:
        return {
            "model": model or "(not set)",
            "stream_supported": False,
            "real_streaming": False,
            "ttft_short_ms": None,
            "ttft_medium_ms": None,
            "ttft_long_ms": None,
            "ttft_mean_ms": None,
            "tokens_per_sec_mean": None,
            "errors": 1,
            "error_detail": f"Missing env var {cfg['api_key_env']}",
        }

    client_kwargs: dict = {"api_key": api_key}
    base_url = None
    if cfg.get("base_url_env"):
        base_url = os.environ.get(cfg["base_url_env"]) or cfg.get("base_url_default")
    else:
        base_url = cfg.get("base_url_default")
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs)

    results: dict[str, dict] = {}
    for length, prompt in PROMPTS.items():
        results[length] = await _run_stream(client, model, prompt, length)

    ttft_values = [
        r["ttft_ms"] for r in results.values() if r["ttft_ms"] is not None
    ]
    tps_values = [
        r["tokens_per_sec"] for r in results.values() if r["tokens_per_sec"] is not None
    ]
    total_errors = sum(r["errors"] for r in results.values())
    stream_ok = all(r["stream_supported"] for r in results.values())
    real_ok = all(r["real_streaming"] for r in results.values() if r["stream_supported"])

    return {
        "model": model,
        "stream_supported": stream_ok,
        "real_streaming": real_ok if stream_ok else False,
        "ttft_short_ms": results["short"]["ttft_ms"],
        "ttft_medium_ms": results["medium"]["ttft_ms"],
        "ttft_long_ms": results["long"]["ttft_ms"],
        "ttft_mean_ms": round(sum(ttft_values) / len(ttft_values), 1) if ttft_values else None,
        "tokens_per_sec_mean": round(sum(tps_values) / len(tps_values), 1) if tps_values else None,
        "errors": total_errors,
    }


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def _decision_hint(results: dict) -> str:
    """Derive streaming viability from benchmark results."""
    supported = [r for r in results.values() if r["stream_supported"]]
    if not supported:
        return "streaming_blocked"

    mean_ttfts = [r["ttft_mean_ms"] for r in supported if r["ttft_mean_ms"] is not None]
    if not mean_ttfts:
        return "streaming_blocked"

    worst_ttft = max(mean_ttfts)
    if worst_ttft < 3000:
        return "streaming_viable"
    if worst_ttft < 5000:
        return "streaming_marginal"
    return "streaming_blocked"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_table(results: dict) -> None:
    """Print markdown summary table to stdout."""
    print()
    print("| Provider | Stream? | Real-stream? | TTFT (mean) | Throughput | Errors |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        stream_icon = "✓" if r["stream_supported"] else "✗"
        real_icon = "✓" if r["real_streaming"] else "✗"
        ttft = f"{int(r['ttft_mean_ms'])}ms" if r["ttft_mean_ms"] else "N/A"
        tps = f"{r['tokens_per_sec_mean']} tok/s" if r["tokens_per_sec_mean"] else "N/A"
        print(f"| {name} | {stream_icon} | {real_icon} | {ttft} | {tps} | {r['errors']} |")
    print()


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load .env if present
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    results: dict[str, dict] = {}
    for name, cfg in PROVIDERS.items():
        print(f"Benchmarking {name} ...")
        results[name] = await _benchmark_provider(name, cfg)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": results,
        "decision_tree_hint": _decision_hint(results),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"phase0_streaming_support_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    _print_table(results)
    print(f"Decision hint: {report['decision_tree_hint']}")


if __name__ == "__main__":
    asyncio.run(main())
