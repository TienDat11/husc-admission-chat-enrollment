"""
benchmark_provider_latency.py — Phase 0 Verification (sprint-0-verification)

Wraps: rag2025/.omc/plans/latency_reduction_plan_v2.md Section 0.1 (G1, G2)
Spec: docs/rag-chatbot-husc-spec.md S7 (latency reduction)

Benchmarks 3 LLM providers head-to-head with 10 router-style classification prompts.
Output: writes results to results/phase0_provider_latency_<timestamp>.json
        prints markdown table to stdout (paste into results/phase0_benchmarks.md)

Usage:
    python rag2025/scripts/benchmark_provider_latency.py

Required env vars (from .env):
    RAMCLOUDS_KEY, RAMCLOUDS_BASE_URL (optional, defaults https://ramclouds.me/v1)
    RAMCLOUDS_GEN_MODEL (e.g. gpt-5.3)
    OPENAI_KEY (optional — skip if missing)
    GOOGLE_KEY (optional — skip if missing)

Decision tree fork (latency plan H2):
    - ramclouds p50 > openai/gemini p50 by >= 5s -> a4_priority (provider direct wins)
    - all 3 providers ~similar -> b2_priority (merged router takes precedence)
    - all 3 fast (< 3s p50) -> streaming_priority (focus B4 SSE)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

WARMUP_CALLS = 1
BENCHMARK_CALLS = 10
MAX_TOKENS = 200
TEMPERATURE = 0.1
TIMEOUT_S = 60.0

ROUTER_PROMPTS: list[str] = [
    "Phân loại: 'Học phí ngành Công nghệ thông tin năm 2025?' → category?",
    "Phân loại: 'Điểm chuẩn ngành Ngôn ngữ Anh năm 2024?' → category?",
    "Phân loại: 'Thời hạn nộp hồ sơ xét tuyển đợt 1?' → category?",
    "Phân loại: 'Quy trình chuyển trường vào HUSC?' → category?",
    "Phân loại: 'Học bổng tài trợ năm đầu tiên điều kiện gì?' → category?",
    "Phân loại: 'Chỗ ở ký túc xá cho sinh viên năm nhất?' → category?",
    "Phân loại: 'Chương trình đào tạo ngành Toán học?' → category?",
    "Phân loại: 'Tuyển sinh chương trình chất lượng cao?' → category?",
    "Phân loại: 'Chỉ tiêu ngành Vật lý năm 2025?' → category?",
    "Phân loại: 'Mã trường đại học khoa học Huế?' → category?",
]

SYSTEM_PROMPT = (
    "Bạn là bộ phân loại truy vấn tuyển sinh HUSC. "
    "Trả lời ngắn gọn 1 từ: hoc_phi|diem_chuan|ho_so|chuyen_truong|"
    "hoc_bong|ky_tuc_xa|chuong_trinh|tuyen_sinh|chi_tieu|ma_truong|khac"
)

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

def _build_providers() -> dict[str, dict[str, Any]]:
    """Build provider configs from env. Skip providers with missing keys."""
    providers: dict[str, dict[str, Any]] = {}

    rc_key = os.environ.get("RAMCLOUDS_KEY")
    if rc_key:
        providers["ramclouds"] = {
            "api_key": rc_key,
            "base_url": os.environ.get("RAMCLOUDS_BASE_URL", "https://ramclouds.me/v1"),
            "model": os.environ.get("RAMCLOUDS_GEN_MODEL", "gpt-4o-mini"),
        }
    else:
        print("[WARN] RAMCLOUDS_KEY not set — skipping ramclouds")

    oai_key = os.environ.get("OPENAI_KEY")
    if oai_key:
        providers["openai"] = {
            "api_key": oai_key,
            "base_url": None,  # default OpenAI endpoint
            "model": os.environ.get("OPENAI_GEN_MODEL", "gpt-4o-mini"),
        }
    else:
        print("[WARN] OPENAI_KEY not set — skipping openai")

    google_key = os.environ.get("GOOGLE_KEY")
    if google_key:
        providers["gemini"] = {
            "api_key": google_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": os.environ.get("GOOGLE_GEN_MODEL", "gemini-2.0-flash"),
        }
    else:
        print("[WARN] GOOGLE_KEY not set — skipping gemini")

    return providers


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

async def _call_provider(
    client: AsyncOpenAI, model: str, prompt: str
) -> tuple[float, bool]:
    """Single timed call. Returns (latency_seconds, is_error)."""
    try:
        t0 = time.perf_counter()
        await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=TIMEOUT_S,
        )
        elapsed = time.perf_counter() - t0
        return elapsed, False
    except (APIError, asyncio.TimeoutError, Exception) as exc:
        print(f"  [ERR] {type(exc).__name__}: {exc}")
        return 0.0, True


async def _benchmark_provider(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run warmup + benchmark for one provider. Sequential to avoid rate limits."""
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
    )
    model = cfg["model"]

    # Warmup (discarded)
    print(f"[{name}] Warmup (1 call)...")
    await _call_provider(client, model, ROUTER_PROMPTS[0])

    # Benchmark
    latencies: list[float] = []
    errors = 0
    for i, prompt in enumerate(ROUTER_PROMPTS[:BENCHMARK_CALLS], 1):
        print(f"[{name}] Call {i}/{BENCHMARK_CALLS}...")
        elapsed, is_err = await _call_provider(client, model, prompt)
        if is_err:
            errors += 1
        else:
            latencies.append(elapsed)

    await client.close()

    if not latencies:
        return {"model": model, "n_runs": 0, "mean_s": 0, "p50_s": 0,
                "p90_s": 0, "p99_s": 0, "errors": errors}

    latencies.sort()
    n = len(latencies)

    def _percentile(data: list[float], p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return round(data[idx], 2)

    return {
        "model": model,
        "n_runs": n,
        "mean_s": round(sum(latencies) / n, 2),
        "p50_s": _percentile(latencies, 50),
        "p90_s": _percentile(latencies, 90),
        "p99_s": _percentile(latencies, 99),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

def _decision_tree(results: dict[str, dict[str, Any]]) -> str:
    """Apply H2 decision tree from latency plan."""
    p50s = {k: v["p50_s"] for k, v in results.items() if v["n_runs"] > 0}
    if not p50s:
        return "inconclusive"

    all_fast = all(v < 3.0 for v in p50s.values())
    if all_fast:
        return "streaming_priority"

    if "ramclouds" in p50s:
        other_min = min(v for k, v in p50s.items() if k != "ramclouds") if len(p50s) > 1 else 0
        if p50s["ramclouds"] - other_min >= 5.0:
            return "a4_priority"

    return "b2_priority"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(results: dict[str, dict[str, Any]]) -> None:
    print("\n| Provider | Model | mean | p50 | p90 | p99 | errors |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(
            f"| {name} | {r['model']} | {r['mean_s']}s | {r['p50_s']}s "
            f"| {r['p90_s']}s | {r['p99_s']}s | {r['errors']} |"
        )


def _write_json(results: dict[str, dict[str, Any]], decision: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"phase0_provider_latency_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "providers": results,
        "decision_tree_hint": decision,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    providers = _build_providers()
    if not providers:
        print("[FATAL] No provider keys found — cannot benchmark.")
        return 1

    # Run providers in parallel, prompts within provider are sequential
    tasks = {name: _benchmark_provider(name, cfg) for name, cfg in providers.items()}
    results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

    decision = _decision_tree(results)

    _print_table(results)

    print(f"\nphase0_decision: {decision}")
    if decision == "a4_priority":
        print("Reasoning: ramclouds p50 >= 5s slower than alternatives → provider direct wins.")
    elif decision == "b2_priority":
        print("Reasoning: providers similar latency → merged router takes precedence.")
    elif decision == "streaming_priority":
        print("Reasoning: all providers fast (< 3s p50) → focus B4 SSE streaming.")
    else:
        print("Reasoning: insufficient data.")

    out_path = _write_json(results, decision)
    print(f"\nResults written to: {out_path}")

    success = any(v["n_runs"] > 0 for v in results.values())
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
