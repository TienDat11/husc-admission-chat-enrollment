"""
Throwaway benchmark: measure end-to-end latency of the GROQ-hosted
llama-3.1-8b-instant vs llama-3.3-70b-versatile for the guardrail
scope-classification call (real call shape from GuardrailService.precheck).

Run:  D:/miniconda3/python.exe scripts/bench_guardrail_groq.py
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq

# ---------- env ----------
# mirror how rag2025 config/settings.py loads (env_file=".env" relative to cwd)
ROOT = Path(__file__).resolve().parent.parent  # rag2025/
load_dotenv(ROOT / ".env")

GROQ_KEY = os.getenv("GROQ_API_KEY")
if GROQ_KEY:
    REDACTED = GROQ_KEY[:4] + "***" + GROQ_KEY[-4:] if len(GROQ_KEY) > 12 else "***"
else:
    REDACTED = "<missing>"

MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

# Same call shape as rag2025/src/services/guardrail.py:precheck
SYSTEM_PROMPT = (
    "Bạn là bộ lọc truy vấn cho chatbot tuyển sinh HUSC. "
    "Phân loại câu hỏi thuộc phạm vi tuyển sinh HUSC hay không. "
    'Trả về JSON: {"is_in_scope": boolean, "reason": string}.'
)

QUERIES: list[tuple[str, bool, str]] = [
    # (query, expected_in_scope, label)
    ("Vật lý học - Chương trình Công nghệ Bán dẫn có cái gì hay hả bạn.", True,
     "vat_ly_ban_dan (must-pass: real 2026 HUSC major)"),
    ("điểm chuẩn ngành Công nghệ thông tin 2026 là bao nhiêu?", True, "diem_chuan_cntt"),
    ("học phí ngành Khoa học dữ liệu?", True, "hoc_phi_khdl"),
    ("ngành mới năm 2026 của trường là gì?", True, "nganh_moi_2026"),
    ("thời tiết Huế hôm nay thế nào?", False, "thoi_tiet (oos)"),
    ("cách nấu phở bò?", False, "cach_nau_pho (oos)"),
    ("Trường có ngành Triết học không?", True, "triet_hoc"),
    ("so sánh ngành CNTT và Kỹ thuật phần mềm", True, "so_sanh_cntt_ktpm"),
]


def redact_payload(payload: Any) -> str:
    s = json.dumps(payload, ensure_ascii=False)
    # belt-and-braces: redact any key value that looks like a GROQ key
    if GROQ_KEY and GROQ_KEY in s:
        s = s.replace(GROQ_KEY, REDACTED)
    return s


async def call_one(client: AsyncGroq, model: str, query: str) -> tuple[float, bool, str]:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # fall back to extracting a JSON object substring (same trick as guardrail.py)
        a = content.find("{")
        b = content.rfind("}")
        data = json.loads(content[a:b + 1]) if a >= 0 and b > a else {}
    is_in_scope = bool(data.get("is_in_scope", False))
    reason = str(data.get("reason", ""))
    return dt_ms, is_in_scope, reason


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def print_table(rows: list[dict], header: str) -> None:
    print()
    print("=" * 88)
    print(header)
    print("=" * 88)
    fmt = "{:>9}  {:<6}  {:<8}  {:<6}  {:<36}"
    print(fmt.format("latency", "expect", "got", "match", "label"))
    print("-" * 88)
    for r in rows:
        print(fmt.format(
            f"{r['ms']:7.1f}ms",
            "IN" if r["expected"] else "OUT",
            "IN" if r["got"] else "OUT",
            "ok" if r["match"] else "MISS",
            r["label"][:36],
        ))


def summarise(latencies: list[float], rows: list[dict], model: str) -> tuple[bool, bool, str]:
    if not latencies:
        return False, False, "no samples"
    correct = sum(1 for r in rows if r["match"])
    total = len(rows)
    acc = correct / total * 100.0
    p95 = pct(latencies, 0.95)
    p50 = statistics.median(latencies)
    meets_1s = p95 <= 1000.0
    meets_2s = p95 <= 2000.0
    print()
    print(f"--- aggregates for {model} ---")
    print(f"  n          : {len(latencies)}")
    print(f"  min        : {min(latencies):7.1f} ms")
    print(f"  median(p50): {p50:7.1f} ms")
    print(f"  p95        : {p95:7.1f} ms")
    print(f"  max        : {max(latencies):7.1f} ms")
    print(f"  accuracy   : {correct}/{total}  ({acc:.1f}%)")
    print(f"  p95 <= 1000ms (1s target) : {'PASS' if meets_1s else 'FAIL'}")
    print(f"  p95 <= 2000ms (2s target) : {'PASS' if meets_2s else 'FAIL'}")
    return meets_1s, meets_2s, f"{correct}/{total}"


async def bench_model(model: str) -> dict | None:
    if not GROQ_KEY:
        print(f"BLOCKED: GROQ_API_KEY missing in {ROOT / '.env'}")
        return None
    client = AsyncGroq(api_key=GROQ_KEY)
    rows: list[dict] = []
    latencies: list[float] = []
    try:
        # 1-call warmup, discarded
        print(f"\n[{model}] warmup...")
        try:
            await call_one(client, model, "warmup query")
        except Exception as e:
            print(f"[{model}] warmup FAILED: {type(e).__name__}: {e}")
            return {"model": model, "blocked": True, "error": f"{type(e).__name__}: {e}"}

        for q, expected, label in QUERIES:
            try:
                ms, got, reason = await call_one(client, model, q)
                rows.append({
                    "label": label, "expected": expected, "got": got,
                    "match": got == expected, "ms": ms, "reason": reason,
                })
                latencies.append(ms)
            except Exception as e:
                # report inline and stop this model
                print(f"[{model}] call FAILED on '{label}': {type(e).__name__}: {e}")
                return {
                    "model": model, "blocked": True,
                    "error": f"{type(e).__name__}: {e}",
                    "partial_rows": rows,
                }
    finally:
        await client.close()

    return {"model": model, "blocked": False, "rows": rows, "latencies": latencies}


async def main() -> int:
    print(f"GROQ_API_KEY loaded: {REDACTED}")
    if not GROQ_KEY:
        return 2

    results: list[dict] = []
    for m in MODELS:
        r = await bench_model(m)
        results.append(r)

    for r in results:
        if r is None:
            continue
        m = r["model"]
        if r.get("blocked"):
            err = r.get("error", "unknown")
            print(f"\nBLOCKED on {m}: {err}")
            # if we got partial rows, show them
            for row in r.get("partial_rows", []):
                print(f"  partial: {row['label']} -> {row['ms']:.1f}ms "
                      f"got={'IN' if row['got'] else 'OUT'} match={'ok' if row['match'] else 'MISS'}")
            continue
        print_table(r["rows"], f"{m} — per-query latency & classification")
        summarise(r["latencies"], r["rows"], m)

    # verdict
    by_model = {r["model"]: r for r in results if r}
    verdict_lines = []
    for m in MODELS:
        r = by_model.get(m)
        if not r or r.get("blocked"):
            verdict_lines.append(f"{m}: BLOCKED ({r.get('error', '?') if r else 'no result'})")
            continue
        lats = r["latencies"]
        correct = sum(1 for x in r["rows"] if x["match"])
        p95 = pct(lats, 0.95)
        p50 = statistics.median(lats)
        ok1s = p95 <= 1000.0
        ok2s = p95 <= 2000.0
        # special check: Vat-ly ban-dan (index 0)
        vlb = r["rows"][0]
        vlb_ok = vlb["match"]
        verdict_lines.append(
            f"{m}: p50={p50:.0f}ms p95={p95:.0f}ms  "
            f"1s-target={'PASS' if ok1s else 'FAIL'}  2s-target={'PASS' if ok2s else 'FAIL'}  "
            f"acc={correct}/{len(r['rows'])}  "
            f"vat_ly_ban_dan={'PASS' if vlb_ok else 'MISS (got OUT)'}"
        )
    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    for line in verdict_lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
