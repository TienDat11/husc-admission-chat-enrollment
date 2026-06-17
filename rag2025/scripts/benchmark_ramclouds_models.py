"""Benchmark ramclouds.me models for 3 pipeline roles: NER, HYDE, Generation."""
import asyncio
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI

API_KEY = os.getenv("RAMCLOUDS_API_KEY", "")
BASE_URL = os.getenv("RAMCLOUDS_BASE_URL", "https://ramclouds.me/v1")

CANDIDATES = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemini-3-flash",
    "gemini-3.1-pro-high",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5.5",
]

NER_PROMPT = """Trích xuất thực thể (NER) từ đoạn văn bản tuyển sinh sau. Trả JSON array, mỗi phần tử: {"entity": "...", "type": "SCHOOL|MAJOR|SUBJECT|FEE|YEAR|LOCATION|PROGRAM"}.
Văn bản: Trường Đại học Khoa học - Đại học Huế tuyển sinh năm 2026 với 42 ngành đào tạo. Học phí khối VII 525.000 VNĐ/tín chỉ, khối IV 540.000 VNĐ/tín chỉ, khối V 600.000 VNĐ/tín chỉ. Điểm chuẩn ngành CNTT 2025 là 24.5 điểm. Địa chỉ: 77 Nguyễn Huệ, TP Huế."""

HYDE_PROMPT = """Sinh 3 biến thể truy vấn (query variants) cho câu hỏi sau để cải thiện retrieval. Mỗi biến thể phải mang góc nhìn khác nhau. Trả JSON array of strings.
Câu hỏi: Học phí Đại học Khoa học Huế năm 2026 là bao nhiêu?"""

GEN_PROMPT = """Trả lời câu hỏi tuyển sinh sau dựa trên context. Ngắn gọn, chính xác, đầy đủ thông tin.
Context: Học phí 2026 theo khối ngành tại HUSC: Khối VII 525.000đ/tín chỉ, Khối IV 540.000đ/tín chỉ, Khối V 600.000đ/tín chỉ. Triết học miễn học phí.
Câu hỏi: Học phí HUSC năm 2026 bao nhiêu?"""


async def test_model(client: AsyncOpenAI, model: str, role: str, prompt: str) -> dict:
    """Test one model for one role. Returns latency + response."""
    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            timeout=30.0,
        )
        elapsed = time.perf_counter() - start
        content = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return {
            "model": model,
            "role": role,
            "latency_s": round(elapsed, 2),
            "tokens": tokens,
            "tps": round(tokens / elapsed, 1) if elapsed > 0 else 0,
            "response": content[:300],
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "model": model,
            "role": role,
            "latency_s": round(elapsed, 2),
            "tokens": 0,
            "tps": 0,
            "response": "",
            "error": str(e)[:200],
        }


async def main():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []

    for model in CANDIDATES:
        print(f"\n{'='*60}\nTesting: {model}\n{'='*60}")
        for role, prompt in [("NER", NER_PROMPT), ("HYDE", HYDE_PROMPT), ("GEN", GEN_PROMPT)]:
            # Run 2 rounds for stability check
            r1 = await test_model(client, model, role, prompt)
            await asyncio.sleep(0.5)
            r2 = await test_model(client, model, role, prompt)
            avg_latency = (r1["latency_s"] + r2["latency_s"]) / 2
            avg_tps = (r1["tps"] + r2["tps"]) / 2

            entry = {
                "model": model,
                "role": role,
                "avg_latency_s": round(avg_latency, 2),
                "avg_tps": round(avg_tps, 1),
                "round1": r1,
                "round2": r2,
                "stable": r1["error"] is None and r2["error"] is None,
            }
            results.append(entry)
            status = "OK" if entry["stable"] else "ERR"
            print(f"  {role}: avg={avg_latency:.2f}s, tps={avg_tps:.0f}, {status}")

    # Summary table
    print("\n\n" + "="*80)
    print("SUMMARY - Best model per role")
    print("="*80)

    for role in ["NER", "HYDE", "GEN"]:
        role_results = [r for r in results if r["role"] == role and r["stable"]]
        if role_results:
            # Sort by avg_latency (fastest first)
            role_results.sort(key=lambda x: x["avg_latency_s"])
            print(f"\n{role} (sorted by latency):")
            for r in role_results:
                resp_preview = r["round1"]["response"][:80].replace("\n", " ")
                print(f"  {r['model']:25s} {r['avg_latency_s']:6.2f}s  {r['avg_tps']:6.0f} tps  {resp_preview}...")
            best = role_results[0]
            print(f"  >>> BEST for {role}: {best['model']} ({best['avg_latency_s']}s)")

    # Quality check: NER must return JSON, HYDE must return variants, GEN must mention khối
    print("\n\nQUALITY CHECK:")
    for r in results:
        if not r["stable"]:
            continue
        resp = r["round1"]["response"]
        role = r["role"]
        model = r["model"]
        if role == "NER":
            has_json = "[" in resp and "{" in resp and "entity" in resp.lower()
            print(f"  {model:25s} NER JSON: {'PASS' if has_json else 'FAIL'}")
        elif role == "HYDE":
            has_variants = resp.count('"') >= 6 or resp.count("'") >= 6 or resp.count("\n") >= 2
            print(f"  {model:25s} HYDE variants: {'PASS' if has_variants else 'FAIL'}")
        elif role == "GEN":
            mentions_khoi = "khối" in resp.lower() or "khoi" in resp.lower()
            mentions_price = "525" in resp or "540" in resp or "600" in resp
            print(f"  {model:25s} GEN accuracy: {'PASS' if mentions_khoi and mentions_price else 'FAIL'}")

    # Save full results
    out_path = Path("data/model_benchmark_results.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
