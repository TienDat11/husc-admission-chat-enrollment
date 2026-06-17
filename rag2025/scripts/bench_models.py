"""Benchmark ramclouds.me models for NER / HYDE / GEN pipeline roles."""
import os, time, json, requests, sys

BASE = os.getenv("RAMCLOUDS_BASE_URL", "https://ramclouds.me/v1")
KEY  = os.getenv("RAMCLOUDS_API_KEY", "")

# Excluded: claude-opus-4.6, claude-opus-4.7, claude-opus-4.7-max,
#           claude-sonnet-4.6, glm-5.1, kimi-k2.6, minimax-m2.7,
#           opuscursor, image models
MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemini-3-flash",
    "gemini-3.1-pro-high",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5.5",
]

NER_PROMPT = """Trích xuất thực thể (NER) từ văn bản sau. Trả về JSON array, mỗi phần tử có {entity, type}.
Các type: NGANH, TRUONG, DIEM_CHUAN, HOC_PHI, PHUONG_THUC, NAM, CHI_TIEU, KHOA, DOI_TUONG.

Văn bản: Trường Đại học Khoa học Huế tuyển sinh năm 2025 ngành Công nghệ thông tin với chỉ tiêu 150, điểm chuẩn 18.5, học phí 950000/tín chỉ. Phương thức xét tuyển: theo học bạ và thi THPT."""

HYDE_PROMPT = """Sinh một đoạn văn bản giả định (HyDE) mà một tài liệu chứa đáp án cho câu hỏi sau sẽ có. Viết bằng tiếng Việt, 3-5 câu.

Câu hỏi: Điểm chuẩn ngành Công nghệ thông tin HUSC 2025 là bao nhiêu?"""

GEN_PROMPT = """Trả lời câu hỏi tuyển sinh sau một cách chính xác, đầy đủ, hữu ích cho thí sinh. Nếu không có đủ thông tin, hãy nói rõ.

Câu hỏi: HUSC có những phương thức xét tuyển nào năm 2025 và điều kiện từng phương thức?"""

ROLES = [
    ("NER",  NER_PROMPT),
    ("HYDE", HYDE_PROMPT),
    ("GEN",  GEN_PROMPT),
]

def bench_one(model: str, role: str, prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=headers,
                          json=payload, timeout=120)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return {"model": model, "role": role, "ok": False,
                    "error": f"HTTP {r.status_code}", "time": dt}
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        total_tok = usage.get("total_tokens", 0)
        # Quality heuristics
        q_ok = True
        q_note = ""
        if role == "NER":
            try:
                # Try to find JSON in response
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, list) and len(parsed) >= 3:
                        q_note = f"{len(parsed)} entities"
                    else:
                        q_ok = len(parsed) > 0
                        q_note = f"only {len(parsed)} entities"
                else:
                    q_ok = False
                    q_note = "no JSON array"
            except Exception:
                q_ok = False
                q_note = "parse fail"
        elif role == "HYDE":
            sentences = content.count('.') + content.count('。')
            q_ok = sentences >= 2
            q_note = f"{sentences} sentences"
        elif role == "GEN":
            q_ok = len(content) >= 100
            q_note = f"{len(content)} chars"
        return {"model": model, "role": role, "ok": True,
                "time": round(dt, 2), "tokens": total_tok,
                "quality_ok": q_ok, "quality_note": q_note,
                "preview": content[:120].replace('\n', ' ')}
    except requests.exceptions.Timeout:
        dt = time.perf_counter() - t0
        return {"model": model, "role": role, "ok": False,
                "error": "TIMEOUT>120s", "time": round(dt, 2)}
    except Exception as e:
        dt = time.perf_counter() - t0
        return {"model": model, "role": role, "ok": False,
                "error": str(e)[:80], "time": round(dt, 2)}

def main():
    print("=" * 70)
    print("  RAMCLOUDS MODEL BENCHMARK  (NER / HYDE / GEN)")
    print("=" * 70)

    results = []
    for model in MODELS:
        for role, prompt in ROLES:
            print(f"  Testing {model:25s} [{role}] ...", end="", flush=True)
            r = bench_one(model, role, prompt)
            results.append(r)
            if r["ok"]:
                q = "PASS" if r.get("quality_ok") else "WEAK"
                print(f"  {r['time']:6.1f}s  tok={r['tokens']:5d}  quality={q} ({r.get('quality_note','')})")
            else:
                print(f"  FAIL  {r.get('error','?')}  ({r['time']:.1f}s)")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY BY ROLE")
    print("=" * 70)
    for role in ["NER", "HYDE", "GEN"]:
        print(f"\n  --- {role} ---")
        role_results = [r for r in results if r["role"] == role and r["ok"]]
        role_results.sort(key=lambda x: x["time"])
        for r in role_results:
            q = "OK" if r.get("quality_ok") else "WEAK"
            print(f"  {r['model']:25s}  {r['time']:6.1f}s  quality={q:4s}  {r.get('quality_note','')}")

    # Recommendations
    print("\n" + "=" * 70)
    print("  RECOMMENDED CONFIG")
    print("=" * 70)
    for role in ["NER", "HYDE", "GEN"]:
        role_results = [r for r in results if r["role"] == role and r["ok"] and r.get("quality_ok")]
        if not role_results:
            # fallback to any ok
            role_results = [r for r in results if r["role"] == role and r["ok"]]
        if role_results:
            best = min(role_results, key=lambda x: x["time"])
            env_key = f"RAMCLOUDS_{'NER' if role=='NER' else 'HYDE' if role=='HYDE' else 'GEN'}_MODEL"
            print(f"  {env_key}={best['model']}  ({best['time']:.1f}s)")
        else:
            print(f"  {role}: NO MODEL PASSED")

if __name__ == "__main__":
    main()
