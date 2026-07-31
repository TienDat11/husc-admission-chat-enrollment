"""Rerun ONLY the 15 items that errored in the previous run (msg072..msg086).
Reads results/86q_thi_sinh_results_final_2026.json, identifies items where
route=='error', and re-issues the same query against /v2/query.
Writes results/86q_thi_sinh_results_final_2026_rerun.json (merged copy).
"""
import json
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "86q_thi_sinh_results_final_2026.json"
EVAL = ROOT / "data" / "eval" / "husc_thi_sinh_thuc.json"
OUT = ROOT / "results" / "86q_thi_sinh_results_final_2026_rerun.json"
API = "http://127.0.0.1:8000/v2/query"


def has_answer(answer: str) -> bool:
    s = (answer or "").strip().lower()
    if not s or len(s) < 50:
        return False
    bad = [
        "tôi không tìm thấy",
        "không tìm thấy",
        "không tìm thấy thông tin chính xác",
        "không tìm thấy thông tin này trong tài liệu",
        "tôi không cung cấp",
    ]
    if any(s.startswith(b) for b in bad):
        return False
    if "không tìm thấy thông tin chính xác" in s:
        return False
    if "không tìm thấy thông tin này trong tài liệu" in s:
        return False
    return True


def main():
    with RESULTS.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    items = bundle["results"]
    error_ids = [x["id"] for x in items if x.get("route") == "error"]
    print(f"Found {len(error_ids)} error items: {error_ids}")

    with EVAL.open("r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    # backup current
    bak = OUT.with_suffix(".bak.json")
    if not bak.exists():
        bak.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Backup -> {bak}")

    by_id = {x["id"]: x for x in items}
    for i, qid in enumerate(error_ids, 1):
        q = questions[qid]
        try:
            with httpx.Client(timeout=180.0) as client:
                t0 = time.perf_counter()
                r = client.post(API, json={"query": q["question"], "top_k": 5})
                dt = time.perf_counter() - t0
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            print(f"[{i}/{len(error_ids)}] {qid}: ERROR {exc}")
            continue
        ans = data.get("answer", "")
        new = {
            "id": qid,
            "question": q["question"],
            "answer": ans,
            "has_answer": has_answer(ans),
            "route": data.get("route", "?"),
            "router_info": data.get("router_info", {}),
            "latency_s": round(dt, 2),
            "sources": data.get("sources", []),
            "confidence": data.get("confidence", 0.0),
        }
        by_id[qid] = new
        stat = "HAS_ANS" if new["has_answer"] else "FALLBACK"
        print(f"[{i}/{len(error_ids)}] {qid}: {stat} | route={new['route']} | {dt:.2f}s")

    merged = list(by_id.values())
    merged.sort(key=lambda x: x["id"])
    out = {
        "metadata": {
            "api_endpoint": API,
            "top_k": 5,
            "total": len(merged),
            "completed": len(merged),
            "rerun": True,
            "rerun_ids": error_ids,
        },
        "results": merged,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    has = sum(1 for x in merged if x.get("has_answer"))
    grd = sum(1 for x in merged if x.get("route") == "guardrail")
    err = sum(1 for x in merged if x.get("route") == "error")
    print(f"FINAL: has_answer={has}/{len(merged)} guardrail={grd} error={err}")


if __name__ == "__main__":
    main()
