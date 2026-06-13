"""
check_gen_token_truncation.py — Empirical measurement: does MAX_GEN_TOKENS=800
truncate real admission answers?

Strategy
--------
1) Load .env (so MAX_GEN_TOKENS=800 / MAX_GEN_TOKENS_ENUM=8000 are live).
2) Use a small set of representative system-prompt + context strings that
   mirror the production generation path's prompt shape (system = generation
   prompt file; user = CONTEXT block + question; max_tokens branched on
   is_program_list_query).
3) Hit ramclouds.me via the SAME provider path the app uses, but call
   OpenAI non-streamed so we can read `response.choices[0].finish_reason`
   deterministically (the streaming path in llm_client.py captures
   finish_reason but does NOT return it to callers). The non-streamed call
   hits the same provider / model / key.
4) For each query, record: finish_reason, output chars, output est-tokens
   (chars/3 heuristic), and whether the response ended mid-sentence.
5) For 3 of the longest non-enum queries, re-run at max_tokens=1200 and
   diff the last 200 chars — to see if the 800-budget answer cut content
   that the 1200-budget answer continued.

Outputs a per-query truncation table + a 800-vs-1200 last-200-char diff
section, then a verdict.

NOTE: this script does NOT modify .env or production source. It only
emits. The user already set MAX_GEN_TOKENS=800.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Repo path wiring ----
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env BEFORE importing anything that reads env.
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import httpx
from loguru import logger

# Mirror the prompt shape used by llm_generator.generate_answer (line 583-585).
# We load the file directly so this script is self-contained.
PROMPT_PATH = REPO_ROOT / "prompts" / "generation_system_prompt.txt"
GEN_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else (
    "Bạn là chuyên gia tư vấn tuyển sinh Trường Đại học Khoa học - ĐH Huế (HUSC)."
)

RAMCLOUDS_API_KEY = os.environ["RAMCLOUDS_API_KEY"]
RAMCLOUDS_BASE_URL = os.environ.get("RAMCLOUDS_BASE_URL", "https://ramclouds.me/v1")
RAMCLOUDS_GEN_MODEL = os.environ.get("RAMCLOUDS_GEN_MODEL", "gemini-3.5-flash-low")
MAX_GEN_TOKENS = int(os.getenv("MAX_GEN_TOKENS", "800"))
MAX_GEN_TOKENS_ENUM = int(os.getenv("MAX_GEN_TOKENS_ENUM", "8000"))


# ---- Realistic, long-ish context per query (mimics retrieved chunks) ----
# These are deliberately plausible HUSC snippets (made-up but shaped like
# the canonical chunks) so the LLM has substantive content to reason over.
PROGRAM_LIST_CONTEXT = (
    "DANH SÁCH CÁC NGÀNH ĐÀO TẠO HUSC 2026 (tổng cộng 28 ngành):\n"
    "1. 7310101 - Kinh tế (Kinh tế phát triển) - tổ hợp A00, A01, D01, D07\n"
    "2. 7310201 - Quản trị Kinh doanh - tổ hợp A00, A01, C01, D01\n"
    "3. 7310301 - Kế toán - tổ hợp A00, C01, D01, D07\n"
    "4. 7340101 - Quản trị nhân lực - tổ hợp A00, C00, D01, C20\n"
    "5. 7340201 - Tài chính - Ngân hàng - tổ hợp A00, A01, C01, D01\n"
    "6. 7340301 - Kiểm toán - tổ hợp A00, A01, C01, D01\n"
    "7. 7380101 - Luật - tổ hợp A00, C00, D01, D66\n"
    "8. 7380107 - Luật Kinh tế - tổ hợp A00, C00, D01, D66\n"
    "9. 7460101 - Toán học - tổ hợp A00, A01, B00, D07\n"
    "10. 7460102 - Toán ứng dụng - tổ hợp A00, A01, B00, D07\n"
    "11. 7460117 - Toán Cơ - tổ hợp A00, A01, B00, D07\n"
    "12. 7460201 - Vật lý - tổ hợp A00, A01, B00, D07\n"
    "13. 7460202 - Vật lý kỹ thuật - tổ hợp A00, A01, B00, D07\n"
    "14. 7460301 - Hóa học - tổ hợp A00, B00, D07, A06\n"
    "15. 7460302 - Hóa dược - tổ hợp A00, B00, D07, A06\n"
    "16. 7460401 - Sinh học - tổ hợp A00, B00, D08, A02\n"
    "17. 7460402 - Công nghệ Sinh học - tổ hợp A00, B00, D08, A02\n"
    "18. 7480101 - Khoa học Máy tính - tổ hợp A00, A01, B00, D07\n"
    "19. 7480102 - Mạng máy tính & Truyền thông dữ liệu - tổ hợp A00, A01, B00, D07\n"
    "20. 7480103 - Kỹ thuật Phần mềm - tổ hợp A00, A01, B00, D07\n"
    "21. 7480104 - Hệ thống Thông tin - tổ hợp A00, A01, B00, D07\n"
    "22. 7480106 - Khoa học Dữ liệu - tổ hợp A00, A01, B00, D07\n"
    "23. 7480201 - Công nghệ Thông tin (CNTT định hướng ứng dụng) - tổ hợp A00, A01, B00, D07\n"
    "24. 7520201 - Kỹ thuật Điện - tổ hợp A00, A01, B00, D07\n"
    "25. 7520207 - Kỹ thuật Điện tử - Viễn thông - tổ hợp A00, A01, B00, D07\n"
    "26. 7540101 - Công nghệ Thực phẩm - tổ hợp A00, B00, D07, A06\n"
    "27. 7580101 - Kiến trúc - tổ hợp A00, V00, V01, V02\n"
    "28. 7220201 - Ngôn ngữ Anh - tổ hợp D01, D14, D66, D78\n"
    "(Học phí: 13.5 - 18.5 triệu VNĐ/năm tùy ngành; điểm chuẩn 2025: 17.0 - 25.0; chỉ tiêu 2026: khoảng 2,400.)"
)

COMPARE_CONTEXT = (
    "BẢNG SO SÁNH NGÀNH CNTT vs KỸ THUẬT PHẦN MỀM — HUSC 2026:\n\n"
    "KHOA HỌC MÁY TÍNH (7480101):\n"
    "- Điểm chuẩn 2025: 24.50 (A00), 22.80 (D07)\n"
    "- Học phí: 18.500.000 VNĐ/năm\n"
    "- Tổ hợp: A00, A01, B00, D07\n"
    "- Cơ hội việc làm: AI Engineer, Data Scientist, Research Scientist, Giảng viên; mức lương khởi điểm 18-35 triệu/tháng\n"
    "- Chỉ tiêu 2026: 80\n\n"
    "KỸ THUẬT PHẦN MỀM (7480103):\n"
    "- Điểm chuẩn 2025: 23.75 (A00), 22.10 (D07)\n"
    "- Học phí: 18.500.000 VNĐ/năm\n"
    "- Tổ hợp: A00, A01, B00, D07\n"
    "- Cơ hội việc làm: Backend/Frontend/Mobile Developer, DevOps, Tech Lead; mức lương khởi điểm 15-30 triệu/tháng\n"
    "- Chỉ tiêu 2026: 90\n\n"
    "Điểm khác biệt chính: KHMT thiên về nền tảng toán-lý-sâu, nghiên cứu; KTPM thiên về ứng dụng, dev thực chiến, thực tập doanh nghiệp từ năm 3."
)

TUITION_TABLE_CONTEXT = (
    "HỌC PHÍ TẤT CẢ CÁC NGÀNH HUSC 2026 (đơn vị: triệu VNĐ/năm học):\n"
    "- Khối Khoa học Tự nhiên: Toán 13.5, Toán UD 13.5, Toán Cơ 13.5, Vật lý 13.5, VLKT 15.0, Hóa 13.5, Hóa dược 15.0, Sinh 13.5, CNSH 15.0\n"
    "- Khối CNTT: KHMT 18.5, MMTT 18.5, KTPM 18.5, HTTT 18.5, KHDL 18.5, CNTT 18.5\n"
    "- Khối Kỹ thuật: KT Điện 16.0, KT Điện tử-VT 16.0\n"
    "- Khối Kinh tế: Kinh tế 14.5, QTKD 14.5, Kế toán 14.5, QLTNLĐ 14.5, Tài chính-NH 14.5, Kiểm toán 14.5\n"
    "- Khối Luật: Luật 14.0, Luật KT 14.0\n"
    "- Khối Nông-Lâm-Thủy sản: CNTP 15.0\n"
    "- Khối Xây dựng/Kiến trúc: Kiến trúc 17.0\n"
    "- Khối Ngoại ngữ: Ngôn ngữ Anh 16.5\n"
    "(Ghi chú: mức học phí trên áp dụng cho năm học 2026-2027, có thể điều chỉnh 5-10% mỗi năm theo Nghị định 81/2021/NĐ-CP.)"
)

HOCBA_PROCEDURE_CONTEXT = (
    "QUY TRÌNH XÉT TUYỂN HỌC BẠ TẠI HUSC 2026:\n\n"
    "Bước 1 (15/2 - 30/4/2026): Đăng ký trực tuyến tại https://tuyensinh.husc.edu.vn. Tạo tài khoản, điền thông tin cá nhân, chọn ngành, tổ hợp.\n"
    "Bước 2 (1/3 - 10/5/2026): Nộp hồ sơ gồm: (a) Học bạ THPT photo công chứng; (b) Bằng tốt nghiệp THPT photo (nếu tốt nghiệp 2025 trở về trước); (c) CCCD photo; (d) Phiếu đăng ký theo mẫu HUSC (in từ hệ thống).\n"
    "Bước 3 (15/5 - 30/5/2026): HUSC xét duyệt hồ sơ theo tổ hợp. Công thức: Điểm xét tuyển = Tổng 3 môn tổ hợp (lấy điểm cả năm lớp 12, không lấy điểm học kỳ).\n"
    "Bước 4 (5/6 - 15/6/2026): Công bố danh sách trúng tuyển đợt 1 trên website và qua email.\n"
    "Bước 5 (20/6 - 30/6/2026): Thí sinh xác nhận nhập học trực tuyến, nộp giấy tờ gốc đối chiếu.\n"
    "Bước 6 (1/7 - 15/7/2026): Nhập học, đóng học phí học kỳ 1, lấy thẻ sinh viên.\n"
    "Hồ sơ bổ sung (nếu thiếu): Giấy chứng nhận ưu tiên, giải HSG cấp tỉnh/quốc gia (nếu có), đơn xin xét đặc cách.\n"
    "Thời gian xét tuyển các đợt bổ sung (nếu còn chỉ tiêu): đợt 2 tháng 7, đợt 3 tháng 8."
)

SHORT_FACT_CONTEXTS = {
    "diem_chuan_cntt": "Điểm chuẩn ngành Khoa học Máy tính (7480101) HUSC năm 2025 theo tổ hợp A00 là 24.50; tổ hợp D07 là 22.80. Năm 2026 chưa công bố.",
    "hoc_phi_vh": "Học phí ngành Ngôn ngữ Anh (khối Ngoại ngữ) HUSC năm 2026: 16.500.000 VNĐ/năm học.",
    "tohop_khdl": "Ngành Khoa học Dữ liệu (7480106) HUSC xét tuyển bằng các tổ hợp: A00 (Toán-Lý-Hóa), A01 (Toán-Lý-Anh), B00 (Toán-Hóa-Sinh), D07 (Toán-Hóa-Anh).",
    "chi_tieu": "Chỉ tiêu tuyển sinh đại học chính quy HUSC năm 2026 dự kiến: khoảng 2.400 sinh viên cho tất cả 28 ngành. Trong đó CNTT (khối 748) chiếm khoảng 35%.",
    "hoc_phi_cntt": "Học phí ngành Khoa học Máy tính HUSC năm 2026: 18.500.000 VNĐ/năm học. Áp dụng chung cho các ngành CNTT (khối 748).",
    "diem_chuan_ktpm": "Điểm chuẩn ngành Kỹ thuật Phần mềm (7480103) HUSC năm 2025 theo A00 là 23.75; theo D07 là 22.10. Năm 2026 chưa công bố.",
}


# ---- Test cases ----
# (label, question, context, is_program_list_query, expected_branch)
TEST_CASES: List[Tuple[str, str, str, bool, str]] = [
    # ENUM path (uses 8000 budget)
    ("[ENUM] Liet ke 28 nganh",
     "Liệt kê tất cả các ngành đào tạo của trường Đại học Khoa học - ĐH Huế năm 2026.",
     PROGRAM_LIST_CONTEXT, True, "ENUM"),
    # Long comparison (should use 800 budget — comparison, not enumeration)
    ("[LONG] So sanh CNTT vs KTPM",
     "So sánh ngành Khoa học Máy tính và Kỹ thuật phần mềm của HUSC về điểm chuẩn, học phí, tổ hợp xét tuyển, cơ hội việc làm và mức lương khởi điểm.",
     COMPARE_CONTEXT, False, "STD"),
    # Long tuition table
    ("[LONG] Bang hoc phi",
     "Cho tôi xem học phí tất cả các ngành đào tạo năm 2026 của HUSC theo từng khối ngành.",
     TUITION_TABLE_CONTEXT, False, "STD"),
    # Long procedure
    ("[LONG] Quy trinh hoc ba",
     "Mô tả chi tiết quy trình xét tuyển học bạ tại HUSC năm 2026, từng bước một, bao gồm thời gian và hồ sơ cần thiết.",
     HOCBA_PROCEDURE_CONTEXT, False, "STD"),
    # Normal short queries
    ("[SHORT] Diem chuan CNTT",
     "Điểm chuẩn ngành Khoa học Máy tính HUSC năm 2025 là bao nhiêu?",
     SHORT_FACT_CONTEXTS["diem_chuan_cntt"], False, "STD"),
    ("[SHORT] Hoc phi Van hoc",
     "Học phí ngành Ngôn ngữ Anh HUSC năm 2026 là bao nhiêu?",
     SHORT_FACT_CONTEXTS["hoc_phi_vh"], False, "STD"),
    ("[SHORT] Tohop KHDL",
     "Ngành Khoa học Dữ liệu HUSC xét tuyển bằng những tổ hợp nào?",
     SHORT_FACT_CONTEXTS["tohop_khdl"], False, "STD"),
    ("[SHORT] Chi tieu",
     "Chỉ tiêu tuyển sinh HUSC năm 2026 là bao nhiêu?",
     SHORT_FACT_CONTEXTS["chi_tieu"], False, "STD"),
    ("[SHORT] Hoc phi CNTT",
     "Học phí ngành Khoa học Máy tính HUSC năm 2026 là bao nhiêu?",
     SHORT_FACT_CONTEXTS["hoc_phi_cntt"], False, "STD"),
    ("[SHORT] Diem chuan KTPM",
     "Điểm chuẩn ngành Kỹ thuật Phần mềm HUSC năm 2025 là bao nhiêu?",
     SHORT_FACT_CONTEXTS["diem_chuan_ktpm"], False, "STD"),
    # Long answer with many subparts but still NORMAL branch
    ("[LONG] Tong quan 5 nganh hot",
     "Hãy giới thiệu tổng quan 5 ngành hot nhất tại HUSC: CNTT, Kế toán, Luật, Ngôn ngữ Anh, Kiến trúc — mỗi ngành gồm: điểm chuẩn 2025, học phí 2026, cơ hội việc làm, và một gợi ý phù hợp với thí sinh nào.",
     TUITION_TABLE_CONTEXT + "\n\n" + PROGRAM_LIST_CONTEXT, False, "STD"),
    # Bonus: ensure ENUM uses ENUM budget for "so sánh" via list-like
    ("[ENUM] Liet ke nganh khoi CNTT",
     "Liệt kê chi tiết các ngành thuộc khối Công nghệ Thông tin (CNTT) mà HUSC đang đào tạo năm 2026.",
     PROGRAM_LIST_CONTEXT, True, "ENUM"),
]


# Queries we'll also re-run at max_tokens=1200 to diff against 800
DIFF_1200_QUERIES = {
    "[LONG] So sanh CNTT vs KTPM",
    "[LONG] Quy trinh hoc ba",
    "[LONG] Tong quan 5 nganh hot",
}


def _redact_key(k: str) -> str:
    if not k or len(k) < 8:
        return "***"
    return k[:4] + "..." + k[-4:]


async def call_ramclouds(
    user_message: str,
    system_message: str,
    max_tokens: int,
    temperature: float = 0.1,
    timeout_s: float = 45.0,
) -> Dict[str, Any]:
    """Single non-streamed call so we can read finish_reason deterministically.

    Mirrors what llm_client._call_provider does but skips the stream wrapping.
    """
    headers = {
        "Authorization": f"Bearer {RAMCLOUDS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RAMCLOUDS_GEN_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{RAMCLOUDS_BASE_URL.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        t0 = time.time()
        resp = await client.post(url, headers=headers, json=payload)
        elapsed = time.time() - t0
    if resp.status_code >= 400:
        body = resp.text
        return {
            "ok": False,
            "status": resp.status_code,
            "body": body[:500],
            "elapsed_s": round(elapsed, 2),
        }
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "status": resp.status_code,
        "elapsed_s": round(elapsed, 2),
        "finish_reason": choice.get("finish_reason"),
        "content": choice.get("message", {}).get("content", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def est_tokens(text: str) -> int:
    """Rough char-based token estimate (1 token ~ 3 chars for mixed VI)."""
    return max(1, len(text) // 3)


def looks_truncated_mid_sentence(text: str) -> bool:
    """Heuristic: ends without terminal punctuation OR with dangling article."""
    if not text:
        return False
    t = text.rstrip()
    if not t:
        return False
    last = t[-1]
    if last in ".!?\"'”’\n":
        return False
    # dangling list/csv
    if t.endswith((",", ";", ":", "-", "—", "và", "và ")):
        return True
    return True  # any non-terminal char ending


def truncated_marker(finish_reason: Optional[str], text: str) -> bool:
    """Deterministic truncation signal: finish_reason == 'length'."""
    return (finish_reason == "length") or (
        finish_reason is None and looks_truncated_mid_sentence(text)
    )


async def main():
    print("=" * 80)
    print("MAX_GEN_TOKENS TRUNCATION CHECK")
    print("=" * 80)
    print(f"API key (redacted):  {_redact_key(RAMCLOUDS_API_KEY)}")
    print(f"Base URL:            {RAMCLOUDS_BASE_URL}")
    print(f"Model:               {RAMCLOUDS_GEN_MODEL}")
    print(f"MAX_GEN_TOKENS:      {MAX_GEN_TOKENS}  (normal answers)")
    print(f"MAX_GEN_TOKENS_ENUM: {MAX_GEN_TOKENS_ENUM}  (program list answers)")
    print(f"Test cases:          {len(TEST_CASES)}")
    print()

    # ---- Pass 1: all queries at the configured budget ----
    results: Dict[str, Dict[str, Any]] = {}
    for i, (label, question, context, is_enum, branch) in enumerate(TEST_CASES, 1):
        budget = MAX_GEN_TOKENS_ENUM if is_enum else MAX_GEN_TOKENS
        user_msg = (
            f"CONTEXT:\n{context}\n\n---\n\nCÂU HỎI: {question}\n\n"
            "Hãy trả lời dựa trên context trên. Sử dụng số liệu cụ thể từ context."
        )
        print(f"[{i}/{len(TEST_CASES)}] {label}  (budget={budget}, branch={branch})")
        r = await call_ramclouds(user_msg, GEN_SYSTEM_PROMPT, max_tokens=budget)
        if not r["ok"]:
            print(f"  ERROR status={r['status']} body={r['body']}")
            results[label] = {"error": r, "question": question, "context": context,
                              "is_enum": is_enum, "branch": branch, "budget": budget}
            continue
        text = r["content"] or ""
        results[label] = {
            "question": question, "context": context,
            "is_enum": is_enum, "branch": branch, "budget": budget,
            "finish_reason": r["finish_reason"],
            "elapsed_s": r["elapsed_s"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
            "chars": len(text),
            "est_tokens": est_tokens(text),
            "text": text,
            "truncated_finish_reason": r["finish_reason"] == "length",
            "truncated_heuristic": looks_truncated_mid_sentence(text),
        }
        flag = "TRUNC" if results[label]["truncated_finish_reason"] else "ok"
        print(f"  finish_reason={r['finish_reason']!r}  "
              f"chars={len(text)}  est_tokens={est_tokens(text)}  "
              f"elapsed={r['elapsed_s']}s  [{flag}]")
    print()

    # ---- Pass 2: diff 800 vs 1200 for longest non-enum queries ----
    diff_results: Dict[str, Dict[str, Any]] = {}
    for label in DIFF_1200_QUERIES:
        if label not in results or "error" in results[label]:
            continue
        rec = results[label]
        if rec["is_enum"]:
            continue  # ENUM branch uses 8000 budget, not 800 vs 1200
        user_msg = (
            f"CONTEXT:\n{rec['context']}\n\n---\n\nCÂU HỎI: {rec['question']}\n\n"
            "Hãy trả lời dựa trên context trên. Sử dụng số liệu cụ thể từ context."
        )
        print(f"[DIFF 1200] {label}")
        r1200 = await call_ramclouds(user_msg, GEN_SYSTEM_PROMPT, max_tokens=1200)
        if not r1200["ok"]:
            print(f"  ERROR status={r1200['status']} body={r1200['body']}")
            continue
        text1200 = r1200["content"] or ""
        diff_results[label] = {
            "at_1200": {
                "finish_reason": r1200["finish_reason"],
                "chars": len(text1200),
                "est_tokens": est_tokens(text1200),
                "truncated": r1200["finish_reason"] == "length",
                "text": text1200,
            },
            "at_800": {
                "finish_reason": rec["finish_reason"],
                "chars": rec["chars"],
                "est_tokens": rec["est_tokens"],
                "truncated": rec["truncated_finish_reason"],
                "text": rec["text"],
            },
        }
        flag = "TRUNC" if diff_results[label]["at_1200"]["truncated"] else "ok"
        print(f"  finish_reason@1200={r1200['finish_reason']!r}  "
              f"chars={len(text1200)}  est_tokens={est_tokens(text1200)}  [{flag}]")
    print()

    # ---- Render truncation table ----
    print("=" * 80)
    print("PER-QUERY TRUNCATION TABLE (configured-budget pass)")
    print("=" * 80)
    header = f"{'query':<40} {'branch':<6} {'budget':<7} {'finish':<8} {'chars':<7} {'~tok':<6} {'TRUNC?':<7}"
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        if "error" in r:
            print(f"{label:<40} ERROR  status={r['error'].get('status')}")
            continue
        flag = "YES" if r["truncated_finish_reason"] else (
            "he" if r["truncated_heuristic"] else "no")
        print(f"{label:<40} {r['branch']:<6} {r['budget']:<7} "
              f"{str(r['finish_reason']):<8} {r['chars']:<7} {r['est_tokens']:<6} {flag:<7}")
    print()

    # ---- Render 800 vs 1200 diff section ----
    print("=" * 80)
    print("800 vs 1200 BUDGET DIFF (last 200 chars each side)")
    print("=" * 80)
    for label, d in diff_results.items():
        a, b = d["at_800"], d["at_1200"]
        print(f"\n--- {label} ---")
        print(f"  @800  finish={a['finish_reason']!r}  chars={a['chars']}  est_tokens={a['est_tokens']}  "
              f"truncated={a['truncated']}")
        print(f"  @800  LAST 200 CHARS: ...{(a['text'] or '')[-200:]!r}")
        print(f"  @1200 finish={b['finish_reason']!r}  chars={b['chars']}  est_tokens={b['est_tokens']}  "
              f"truncated={b['truncated']}")
        print(f"  @1200 LAST 200 CHARS: ...{(b['text'] or '')[-200:]!r}")
        # length delta
        if b['chars'] and a['chars']:
            print(f"  delta chars(1200-800) = +{b['chars']-a['chars']}  delta tokens(est) = +{b['est_tokens']-a['est_tokens']}")

    # ---- Verdict ----
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    normal_truncated: List[str] = []
    enum_uses_enum_budget = True
    max_normal_est_tokens = 0
    for label, r in results.items():
        if "error" in r:
            continue
        if r["is_enum"]:
            if r["branch"] != "ENUM" or r["budget"] != MAX_GEN_TOKENS_ENUM:
                enum_uses_enum_budget = False
        else:
            if r["truncated_finish_reason"]:
                normal_truncated.append(label)
            max_normal_est_tokens = max(max_normal_est_tokens, r["est_tokens"])

    print(f"Normal (non-enum) answers truncated at 800 (finish_reason=length): {len(normal_truncated)}")
    for q in normal_truncated:
        print(f"  - {q}")
    print(f"Enumeration answers use the 8000 ENUM budget: {enum_uses_enum_budget}")
    print(f"Max est-tokens among normal answers: {max_normal_est_tokens}  (budget=800)")

    headroom = 800 - max_normal_est_tokens
    if len(normal_truncated) == 0 and headroom > 0:
        rec = f"800 SAFE — max normal answer ≈{max_normal_est_tokens} est-tokens, {headroom} headroom. No truncation observed."
    elif len(normal_truncated) == 0 and headroom <= 0:
        rec = f"800 MARGINAL — max normal answer ≈{max_normal_est_tokens} est-tokens. Heuristic-only, but no finish_reason=length. Consider raising to 1000 for safety."
    else:
        rec = f"800 UNSAFE — {len(normal_truncated)} normal answer(s) truncated. Recommend raising MAX_GEN_TOKENS to at least 1500."

    print(f"RECOMMENDATION: {rec}")

    # write machine-readable results for the parent to parse
    out = {
        "config": {
            "max_gen_tokens": MAX_GEN_TOKENS,
            "max_gen_tokens_enum": MAX_GEN_TOKENS_ENUM,
            "model": RAMCLOUDS_GEN_MODEL,
            "base_url": RAMCLOUDS_BASE_URL,
            "api_key_redacted": _redact_key(RAMCLOUDS_API_KEY),
        },
        "results": {
            label: ({
                "branch": r["branch"],
                "budget": r["budget"],
                "finish_reason": r["finish_reason"],
                "chars": r["chars"],
                "est_tokens": r["est_tokens"],
                "elapsed_s": r["elapsed_s"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "truncated_finish_reason": r["truncated_finish_reason"],
                "truncated_heuristic": r["truncated_heuristic"],
                "text": r["text"],
            } if "error" not in r else {"error": r["error"]})
            for label, r in results.items()
        },
        "diff_800_vs_1200": diff_results,
        "verdict": {
            "normal_truncated_count": len(normal_truncated),
            "normal_truncated_labels": normal_truncated,
            "enum_uses_enum_budget": enum_uses_enum_budget,
            "max_normal_est_tokens": max_normal_est_tokens,
            "recommendation": rec,
        },
    }
    out_path = REPO_ROOT / "scripts" / "check_gen_token_truncation_results.json"
    # The text/field may contain non-ASCII; ensure UTF-8
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMachine-readable results: {out_path}")


if __name__ == "__main__":
    logger.remove()
    asyncio.run(main())
