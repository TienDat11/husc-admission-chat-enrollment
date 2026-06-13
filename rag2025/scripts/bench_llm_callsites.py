"""LLM call-site latency benchmark — REAL gateway calls to ramclouds.me.

Measures wall-clock latency of EVERY LLM round-trip on the /v2 hot path,
using the REAL UnifiedLLMClient. Reports p50/p95/max per call site +
success/timeout/error counts + raw model id.

REDACT RAMCLOUDS_API_KEY in all output. Do NOT edit production source.

Usage:
    cd rag2025 && D:/miniconda3/python.exe scripts/bench_llm_callsites.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup so we can import src.services.* like the rest of the project.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
for _p in (str(_PROJECT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env BEFORE importing anything that reads env vars.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_PROJECT / ".env")
except Exception:
    pass

from src.services.llm_client import UnifiedLLMClient, get_llm_client  # noqa: E402

# Reduce httpx + openai noise during the bench.
os.environ.setdefault("LOGURU_LEVEL", "WARNING")

# ─── Redaction helpers ──────────────────────────────────────────────────────

_KEY_RX = re.compile(r"sk-[A-Za-z0-9_-]{6,}")
_KEY_TAIL_RX = re.compile(r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+")


def _redact(text: str) -> str:
    """Replace any 'sk-...' style key in text with 'sk-***REDACTED***'."""
    if not text:
        return text
    text = _KEY_TAIL_RX.sub(r"\1***REDACTED***", text)
    text = _KEY_RX.sub("sk-***REDACTED***", text)
    return text


def _redact_env() -> Dict[str, str]:
    """Return current model env (key redacted)."""
    keys = [
        "RAMCLOUDS_API_KEY",
        "RAMCLOUDS_BASE_URL",
        "RAMCLOUDS_MODEL",
        "RAMCLOUDS_NER_MODEL",
        "RAMCLOUDS_HYDE_MODEL",
        "RAMCLOUDS_GEN_MODEL",
        "RAMCLOUDS_GEN_FALLBACK_MODEL",
        "RAMCLOUDS_STREAM",
        "MAX_GEN_TOKENS",
        "MAX_GEN_TOKENS_ENUM",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        v = os.getenv(k, "<unset>")
        if "API_KEY" in k and v and v != "<unset>":
            tail = v[-4:] if len(v) >= 4 else "****"
            out[k] = f"<redacted:...{tail}>"
        else:
            out[k] = v
    return out


# ─── Latency stats ──────────────────────────────────────────────────────────


def _stats(samples_ms: List[float]) -> Dict[str, Any]:
    if not samples_ms:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None, "min": None}
    s = sorted(samples_ms)
    n = len(s)

    def _pct(p: float) -> float:
        # Nearest-rank percentile.
        if n == 1:
            return s[0]
        idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        return s[idx]

    return {
        "n": n,
        "p50": _pct(50),
        "p95": _pct(95),
        "max": s[-1],
        "mean": round(statistics.fmean(s), 1),
        "min": s[0],
    }


# ─── Per-call-site runners ──────────────────────────────────────────────────

# Match the prompts in production so the benchmark is faithful.
_CLASSIFY_STEPBACK_SYSTEM = """Bạn là hệ thống phân loại câu hỏi tuyển sinh đại học Việt Nam (HUSC - Đại học Khoa học Huế).

Phân tích câu hỏi và trả về JSON DUY NHẤT với cấu trúc:
{
  "route": "padded" | "hybrid" | "graph",
  "complexity": 1-5,
  "intent": "diem_chuan" | "hoc_phi" | "nganh_hoc" | "to_hop" | "chinh_sach" | "thu_tuc" | "so_sanh" | "da_hop" | "liet_ke" | "general",
  "reasoning": "lý do ngắn gọn (≤20 từ)",
  "step_back": "câu hỏi đã được trừu tượng hóa về nguyên lý/khái niệm nền tảng (tiếng Việt). Nếu câu hỏi đã là single-fact lookup cụ thể (đã có NGÀNH + NĂM ràng buộc), echo nguyên câu hỏi."
}

QUY TẮC ROUTE (theo thứ tự ưu tiên):
1) SO SÁNH / ĐA-THỰC-THỂ / QUAN HỆ / LIỆT KÊ nhiều ngành → route = "graph", complexity = 4-5
2) TRA CỨU 1 THỰC THỂ + 1 THUỘC TÍNH (1-hop đơn giản) → route = "padded", complexity = 1
3) MỌI TRƯỜNG HỢP KHÁC → route = "hybrid", complexity = 2-3

CHỈ trả về JSON hợp lệ, không có text bên ngoài. KHÔNG bịa thêm field ngoài schema."""


_NER_SYSTEM = """Bạn là hệ thống trích xuất thông tin cấu trúc từ văn bản tuyển sinh đại học Việt Nam.

Nhiệm vụ: Từ đoạn văn bản được cung cấp, trích xuất:
1. **Thực thể** (entities): tên ngành, mã ngành, tổ hợp môn, điểm chuẩn, học phí.
2. **Quan hệ** (triples): bộ ba (thực thể_đầu, quan_hệ, thực thể_cuối).

Loại thực thể hợp lệ: NGANH, TO_HOP, DIEM_CHUAN, HOC_PHI, THOI_GIAN, TO_CHUC, CHINH_SACH, UNKNOWN
Loại quan hệ hợp lệ: CO_TO_HOP, CO_DIEM, THUOC_TRUONG, YEU_CAU, LIEN_QUAN

QUY TẮC BẮT BUỘC:
- Chỉ trả về JSON hợp lệ, không có markdown, không có văn bản bên ngoài.
- "normalized": chuỗi viết thường, không dấu, dùng dấu gạch dưới thay khoảng trắng.
- Nếu không tìm thấy: {"entities": [], "triples": []}.

CHỈ trả về JSON hợp lệ."""


_GEN_SYSTEM = (
    "Bạn là chuyên gia tư vấn tuyển sinh Trường Đại học Khoa học - Đại học Huế (HUSC). "
    "Trả lời câu hỏi DỰA HOÀN TOÀN trên CONTEXT được cung cấp. "
    "KHÔNG bịa số liệu, mã ngành, học phí, điểm chuẩn nằm ngoài context. "
    "Toàn bộ tiếng Việt, không trộn tiếng Anh. "
    "Khi KHÔNG có thông tin trong context: "
    "\"Tôi không tìm thấy thông tin này trong tài liệu hiện có.\""
)


# A small but realistic retrieved context (5 fake chunks) for the gen bench.
_FAKE_CHUNKS = [
    {
        "text": "Ngành Công nghệ thông tin (mã ngành 7480201) xét tuyển các tổ hợp A00, A01, A02, B00 năm 2026. "
        "Chỉ tiêu: 120. Điểm chuẩn năm 2025: 22.5 (A00). Học phí: 1.200.000 VNĐ/tín chỉ.",
        "summary": "CNTT 7480201 — tổ hợp A00/A01/A02/B00, chỉ tiêu 120, học phí 1.2tr/tín chỉ.",
        "chunk_id": "husc_nganh_7480201",
        "metadata": {
            "data_year": "2026",
            "notification_id": 12,
            "source": "tuyensinh.husc.edu.vn",
            "info_type": "nganh_hoc",
        },
    },
    {
        "text": "Ngành Kỹ thuật phần mềm (mã ngành 7480103) tổ hợp A00, A01, A02, B00, D01. "
        "Chỉ tiêu 80. Điểm chuẩn 2025: 23.0. Học phí 1.200.000 VNĐ/tín chỉ.",
        "summary": "KTPM 7480103 — tổ hợp A00/A01/A02/B00/D01, chỉ tiêu 80.",
        "chunk_id": "husc_nganh_7480103",
        "metadata": {"data_year": "2026", "notification_id": 12,
                     "source": "tuyensinh.husc.edu.vn", "info_type": "nganh_hoc"},
    },
    {
        "text": "Học phí các ngành HUSC năm 2026 áp dụng chung: 1.200.000 VNĐ/tín chỉ đối với bậc đại học chính quy. "
        "Một năm học trung bình 40 tín chỉ, tổng học phí khoảng 48.000.000 VNĐ.",
        "summary": "Học phí chung 1.2tr/tín chỉ, ~48tr/năm học.",
        "chunk_id": "hocphi_2026_tong_hop",
        "metadata": {"data_year": "2026", "notification_id": 15,
                     "source": "tuyensinh.husc.edu.vn", "info_type": "hoc_phi"},
    },
    {
        "text": "Phương thức xét tuyển năm 2026: (1) THPT 2026 (chiếm 70% chỉ tiêu), "
        "(2) Học bạ THPT (20%), (3) ĐGNL ĐHQG (10%). Mỗi thí sinh được đăng ký tối đa 5 nguyện vọng.",
        "summary": "PTXT 2026: THPT 70%, HB 20%, ĐGNL 10%, max 5 NV.",
        "chunk_id": "phuongthuc_2026",
        "metadata": {"data_year": "2026", "notification_id": 10,
                     "source": "tuyensinh.husc.edu.vn", "info_type": "phuong_thuc"},
    },
    {
        "text": "Thời gian xét tuyển: nhận hồ sơ từ 01/6/2026 đến 30/6/2026. Công bố kết quả đợt 1: 15/8/2026. "
        "Lệ phí xét tuyển: 30.000 VNĐ/nguyện vọng.",
        "summary": "Thời gian XT 1/6-30/6, công bố 15/8, lệ phí 30k/NV.",
        "chunk_id": "thutuc_2026",
        "metadata": {"data_year": "2026", "notification_id": 11,
                     "source": "tuyensinh.husc.edu.vn", "info_type": "thu_tuc"},
    },
]


def _build_gen_user_msg(query: str, chunks: List[Dict[str, Any]]) -> str:
    parts = ["CONTEXT:"]
    for i, c in enumerate(chunks, 1):
        text = c.get("text", "") or c.get("summary", "")
        parts.append(f"[Đoạn {i}] {text}")
    parts.append("\n---\n")
    parts.append(f"CÂU HỎI: {query}\n")
    parts.append("Hãy trả lời dựa trên context trên. Sử dụng số liệu cụ thể từ context.")
    return "\n".join(parts)


async def _time_call(fn) -> Tuple[float, Any, Optional[BaseException]]:
    """Run an async fn, return (wall_ms, result, exc)."""
    t0 = time.perf_counter()
    try:
        r = await fn()
        return (time.perf_counter() - t0) * 1000.0, r, None
    except BaseException as exc:  # noqa: BLE001
        return (time.perf_counter() - t0) * 1000.0, None, exc


# ─── 1. Router classify ─────────────────────────────────────────────────────


async def bench_router_classify(
    client: UnifiedLLMClient, model_id: str, runs: int
) -> Dict[str, Any]:
    """chat_json call: _CLASSIFY_STEPBACK_SYSTEM + sample query.
    Effective model on hot path = RAMCLOUDS_HYDE_MODEL (router uses default
    client which reads RAMCLOUDS_MODEL)."""
    queries = [
        "học phí ngành CNTT 2026?",                                # 1-fact
        "cách xét tuyển bằng học bạ và thời gian nộp hồ sơ?",     # policy
        "so sánh ngành CNTT và ngành Khoa học dữ liệu?",           # comparison
    ]

    latencies: List[float] = []
    successes = 0
    timeouts = 0
    errors = 0
    json_valid = 0
    raw_errors: List[str] = []

    for q in queries:
        for r in range(runs):
            ms, resp, exc = await _time_call(
                lambda q=q: client.chat_json(
                    user_message=f"Câu hỏi gốc: {q}",
                    system_message=_CLASSIFY_STEPBACK_SYSTEM,
                    temperature=0.1,
                    max_tokens=512,
                )
            )
            latencies.append(ms)
            if exc is None:
                successes += 1
                if isinstance(resp, dict) and "route" in resp:
                    json_valid += 1
            elif "Timeout" in type(exc).__name__ or "timeout" in str(exc).lower():
                timeouts += 1
            else:
                errors += 1
                raw_errors.append(f"{type(exc).__name__}: {_redact(str(exc))[:200]}")
            # small jitter so we don't hit the gateway in lockstep
            await asyncio.sleep(0.1)

    return {
        "call_site": "router_classify",
        "model": model_id,
        "system_prompt_chars": len(_CLASSIFY_STEPBACK_SYSTEM),
        "max_tokens": 512,
        "temperature": 0.1,
        "n_queries": len(queries),
        "runs_per_query": runs,
        "total_attempts": len(queries) * runs,
        "success": successes,
        "json_valid": json_valid,
        "timeout": timeouts,
        "error": errors,
        "latency_ms": _stats(latencies),
        "sample_errors": raw_errors[:5],
    }


# ─── 2. Generation (with TTFT for SSE) ──────────────────────────────────────


async def _gen_with_ttft(
    client: UnifiedLLMClient, user_msg: str, system: str, max_tokens: int
) -> Tuple[float, Optional[float], str, Optional[BaseException]]:
    """Measure TTFT + total for a streaming call (always, since RAMCLOUDS_STREAM=true).

    Returns (total_ms, ttft_ms_or_None, content, exc).
    """
    # We call the public `chat` (which goes through _call_provider's SSE path
    # when RAMCLOUDS_STREAM is on). To measure TTFT we patch the underlying
    # OpenAI client call with a one-shot wrapper that records when the first
    # delta arrives. Easier path: replicate the SSE collection inline and
    # time the first chunk ourselves.
    import httpx
    from openai import AsyncOpenAI
    # Build a client using the same provider config as the real one.
    prov = client._providers[0]
    oa = AsyncOpenAI(api_key=prov.api_key, base_url=prov.base_url)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    kwargs: Dict[str, Any] = {
        "model": prov.model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    chunks: List[str] = []
    finish_reason = None
    exc: Optional[BaseException] = None
    try:
        stream = await oa.chat.completions.create(**kwargs)
        async for ev in stream:
            if not ev.choices:
                continue
            choice = ev.choices[0]
            delta = getattr(choice, "delta", None)
            piece = getattr(delta, "content", None) if delta else None
            if piece:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                chunks.append(piece)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
    except BaseException as e:  # noqa: BLE001
        exc = e
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, ttft, "".join(chunks), exc


async def _gen_non_stream(
    client: UnifiedLLMClient, user_msg: str, system: str, max_tokens: int
) -> Tuple[float, str, Optional[BaseException]]:
    """One non-stream gen call. Reuses the same provider config."""
    import httpx
    from openai import AsyncOpenAI
    prov = client._providers[0]
    oa = AsyncOpenAI(api_key=prov.api_key, base_url=prov.base_url)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    t0 = time.perf_counter()
    chunks: List[str] = []
    finish_reason = None
    exc: Optional[BaseException] = None
    try:
        stream = await oa.chat.completions.create(
            model=prov.model, messages=messages, temperature=0.1,
            max_tokens=max_tokens, stream=False,
        )
        if stream.choices:
            chunks.append(stream.choices[0].message.content or "")
            finish_reason = stream.choices[0].finish_reason
    except BaseException as e:  # noqa: BLE001
        exc = e
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, "".join(chunks), exc


async def bench_generation(
    client: UnifiedLLMClient, model_id: str, runs: int
) -> Dict[str, Any]:
    """Answer-gen: realistic 5-chunk prompt at MAX_GEN_TOKENS=1200.
    Measure SSE (TTFT+total) and non-SSE (total) variants."""
    query = "Học phí ngành Công nghệ thông tin năm 2026 tại HUSC là bao nhiêu?"
    user_msg = _build_gen_user_msg(query, _FAKE_CHUNKS)
    max_tokens = int(os.getenv("MAX_GEN_TOKENS", "1200"))

    # SSE (matches RAMCLOUDS_STREAM=true default in production)
    sse_total: List[float] = []
    sse_ttft: List[float] = []
    sse_ok = 0
    sse_err = 0
    sse_errs: List[str] = []
    for _ in range(runs):
        total, ttft, content, exc = await _gen_with_ttft(
            client, user_msg, _GEN_SYSTEM, max_tokens
        )
        sse_total.append(total)
        if ttft is not None:
            sse_ttft.append(ttft)
        if exc is None and content:
            sse_ok += 1
        else:
            sse_err += 1
            if exc is not None:
                sse_errs.append(f"{type(exc).__name__}: {_redact(str(exc))[:200]}")
        await asyncio.sleep(0.1)

    # Non-SSE: temporarily flip RAMCLOUDS_STREAM off via env (we replicate the
    # call inline so we don't have to mutate the singleton's behavior).
    nons_total: List[float] = []
    nons_ok = 0
    nons_err = 0
    nons_errs: List[str] = []
    for _ in range(runs):
        total, content, exc = await _gen_non_stream(
            client, user_msg, _GEN_SYSTEM, max_tokens
        )
        nons_total.append(total)
        if exc is None and content:
            nons_ok += 1
        else:
            nons_err += 1
            if exc is not None:
                nons_errs.append(f"{type(exc).__name__}: {_redact(str(exc))[:200]}")
        await asyncio.sleep(0.1)

    return {
        "call_site": "answer_generation",
        "model": model_id,
        "prompt_chars": len(_GEN_SYSTEM) + len(user_msg),
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "runs": runs,
        "sse": {
            "total_ms": _stats(sse_total),
            "ttft_ms": _stats(sse_ttft),
            "success": sse_ok,
            "error": sse_err,
            "sample_errors": sse_errs[:5],
        },
        "non_sse": {
            "total_ms": _stats(nons_total),
            "success": nons_ok,
            "error": nons_err,
            "sample_errors": nons_errs[:5],
        },
    }


# ─── 3. NER fallback (gpt-5.5-chat) ────────────────────────────────────────


async def bench_ner(
    client: UnifiedLLMClient, model_id: str, runs: int
) -> Dict[str, Any]:
    """chat_json call with _NER_SYSTEM_PROMPT. Uses a small Vietnamese
    admission paragraph as input. The NER ladder on the graph hot path
    fires for graph_rag routes; for padded/hybrid it is gated by the
    build_graph step, not /v2 query."""
    sample_text = (
        "Ngành Công nghệ thông tin (mã ngành 7480201) xét tuyển tổ hợp A00, A01. "
        "Điểm chuẩn năm 2025 là 22.5. Học phí 1.200.000 VNĐ/tín chỉ. "
        "Ngành Kỹ thuật phần mềm (mã ngành 7480103) tổ hợp A00."
    )
    latencies: List[float] = []
    successes = 0
    timeouts = 0
    errors = 0
    json_valid = 0
    raw_errors: List[str] = []
    for _ in range(runs):
        ms, resp, exc = await _time_call(
            lambda: client.chat_json(
                user_message=sample_text,
                system_message=_NER_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
        )
        latencies.append(ms)
        if exc is None:
            successes += 1
            if isinstance(resp, dict) and "entities" in resp:
                json_valid += 1
        elif "Timeout" in type(exc).__name__ or "timeout" in str(exc).lower():
            timeouts += 1
        else:
            errors += 1
            raw_errors.append(f"{type(exc).__name__}: {_redact(str(exc))[:200]}")
        await asyncio.sleep(0.1)

    return {
        "call_site": "ner_fallback",
        "model": model_id,
        "input_chars": len(sample_text),
        "max_tokens": 1024,
        "temperature": 0.1,
        "runs": runs,
        "success": successes,
        "json_valid": json_valid,
        "timeout": timeouts,
        "error": errors,
        "latency_ms": _stats(latencies),
        "sample_errors": raw_errors[:5],
    }


# ─── 4. Fallback model (RAMCLOUDS_GEN_FALLBACK_MODEL) ──────────────────────


async def bench_fallback_model(
    primary_client: UnifiedLLMClient,
    fallback_model: str,
    runs: int,
) -> Dict[str, Any]:
    """Re-measure the same gen prompt but with the GEN_FALLBACK_MODEL."""
    fb_client = UnifiedLLMClient(force_model=fallback_model)
    query = "Học phí ngành Công nghệ thông tin năm 2026 tại HUSC là bao nhiêu?"
    user_msg = _build_gen_user_msg(query, _FAKE_CHUNKS)
    max_tokens = int(os.getenv("MAX_GEN_TOKENS", "1200"))

    total: List[float] = []
    ttft: List[float] = []
    ok = 0
    err = 0
    errs: List[str] = []
    for _ in range(runs):
        t, ft, content, exc = await _gen_with_ttft(
            fb_client, user_msg, _GEN_SYSTEM, max_tokens
        )
        total.append(t)
        if ft is not None:
            ttft.append(ft)
        if exc is None and content:
            ok += 1
        else:
            err += 1
            if exc is not None:
                errs.append(f"{type(exc).__name__}: {_redact(str(exc))[:200]}")
        await asyncio.sleep(0.1)
    await fb_client.close()

    return {
        "call_site": "answer_generation_fallback",
        "model": fallback_model,
        "prompt_chars": len(_GEN_SYSTEM) + len(user_msg),
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "runs": runs,
        "sse": {
            "total_ms": _stats(total),
            "ttft_ms": _stats(ttft),
            "success": ok,
            "error": err,
            "sample_errors": errs[:5],
        },
    }


# ─── Main ───────────────────────────────────────────────────────────────────


async def main() -> None:
    runs = int(os.getenv("BENCH_RUNS", "5"))
    print("=" * 80)
    print("LLM CALL-SITE BENCHMARK — REAL ramclouds.me calls")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}  cwd: {os.getcwd()}")
    print("ENV (redacted):")
    for k, v in _redact_env().items():
        print(f"  {k}={v}")
    print()

    if not os.getenv("RAMCLOUDS_API_KEY"):
        print("FATAL: RAMCLOUDS_API_KEY not set in .env — aborting")
        sys.exit(2)

    # Build a singleton client that the production /v2 hot path uses
    # (so the production provider chain is exercised, not a synthetic one).
    client = get_llm_client()
    if not client._providers:
        print("FATAL: UnifiedLLMClient has no providers configured")
        sys.exit(2)

    primary_model = client._providers[0].model
    print(f"Primary model from RAMCLOUDS_MODEL = {primary_model!r}")
    print(f"Runs per call site = {runs}  (override via BENCH_RUNS env)")
    print()

    results: List[Dict[str, Any]] = []

    # 1. Router classify
    print("[1/4] router_classify (RAMCLOUDS_HYDE_MODEL / primary) ...")
    results.append(await bench_router_classify(client, primary_model, runs))
    print(f"  -> {_redact(json.dumps(results[-1], ensure_ascii=False))[:300]}...")

    # 2. Generation
    print("\n[2/4] answer_generation (RAMCLOUDS_GEN_MODEL) ...")
    results.append(await bench_generation(client, primary_model, runs))
    sse = results[-1]["sse"]
    nons = results[-1]["non_sse"]
    print(f"  SSE  total p50={sse['total_ms'].get('p50')}ms "
          f"ttft p50={sse['ttft_ms'].get('p50')}ms ok={sse['success']}/{runs} err={sse['error']}")
    print(f"  nonS total p50={nons['total_ms'].get('p50')}ms ok={nons['success']}/{runs} err={nons['error']}")

    # 3. NER fallback (gpt-5.5-chat)
    ner_model = os.getenv("RAMCLOUDS_NER_MODEL", "deepseek-v4-flash")
    print(f"\n[3/4] ner_fallback (RAMCLOUDS_NER_MODEL={ner_model}) ...")
    ner_client = UnifiedLLMClient(force_model=ner_model)
    results.append(await bench_ner(ner_client, ner_model, runs))
    print(f"  -> latency p50={results[-1]['latency_ms'].get('p50')}ms "
          f"ok={results[-1]['success']}/{runs} json_valid={results[-1]['json_valid']} "
          f"timeout={results[-1]['timeout']} err={results[-1]['error']}")
    await ner_client.close()

    # 4. Fallback model
    fb_model = os.getenv("RAMCLOUDS_GEN_FALLBACK_MODEL", "deepseek-v4-flash")
    print(f"\n[4/4] answer_generation_fallback (RAMCLOUDS_GEN_FALLBACK_MODEL={fb_model}) ...")
    results.append(await bench_fallback_model(client, fb_model, runs))
    sse = results[-1]["sse"]
    print(f"  -> SSE total p50={sse['total_ms'].get('p50')}ms "
          f"ttft p50={sse['ttft_ms'].get('p50')}ms ok={sse['success']}/{runs} err={sse['error']}")

    # ── Save raw report ──
    out_dir = _PROJECT / "results" / "llm_callsite_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs_per_site": runs,
        "env": _redact_env(),
        "results": results,
    }
    out_path.write_text(
        _redact(json.dumps(report, ensure_ascii=False, indent=2)),
        encoding="utf-8",
    )
    print(f"\nRaw report: {out_path}")

    await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
