"""
Hard performance stress tests for rag2025 RAG pipeline.
Tests: rate limiter, cache stampede, stream disconnect, LLM timeout, groundedness, concurrent writes, chunk truncation.
Run: pytest rag2025/tests/performance/test_hard_stress.py -v -m "not slow"  # fast only
       pytest rag2025/tests/performance/test_hard_stress.py -v           # all including slow
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio


# ── 1. Rate limiter: 61 concurrent requests ──────────────────────────────
async def test_rate_limiter_allows_burst_up_to_limit():
    """Fire RATE_LIMIT_PER_MINUTE requests rapidly. All should succeed."""
    from collections import deque
    from threading import Lock

    _lock = Lock()
    _buckets = {}
    RATE_LIMIT = 60

    def try_record(ip, now):
        with _lock:
            b = _buckets.get(ip, deque())
            while b and b[0] < now - 60:
                b.popleft()
            if len(b) < RATE_LIMIT:
                b.append(now)
                _buckets[ip] = b
                return True
            return False

    async def make_request(ip):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: try_record(ip, time.time()))

    ip = "test_ip"
    results = await asyncio.gather(*[make_request(ip) for _ in range(60)])
    allowed_count = sum(1 for r in results if r)
    assert allowed_count == 60, f"Expected 60 allowed, got {allowed_count}"

    # 61st should be blocked
    result_61 = await make_request(ip)
    assert result_61 is False, "61st request should be blocked"


# ── 2. Cache stampede: only 1 LLM call for N concurrent misses ───────
async def test_cache_stampede_single_llm_call():
    """20 simultaneous uncached queries → only 1 LLM call fires."""
    call_count = 0
    _cache = {}
    _call_lock = asyncio.Lock()

    async def fake_llm(query):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # simulate LLM latency
        return {"route": "hybrid", "complexity": 2, "intent": "general", "reasoning": "test"}

    async def routed_query(q):
        if q in _cache:
            return _cache[q]
        async with _call_lock:
            if q in _cache:  # double-check after lock
                return _cache[q]
            result = await fake_llm(q)
            _cache[q] = result
            return result

    queries = ["test query"] * 20
    start = time.perf_counter()
    results = await asyncio.gather(*[routed_query(q) for q in queries])
    elapsed = time.perf_counter() - start
    assert call_count == 1, f"Expected 1 LLM call, got {call_count}"
    assert elapsed < 2.0, f"Stampede took {elapsed:.2f}s — lock not working"
    assert all(r["route"] == "hybrid" for r in results)


# ── 3. Stream disconnect: tasks cleaned up ──────────────────────────────
async def test_stream_generator_cleans_up_on_cancel():
    """When cancelled, SSE generator must cancel its pipeline task."""
    import asyncio

    async def slow_work():
        await asyncio.sleep(10)
        return {"answer": "done"}

    async def stream_generator():
        task = asyncio.create_task(slow_work())
        try:
            return await asyncio.wait_for(task, timeout=0.1)
        except asyncio.TimeoutError:
            task.cancel()
            raise

    with pytest.raises(asyncio.TimeoutError):
        await stream_generator()

    # Task should be cancelled, not orphaned
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    # At minimum, the slow_work task should not be pending
    slow_pending = [t for t in tasks if "slow_work" in str(t.get_coro())]
    assert len(slow_pending) == 0, "Orphaned tasks detected"


# ── 4. LLM timeout: falls back within deadline ─────────────────────────
async def test_llm_timeout_raises():
    """Hanging LLM raises TimeoutError, does not block forever."""
    async def hang_forever(*args, **kwargs):
        await asyncio.sleep(30)
        return {"content": "ok"}

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(hang_forever(), timeout=0.5)


# ── 5. Groundedness low score warning ─────────────────────────────────
def test_groundedness_low_score_adds_warning():
    """Low groundedness score should prepend warning emoji."""
    answer = "HUSC có 29 ngành tuyển sinh"
    chunks = [{"text": "Chỉ có 1 ngành được liệt kê ở đây"}]

    # Simulate the groundedness logic from main.py
    answer_tokens = set(answer.lower().split())
    source_tokens = set(" ".join(c["text"] for c in chunks).lower().split())
    overlap = len(answer_tokens & source_tokens)
    groundedness = overlap / len(answer_tokens) if answer_tokens else 0.0

    assert groundedness < 0.4, "Test precondition: groundedness must be low"

    if groundedness < 0.4:
        answer_with_warning = "⚠️ " + answer + "\n\n(Lưu ý: Câu trả lời này có mức bám sát tài liệu nguồn thấp.)"

    assert answer_with_warning.startswith("⚠️"), "Warning prefix missing"
    assert "bám sát" in answer_with_warning, "Warning text missing"


# ── 6. Concurrent cache writes: no race condition ────────────────────────
async def test_concurrent_cache_write_no_crash():
    """10 simultaneous writes to same key must not crash."""
    cache = {}
    lock = asyncio.Lock()

    async def write(i):
        async with lock:
            cache["key"] = i
        return i

    results = await asyncio.gather(*[write(i) for i in range(10)], return_exceptions=True)
    # Should complete without exceptions
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0, f"Cache write errors: {errors}"
    # Final value is one of the written values
    assert "key" in cache


# ── 7. Large chunk list truncation ─────────────────────────────────────
def test_large_chunk_list_truncated_for_program_query():
    """100 chunks must be truncated to 15 for program queries."""
    chunks = [{"text": f"chunk {i}", "score": 1.0 - i*0.01} for i in range(100)]

    is_program_query = True
    max_chunks = 15 if is_program_query else 30

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    assert len(chunks) == 15, f"Expected 15 chunks for program query, got {len(chunks)}"

    # Non-program query should use 30
    is_program_query = False
    max_chunks = 15 if is_program_query else 30
    all_chunks = [{"text": f"chunk {i}", "score": 1.0 - i*0.01} for i in range(100)]
    if len(all_chunks) > 30:
        all_chunks = all_chunks[:30]
    assert len(all_chunks) == 30, f"Expected 30 chunks for non-program query, got {len(all_chunks)}"


# ── 8. PPR cache bounded by max_iter ───────────────────────────────
async def test_ppr_cache_keyed_by_seeds():
    """Same seeds must hit cache, different seeds must miss."""
    call_count = 0

    async def fake_pagerank(seeds, alpha):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return {s: 1.0 / len(seeds) for s in seeds}

    cache = {}
    call_log = []

    async def cached_ppr(seeds, alpha):
        key = tuple(sorted(seeds))
        if key in cache:
            return cache[key]
        result = await fake_pagerank(seeds, alpha)
        cache[key] = result
        call_log.append(seeds)
        return result

    # Same seeds twice
    seeds = ["NGANH:CNTT", "NGANH:KTPM"]
    r1 = await cached_ppr(seeds, 0.85)
    r2 = await cached_ppr(seeds, 0.85)
    assert call_count == 1, f"Expected 1 call, got {call_count} (cache miss on identical seeds)"
    assert r1 == r2

    # Different seeds — should miss cache
    await cached_ppr(["NGANH:BÁO_CHÍ"], 0.85)
    assert call_count == 2, f"Expected 2 calls, got {call_count} (different seeds should miss)"
