# Durability Audit — HUSC Admission RAG
**Auditor:** reliability-auditor subagent  
**Date:** 2026-06-07  
**Scope:** 10-year rollover (2026→2036), read-only  
**Test run:** 96/96 passed (`test_temporal_rollover_e2e`, `test_prompt_durability_10yr`, `test_season`, `test_temporal_authority`, `test_year_normalizer`, `test_admission_context`, `test_year_facts`)

---

## Test Depth Assessment

Tests are **REAL**, not shallow. `test_temporal_rollover_e2e.py` drives 3 explicit phases:
- P1 Jul-2026 (in-season 2026), P2 Oct-2026 (gap waiting for 2027), P3 Apr-2027 (in-season 2027).
- Asserts 2025 rows are hard-dropped in P1, KEPT in P2, 2026 rows dropped in P3.
- `test_prompt_durability_10yr.py` enforces zero `\b20\d{2}\b` literals in the generation prompt and router constants.

However: **the tests do NOT cover `aggregation_booster.py` patterns** — this is the largest untested attack surface.

---

## TIME-BOMB Table (ranked by severity)

| # | Severity | File : Line | What breaks / Which year | Fail-safe or Fail-wrong |
|---|----------|-------------|--------------------------|-------------------------|
| 1 | **CRITICAL** | `src/services/aggregation_booster.py` lines 63–129 and ~50 chunk-id strings throughout | ~30 regex patterns contain literal `2025`/`2026` (e.g. `r"học\s+phí.*2026"`, `r"điểm\s+mới.*2026"`). In 2027 a user asking `"học phí năm 2027"` or `"thay đổi 2027"` **silently gets zero boost** — no error, no log, no disclaimer, just weaker retrieval. Pattern `r"học\s+phí.*hiện\s+tại"` is the sole year-agnostic escape valve for tuition. | **FAIL WRONG** (silent de-match; answer served with lower-quality chunks) |
| 2 | **CRITICAL** | `src/services/aggregation_booster.py` lines 42–44, 56–57, 96, 106, 110, 118, 128, 152–381 — ~60 chunk_ids like `liet_ke_28_nganh_2026_v2`, `hocphi_2026`, `chinhsach_2026`, `phuongthuc_xettuyen_2026_v2`, `to_hop_a00_full_2026`, etc. | Chunk IDs are hardcoded with `_2026` suffix. When 2027 data is ingested under `_2027` IDs, `fetch_by_id("liet_ke_28_nganh_2026_v2")` returns `None` silently (see booster line 447: `logger.debug` only). All `_2026`-suffixed canonical chunks go dark for 2027+. | **FAIL WRONG** (silent null-fetch, user gets no aggregation summary) |
| 3 | **HIGH** | `src/services/lancedb_retrieval.py` line 39 | `if max_year >= 2025: return max_year` — the literal `2025` is a sanity floor. This is **safe today** and safe through 2035+ because max_year of real data will always be ≥ 2025. However the companion comment says "fallback: use system clock year (admissions happen mid-year)" — the real fallback is `now.year if now.month >= 1 else now.year - 1`, which is **always `now.year`** (condition `month >= 1` is always True). Dead code, not a bomb, but misleading. Module-level `CURRENT_ADMISSION_YEAR` is superseded by per-call `temporal_authority.get_current_admission_year()` (line 280), so the `>= 2025` floor is inert. | **FAIL SAFE** (floor always satisfied; per-call authority overrides) |
| 4 | **HIGH** | `src/services/query_router.py` line 110 | `CONTACT_BLOCK` hardcodes `"📞 Hotline: 0234.3823290 (Phòng Đào tạo)"`. This phone number is served verbatim to every contact query forever. If HUSC changes their hotline, every user gets stale contact info. No data-driven path exists. | **FAIL WRONG** (wrong phone, no versioning, no fallback to chunk) |
| 5 | **HIGH** | `uni-guide-ai/src/lib/chat-types.ts` line 58 | `suggestedQuestions` hardcodes `"Học phí năm học 2025-2026 là bao nhiêu?"` and line 111 hardcodes `'Thông báo tuyển sinh 2025'`. FE `YearBanner` correctly fetches `/api/meta` for the banner year, but **these UI suggestion chips are never updated** — they show 2025-2026 strings in 2027+. | **FAIL WRONG** (stale suggestions mislead users into asking about wrong year) |
| 6 | **MEDIUM** | `src/services/major_code_validator.py` line 101 | `_LEGACY_DEFAULT_YEAR = 2026`. When `get_whitelist(year=None)` fails to import `temporal_authority` (e.g. import error in CI or isolated env), it falls back to 2026 whitelist forever. For 2027+ this means codes like `7460108` (new in 2026) may be validated against a stale set. BUT: the design is explicit fallback — acceptable if `data/major_codes/2027.json` is never created. | **FAIL SAFE** (documented fallback, conservative — accepts 2026 codes) |
| 7 | **MEDIUM** | `src/services/llm_client.py` lines 280–290 | Provider fallback chain: ramclouds → groq → compat. When primary 503s, it tries all providers and raises `RuntimeError("All LLM providers failed")` if all fail. No circuit-breaker / cached response / graceful degradation. A ramclouds model retirement (e.g. `gemini-2.5-flash` deprecated) with no GROQ_API_KEY set causes a hard 500 to the user on every query. | **FAIL WRONG** (hard RuntimeError bubbles to HTTP 500 with no fallback message) |
| 8 | **LOW** | `src/services/aggregation_booster.py` lines 73–75 | Pattern `r"học\s+phí.*2025"` boosts chunk `hocphi_2025`. In 2028+, a user genuinely asking about 2025 historical tuition will still get the boost (correct). But `r"học\s+phí.*2026"` (line 78) won't match `"học phí 2027"` at all — asymmetric: 2025 history works, 2027 current does not. | **FAIL WRONG** (asymmetric; 2026 tuition queries work, 2027 don't) |
| 9 | **LOW** | `src/services/lancedb_retrieval.py` lines 52–54 | `HISTORICAL_QUERY_PATTERN` hardcodes years `2024|2023|2022|2021` in the regex. In 2028+, a user asking "so sánh năm 2025 vs 2026" won't hit this pattern (2025/2026 not in the list), so it won't be classified as `is_historical_query=True`, and 2025/2026 chunks may be hard-dropped in the IN_SEASON path. | **FAIL WRONG** (historical queries for older-but-not-ancient years misclassified) |

---

## Key Findings Summary

**Biggest real bomb:** `aggregation_booster.py` is an ~380-line static lookup table of year-literal regex patterns and `_2026`-suffixed chunk IDs. It has **zero year-parameterization**. It is entirely untested by the durability suite. When admission data for 2027 arrives under new chunk IDs, all ~60 canonical boost targets silently miss. Users get answers from weaker semantic-similarity retrieval only — no error, no warning, no disclaimer.

**Architecture gap:** The temporal authority, season machine, and retrieval layer are well-designed and year-agnostic. The aggregation booster layer is not — it was never brought into the year-parameterization contract.

**Tests assess 2026→2027 rollover correctly** but never invoke `detect_aggregation_chunks()` with years beyond 2026. Adding one parametrized test for `query="học phí năm 2027"` at simulated clock 2027 would immediately surface bomb #1.

**FE suggested questions** are purely cosmetic but create support confusion: users click "Học phí năm học 2025-2026" in 2028 and get confusing answers.

**Model retirement** is mitigated by the provider chain but has no last-resort cached/static fallback — a total outage (all 3 providers down or retired) raises an unhandled RuntimeError with no graceful user-facing message.

---

## Recommended Fixes (priority order)

1. **aggregation_booster.py** — replace hardcoded `_2026` chunk ID suffixes with a runtime year parameter: `f"liet_ke_28_nganh_{year}_v2"`. Replace year-literal patterns (`r"học\s+phí.*2026"`) with `r"học\s+phí.*hiện\s+tại"` + a year-variable pattern built at startup from `get_current_admission_year()`.
2. **query_router.py CONTACT_BLOCK** — move hotline into a data chunk (`husc_info`) and fetch dynamically, or at least expose via `/api/meta`.
3. **chat-types.ts suggestedQuestions** — derive from `/api/meta` response (same pattern as `YearBanner`).
4. **llm_client.py** — add a static fallback response (`"Hệ thống đang bảo trì, vui lòng thử lại sau"`) when all providers raise, instead of letting RuntimeError propagate.
5. **lancedb_retrieval.py:52-54** — extend `HISTORICAL_QUERY_PATTERN` to use `r"20\d{2}"` (any 4-digit year) rather than an enumerated list of 2021-2024.
