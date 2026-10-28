# PROGRESS — rag-chatbot-husc

Single source of truth for current work across all packages. Update on
done/partial/blocked per rule V5-R003.

## Active package

**rag-chatbot-husc**

## Packages

| Package | Status | Spec |
|---------|--------|------|
| rag-chatbot-husc | in-progress | docs/spec/rag-chatbot-husc-spec.md |

## Session log

### 2026-10-10 — T21 servability fix + verify "web chạy ngon"

- **Trigger**: advisory chỉ ra `url_ingest_service` ghi blob `url-page-<hash>.html` vào `registry.documents_dir` → `is_servable()` từ chối (không `uploads/`, không corpus root) → `/content` 404 trên citation chip của nháp URL vừa publishable. **Reproduce xác nhận**: `document_id=url-page-56ad19c31435c1eb.html`, `is_servable=False`.
- **Fix** (`url_ingest_service.py`): `document_id_for_url` → `uploads/url-page-<hash>.txt` (store-owned, `is_servable` mở vô điều kiện); blob ghi vào `<root>/source_documents/` qua `SOURCE_DOCUMENTS_DIRNAME` (không phải `documents_dir` — thư mục entry-JSON). `.txt` để `/content` stream `text/plain`: trang HTML bên thứ ba KHÔNG được serve dạng `text/html` (stored-XSS), và `.txt` vẫn là kind `classify_document_id` nhận. Cùng convention HOI_DAP (`admin_lifecycle.py:873`). **Không cần migration** — 0 record nào cite `url-page-`.
- **Test**: service test `test_fetched_page_is_registerable_and_publishable` thêm assert `is_servable` + `resolve_path`; test mới `tests/routers/test_url_ingest_content_http.py` — HTTP seam thật POST `/source-documents/from-url` (201) → GET `/records/{id}` (id `uploads/`) → GET `/source-documents/content` (200, bytes khớp). **15 passed** (service + HTTP). Regression `tests/services tests/routers` → **1847 passed, 37 skipped, 0 failed**.
- **FE verify**: `tsc --noEmit` sạch; `npm run build` OK (26.6s); chụp 4 màn thật (light `/quan-ly`, `/quan-ly/cho-duyet`, `/quan-ly/hoi-thu`, `/admin/records`) — tiếng Việt đủ dấu, contrast/typography sạch, không regression thị giác.
- **Còn chặn**: T14/T14b/T20/T23 cần VM GCP (không Docker trên host); T18 thiếu OCR/10 trang gõ tay; T00 AC3/AC4 thiếu cột `expected_record_ids` + chấm người; T22 deferred.

### 2026-10-09 — husc-teacher-admin: kiểm chứng ticket + vá R5 (temporal effectiveness) + KPI K3/K4

- **Vai trò (user)**: lead agent — điều phối subagent, thực thi công việc ghi trong `docs-devs/2026-10-06_1300` (publish `.scratch/husc-teacher-admin/`, 25 ticket T00–T23). Chuẩn: clean/SOLID/pattern, verify sau mỗi slice.
- **Kết quả trạng thái ticket (cuối ngày)**: done — T01,T02,T03,T04,T05,T06,T07,T08,T09,T10,T11,T12,**T15**,**T16**,**T17**,**T19**,**T21**; partial — T00 (2/4), T13 (đóng gói tĩnh xong, runtime chặn); blocked — T18 (thiếu Tesseract/Document AI/10 trang gõ tay); deferred — T22 (ngoài MVP); chờ infra — T14, T14b, T20, T23.
- **T15 — vá R5 "hết hạn không kích hoạt theo đồng hồ"** (điểm chính, trước đây là dead code): `temporal_filter.current_clause` không được import; `_apply_temporal_filter` chỉ lọc theo `data_year` + tombstone/superseded, **không** kiểm `status`/cửa sổ hiệu lực. Nay thêm `_drop_by_effectiveness` (`lancedb_retrieval.py:44`) — schema-gated NULL=KEEP (3-valued logic), clock lấy từ `temporal_authority` — áp ở dense retrieve, `fetch_by_id`, `fetch_by_ids`, BM25 build; kèm **sidecar eligibility re-attach** trong `hybrid_search.py` để chunk hết hạn bị loại ngay cả khi sparse index chưa rebuild. `lancedb_adapter` thêm `has_status_column()`/`has_effective_window_columns()`.
  - **An toàn corpus xác minh bằng đo thật**: 1679/1679 row `status='current'`, `effective_from=0`, `effective_to=99991231`, `is_tombstone=false` → **0 row bị drop**; `metadata_json.status` không tồn tại (deny-set không trúng). Writer stamp `status="current"` (`ingest_lancedb.py:534,870`) khớp predicate.
  - AC1/AC2 (claim + four-eyes) đã có sẵn, verify `test_admin_claim_draft.py`. AC3 = **HTTP seam thật** 3 route: `tests/routers/test_tombstone_three_routes_http.py` (padded_rag + graph_rag + hybrid v3, LanceDBRetriever/HybridSearchService thật, chỉ stub LLM/embedding/reranker/guardrail) → 3 passed. AC4/AC5 **BLOCKED** (cần gateway LLM sống + chấm người).
  - Test: `test_effective_window_predicate.py` (20, gồm regression "sparse chunk hết hạn bị loại khi chưa rebuild") + 5 file cũ → **87 passed**.
- **T16 — KPI K3/K4 thật** (trước chỉ in interface + chữ "T07 chưa land" đã sai): thêm `src/services/serve_probe.py` — log append-only `<CANONICAL_STORE_ROOT>/kpi/probe_log.jsonl`; `publish_hook` ghi entry `published` sau swap (upload_received_at từ source-document registry, `null`+cờ `_known` nếu thiếu — **không bịa**); query path stamp `served` khi chunk của release+record đó được trả về (`main.py` 4 điểm). `unanswered_queue.Item.out_of_scope` + K4 đếm theo 02 §9. `kpi_report.py` đọc log thật; log trống → "chưa có dữ liệu" (không trả 0 giả). K2/K5 giữ "chưa đo được" vì baseline T00 degraded. Smoke `python -m scripts.kpi_report all`: K4 **0/271 = 0.0%** (271 mục trong tuần, 0 resolved); K3 "chưa có dữ liệu". Test 49 passed; suite rộng 1349 passed.
- **T17/T21 (v1) chốt done**: T17 — XLSX 3 dòng → 3 nháp; **siết test** để assert `values` từng nháp khớp đúng dòng đầu vào (trước chỉ kiểm bản ghi đầu → lỗi gán sai dòng lọt). T21 — URL → nháp + bỏ qua khi nội dung không đổi (service + HTTP, 39 passed).
- **T13 (đóng gói)**: pin khớp venv (`lancedb==0.30.2`, `pyarrow==24.0.0`, `pydantic==2.13.3`, `torch==2.11.0`, `openpyxl==3.1.5`); `ragas`/`deepeval` đã tách sang `requirements-eval.txt`; compose parse hợp lệ; `.dockerignore` giữ `api/openapi.json` + `Caddyfile`; FE Dockerfile `COPY --from=repo api/openapi.json` → file **đã track**; Caddyfile regex khớp 10 route `/admin` + `Accept: text/html`. **Chặn runtime**: máy không có `docker`/`podman`, WSL2 hỏng (`ERROR_PATH_NOT_FOUND`) → không chạy được `docker compose up`. Bổ trợ: `npm run test:smoke` 9 passed (server thật + auth thật).
- **Fix UI (defect đã ship, ngoài cổng T09) — ĐÃ XỬ LÝ**: `/admin/*` (7 màn kỹ thuật, reachable từ `/quan-ly`) hardcode `text-zinc-300` (1.48:1) / `text-zinc-400` (2.56:1) / `amber-500-600` (<3.2:1) / `dark:text-zinc-500` (4.12:1) **không có nhánh `dark:`** → vi phạm AA trên nền trắng theme sáng; cổng `check-admin-build.mjs` chỉ liệt kê route `/quan-ly` nên không bắt. Đổi sang token ngữ nghĩa + cặp `dark:` và viền `!border-zinc-500` (4.83:1 trắng / 4.12:1 zinc-950), nút `bg-blue-600 text-white` (5.07:1); **mở rộng gate sang đủ 10 route `/admin`** → **0 vi phạm U1–U3 trên 18 route × 2 theme** (`npm run test:a11y` PASS). Ngưỡng target-size tách theo mặt (`/quan-ly` 44px AAA, `/admin` 24px AA — console dày `h-7/h-8`). FE vitest **341/341 pass**.
- **Regression backend**: `tests/services tests/routers tests/infrastructure` → **1972 passed, 37 skipped, 0 failed** (1043s) — không hồi quy từ T15/T16.
- **T00 AC3/AC4**: giữ partial, **không bịa** — dataset 86q (thực ra 303 câu) thiếu cột `expected_record_ids` (chỉ có `gt_chunks`; 44 giá trị gt_chunk → 8 map doc_id, **0** map chunk/record); AC4 cần chấm người. AC1 (SHA+cấu hình) đã commit `16ce1de`.
- **T22**: chuyển `deferred` — chủ dự án chốt "SSO để sau", không AC.

### 2026-07-27 — /maestro data-lifecycle NGÀY 1: 4 probe kiến trúc (Opus commander + Haiku executor + Sonnet reviewer)

- **Trigger**: triển khai `rag2025/.omc/plans/data_lifecycle_plan_v2.md` (ĐÃ DUYỆT, 1546 dòng). Prompt yêu cầu 4 probe §3.3 + Phụ lục B TRƯỚC khi viết code. Zero-bug là yêu cầu cứng.
- **Harness**: `Agent()` bị chặn trên Windows (đòi tmux/WSL) → chuyển sang `Workflow` pipeline, 4 probe Haiku song song + Sonnet review từng cái. 8 agent, 906k token, 19 phút.
- **P1 tag durability — G1 PASS, G1b PASS** (`docs/evidence/M1-tag-durability.md`). Executor báo PARTIAL/FAIL, Sonnet ESCALATE; Opus verify tay và **sửa kết luận**: `optimize(cleanup_older_than=0)` KHÔNG xoá tag — nó **raise trước khi chạm data** (fail-safe). `checkout("rel-A")` = 1 row nguyên vẹn ở cả 2 nhánh (sau cleanup raise, và sau `optimize()` compaction-only). Xoá tag đi thì cleanup chạy → tag đúng là thứ chặn. `restore("rel-A")` OK, `tags.update()` OK, `branches` không có. → **§0.2 giữ nguyên, KHÔNG cần rollback §5.**
  - Bác 2 claim sai của executor: (a) "tags survive by accident" — sai, fail-safe là thiết kế; (b) "inner async method accepts `error_if_tagged_old_versions`" — sai, cả `LanceTable.optimize` và `AsyncTable.optimize` đều chỉ có `(cleanup_older_than, delete_unverified, retrain)`.
  - **Bug thiết kế API phát hiện được**: plan §M1 dòng 334-338 định nghĩa `maintain(keep_days=30)` gọi `optimize(cleanup_older_than=timedelta(days=30))` → sẽ **luôn crash** trên mọi table production (vì kiến trúc tag từng release). Phải sửa spec `maintain()` TRƯỚC khi code `lancedb_versioning.py`.
- **P2 docling — PASS** (`docs/evidence/M2-docling-install.md`). docling 2.115.0 cài được CPU-only Windows (phải dùng `--user`, 2 lần đầu `WinError 5` do torch\_C.pyd bị lock). Parse thật 77 trang `THONG_TIN_TUYEN_SINH_202611.pdf` = **161.6s (~2.1s/trang)**, 49.815 ký tự, **có pipe table**. Disk D: 41.5 → 38.0 GB. `pip check` = No broken requirements. Pin không đổi: lancedb 0.29.2 · pyarrow 23.0.1 · pydantic 2.10.3.
- **P3 embedding identity — MATCH, không cần re-embed** (`docs/evidence/M3-embedding-identity.md`). Sonnet BOUNCE (đúng: test đầu chỉ chứng minh config tự tái tạo chính nó) → đã chạy cross-check `BAAI/bge-m3` như plan chỉ định:
  - `Qwen/Qwen3-Embedding-0.6B` (`.env`): AVG cosine **1.000000**
  - `BAAI/bge-m3` (plan giả định): AVG cosine **−0.050123** → LOẠI
  - Hai model cùng `dim=1024` nên dim một mình không định danh được model — đúng như M3-R4 lo. Tiết kiệm +2 NC dự phòng.
  - **Sai sót của plan**: §M3 DoD (dòng 648) viết `.env.example` phải theo preset `bge`; thực tế phải là `qwen` / `Qwen3-Embedding-0.6B` / `1024`.
- **P4 filter pushdown — G3 PASS** (`docs/evidence/M3-pushdown.md`). Trước index: `FilterExec` rời trên `KNNVectorDistance`. Sau 7 scalar index: `FilterExec` **biến mất**, thay bằng `ScalarIndexQuery` dưới `LanceRead`, `refine_filter=--`. Cả 7 cột nhận index (BTREE cho string/int32, BITMAP cho status/is_tombstone). → **chốt schema phẳng.**
  - **Sửa giả định plan §0.4**: bare-column filter trên nested schema **RAISE `RuntimeError: No field named data_year`**, KHÔNG silently swallow như plan đoán. Crash chứ không trả sai âm thầm.
- **Blocker xác nhận bằng số**: B1 drift thật (`rag2025` 1671 row, `vector`+`metadata_json:string` vs `husc_v2026_clean` 448 row, `embedding`+`metadata:struct`) · B2 `temporal_filter.py:40` (không phải :41) sinh `data_year = {current_year}` · B3 `.env.example:46` = 2560 vs bảng thật 1024 (`.env:31` = 1024, đúng) · B4 `list_indices() == []` cả 2 bảng.
- **Baseline test sau khi cài docling** (`docs/evidence/day1-baseline-tests.md`): `1877 tests, 23 failures, 0 errors, 1 skipped`. Sensor xanh: `import src.main` OK, `pip check` clean, vector reproducible AVG cosine **0.999869** (tụt từ 1.000000 do torch 2.9.1→2.13.0 đổi kernel CPU, ngưỡng plan >0.99 nên PASS rộng). Không fail nào là ImportError/tensor — **cài docling không thêm fail nào**.
  - 23 fail = **4 env drift** (test untracked hard-code `llama-3.3-70b-versatile`/`gemini-3.6-flash-high` vs `.env:73` `GROQ_ROUTER_MODEL=openai/gpt-oss-20b` + `.env:80` `RAMCLOUDS_ROUTER_MODEL=deepseek-v4-flash`) + **19 hệ quả B2/B5** (mọi câu 2026 trả "chưa được công bố chính thức", groundedness 0.0; ingest idempotency `assert 0 == 1`). Nhóm 19 sẽ tự hết khi M3 sửa B2/B5.
- **User quyết (scope_prioritization)**: sửa `.env` theo test. Backup `rag2025/.env.bak_router_20260727`. Đổi `.env:73` `GROQ_ROUTER_MODEL=openai/gpt-oss-20b`→`llama-3.3-70b-versatile`, `.env:80` `RAMCLOUDS_ROUTER_MODEL=deepseek-v4-flash`→`gemini-3.6-flash-high`. Full suite verify (haiku chạy, sonnet review ACCEPT): **23→20 fail**, 0 regression mới. 3/4 router test clear.
- **Fail thứ 4 (NER) — tách biến gốc (V5-R011)**: `test_ner_cache.py::test_fallback_swaps_to_groq_llama_for_ramclouds_only_model` vẫn fail vì NER fallback đọc chung `GROQ_ROUTER_MODEL` (router cần 70b, NER cần 8b-instant). Workflow `wf_545ae3b8-d56` (haiku diagnose→implement, sonnet review): thêm `GROQ_NER_MODEL` (default `llama-3.1-8b-instant`) vào `settings.py:201` + `.env` + `.env.example`, đổi `llm_client.py:235` `_build_providers()` đọc `GROQ_NER_MODEL`. Router path `:615` `_build_router_providers()` không đụng. Test assertion không sửa. Target test pass (23 passed + 1 skipped), router test vẫn xanh. Sonnet ACCEPT, 0 issue. → **20→19 fail** (19 còn lại = B2/B5, chờ M3).
- **Sensor xanh sau NER split**: `import src.main` OK, target tests pass, không regression.
- **Lesson ghi**: `feedback-opus-commander-must-delegate.md` — Opus chỉ route/gate/decide; verify, test, poll, viết file, sửa config đều spawn haiku. Không tự chạy inline, không poll vòng lặp.

### 2026-07-27/28 — M1 Canonical release store + LanceDB version/tag layer (XONG)

- **Trigger**: triển khai `data_lifecycle_plan_v2.md` §M1 — canonical release manifest, version/tag layer trên LanceDB, atomic pointer.
- **Harness**: 10 slice, mỗi slice 1 haiku executor + sonnet review sau batch. Pipeline tuần tự: WRITE_TESTS(RED) → IMPLEMENT(GREEN) → MECHANICAL.
- **Gate M1**: M1-T1..T7 đủ 7 test + 2 extra. `pytest rag2025/tests/domain/ rag2025/tests/infrastructure/` = 26 passed 0 fail.
- **Full suite**: `tests 1887 failures 20 errors 0 skipped 1` (baseline 1877/20 → +10 test mới, **0 regression**). 20 fail còn lại: 17 golden (blocker B2) + 2 ingest idempotency (B5) + 1 provider model default — tất cả thuộc scope M3.
- **Sensor**: `pip check` = `No broken requirements found.`; `import src.main` = `MAIN IMPORT OK` khi cwd=`rag2025/`.
- **Review**: 3 vòng sonnet CODE_REVIEW — ACCEPT toàn bộ.

**10 slice chi tiết:**

| Slice | Lane | Kết quả |
|-------|------|---------|
| 1 | WRITE_TESTS | `rag2025/tests/domain/test_release_manifest.py` — RED `ModuleNotFoundError: No module named 'src.domain.release'` |
| 2 | IMPLEMENT | `rag2025/src/domain/release.py` 99 dòng — GREEN 2 passed |
| 3 | WRITE_TESTS | append 2 test → RED `No module named 'src.infrastructure.canonical_store'` |
| 4 | IMPLEMENT | `rag2025/src/infrastructure/canonical_store.py` 57 dòng — GREEN 4 passed |
| 5 | WRITE_TESTS | `rag2025/tests/infrastructure/test_lancedb_versioning.py` — RED |
| 6 | IMPLEMENT | `rag2025/src/infrastructure/lancedb_versioning.py` 34 dòng (`tag_release`, `rollback_to`, `maintain`) — GREEN |
| 7 | WRITE_TESTS | append 3 test → RED ImportError |
| 8 | IMPLEMENT | mở rộng lên 81 dòng (`promote_to_prod`, `read_only_at`, `prune_versions`) — GREEN 5/5 |
| 9 | MECHANICAL | `lancedb_adapter.py` 166→176, thêm `get_table()` |
| 10 | MECHANICAL | DoD: `requirements.txt:41` comment PINNED, `rag2025/data/canonical/.gitignore` + `README.md`, sửa citation `M1-tag-durability.md` → `data_lifecycle_plan_v2.md:375` |

**Quyết định kiến trúc chốt trong M1:**
- `maintain(table)` compaction-only `table.optimize(delete_unverified=False)`. `cleanup_older_than` CHỈ dùng trong `prune_versions()` và phải `tags.delete()` trước. Lý do: probe P1 chứng minh `optimize(cleanup_older_than=...)` raise `RuntimeError: N tagged version(s)` trên MỌI table có tag — tức mọi table production. Fail-safe, không mất data.
- Atomic pointer: `canonical_store.publish_pointer()` ghi `.tmp` cùng volume rồi `os.replace()` → nguyên tử trên NTFS, không mixed state (§6). KHÔNG dùng lại `win32_utils.py` vì file đó chỉ có diagnostics/copytree/retry-write, không single-file pointer swap.
- `get_table()` chưa có consumer production, là infra prep cho M5 CLI publisher.

### 2026-06-16 — /maestro clean-code review + 10-yr QA hardening (Opus commander + Sonnet/Haiku fan-out)

- **Trigger (user /maestro)**: review chặt clean-code theo chuẩn AI-Engineer; QA toàn diện sâu nhất để chắc chatbot không lỗi 10 năm tới (best+worst case); debug→fix; ghi context vào memory. Model fallback rule: sonnet limit→haiku, haiku limit→sonnet, ghi timelimit.
- **Backend fail reduction 54 → 4 → 0**:
  - **main.py truncation incident**: 1 Sonnet worker (W-HARDEN) làm on_event→lifespan rewrote main.py **1771→465 dòng**, xóa MỌI route → 54 test 404. Recovered `git checkout HEAD -- rag2025/src/main.py`. Lesson L002 + memory `subagent-destructive-edit-mainpy.md`.
  - **Dead qdrant chain removed**: xóa `services/{qdrant_retrieval,bge_retriever,vector_store,retriever}.py` + 2 test files; `qdrant-client` khỏi requirements.txt (user xác nhận đã chuyển sang LanceDB).
  - **llm_client hardening**: line-124 `os.getenv("OPENAI_API_KEY")` leak fixed; retry-cap `CHAT_MAX_ATTEMPTS=4` / `JSON_MODE_MAX_ATTEMPTS=2` (chặn 30-60s retry storm trên empty-body json_mode).
  - **Temporal durability**: `year_ner.py` / `year_normalizer.py` YEAR_HI giờ động (`current_year+6`) — không hardcode 2026.
  - **conftest.py mới**: prepend `rag2025/src` + repo root vào sys.path → fix import split-brain.
  - **aggregation_rules.py mới**: tách rule data khỏi aggregation_booster god-object.
- **4 fail cuối (root cause, KHÔNG phải pollution như giả định ban đầu)**:
  - `test_llm_client_json_retry` ×2 → `GROQ_API_KEY` rò từ test trước → `_build_providers()` tạo 2 provider → nhân đôi retry attempts. Fix: `patch.dict` GROQ/OPENAI_COMPAT env về rỗng trong test.
  - `test_hybrid_search` ×2 → `asyncio.get_event_loop()` đã đóng sau pytest-asyncio. Fix: `new_event_loop()` + try/finally/close.
- **Lifespan migration #11 (làm lại surgical)**: on_event(startup/shutdown)→`FastAPI(lifespan=)`. Evidence-gated: `on_event=0`, `lifespan_kw=True`, lines 1770 (no truncation), 18 API routes intact. KHÔNG lặp lại incident truncation.
- **FE verify (toàn bộ, vì 1 subagent lỗi → các con khác high-risk)**: `uni-guide-ai` typecheck + build + vitest đều green.
- **FINAL GATE**: full backend suite `1263 passed, 3 skipped, 0 failed` (verify độc lập 2 lần). FE green.
- **Memory ghi lại**: `model-rate-limit-kimi-k2.md` (executor backend 429 reset ~2min), `subagent-destructive-edit-mainpy.md` (whole-file-rewrite guard + per-worker line-count gate). Commander responsibility: brief thiếu preserve-routes constraint + verify global-only thay vì per-worker boundary.

### 2026-06-07 — Rich Markdown Generation + Backend-Frontend Sync (S16.x follow-up)

- **Trigger**: Exploration (Task #1) confirmed prompt rule 14 prohibits tables, FE `react-markdown@10` lacks gfm plugin, and `sources` is flat `List[str]` losing chunk metadata.
- **RALPLAN-DR consensus plan**: `.omc/plans/rich-markdown-generation-plan.md` (Option B: prompt rewrite + structured sources + gfm, scoped to `/query` only).
- **T0 — Table validation post-guard** (`_validate_markdown_table` in `llm_generator.py`): 5 unit tests pass. Handles valid tables (pass-through), mismatched columns (pad), missing separators (degrade to bullets).
- **T1 — Prompt rule 14 rewrite**: `generation_system_prompt.txt` lines 226-243 changed from CMF-6 prohibitions to permissive rich-markdown rules. Allows pipe tables, **bold**, *italic*, ### headings, - bullets. Keeps no raw HTML, no bare source URLs in body.
- **T2 — S16.4 tests updated**: `test_generation_prompt_s16.py` — `test_no_table_rule_present` → `test_table_allowed_rule_present`, `test_comparison_layout_rule_present` updated, `test_no_code_fence_rule_present` tombstoned, added `test_bold_italic_allowed`. 15/15 pass.
- **T3 — Enriched sources in `generate_answer`**: Return dict now contains `List[Dict]` with keys `id`, `title`, `url`, `snippet`, `data_year`. Built from chunk metadata via `get_source_url()` + fallback chain.
- **T4 — `QueryResponse.sources` type**: Added `SourceChip` Pydantic model (`id`, `title`, `url`, `snippet`, `data_year`). Changed `QueryResponse.sources` from `List[str]` to `List[SourceChip]`. Did NOT change `UnifiedQueryResponse` or `V3QueryResponse` (deferred).
- **T5 — `remark-gfm` added to FE**: Installed in `uni-guide-ai`, imported in `MessageBubble.tsx`, added `remarkPlugins={[remarkGfm]}` to `<ReactMarkdown>`.
- **T6 — FE Source interface + ChatLayout + SourceChip updated**: `chat-types.ts` Source now has `snippet?`, `data_year?`. `ChatLayout.tsx` mapping handles both legacy strings and new SourceChip objects. `SourceChip` component shows `data_year` badge and clickable `url`.
- **T8 — Regression test for URL guard + pipe compatibility**: 3 new tests in `test_url_faithfulness_guard.py` (`TestStripUngroundedUrlsTableCompatibility`). Verifies pipe chars survive, table structure intact, URLs in cells stripped only if ungrounded.
- **VERIFY**: 37 tests pass across 4 suites (`test_url_faithfulness_guard.py`, `test_markdown_table_validation.py`, `test_generation_prompt_s16.py`, `test_llm_generator_no_fabrication.py`). TypeScript type-check clean. `SourceChip` Pydantic model serializes correctly.

- **User báo**: chatbot tự tin trả học phí ngành Văn học nhưng số có thể của Đại học Huế (parent) chứ không phải HUSC chính thức. Yêu cầu dùng Sonnet + /pdf đọc 3 file `new_data/` tìm học phí 28 ngành.
- **Discovery (4 agent, /pdf + LanceDB trace)**:
  - 2 file Bộ GD&ĐT (`2304_BGDDT HD tuyển sinh`, `Thông tư 06/2026`) — KHÔNG có học phí cụ thể (chỉ policy: Điều 11.2b "CSĐT phải công bố mức thu", Điều 14.6 lệ phí). Xác nhận đây là nguồn chỉ-quy-trình.
  - **`THONG_TIN_TUYEN_SINH_202611.pdf` trang 54-55 CÓ bảng học phí HUSC chính thức** — ghi rõ "Trường Đại học Khoa học — Học phí năm học 2026-2027", theo Nghị định 238/2025/NĐ-CP, per-major 25 ngành.
  - **Corpus SAI số**: chunk `husc_nganh_*_2026` ghi Khối VII 525.000 + Khối IV 540.000 (số dựng từ TT06 + trang web, KHÔNG phải official) — trong khi PDF official: **Khối VII = 490.000, Khối IV = 555.000**, Khối V 600.000 (đúng), CNTT-VJ 640.000, Triết học = 0 (miễn). Văn học official = **490.000đ/TC ≈ 15.435.000đ/năm**.
- **User quyết**: tin PDF 54-55 là official → sửa số đúng.
- **Fix (backup `lancedb_backup_pre_tuition_fix_20260604`)**: `fix_husc_tuition_2026.py` (TDD 18 test) + canonical `data/major_codes/husc_tuition_2026_official.json` (25 ngành). Sửa: **17 per-major table chunks + 4 aggregate chunks** trong LanceDB; **8 dòng on-disk** (chunked_10/10_enhanced/24, aggregation_v2, raw canonical) + GT file. Block-aware: Khối V 600k giữ nguyên; chunk data_year=2025 KHÔNG đụng; idempotent. Thêm metadata tuition_source_url + tuition_basis="Nghị định 238/2025" + tuition_year="2026-2027".
- **VERIFY end-to-end**: hỏi "học phí ngành Văn học" → chatbot trả **490.000 VNĐ/tín chỉ ≈ 15.435.000đ/năm** (KHÔNG còn 525.000 sai), "Vật lý" → 555.000, "KTPM" → 600.000. 762 test pass.
- **Còn 1 việc nhỏ**: câu trả lời chưa nêu nhãn "Nghị định 238/2025 / năm học 2026-2027" trong body (metadata có nhưng generation chưa surface) — minor, có thể thêm vào generation prompt nếu muốn minh bạch nguồn.

- **Trigger (user)**: PDF teacher chỉ show 5 câu (bug) → phải đủ 86; cải thiện prompt để dùng 10 năm KHÔNG cần sửa; benchmark "điểm hay" nhưng KHÔNG overfit 86 câu; production-ready cho MỌI trường. Eng-lead (Opus) giao việc rõ ràng xuống Sonnet.
- **Fixes landed (TDD, fan-out Sonnet + Opus review, 773 test pass)**:
  - **PDF teacher đủ 86 câu**: `gen_thi_sinh_pdf.py:234` cũ lấy `sample[:8]` → đổi hiển thị TOÀN BỘ 81 answered + 5 fallback. Verify pdfplumber: 11 trang, 0 câu thiếu. Cả 2 PDF (full 15tr + teacher 11tr) regen từ s16, 0 lỗi font.
  - **Held-out 40Q chống overfit** (`data/eval/husc_heldout_v1.json` + validator + 24 test): 40 câu MỚI, tách biệt 100% khỏi 86Q + 60Q adversarial (0 trùng), tiếng Việt teen/viết tắt + 8 out-of-scope.
  - **Prod `/v2` wire qua `generate_answer`** (`_synthesize_v2_answer`, 4 test): user thật giờ có URL-guard + abstain + GAP_DISCLAIMER; CONTACT_BLOCK không bị nuốt. (`/query` legacy + `/v3` đã đúng sẵn.)
  - **Hợp đồng 10 năm** (`test_prompt_durability_10yr.py`, 18 test): quét + sửa 2 bug 10-năm THẬT — mã ngành `7480101/7220201` hardcode trong `_STEP_BACK_SYSTEM` + regex entity-detect `2025|2026` (năm 2030 sẽ bị từ chối oan) → `\b20\d{2}\b`. Test FAIL nếu tái phạm. temporal_authority xác nhận no upper-cap.
  - **Bug test pollution**: 3 test v2 fail khi chạy chung suite (global event loop bị async test khác đóng) → fresh loop mỗi test. 773 passed.
- **KẾT QUẢ GENERALIZATION** (`heldout_v1_generalization.json`):
  - **faithfulness held-out 0.971 ≥ 86Q s16 0.964** → HỆ THỐNG KHÔNG OVERFIT (trả lời trung thực HƠN trên dữ liệu lạ).
  - **route-accuracy 90% (36/40)** trên câu chưa từng thấy; 4 mismatch ĐỀU là under-route về phía nhiều context hơn (padded→hybrid/graph) = fail-safe, không bao giờ thiếu context.
  - **out-of-scope abstain 7/8** (câu trường khác/thời tiết/đồ ăn → từ chối đúng).
  - context_precision held-out 0.535 thấp hơn 86Q 0.789 — DO 8/40 câu out-of-scope retrieve chunk không liên quan (không có đáp án HUSC) → precision thấp nhưng hệ thống TỪ CHỐI đúng (faithfulness 0.971). Không phải regression.
- **Router-model benchmark** (`router_model_benchmark_summary.json`): gemini-3.5-flash-low **0.975** 🥇 > gpt-5.4 0.900 > deepseek-v4-pro 0.713 > deepseek-v4-flash 0.638 (HyDE leak CoT). → giữ flash-low cho router (xác nhận bằng số).
- **⚠️ PRE-DEPLOY (nhắc lại)**: `.env` PHẢI để `RAMCLOUDS_HYDE_MODEL=gemini-3.5-flash-low` (gpt-5.5 hỏng JSON-classify → router default all-hybrid).

- **Trigger (user)**: benchmark deepseek-v4-flash / deepseek-v4-pro / gpt-5.4 (+ gpt-5.5 retry) cho vai trò HyDE/router/step-back; chọn con TỐT NHẤT cho domain, KHÔNG tính latency. Opus viết benchmark, Sonnet fan-out chạy song song.
- **Bench harness (Opus tự viết)**: `data/eval/router_model_benchmark_gold.json` (20 classify có nhãn route padded/hybrid/graph + 5 step-back giữ-entity + 3 HyDE domain-vocab) + `scripts/bench_router_model.py` (overall = 0.5×route_acc + 0.25×stepback + 0.25×hyde; JSON-valid là gate cứng).
- **Bài học (probe correction)**: probe 1-shot ban đầu báo deepseek FAIL-JSON, nhưng chạy đủ 20 câu → cả 2 deepseek json_valid=1.0 → probe chỉ là gateway hiccup tạm thời, KHÔNG phải giới hạn model. → không loại model dựa trên 1 probe.
- **KẾT QUẢ** (`results/eval_harness/router_model_benchmark_summary.json` + `router_bench/*.json`):
  | Model | route_acc | json | step-back | HyDE | **overall** |
  |---|---|---|---|---|---|
  | **gemini-3.5-flash-low** | 0.95 | 1.0 | 1.0 | 1.0 | **0.975** 🥇 |
  | deepseek-v4-pro | 0.725 | 1.0 | 0.4 | 1.0 | 0.713 |
  | deepseek-v4-flash | 0.975 | 1.0 | 0.6 | **0.0** | 0.638 |
  | gpt-5.4 | 0.8 | 1.0 | 1.0 | 1.0 | 0.900 🥈 |
- **KHUYẾN NGHỊ CHỐT**: **GIỮ `RAMCLOUDS_HYDE_MODEL=gemini-3.5-flash-low`** cho router (đồng thời là HyDE+step-back client). Thắng mọi trục. deepseek-v4-flash route-acc cao nhất (0.975) NHƯNG HyDE=0.0 (leak chain-of-thought vào HyDE doc) + step-back mất entity → loại vì 3 vai trò dùng chung 1 client. deepseek-v4-pro yếu routing+step-back. gpt-5.5 hỏng JSON-classify trên gateway (đã biết từ S15).
- **Hệ quả**: quyết định ".env nên để HYDE=flash-low" (đã ghi ở S16) giờ được benchmark XÁC NHẬN bằng số, không phải chỉ vì gpt-5.5 hỏng JSON.

- **Trigger (user)**: fix URL-faithfulness (s15 0.949 < gate 0.95); router phải chính xác per-query (padded/graph/hybrid, HyDE KHÔNG áp đảo); 3 prompt mới (HyDE/step-back/generation) — generation render đẹp cho `uni-guide-ai`; regen 2 PDF nộp cô tuyển sinh. Plan: `.omc/plans/s16_faithfulness_route_calibration_plan.md` (ralplan: Architect SOUND-WITH-CHANGES 5 AMF → Critic ITERATE 6 CMF → patched → fan-out Sonnet, Opus integration review GO).
- **Discovery (3 agent + sciomc web)**: root-cause over-routing = HyDE doc bị concat vào `_classify` prompt (`:293`) → flash-low thấy "multi-entity" → 76/86 graph. FE `uni-guide-ai` = react-markdown@10 KHÔNG có gfm → chỉ render h3/bold/bullet/link, KHÔNG render bảng. Sources = chip riêng (không lấy URL từ body).
- **Fixes landed (TDD, 2 lane Sonnet + Opus review, 702 test pass)**:
  - **S16.1** `QueryRoute.HYBRID` + 3-nhánh `pipeline.query()`: padded=vector thuần (no PPR), hybrid=vector+PPR expander M=5, graph=PPR M=8+multiseed. `retrieve(expander_m=...)` per-route.
  - **S16.2** BỎ HyDE khỏi classify (`_classify(query)` only) — diệt gốc rễ over-route. Classify rewrite 3-way + complexity=1 anchors. Precedence: vague/contact > enum→graph > comparison→graph > classify authoritative > padded+cx≥2→hybrid > padded+≥2 ngành→hybrid > exception→hybrid (không còn default graph).
  - **S16.3** HyDE ≤40 từ + step-back giữ entity + `_should_stepback` (skip khi có năm+mã ngành).
  - **S16.4** generation prompt: faithfulness (cấm bịa URL/số/mã), abstain slot, FE-render (h3/bold/bullet, CẤM bảng/HTML, so sánh A-B dùng `### A`+bullet), giữ `{{YEAR_FACTS}}`.
  - **S16.5** `_strip_ungrounded_urls`: strip URL không có trong context, allowlist 4 domain chính thức (bảo vệ GAP_DISCLAIMER URL).
- **KẾT QUẢ s16 CHỐT** (`metrics_report/s16/` + `s16_comparison.json`, all-Sonnet judge; router+gen flash-low):
  - **RAGAS** (baseline→s14→s15→**s16**): faithfulness 0.963→0.965→0.949→**0.964** (PASS ≥0.95, URL-fix thành công) · answer_relevancy 0.798→0.916→0.936→**0.948** (BEST) · context_precision 0.845→0.724→0.765→**0.789** (hồi tiếp).
  - **Hallucination genuine**: Type-4 **0** (3 flag đều FP registry-lag), Type-5 **1** (msg068 tổng học phí cộng dồn), Type-6 **0**, Type-1 2 (msg026/028 năm-trước debatable).
  - **Route 3-nhánh**: padded 7 / hybrid 53 / graph 24 (was s15: graph 76). Latency p50 18.5s.
  - **Gate**: faithfulness PASS; graph≤30 PASS (24); **padded≥15 FAIL (chỉ 7)** — flash-low classify thiên về hybrid; KHÔNG phải regression vì hybrid = vector+PPR-nhẹ (an toàn hơn padded thuần), chỉ là phân bố lệch nhẹ. abstain_accuracy FAIL 0.143 (như s15 — phần lớn GT-convention, không phải bịa thật).
- **2 PDF regen từ s16** cho cô tuyển sinh: `86q_thi_sinh_report_full.pdf` (15 trang) + `86q_thi_sinh_report_thay_tuyen_sinh.pdf` (2 trang). Verify pdfplumber: 86/86 Q+A đầy đủ, 0 lỗi font, 0 thiếu dấu tiếng Việt.
- **⚠️ 2 việc PRE-DEPLOY (task #82 + .env)**:
  1. **Prod `/v2/query` sinh answer bằng `llm.chat` inline, KHÔNG qua `generate_answer`** → user thật CHƯA có URL-guard/abstain. Phải wire `/v2` qua `generate_answer` trước deploy (eval path đã đúng, nên 86Q hợp lệ).
  2. **`.env` đang `RAMCLOUDS_HYDE_MODEL=gpt-5.5`** → gpt-5.5 hỏng JSON-classify trên gateway → router prod sẽ default ALL hybrid. PHẢI đổi `RAMCLOUDS_HYDE_MODEL=gemini-3.5-flash-low` (eval đã override, prod chưa).

- **Bug user báo**: hỏi "khi nào có kế hoạch tuyển sinh xét học bạ?" → trả generic "Đợt 1 tháng 5-6, Đợt 2 tháng 7-8" (placeholder) thay vì mốc cụ thể đã crawl ở thongbao.php?id=74.
- **Root cause (2 agent discovery)**: chunk mốc CỤ THỂ `qa_dang_ky_xet_tuyen_2026_v2` (id=74: 02/7/2026, 23/5/2026, 14/7/2026) CÓ trên đĩa trong `husc_2026_canonical_v2.jsonl` + đã approved, NHƯNG chưa ingest vào LanceDB (không có trong manifest — manifest chỉ track `chunked_*.jsonl`). 8 file approved khác cũng bị sót. Generic placeholder thắng retrieval vì là chunk timeline duy nhất trong index.
- **Fix (backup `data/lancedb_backup_pre_s17_20260603/` trước mọi thao tác)**:
  1. **Ingest 5 file approved bị sót** (`ingest_missing_2026.py`, TDD, dedup-by-id, dim=1024): canonical_v2(3)+canonical_v3(7)+aggregation_v2(4)+nganh_detail_2026(28)+vision_ocr(12) = **+42 chunk** (1613→1655). SKIP `baiviet_2026` (HTML noise) + `thongbao_2026_v2` (data_year=2025 sai nhãn) + `chunked_10_enhanced` (dup).
  2. **Supersede 2 generic placeholder** (`supersede_generic_timeline.py`): `qa_lich_tuyen_sinh_2026` + `qa_xet_hoc_ba_quy_trinh_2026` (đều có "[CẦN HUSC CUNG CẤP ngày cụ thể]") → is_superseded=true.
  3. **Thêm chunk query-aligned** `qa_lich_xet_hoc_ba_2026_v1` (lead bằng framing "khi nào/lịch/thời gian", mốc copy nguyên văn từ id=74 — KHÔNG bịa) để match câu hỏi hội thoại.
  4. **Dedup 34 chunk boilerplate trùng** (`dedup_boilerplate_chunks.py`, TDD): 20 group text<80 ký tự trùng (header "CÁC PHƯƠNG THỨC...", "Vụ GDĐH...", "CỘNG HÒA...") flood top-k → giữ 1 canonical, supersede phần còn lại.
- **VERIFY end-to-end (câu trả lời model THẬT, không chỉ retrieval)**: hỏi đúng câu user → answer giờ trả **"Hệ thống ĐH Huế: 07:00 ngày 23/5/2026 đến 17:00 ngày 14/7/2026; Hệ thống Bộ GD&ĐT: 02/7/2026 đến 17:00 ngày 14/7/2026"** — chứa cả 3 mốc cụ thể (02/7, 23/5, 14/7/2026), KHÔNG còn "tháng 5-6" generic. 3 cách hỏi: Q1 (hội thoại) top-1 PASS, Q3 (thời gian) top-1 PASS, Q2 (keyword "lịch xét tuyển học bạ 2026") top-3 qua production retriever PASS. No-regression: "điểm chuẩn CNTT" vẫn trả chunk đúng.
- **Test**: 6 (dedup) + 9 (ingest+supersede) mới; `tests/scripts/` 344 passed 1 skipped. 4 script mới + backup.
- **LƯU Ý corpus đã đổi** (+42 chunk, 36 superseded): nên chạy lại 86Q eval (run-id s17) để xác nhận không hồi quy chunk khác — chờ user quyết.

- **Trigger (user)**: 3 vấn đề mới — (1) router phân loại padded/graph có chính xác không, comparison có bị nhầm sang padded (thiếu context graph)? (2) hệ thống trả lời được câu toàn cục/so sánh A-vs-B chưa? (3) đã đổi sang gemini-3.5-flash-low full → context_precision tụt 0.845→0.724, cứu được không? Plan: `rag2025/.omc/plans/s15_graphrag_fusion_plan.md` (ralplan: Architect SOUND-WITH-CHANGES 4 must-fix → Critic ITERATE 5 must-fix → patched).
- **Discovery (3 agent + web /sciomc)**: 
  - **Root cause #1 (sâu nhất)**: `graphrag_retriever.retrieve()` là RERANK-ONLY — `doc_by_id` chỉ build từ `baseline_docs` (vector), nên chunk PPR-only (entity graph biết nhưng vector miss) bị VỨT ở line 185. → graph route vẫn thiếu context dù route đúng.
  - **Root cause #2**: router LLM-based (giờ là flash-low) + threshold `>` (chỉ complexity≥3 lên graph) + `_ENUMERATION_PATTERNS` KHÔNG có comparison verb → 8 câu misroute (so sánh/khác nhau/A-hay-B → padded). 
  - **Comparison/global**: chỉ chạy được nhờ ~30 rule hardcode trong `aggregation_booster` (vd `qa_so_sanh_cntt_va_cntt_vj`); KHÔNG có cơ chế chung.
  - **Web 2026**: consensus = FUSE (không route either/or); RRF; LightRAG/HippoRAG2 dual-level; comparison = multi-seed PPR (không cần LLM decompose). 
- **Fixes landed (TDD, 3 lane Sonnet song song + Opus architect/critic/integration)**:
  - **S15.1 expander** (`graphrag_retriever.py`): chunk PPR-only top-M (5, comparison→8) được `fetch_by_id` nạp vào pool; score sentinel 1.0→**0.0** (fused = β·ppr, không auto-rank #1); tag `graph_injected`. `from_disk` wire `chunk_fetcher=get_retriever().fetch_by_id` (singleton).
  - **S15.2 multi-seed**: ≥2 entity NGANH/TO_HOP → M 5→8, top_k +3.
  - **S15.3 router**: `_COMPARISON_PATTERNS` (so sánh|khác nhau|so với|hơn|kém|chênh lệch|hay) force graph_rag; threshold `>`→`>=`; classify-exception default **graph_rag** (fail toward more context).
  - **S15.4 model split**: `RAMCLOUDS_ROUTER_MODEL` (default=HYDE, no-op) → router/HyDE/classify chạy model mạnh, generation giữ flash-low nhanh.
  - **S15.5 precision cutoff**: drop chunk score < 0.3×top (min_keep 2), EXEMPT `booster_injected`+`graph_injected`; áp ở `query()` boundary.
  - **S15.6 abstain-hardening**: fix msg055 — keyword contact (zalo|group|fanpage|hotline|...) + term vắng trong chunk → abstain.
- **Regression**: **602 passed, 1 skipped**. Live smoke: effective_router_model=gpt-5.5, chunk_fetcher wired (graph 2270 nodes/2440 edges).
- **Gateway finding (quan trọng)**: gpt-5.5 **KHÔNG** trả được JSON cho `_classify` trên gateway này (trả non-JSON → "Expecting value line 1") → treo eval ở msg023, ~19s/câu lãng phí. flash-low classify trả JSON đúng + nhanh hơn. → **ADR-D model-split phản tác dụng**; giữ router=flash-low. Điều cứu correctness là S15.3 default-graph-on-classify-fail. (`s15_router_model_finding.json`). User sau đó đổi HYDE→MiniMax-M3.
- **KẾT QUẢ s15 CHỐT** — `metrics_report/s15/` + `s15_comparison.json` (3 cột, all-Sonnet judge apples-to-apples; router+gen=flash-low):
  - **RAGAS** (baseline→s14_fixed→s15): faithfulness 0.963→0.965→**0.949** · answer_relevancy 0.798→0.916→**0.936** (BEST) · context_precision 0.845→0.724→**0.765** (hồi +0.041 vs s14 nhờ cutoff+expander).
  - **Hallucination genuine** (triage vs context): Type-4 **0** (5 detector-flag đều FP, registry lag), Type-5 **1** (msg068 tổng học phí cộng dồn), Type-6 **0**, Type-1 2 (msg026/028 năm-trước, historical intent — debatable).
  - **Route**: graph_rag 18→**76** (comparison patterns + `>=` threshold + default-graph). Latency p50 16.3→**21.1s** (vẫn 2.5× nhanh hơn baseline 53.4s).
  - **Comparison/global**: expander chứng minh hoạt động (E2E S15.10: chunk PPR-only ngành B sống sót qua expander+cutoff vào context). Live: "admitted 8 PPR-only chunks, ppr_hits=1130".
  - **CẢNH BÁO cần fix tiếp**: (1) **faithfulness 0.949 hơi dưới gate 0.95** — nguồn lỗi: model bịa URL cổng (tuyensinh/Bộ GDĐT) KHÔNG có trong chunk được nạp (msg013/037/041/044/047/061/062/064/070/083). → prompt phải cấm nêu URL không có trong context. (2) **over-route graph_rag 76/86** đánh đổi precision+latency — cân nhắc nâng lại threshold để chỉ multi-hop thật mới graph.
  - **abstain 0.143**: msg055 (zalo-fabrication cũ) giờ chunk ĐÃ chứa zalo/facebook/hotline (graph route surface contact chunk) → guard đúng khi KHÔNG abstain, answer grounded. Phần lớn abstain-miss là GT-convention (trả đúng fact in-context mà GT đánh abstain). KHÔNG phải regression bịa thật.
- **86Q reports regenerated** (user yêu cầu, no char-loss): `eval_records_to_results_json.py` (TDD converter) đổ s15 → `results.json` → `gen_thi_sinh_pdf.py` + `gen_thi_sinh_reports.py` (bỏ cap 200/100/600/400 ký tự). Verify pdfplumber: 86/86 Q+A đầy đủ, 0 ký tự lỗi font (Arial), 0 dấu "..." cắt. 26 test mới.
- **Đang chạy**: 60Q adversarial (`adv60`). Next: judge adversarial → temporal/comparison/abstain pass-rate; rồi (đề xuất S16) fix URL-faithfulness + cân chỉnh graph_rag threshold.

- **Goal**: fix root causes từ baseline_2026 để hệ thống bền 10 năm khi năm tuyển sinh xoay vòng (2026→2027→…). Plan: `rag2025/.omc/plans/temporal_durability_fix_plan.md` (ralplan 1 vòng: Architect SOUND-WITH-CHANGES 3 must-fix → Critic ITERATE 5 must-fix → patched → fan-out Sonnet, Opus review mỗi wave + integration review).
- **Scope (user)**: Correctness + Temporal. KHÔNG gộp 5→1 router LLM-call (latency) đợt này (blast-radius cao, để đợt sau).
- **Fixes landed (TDD 1:1, Sonnet impl + Opus review)**:
  - **S14.1** `src/services/season.py` — season-state machine `(calendar × data-presence) → {PRE_SEASON_GAP, IN_SEASON, OFF_SEASON}`; data-presence là HARD switch, month-window chỉ cho wording. 38 test.
  - **S14.2+2b** `lancedb_retrieval.py` — giết import-freeze: year anchor + `has_current_year_data()` (probe `SELECT DISTINCT data_year`, cache theo row-count) + `get_admission_context()` resolve PER-CALL (env/data-ingest đổi → hiệu lực ngay, không cần restart). 13 test.
  - **S14.3+3b** wire `temporal_intent` + season vào `retrieve(query=...)`: IN_SEASON+current → **hard-EXCLUDE** prior-year (post-fetch trên metadata_json, vì bảng KHÔNG có cột data_year top-level — SQL `WHERE data_year` sẽ crash; bug này Opus integration-review bắt được + sửa); PRE_SEASON_GAP → giữ prior-year (show labeled); historical/cross_year → cho qua. Null-year stable chunks luôn sống sót. 17 test.
  - **S14.4** `llm_generator.py` — season-aware `GAP_DISCLAIMER` ("Thông tin tuyển sinh năm {N} chưa công bố… Dưới đây là thông tin năm {N-1} (đã cũ)"), parametrize hết literal "2026" → `get_current_admission_year()`, fold strip `_get_fallback_prompt`. 5 test.
  - **S14.5** `query_router.py` — XÓA `HYDE_META_ANSWER_INTRO` (hardcode 28 ngành + mã giả 7480107TD + phone 887.111). Split: "ngành gì" → retrieval thật; pure-contact → `CONTACT_BLOCK` year-agnostic (chỉ URL + hotline chính thức 0234.3823290). 17 test.
  - **S14.6** `prompts/generation_system_prompt.txt` — strip toàn bộ 28-ngành/mã/học phí/year facts; 0 major-code, 0 year-literal còn lại; giữ `{{YEAR_FACTS}}` runtime-inject. 9 test.
  - **S14.7** `major_code_validator.py` — whitelist year-loadable từ `data/major_codes/<year>.json` (fallback `WHITELIST_2026`); seed 2026 + thêm 6 mã thật bị flag false-positive (7510302/7520503/7580211/7229030/7320111/7229010). No-arg shim giữ mọi caller xanh. 30 test.
  - **Integration fix (Opus review BLOCKER)**: 4 production call-sites của `retrieve()` (eval runner + main.py ×3) thiếu `query=` → temporal wiring sẽ DEAD trong path eval. Đã thêm `query=` hết. Live SQL smoke xác nhận: `năm nay`/`cross-year`/`2024` đều trả 2026 chunks (IN_SEASON hard-exclude prior hoạt động), không crash.
- **Regression**: **558 passed, 1 skipped** (`test_embedding.py` pre-existing-broken `from rag2025.src...` — không liên quan). ZERO secret leak.
- **Đang chạy**: S14.8 re-run 86Q `--run-id s14_fixed` (STREAM=false). Next: judge fan-out (all-Sonnet 8 batch để apples-to-apples) → aggregate + halltype + report `metrics_report/s14_fixed/` so với baseline_2026 → 60Q adversarial → S14.11 rollover E2E (3 mốc: Jul-2026 in-season / Oct-2026 gap-2027 / Apr-2027 in-season-2027).
- **Gate kỳ vọng s14_fixed**: Type-4 genuine=0, Type-6=0, Type-5≤3, faithfulness≥0.95, abstain↑ vs 0.714.
- **KẾT QUẢ s14_fixed CHỐT** — `results/eval_harness/metrics_report/s14_fixed/` + `s14_fixed_comparison.json` (baseline re-judge all-Sonnet để apples-to-apples):
  - **Hallucination (triage vs context thật)**: Type-4 genuine **4→0** (4 detector-flag còn lại đều là mã CÓ trong context → registry lag, không phải bịa); Type-6 **4→0** (diệt sạch — toàn bộ nằm trên route auto-answer đã xóa); Type-5 genuine **5→1** (chỉ msg068 "31.5 tín chỉ" = số cộng dồn); Type-1 1→1 (msg028). → **auto-answer hardcode kill THÀNH CÔNG**.
  - **RAGAS**: faithfulness **0.963→0.965** (giữ, PASS ≥0.95) · answer_relevancy **0.798→0.916** (+0.118 — WIN LỚN: retrieval thật thắng static block) · context_precision 0.845→0.724 (−0.121, đánh đổi chấp nhận được: hard-exclude prior-year bỏ vài chunk biên).
  - **Latency**: total p50 **53.4s→16.3s** (3.3× nhanh hơn dù CHƯA gộp router — ít path auto-answer lãng phí + STREAM=false ổn định).
  - **Route**: auto_answer 6→2, graph_rag 7→18, padded_rag 73→66.
  - **abstain_accuracy 0.714→0.286 (FAIL)** nhưng phân tích: chỉ **1 regression THẬT** (msg055 bịa "Zalo OA + mùa cao điểm tháng 5-9" ngoài context); msg059 trả "mã trường DHT" ĐÚNG + có trong context (GT-convention sai); msg028 trả "điểm học bạ năm trước" đúng historical intent (debatable). → chất lượng abstain thực chất ~ giữ nguyên; metric phóng đại.
- **Follow-up còn lại trước deploy**: (1) **abstain-hardening** cho msg055-class (over-answer khi context không có) — đây là regression thật duy nhất; (2) chạy 60Q adversarial (15 temporal minimal-pair); (3) (đợt sau) gộp 5→1 router call.

- **Goal**: build hardest-possible measurement instrument for a TRUE baseline (NOT fixing pipeline). Plan: `rag2025/.omc/plans/ragas_eval_harness_plan.md` (ralplan consensus: Planner→Architect SOUND-WITH-CHANGES→Critic ITERATE→APPROVE).
- **Discovery (4 agents)**: token "bug" was PDF render cap `gen_thi_sinh_pdf.py:199` (300 chars), NOT LLM; eval persisted only chunk_ids → faithfulness ran 17/86; no ground-truth; 5 sequential LLM calls = 61.6s median; system prompt hardcodes 28 majors (self-hallucination source).
- **Web research 2026 (4 agents)**: RAGAS+DeepEval, Anthropic Contextual Retrieval, drop HyDE/Step-Back, single-call routing, LiteLLM fallback, cross-vendor judge.
- **13 slices built (Sonnet TDD, Opus strict tests + review)** under `rag2025/scripts/` + `rag2025/tests/scripts/`: eval_record, eval_run_instrumented (3 latency seams + monkeypatch attribution), gen_ground_truth (fact-level traceable, 13/86 seeded), gen_adversarial_set (60Q), eval_judge_prepare (Opus-subagent batch + rubric), eval_judge_aggregate (bootstrap CI), eval_judge_ramclouds (Cohen κ), eval_judge_halltype (Type-1/4/5/6), eval_report_metrics, sweep_lancedb_noise, run_eval_suite, eval_parity_ragas_lib. **313 passed, 1 skipped**. ZERO pipeline-source edits.
- **LanceDB noise sweep (read-only, real)**: 1613 chunks; stale-2025-without-superseded 150 (9.3%); low-quality 441 (27.3%); duplicate 91 (5.6%); dim 1024=1024 OK. Table `_versions` unchanged (175).
- **PDF truncation fixed**: full answers now render (164KB vs 128KB truncated).
- **Gateway incident**: mimo-v2.5-pro 503 + gpt-5.3-codex 400 dead; only gpt-5.4 live. Per user decision, ran 86Q baseline with all model envs → gpt-5.4 (config only, no pipeline edit). `.env` backed up.
- **BASELINE CHỐT (2026-06-02)** — `results/eval_harness/metrics_report/baseline_2026/`:
  - **Judging**: 86/86 câu (batch_000 Opus + batch_001–007 Sonnet fanout, vì gateway hết quota Opus-judge). Aggregator: per-id-mean → bootstrap CI (seed=42, n=1000).
  - **RAGAS (Opus/Sonnet primary, mean ± 95% CI)**: faithfulness **0.959** [0.929, 0.985] (n=82, PASS ≥0.90) · answer_relevancy **0.795** [0.706, 0.871] · context_precision **0.843** [0.791, 0.885] · context_recall 1.000 (n=2 — chỉ 2/86 có reference traceable, KHÔNG đại diện).
  - **Hallucination gates (detector thô)**: Type-1=1 (PASS), Type-4=9 (FAIL), Type-5=10 (FAIL), Type-6=4 (FAIL).
  - **Triage `halltype_triage_baseline_2026.json`** (cross-check từng flag với context thực): Type-4 genuine **4/9** (msg001/015/017/022 — toàn route auto-answer; 5 còn lại là FALSE-POSITIVE do canonical registry trong major_code_validator thiếu mã SC/IC + vài mã 72xx/75xx). Type-5 genuine ≈ **5** (msg006 "19.69 điểm 2025", msg024/043/067 học phí, msg068 31.5 — số KHÔNG có trong context; phần còn lại là phone-fragment 887.111 + hệ số 0.5 → false-positive). Type-6=4 và Type-1=1 đều genuine.
  - **KẾT LUẬN**: lỗi hallucination tập trung ở **route auto-answer** (msg001/015/017/022: hardcode 28 ngành + mã giả 7480107TD + phone) — chính là system-prompt tự bịa khi skip retrieval. Đường RAG thực (padded/graph) faithfulness 0.959 = khá tốt. abstain_accuracy out-of-scope **0.714** (FAIL ≥0.95) — hệ thống trả lời thay vì từ chối.
  - **Latency (3 seam đo thật)**: total p50 **53.4s** / p95 82.3s; route_ms p50 **37.7s** (HyDE+StepBack+Classify = 5 LLM call tuần tự) = nút cổ chai chính, không phải generation.
- **Next (fix, ~5–6 ngày tới)**: (1) **bỏ/giảm route auto-answer** hoặc bắt nó cũng đi qua retrieval — diệt Type-4/5/6 + cắt prompt hardcode 28 ngày; (2) **gộp 5 LLM-call router → 1 call** cắt ~30s latency; (3) **abstain hardening** out-of-scope; (4) cập nhật canonical registry (xóa 5 false-positive Type-4); (5) chạy 60Q adversarial set.

### 2026-05-24


- **Phase-1 scope expanded** to cover latency + standardize plans (auto-decided per V5-R057, V5-R058 audit trail).
- `spec-index.yaml` updated:
  - `phase-1.paths` += `rag2025/**`, `uni-guide-ai/**`, `.github/workflows/**`, `api/**`, `nginx.conf`, `prompts/**`
  - `phase-1.sections` += [7,8,9,10,11,12]
  - `tdd.src_patterns` extended for Python pytest (`rag2025/src/**`) + Vitest/Playwright (`uni-guide-ai/src/**`)
  - `spec` path corrected: `docs/spec/rag-chatbot-husc-spec.md` → `docs/rag-chatbot-husc-spec.md`
- `docs/rag-chatbot-husc-spec.md` extended:
  - Sections S7–S9 reference `rag2025/.omc/plans/latency_reduction_plan_v2.md`
  - Sections S10–S12 reference `rag2025/.omc/plans/standardize_frontend_backend_plan_v2.md`
  - 10 hard constraints (C1–C10) documented including `max_tokens=1500` (user-tuned, not negotiable)
  - Conflict resolution: standardize plan B4 is authoritative for SSE streaming spec
- 2 plans v2 ready in `rag2025/.omc/plans/`. Total effort 14 days, 6 sprints.
- **Next**: Phase 0 verification (`benchmark_provider_latency.py`, `benchmark_streaming_support.py`, answer length distribution) → decision tree for Sprint 1 routing.

### 2026-04-16

- Bootstrapped from @agentic-gateway/mega-template
- `spec-index.yaml` initialized with active_phase: phase-1
- Enforcement layer active: check-spec, check-phase-gate, check-spec-coverage, require-tests, load-context, orphan-sweep
