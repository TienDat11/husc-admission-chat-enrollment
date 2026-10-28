# T00 — Ghim baseline (commit SHA + cấu hình)

**Mục đích:** mốc hồi quy cho mọi so sánh sau này (T15, T20). Đây là bản ghim
được **track trong git**; bộ artifact đo đầy đủ nằm cạnh file này nhưng bị
`.gitignore` loại (`rag2025/results/**/*.{md,json,jsonl}`).

## Định danh cây đo

| Trường | Giá trị |
|---|---|
| Commit | `e18c0fcf67a24278f3d3e5636d094fae5ecb9f21` |
| Ngày chạy | 2026-10-06 |
| Harness | `rag2025/scripts/eval_run_instrumented.py` (pipeline thật: `UnifiedRAGPipeline.from_disk` + `LanceDBRetriever.from_env` + `EmbeddingService` + `LLMGenerator`) |
| Trạng thái cây | **Bẩn — 75 file chưa commit, không liên quan ticket. Không sửa gì, không commit.** |

> ⚠️ **Cảnh báo khi dùng lại:** SHA trên là commit *HEAD* tại thời điểm đo, không
> phải cây làm việc. Cây làm việc khi đó có 75 file đã sửa chưa commit, nên SHA
> một mình **không** tái lập được đúng mã đã chạy. Muốn tái lập chính xác phải
> checkout `e18c0fc` **rồi** áp cùng bộ thay đổi chưa commit (không còn lưu lại).
> Coi đây là mốc *số đo*, không phải mốc *mã nguồn*.

## Cấu hình đo (snapshot `config_snapshot.json`, API key đã che)

| Key | Giá trị |
|---|---|
| `EMBEDDING_PROVIDER` | qwen |
| `EMBEDDING_MODEL` | Qwen/Qwen3-Embedding-0.6B |
| `EMBEDDING_DIM` | 1024 |
| `RERANKER_ENABLED` | true |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 |
| `RERANKER_WEIGHT` | 0.35 |
| `USE_HYBRID_RETRIEVAL` | false |
| `LANCEDB_URI` | ./data/lancedb |
| `LANCEDB_TABLE` | husc_chunks |
| `LANCEDB_ENTITY_TABLE` | husc_entities |
| `RAMCLOUDS_MODEL` / NER / HYDE / GEN / GEN_FALLBACK / ROUTER | gemini-3.8-flash |
| `RAMCLOUDS_BASE_URL` | https://ramclouds.me/v1 |
| `GRAPHRAG_SIMPLE_THRESHOLD` | null |

## Artifact đo (không track)

| File | Nội dung |
|---|---|
| `husc86q_records_t00.jsonl` | 303 dòng (tên file "86q", thực chứa 303 câu) — route, `sources`, có trích dẫn, câu trả lời |
| `heldout40_records_t00.jsonl` | 40 câu |
| `baseline_T00_detail.json` | chi tiết từng câu |
| `config_snapshot.json` | snapshot cấu hình đầy đủ |
| `SUMMARY.md` | tổng hợp + cảnh báo |

## Trạng thái tiêu chí

- [x] **AC1** — SHA + cấu hình nằm ở đường dẫn được track (file này).
- [x] **AC2** — mỗi câu ghi route, `sources` chunk id, có trích dẫn hay không, câu trả lời.
- [ ] **AC3** — expected-record recall@5: **KHÔNG THỰC HIỆN ĐƯỢC** — dataset 86q thiếu cột `expected_record_ids` (chỉ có `gt_chunks`); map `gt_chunks`→record 0/44. Chỉ đo được chunk-level recall@5 trên 87/303 câu: macro=micro=**0.1149**.
- [ ] **AC4** — tỷ lệ đúng sự kiện do chuyên gia chấm: **BLOCKED** — cần người chấm.

**Cảnh báo dùng lại:** provider LLM hết quota lúc chạy → 281/303 câu trả lời là
fallback. ROUTE/RETRIEVAL/RERANK/LATENCY vẫn 100% thật.
