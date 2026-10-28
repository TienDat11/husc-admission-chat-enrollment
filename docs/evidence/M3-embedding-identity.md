# Probe 3: Embedding Identity Report

## STEP 1 — LanceDB Table Inventory

### Table: `rag2025`
- **Rows:** 1671
- **Version:** 396
- **Schema:**
  ```
  id: string
  chunk_id: string
  text: string
  source: string
  faq_type: string
  metadata_json: string
  vector: fixed_size_list<item: float>[1024]
  ```
- **Indices:** [] (empty)

### Table: `husc_v2026_clean`
- **Rows:** 448
- **Version:** 2
- **Schema:**
  ```
  id: string
  text: string
  summary: string
  embedding: fixed_size_list<item: float>[1024]
  metadata: struct<approved_by: string, approved_date: string, audience: string, ... 19 more fields>
  ```
- **Indices:** [] (empty)

### Blocker B1 (schema drift) — CONFIRMED
- `rag2025` uses column name `vector` + `metadata_json:string`
- `husc_v2026_clean` uses column name `embedding` + `metadata:struct` (rich schema with 20 fields)
- Both have dim=1024

### Blocker B4 (no indices) — CONFIRMED
- `list_indices()` returns `[]` for both tables

## STEP 2 — Code Evidence Chain

| Setting | File | Line | Value |
|---------|------|------|-------|
| EMBEDDING_PROVIDER | `.env` | 29 | `qwen` |
| EMBEDDING_MODEL | `.env` | 30 | `Qwen/Qwen3-Embedding-0.6B` |
| EMBEDDING_DIM | `.env` | 31 | `1024` |
| EMBEDDING_PROVIDER | `.env.example` | 44 | `qwen` |
| EMBEDDING_MODEL | `.env.example` | 45 | `Qwen/Qwen3-Embedding-4B` |
| EMBEDDING_DIM | `.env.example` | 46 | `2560` |
| Default model | `config/settings.py` | 42-43 | `Qwen/Qwen3-Embedding-4B` |
| Default dim | `config/settings.py` | 46 | `2560` |
| Load path | `src/services/embedding.py` | 54-60 | `SentenceTransformer(model_name, device="cpu")` |

**Active model in production (.env):** `Qwen/Qwen3-Embedding-0.6B`, dim=1024, provider=qwen.
**Model cached at:** `D:/chunking/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B`

## STEP 3 — Empirical Cosine Check

Re-embedded 5 chunks from `rag2025` table using `EmbeddingService.encode_documents()` (same function the app uses).

| Chunk ID | Cosine(stored, re-embedded) |
|----------|---------------------------|
| all_bogddt_tt_08_header_2025 | 1.000000 |
| all_bogddt_tt_08_dieu_2_nguong_dau_vao_2025 | 1.000000 |
| csdt_admin_bogddt_tt_08_dieu_2_xu_ly_nguyen_vong_2025 | 1.000000 |
| thi_sinh_bogddt_tt_08_dieu_2_diem_uu_tien_2025 | 1.000000 |
| thi_sinh_bogddt_tt_08_dieu_2_xet_tuyen_thang_2025 | 1.000000 |

- **AVG_COSINE:** 1.000000
- **MIN_COSINE:** 1.000000
- **MAX_COSINE:** 1.000000

## STEP 4 — EMBEDDING_DIM Consistency

| Source | File:Line | Value |
|--------|-----------|-------|
| .env.example | line 46 | 2560 |
| .env | line 31 | 1024 |
| Real vector dim (step 1) | — | 1024 |

`.env` value (1024) matches the actual stored vector dimension. `.env.example` defaults to 2560 (Qwen3-Embedding-4B preset) but was overridden in `.env` to 1024 (Qwen3-Embedding-0.6B).

## STEP 3b — Cross-check với bge-m3 (đóng review BOUNCE)

Reviewer (Sonnet) BOUNCE đúng nguyên tắc: test ở STEP 3 chỉ chứng minh config hiện tại tự tái tạo được chính nó, chưa loại được model ứng viên khác. Plan §3.3 item 3 + Phụ lục B item 3 chỉ định `bge-m3`. Đã chạy cross-check với `BAAI/bge-m3` (đã cache tại `D:/chunking/.cache/huggingface/hub/models--BAAI--bge-m3`, khai trong `.env:39-40` là `BGE_EMBEDDING_MODEL`/`BGE_EMBEDDING_DIM=1024`) trên **cùng 5 chunk**:

```
bge-m3 out dim: (5, 1024)
BGE  cos -0.074073 all_bogddt_tt_08_header_2025
BGE  cos -0.076333 all_bogddt_tt_08_dieu_2_nguong_dau_vao_2025
BGE  cos -0.043909 csdt_admin_bogddt_tt_08_dieu_2_xu_ly_nguyen_vong_2025
BGE  cos -0.037908 thi_sinh_bogddt_tt_08_dieu_2_diem_uu_tien_2025
BGE  cos -0.018391 thi_sinh_bogddt_tt_08_dieu_2_xet_tuyen_thang_2025
BGE AVG -0.050123
```

| Model ứng viên | AVG cosine vs vector đã lưu | Kết luận |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` (`.env` hiện tại) | **1.000000** | model đã tạo dữ liệu |
| `BAAI/bge-m3` (plan giả định) | **−0.050123** | LOẠI — cùng dim 1024 nhưng không gian vector khác hoàn toàn |

Hai model cùng `dim=1024` nên dim một mình KHÔNG định danh được model — đúng như M3-R4 lo ngại. Cosine bây giờ mới discriminative: 1.000000 vs −0.05 là hai cực. bge-m3 bị loại bằng số, không phải bằng suy luận.

Hệ quả cho plan: `.env.example` phải sửa `EMBEDDING_PROVIDER=qwen` + `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B` + `EMBEDDING_DIM=1024` — **không** phải preset `bge` như plan §M3 DoD (dòng 648) viết. Đó là một sai sót của plan, phát hiện từ probe này.

## VERDICT

**MATCH — không cần re-embed.** Vector đã lưu do `Qwen/Qwen3-Embedding-0.6B` (dim=1024) tạo, khớp `.env` hiện tại (cosine 1.000000). `BAAI/bge-m3` đã bị loại bằng cross-check (cosine −0.050). Tiết kiệm được +2 NC mà M3-R4 dự phòng.
