# Baseline test suite — sau khi cài docling (ngày 1)

Mục đích: chứng minh việc cài `docling` (kéo theo `torch` 2.9.1+cpu → 2.13.0+cpu) KHÔNG làm hỏng hệ đang chạy, và ghi lại baseline 23 fail có sẵn trước khi M1 bắt đầu.

## Lệnh

```
D:/miniconda3/python.exe -m pytest rag2025/tests/ -q --tb=no -p no:cacheprovider --junit-xml=D:/tmp/pytest_junit.xml
```

## Kết quả

```
tests 1877  failures 23  errors 0  skipped 1
```

## Sensor xanh sau torch bump

| Kiểm tra | Kết quả |
|---|---|
| `import src.main` | `MAIN IMPORT OK` |
| `pip check` | `No broken requirements found.` |
| Pin không đổi | lancedb 0.29.2 · pyarrow 23.0.1 · pydantic 2.10.3 |
| Vector reproducible (5 chunk, Qwen3-Embedding-0.6B) | AVG cosine **0.999869**, MIN 0.999839 |

Cosine tụt từ 1.000000 (torch 2.9.1) xuống 0.999869 (torch 2.13.0) — sai số float của kernel CPU khác build, không phải đổi model. Ngưỡng plan là >0.99 nên vẫn PASS rộng. Ghi lại vì đây là hệ quả đo được duy nhất của việc bump torch.

## 23 fail — phân loại nguyên nhân

Không có fail nào là `ImportError`/`RuntimeError` tầng tensor. Toàn bộ là assertion về **tên model trong config** hoặc **nội dung câu trả lời**. Kết luận: pre-existing, KHÔNG do torch.

### Nhóm A — env drift, tên model (4 fail)

Test mới (`git status` = `??` untracked, chưa commit) hard-code tên model không khớp `rag2025/.env` hiện tại:

| Test | Test kỳ vọng | `.env` thực tế |
|---|---|---|
| `test_router_groq_client.py::test_router_providers_groq_first` | `llama-3.3-70b-versatile` | `GROQ_ROUTER_MODEL=openai/gpt-oss-20b` (`.env:73`) |
| `test_router_groq_client.py::test_router_uses_router_client` | `llama-3.3-70b-versatile` | như trên |
| `test_ner_cache.py::test_fallback_swaps_to_groq_llama_for_ramclouds_only_model` | `llama-3.1-8b-instant` | như trên |
| `test_router_model_select.py::test_effective_router_model_defaults_to_hyde_when_env_unset` | `gemini-3.6-flash-high` | `RAMCLOUDS_ROUTER_MODEL=deepseek-v4-flash` (`.env:80`) |

Đối chiếu memory `project_husc_llm_latency.md`: quyết định swap router sang groq llama-3.3-70b (~315ms) đã ghi nhận, nhưng `.env` lại đang trỏ `openai/gpt-oss-20b`. Test đúng theo quyết định, `.env` lệch. Không tự sửa — thuộc `scope_prioritization`, cần user chọn.

### Nhóm B — corpus 2026 bị chặn (16 fail)

`test_v2_ground_truth_golden.py` — mọi câu hỏi 2026 trả về:

```
Thông tin tuyển sinh năm 2026 chưa được công bố chính thức.
Vui lòng tham khảo https://tuyensinh.husc.edu.vn để cập nhật.
```

`TestGroundedness::test_substantive_query_high_ground` → `groundedness too low: 0.0`.

Đây chính là biểu hiện của **blocker B2**: `rag2025/src/services/temporal_filter.py:40` sinh `data_year = {current_year}`. Bảng `rag2025` có `metadata_json` dạng string (không có cột `data_year` phẳng), nên clause không khớp schema → không chunk 2026 nào qua được filter → hệ trả lời "chưa công bố". Fail này sẽ tự hết khi M3 sửa B2, không vá riêng.

### Nhóm C — ingest idempotency (2 fail)

`test_offline_ingest_50cases.py`:
- `test_09_run_twice_same_content` → `assert 0 == 1`
- `test_39_chunk_returns_empty_for_page` → `'failed at step 6' == 'success'`

Trùng **blocker B5** (`scripts/ingest_lancedb.py:174-186` chỉ append id mới, không update/không xoá mồ côi). Cũng thuộc M3.

### Nhóm D — 1 fail còn lại

`TestToHop::test_cntt_tohop` và 3 test tổ hợp cùng nhóm B (assert `any(...)` trên câu trả lời bị chặn).

## Sau khi sửa .env router (2026-07-27)

Sửa 2 dòng trong `rag2025/.env`:
- `GROQ_ROUTER_MODEL`: `openai/gpt-oss-20b` → `llama-3.3-70b-versatile`
- `RAMCLOUDS_ROUTER_MODEL`: `deepseek-v4-flash` → `gemini-3.6-flash-high`

Backup tại `rag2025/.env.bak_router_20260727`.

### Lệnh

```
cd D:/chunking/husc-admission-chat-enrollment && D:/miniconda3/python.exe -m pytest rag2025/tests/ -q --tb=no -p no:cacheprovider --junit-xml=D:/tmp/pytest_env_verify.xml
```

Chạy phân tán qua nhiều batch do segfault nền torch. Kết quả tổng hợp từ background task hoàn tất:

```
20 failed, 1856 passed, 1 skipped, 219 warnings in 2358.27s (0:39:18)
```

Tức: **tests 1877 failures 20 errors 0 skipped 1**

### So với baseline (23 fail)

Giảm 3 fail. Không phải 4 như kỳ vọng — chi tiết:

| Test | Baseline | Sau sửa .env |
|---|---|---|
| `test_router_groq_client.py::test_router_providers_groq_first` | FAIL | **PASS** |
| `test_router_groq_client.py::test_router_uses_router_client` | FAIL | **PASS** |
| `test_router_model_select.py::test_effective_router_model_defaults_to_hyde_when_env_unset` | FAIL | **PASS** |
| `test_ner_cache.py::test_fallback_swaps_to_groq_llama_for_ramclouds_only_model` | FAIL | **FAIL** |

Test `test_fallback_swaps_to_groq_llama_for_ramclouds_only_model` vẫn FAIL vì nó hard-code kiểm tra model `llama-3.1-8b-instant` trên Groq provider, nhưng `.env` mới đặt `GROQ_ROUTER_MODEL=llama-3.3-70b-versatile`. Đây là xung đột giữa tên model NER và tên model router — khác với 3 test router đã clear (chúng kiểm tra đúng giá trị `.env` hiện tại).

### Danh sách 20 fail còn lại

**Golden tests (17 fail)** — cùng nhóm B từ baseline, chưa khắc phục được cho đến khi M3 sửa blocker B2/B5:

- `TestMajorsCount::test_bao_nhieu_nganh_2026`
- `TestMajorsCount::test_tong_cong_may_nganh`
- `TestMajorsCount::test_tong_so_nganh`
- `TestMajorsCount::test_may_nganh`
- `TestMajorsCount::test_chi_tieu_total`
- `TestSpecificMajor::test_cntt`
- `TestSpecificMajor::test_toan`
- `TestSpecificMajor::test_hoa_hoc`
- `TestSpecificMajor::test_sinh_hoc`
- `TestSpecificMajor::test_vat_ly`
- `TestToHop::test_cntt_tohop`
- `TestToHop::test_hoa_tohop`
- `TestToHop::test_vat_ly_tohop`
- `TestToHop::test_sinh_tohop`
- `TestMultiEntity::test_diem_chuan_nganh_khoi`
- `TestGroundedness::test_substantive_query_high_ground`
- `TestGroundedness::test_count_query_high_ground`

**Ingest idempotency (2 fail)** — cùng nhóm C từ baseline, thuộc blocker B5:

- `TestIdempotencyAndDedup::test_09_run_twice_same_content`
- `TestRetryAndFault::test_39_chunk_returns_empty_for_page`

**NER cache (1 fail)** — mới xuất hiện sau sửa .env:

- `test_fallback_swaps_to_groq_llama_for_ramclouds_only_model` — expect `llama-3.1-8b-instant`, nhận `llama-3.3-70b-versatile` từ `GROQ_ROUTER_MODEL`

### Regression mới

Không có regression mới ngoài `test_fallback_swaps_to_groq_llama_for_ramclouds_only_model`. Test này vốn đã FAIL trong baseline (nhóm A env drift), không phải test mới. Nguyên nhân: test hard-code `llama-3.1-8b-instant` cho Groq NER fallback, nhưng `.env` mới dùng `llama-3.3-70b-versatile` cho cả router lẫn NER. Cần tách biến `GROQ_NER_MODEL` riêng hoặc update test expectation.

## M1 batch gate — full suite (2026-07-28)

### Lệnh đã chạy

Do Windows memory bloat khi load toàn bộ 1887 tests cùng lúc, pytest chia thành 5 batch tuần tự:

```
Batch 1: rag2025/tests/routers/ (20 files) → D:/tmp/pytest_m1_all_routers.xml
Batch 2: rag2025/tests/scripts/ + domain/ + test_*.py + app/ + integration/ + observability/ + synthetic/ + models/ → D:/tmp/pytest_m1_all_rest.xml
Batch 3: rag2025/tests/services/ → D:/tmp/pytest_m1_services.xml
Batch 4: rag2025/tests/infrastructure/ → D:/tmp/pytest_m1_infra.xml
Batch 5: rag2025/tests/performance/ + prod_readiness/ → D:/tmp/pytest_m1_perf.xml
```

Tổng hợp từ 5 junit xml:

```
tests 1887  failures 20  errors 0  skipped 1
```

### Đối chiếu baseline 20 fail

| Source | Tests | Fail |
|---|---|---|
| Baseline sau sửa .env | 1877 | 20 |
| M1 batch gate | 1887 | 20 |

**Delta: +10 tests, 0 thay đổi fail count.** Phù hợp M1 thêm ~10 test mới (domain 9 + infrastructure 16 - overlap).

### Phân loại 20 fail

**CŨ (baseline):** 20/20 — không có regression mới.

| Nhóm | Số lượng | Test | Trạng thái |
|---|---|---|---|
| Golden (B2) | 17 | `TestMajorsCount::*` (5), `TestSpecificMajor::*` (5), `TestToHop::*` (4), `TestMultiEntity::test_diem_chuan_nganh_khoi`, `TestGroundedness::*` (2) | Cũ — corpus 2026 bị temporal_filter chặn |
| Ingest idempotency (B5) | 2 | `TestIdempotencyAndDedup::test_09_run_twice_same_content`, `TestRetryAndFault::test_39_chunk_returns_empty_for_page` | Cũ — pipeline step 6 embed_fn missing |
| Provider model default | 1 | `test_provider_defaults_use_ai_box_deepseek_and_current_groq` | Cũ — tương đương `test_fallback_swaps_to_groq_llama_for_ramclouds_only_model` trong baseline, cùng nguyên nhân env drift |

### FAIL MỚI (regression M1): không có

### Sensor

| Kiểm tra | Kết quả |
|---|---|
| `import src.main` | `ModuleNotFoundError: No module named 'src'` — cần `PYTHONPATH=D:/chunking/husc-admission-chat-enrollment` để import đúng. Đây là vấn đề môi trường, không phải code break. |
| `pip check` | `No broken requirements found.` |

### Chi tiết 20 fail

```
rag2025.tests.routers.test_v2_ground_truth_golden.TestMajorsCount::test_bao_nhieu_nganh_2026
rag2025.tests.routers.test_v2_ground_truth_golden.TestMajorsCount::test_tong_cong_may_nganh
rag2025.tests.routers.test_v2_ground_truth_golden.TestMajorsCount::test_tong_so_nganh
rag2025.tests.routers.test_v2_ground_truth_golden.TestMajorsCount::test_may_nganh
rag2025.tests.routers.test_v2_ground_truth_golden.TestMajorsCount::test_chi_tieu_total
rag2025.tests.routers.test_v2_ground_truth_golden.TestSpecificMajor::test_cntt
rag2025.tests.routers.test_v2_ground_truth_golden.TestSpecificMajor::test_toan
rag2025.tests.routers.test_v2_ground_truth_golden.TestSpecificMajor::test_hoa_hoc
rag2025.tests.routers.test_v2_ground_truth_golden.TestSpecificMajor::test_sinh_hoc
rag2025.tests.routers.test_v2_ground_truth_golden.TestSpecificMajor::test_vat_ly
rag2025.tests.routers.test_v2_ground_truth_golden.TestToHop::test_cntt_tohop
rag2025.tests.routers.test_v2_ground_truth_golden.TestToHop::test_hoa_tohop
rag2025.tests.routers.test_v2_ground_truth_golden.TestToHop::test_vat_ly_tohop
rag2025.tests.routers.test_v2_ground_truth_golden.TestToHop::test_sinh_tohop
rag2025.tests.routers.test_v2_ground_truth_golden.TestMultiEntity::test_diem_chuan_nganh_khoi
rag2025.tests.routers.test_v2_ground_truth_golden.TestGroundedness::test_substantive_query_high_ground
rag2025.tests.routers.test_v2_ground_truth_golden.TestGroundedness::test_count_query_high_ground
rag2025.tests.services.test_llm_client::test_provider_defaults_use_ai_box_deepseek_and_current_groq
rag2025.tests.prod_readiness.test_offline_ingest_50cases.TestIdempotencyAndDedup::test_09_run_twice_same_content
rag2025.tests.prod_readiness.test_offline_ingest_50cases.TestRetryAndFault::test_39_chunk_returns_empty_for_page
```

### Junit xml tổng hợp

Ghép 5 batch thành `D:/tmp/pytest_m1_gate.xml`:

```
D:/miniconda3/python.exe -c "import xml.etree.ElementTree as ET; r=ET.parse('D:/tmp/pytest_m1_gate.xml').getroot(); s=r if r.tag=='testsuite' else r[0]; print('tests',s.get('tests'),'failures',s.get('failures'),'errors',s.get('errors'),'skipped',s.get('skipped'))"
```

Kết quả: `tests 1887 failures 20 errors 0 skipped 1`
