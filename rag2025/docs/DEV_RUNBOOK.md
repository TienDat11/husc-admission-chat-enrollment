# Vận hành nhanh: .env, model, và chạy FastAPI

## 1) Fill biến môi trường bắt buộc
Copy file mẫu:

```bash
copy .env.example .env
```

Mở `.env` và điền tối thiểu:

- `RAMCLOUDS_API_KEY` (LLM chính)
- `GROQ_API_KEY` (guardrail + fallback free)
- `LANCEDB_URI` (mặc định local: `./data/lancedb`)
- `LANCEDB_TABLE` (mặc định: `rag2025`)
- `QWEN_EMBEDDING_MODEL` (`Qwen/Qwen3-Embedding-8B`)
- `ERROR_EXPOSURE_MODE=dev` (để UI thấy lỗi chi tiết khi dev)

## 2) Cài dependencies

```bash
pip install -r requirements.txt
```

## 3) Ingest dữ liệu vào LanceDB

```bash
python scripts/ingest_lancedb.py
python scripts/check_lancedb.py
```

## 4) Kiểm tra model/guardrail đang chạy

- Guardrail model: `GUARDRAIL_MODEL` (default: `llama-3.1-8b-instant`)
- Exposure mode:
  - `dev`: trả `internal_status_code` chi tiết (`NOT_IN_HUSC_SCOPE`, `HUSC_ENTITY_NOT_FOUND`, `INSUFFICIENT_DATA`)
  - `prod`: chỉ trả mã công khai duy nhất

Bạn có thể kiểm tra nhanh bằng API `/health`.

## 5) Chạy FastAPI

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger: `http://localhost:8000/docs`

## 6) Contract response cho UI

`POST /query` trả thêm:

- `status_code`
- `status_reason`
- `data_gap_hints`
- `internal_status_code` (chỉ rõ khi `ERROR_EXPOSURE_MODE=dev`)

## 7) Quan sát nhanh metrics và groundedness

- `GET /metrics`: xem counters nội bộ như:
  - `query_total`, `query_errors`, `query_cache_hits`
  - `query_guardrail_blocks`, `query_pii_blocks`
  - `query_low_groundedness`, `unified_query_low_groundedness`
- `GET /health`: ngoài tình trạng LanceDB còn có thêm snapshot `metrics`
- `trace_id`: mỗi response từ `/query` và `/v2/query` đều có `trace_id` để lần ngược log

### Khi nào cần chú ý
- `query_low_groundedness` tăng: câu trả lời bám nguồn kém, cần kiểm tra chunking/retrieval/reranker
- `query_pii_blocks` tăng: người dùng đang gửi dữ liệu nhạy cảm, cần UX nhắc che bớt thông tin
- `query_errors` hoặc `unified_query_errors` tăng: ưu tiên kiểm tra provider/LanceDB/graph pipeline

## 8) Failure taxonomy tối thiểu

- `SENSITIVE_PII_DETECTED`: query chứa CCCD/email/điện thoại hoặc dữ liệu nhạy cảm tương tự
- `NOT_IN_HUSC_SCOPE`: query ngoài phạm vi tư vấn tuyển sinh HUSC
- `HUSC_ENTITY_NOT_FOUND`: không tìm thấy ngành/sự việc cụ thể trong dữ liệu hiện có
- `INSUFFICIENT_DATA`: dữ liệu hiện có không đủ để trả lời đáng tin cậy
- `LOW_GROUNDEDNESS`: câu trả lời sinh ra có mức bám sát nguồn thấp hơn ngưỡng `GROUNDING_THRESHOLD`

## 9) Canary checklist tối thiểu

Trước khi rollout:
- Chạy `tests/test_api.py`
- Chạy nhóm test retrieval trọng yếu (`test_hybrid_search.py`, `test_graphrag.py`)
- Gọi thử `/health` và `/metrics`
- Kiểm tra response có `trace_id`
- Kiểm tra query có PII bị block đúng với `SENSITIVE_PII_DETECTED`

Sau rollout canary:
- Theo dõi `query_errors`, `unified_query_errors`
- Theo dõi `query_low_groundedness`, `unified_query_low_groundedness`
- Nếu error tăng hoặc groundedness cảnh báo tăng đột biến, rollback ngay build mới

## 10) SLO gợi ý tối thiểu

- `/query`: tỷ lệ lỗi ứng dụng < 2%
- `/v2/query`: tỷ lệ lỗi ứng dụng < 3%
- `query_low_groundedness / query_total` nên giữ ở mức thấp và theo dõi xu hướng theo ngày
- Guardrail PII phải block đúng 100% với các pattern kiểm thử chuẩn
