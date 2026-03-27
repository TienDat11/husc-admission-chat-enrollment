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

UI `@uni-guide-ai` có thể bắt các field này để hiển thị cảnh báo/nhãn dữ liệu thiếu.
