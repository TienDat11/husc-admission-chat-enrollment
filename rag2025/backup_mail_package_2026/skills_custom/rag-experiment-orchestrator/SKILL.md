---
name: rag-experiment-orchestrator
description: Điều phối chạy thực nghiệm RAG/GraphRAG end-to-end (normalize, chunk, ingest, evaluate, export metrics). Dùng skill này khi người dùng muốn chạy benchmark, tái lập kết quả, lấy số liệu điền luận văn/PPT, hoặc chuyển máy mà cần playbook chạy lại.
---

# RAG Experiment Orchestrator

## Input tối thiểu
- Root project path.
- Dữ liệu raw hoặc chunked.
- File cấu hình `.env` hợp lệ.

## Runbook chuẩn
1. Preflight: kiểm tra Python, dependencies, `.env`, thư mục dữ liệu.
2. Data pipeline:
   - `normalize_data.py`
   - `validate_jsonl.py`
   - `chunker.py`
   - `normalize_chunks.py`
3. Retrieval/graph pipeline:
   - ingest vector store
   - build graph
4. Evaluation:
   - chạy script eval
   - xuất `final_metrics.json`
5. Export:
   - tạo CSV summary
   - sinh LaTeX snippet để điền bảng.

## Quy tắc chất lượng
- Luôn lưu artifact trung gian (json/csv/log).
- Mỗi con số báo cáo phải truy ngược được về file nguồn.
- Nếu pipeline fail, trả báo cáo `BLOCKED` + nguyên nhân + bước tiếp theo.

## Output chuẩn
- `RUN_REPORT.md`
- `results/final_metrics.json`
- `results/thesis_eval_summary.csv`
- optional: `results/latex_rows.txt`

## Troubleshooting nhanh
- Lỗi API key: kiểm tra `.env`.
- Lỗi schema: chạy lại `normalize_chunks.py --in-place`.
- Lỗi mạng cloud DB: fallback local-first (LanceDB).
