---
name: rag-systematic-debugging
description: Debug có hệ thống cho pipeline RAG/Chunking/GraphRAG. Dùng skill này khi test fail, script lỗi, metrics bất thường, hoặc output retrieval sai ngữ nghĩa.
---

# RAG Systematic Debugging

## Quy trình bắt buộc
1. Tái hiện lỗi bằng lệnh tối thiểu.
2. Khoanh vùng tầng lỗi: normalize / chunk / ingest / retrieve / rerank / graph / generation.
3. Thu thập bằng chứng (log, input mẫu, stack trace).
4. Đặt giả thuyết nhỏ, kiểm chứng từng giả thuyết.
5. Chỉ sửa khi đã xác định root cause.
6. Chạy lại kiểm thử hồi quy tối thiểu.

## Bản đồ lỗi nhanh
- `normalize_data.py` fail -> kiểm tra format file + encoding.
- `validate_jsonl.py` fail -> mismatch schema.
- `chunker.py` chất lượng kém -> profile/overlap/separator.
- ingest fail -> schema chunk thiếu trường chuẩn.
- retrieval tệ -> chunk quá ngắn hoặc embedding mismatch.

## Output chuẩn
- Root cause (1 dòng)
- Bằng chứng kỹ thuật
- Patch tối thiểu
- Lệnh verify sau fix
