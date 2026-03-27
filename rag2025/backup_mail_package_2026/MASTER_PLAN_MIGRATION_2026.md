# MASTER PLAN DI CHUYỂN & CHẠY THỰC NGHIỆM (MÁY KHÁC)

## Mục tiêu
- Chạy lại toàn bộ pipeline RAG/Chunking trên máy mới.
- Sinh thống kê + chỉ số để điền vào 2 file LaTeX:
  - `rag-2026.tex`
  - `rag-chunking-2026.tex`
- Có notebook (`.ipynb`) để tái lập nhanh và xuất số liệu.

---

## 0) Gói backup này chứa gì
- `rag-2026.tex`
- `rag-chunking-2026.tex`
- `MIGRATION_COMMANDS.md`
- `TREND_2026_NOTES.md` (tóm tắt xu hướng từ Exa + Context7)
- `notebooks/`
  - `01_chunking_stats_for_thesis.ipynb`
  - `02_graphrag_eval_for_thesis.ipynb`
  - `03_latex_fill_helper.ipynb`

---

## 1) Checklist môi trường máy mới
1. Cài Python 3.10+ (khuyến nghị 3.11/3.12).
2. Có đủ dung lượng trống:
   - Tối thiểu 12 GB cho env + cache + dữ liệu.
3. Giải nén package vào thư mục làm việc, ví dụ:
   - `D:\work\rag2025`
4. Tạo/điền `.env` theo `.env.example`.

---

## 2) Luồng chạy chuẩn (khuyến nghị)
1. `fix_deps.bat`
2. `setup_data.bat data\raw`
3. `run_lab.bat`
4. Kiểm tra API tại `http://localhost:8000/docs`
5. Chạy notebooks theo thứ tự 01 -> 02 -> 03

---

## 3) Luồng chạy thủ công (khi cần debug)
1. `python src\normalize_data.py data\raw data\normalized\normalized_2.jsonl`
2. `python src\validate_jsonl.py data\normalized\normalized_2.jsonl config\rag_chunk_schema.json data\validated\validated_2.jsonl`
3. `python src\chunker.py data\validated\validated_2.jsonl data\chunked\chunked_2.jsonl config\chunk_profiles.yaml auto`
4. `python src\normalize_chunks.py data\chunked --in-place`
5. `python scripts\ingest_lancedb.py`
6. `python scripts\build_graph.py`
7. `python scripts\evaluate_graphrag.py`

---

## 4) Điền số vào LaTeX

### 4.1 `rag-chunking-2026.tex`
- Bảng kết quả chính: `tab:main_results`
- Nguồn số:
  - Notebook `02_graphrag_eval_for_thesis.ipynb`
  - `results/final_metrics.json`
- Lưu ý: nếu chỉ có baseline + GraphRAG thì điền trước các dòng có dữ liệu, các dòng còn lại giữ `--` hoặc ghi chú rõ là chưa chạy.

### 4.2 `rag-2026.tex`
- Điền các bảng/chỉ số evaluation chapter 3 bằng output từ notebook 02.
- Notebook 03 hỗ trợ xuất trực tiếp snippet LaTeX row/mini-table.

---

## 5) Tiêu chuẩn chất lượng trước khi chốt số liệu
- Có log chạy thực nghiệm + timestamp.
- Có file kết quả JSON gốc (không chỉ copy số vào tex).
- Có ảnh/charts hoặc bảng tạm trong notebook để truy vết.
- Nếu có simulated metrics thì phải đánh dấu rõ simulated.

---

## 6) Rủi ro thường gặp
- Thiếu API key -> pipeline ngắt ở bước LLM/embedding.
- Sai đường dẫn data raw -> normalize không sinh file.
- Mismatch schema chunk -> ingest lỗi.
- Qdrant/cloud lỗi mạng -> dùng nhánh LanceDB local-first.

---

## 7) Khuyến nghị chiến lược chunking theo xu hướng 2026
- Mặc định: recursive/semantic với overlap vừa phải.
- Nâng cấp: hierarchical parent-child + auto-merge context.
- Truy vấn khó/multi-hop: bổ sung proposition/graph layer.
- Tinh thần tổng quát: **hybrid + hierarchical + adaptive**.

---

## 8) Bàn giao
Sau khi chạy xong trên máy mới, gửi lại:
1. `results/final_metrics.json`
2. export notebook HTML hoặc PDF
3. 2 file tex đã điền số
4. 1 file `RUN_REPORT.md` (thời gian, lệnh chạy, lỗi gặp, cách xử lý)
