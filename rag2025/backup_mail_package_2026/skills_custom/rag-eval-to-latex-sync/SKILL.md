---
name: rag-eval-to-latex-sync
description: Đồng bộ số liệu thực nghiệm từ JSON/CSV/notebook sang bảng LaTeX của luận văn RAG. Dùng khi người dùng cần điền bảng kết quả, cập nhật caption/ghi chú, hoặc sinh snippet LaTeX từ metrics.
---

# RAG Eval to LaTeX Sync

## Mục tiêu
- Truy xuất số liệu đúng nguồn.
- Xuất hàng bảng LaTeX sẵn dán.
- Tránh bịa số liệu và tránh lỗi format.

## Quy trình
1. Đọc nguồn metrics (`results/final_metrics.json`, CSV, notebook output).
2. Kiểm tra trạng thái số liệu (real/simulated/failed).
3. Sinh LaTeX rows theo đúng thứ tự cột bảng mục tiêu.
4. Bổ sung ghi chú nguồn dữ liệu ngay dưới bảng.

## Quy tắc an toàn
- Gặp `EXPERIMENT_FAILED` -> không điền số giả.
- Gặp simulated -> bắt buộc ghi simulated note.
- Không đổi công thức/chỉ số nếu người dùng không yêu cầu.

## Mẫu xuất
- `results/latex_rows.txt`
- `results/thesis_eval_summary.csv`
- đoạn `\noindent\textit{Ghi chú: ...}` cho bảng
