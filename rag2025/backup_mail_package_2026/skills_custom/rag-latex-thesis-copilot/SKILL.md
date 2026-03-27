---
name: rag-latex-thesis-copilot
description: Hỗ trợ viết/chỉnh sửa luận văn LaTeX cho dự án RAG/Chunking/GraphRAG. Luôn dùng skill này khi người dùng yêu cầu viết lại chương, điền bảng kết quả, chỉnh công thức toán, chuẩn hóa văn phong học thuật tiếng Việt, hoặc đồng bộ số liệu thực nghiệm vào file .tex.
---

# RAG LaTeX Thesis Copilot

## Mục tiêu
- Viết học thuật tự nhiên (không máy móc), giữ chặt logic kỹ thuật.
- Giữ nguyên tính đúng đắn công thức toán học.
- Đồng bộ số liệu từ JSON/CSV/notebook vào bảng LaTeX có kiểm chứng nguồn.

## Quy trình chuẩn
1. Đọc đúng file `.tex` mục tiêu và xác định section cần sửa.
2. Rà soát label/ref/cite để tránh lỗi tham chiếu.
3. Chỉ sửa tối thiểu phần được yêu cầu.
4. Nếu có số liệu thực nghiệm, luôn ghi rõ nguồn file (ví dụ `results/final_metrics.json`).
5. Kiểm tra các điểm dễ lỗi:
   - ký tự `%`, `_`, `\` trong table/text;
   - công thức có đóng đủ `{}`;
   - caption/label nhất quán.

## Luật điền bảng kết quả
- Nếu chưa có số liệu thật: giữ `--` và ghi chú trạng thái.
- Nếu có số liệu simulated: bắt buộc gắn nhãn simulated.
- Không tự bịa số liệu.

## Mẫu văn phong
- Ưu tiên câu rõ, ngắn-vừa, tự nhiên.
- Dùng cấu trúc: bối cảnh -> phương pháp -> kết quả -> hàm ý.
- Tránh lạm dụng buzzword.

## Checklist trước khi kết thúc
- [ ] Đúng vị trí section/subsection
- [ ] Công thức biên dịch được
- [ ] Bảng/figure có caption + label
- [ ] Citation hợp lệ
- [ ] Không có thay đổi ngoài phạm vi yêu cầu
