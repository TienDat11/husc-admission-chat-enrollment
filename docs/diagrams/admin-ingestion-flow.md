# Luồng admin nạp tài liệu tuyển sinh

Sơ đồ mô tả luồng chính từ lúc cán bộ tuyển sinh tải tài liệu lên đến khi phát hành release. Nguồn: `.omp/plans/RAG_ADMIN_INGESTION_PLAN.md`, mục 2.2 và 8.

Các file đi kèm: [HTML tự chứa](admin-ingestion-flow.html) · [SVG](admin-ingestion-flow.svg). PNG chưa xuất; xem ghi chú cuối file.

## Ba vai trò

| Lane | Ai | Làm gì |
|---|---|---|
| Cán bộ tuyển sinh | Người soạn | Tải lên, cấu hình cắt đoạn, sửa bản ghi |
| Hệ thống | Máy | Đọc file, cắt đoạn, tạo bản xem trước, build release |
| Người duyệt | Người khác với người soạn | Duyệt, từ chối kèm lý do, phát hành |

## Sáu bước

| # | Bước | Input | Output |
|---|---|---|---|
| 1 | Tải lên | PDF, DOCX hoặc XLSX | Tài liệu đã đăng ký; trùng nội dung thì mở bản cũ |
| 2 | Phân tích | Tài liệu đã đăng ký | Khối nội dung kèm trang; OCR chỉ với trang thiếu chữ |
| 3 | Cấu hình | Khối nội dung | Đoạn cắt theo Điều/Khoản/Điểm, bảng hoặc FAQ |
| 4 | Xem trước | Đoạn đã cắt | Từng đoạn kèm nguồn; cảnh báo vượt ngưỡng |
| 5 | Sửa | Đoạn cần chỉnh | Bản ghi phiên bản mới, gửi duyệt |
| 6 | Duyệt và phát hành | Bản ghi đã gửi | Release mới đang phục vụ, hoặc bị từ chối |

## Điểm kiểm soát

- **Người duyệt phải khác người soạn.** Hệ thống so với `authored_by` và từ chối nếu trùng.
- **Từ chối bắt buộc có lý do.** Bản ghi quay về bước 5.
- **Bản xem trước chưa phải phát hành.** Chỉ bản đã duyệt và phát hành mới vào release phục vụ sinh viên.

## Ngoài phạm vi sơ đồ

Sơ đồ không thể hiện trạng thái job (`queued → … → active`), rollback và API. Các phần đó nằm trong plan, mục 4.3 và 4.2.

> Ghi chú: file PNG chưa được xuất vì môi trường hiện tại chưa có Python để chạy `scripts/export_png.py`. Khi có Python, chạy lại export để tạo `admin-ingestion-flow.png`.
