# Danh sách câu hỏi cần xác nhận với Phòng Tuyển sinh HUSC

> Sinh ra từ eval 86 câu hỏi thí sinh thật. Đây là các điểm corpus **chưa có dữ liệu** hoặc **dữ liệu mơ hồ** → chatbot buộc phải trả lời "không tìm thấy thông tin". Khi tuyển sinh trả lời, ingest vào corpus để chatbot trả lời được.

## A. Corpus THIẾU dữ liệu — chatbot đang abstain (ưu tiên cao)

1. **Thí sinh tự do tốt nghiệp lâu năm (vd 2004)** có được xét học bạ không? Quy định cụ thể cho thí sinh tự do tốt nghiệp các năm trước? *(msg011)*
2. **Xét học bạ có ưu tiên/khác biệt giữa hệ THPT và GDTX không?** Học bạ GDTX có được chấp nhận như THPT? *(msg039)*
3. **Thí sinh tự do KHÔNG đăng ký thi lại THPT năm nay** có được đăng ký xét tuyển vào trường không (qua phương thức nào)? *(msg040)*
4. **Quy trình chuyển/bảo lưu điểm năng khiếu Vẽ mỹ thuật từ trường khác về HUSC** làm thế nào? Hồ sơ, thủ tục, hạn nộp? *(msg045)*
5. **Hết hạn đăng ký mà trường chưa duyệt minh chứng** thì xử lý ra sao? Thí sinh cần làm gì, có bị mất quyền xét tuyển không? *(msg082)*

## B. Dữ liệu MƠ HỒ / cần chốt mốc thời gian (xác nhận để chatbot trả lời chắc chắn hơn)

6. **Điểm chuẩn / điểm sàn năm 2026** — khi nào công bố chính thức? (Hiện chatbot chỉ có 2025, phải nói "2026 chưa công bố".) *(msg026/027/028)*
7. **HUSC có chấp nhận điểm năng khiếu Vẽ từ ĐH Kiến trúc TP.HCM (UAH) / các trường khác không**, ngưỡng bao nhiêu? *(msg034)*
8. **Link/hệ thống đăng ký xét tuyển 2026** đã mở chưa, địa chỉ chính thức? *(msg044)*
9. **Kênh liên hệ chính thức (Zalo OA / group / fanpage)** của phòng tuyển sinh — số/đường link chính xác để chatbot dẫn đúng. *(msg055)*
10. **Mã trường HUSC** dùng trên hệ thống Bộ GD&ĐT 2026 — xác nhận chính xác. *(msg059)*

---
*Nguồn: `results/eval_harness/86q_records_s16.jsonl` + `results/ultraqa_metrics/abstain_oos_triage.json`. Nhóm A = 0 chunk hỗ trợ trong corpus; Nhóm B = có dữ liệu năm cũ hoặc cần chốt 2026.*
