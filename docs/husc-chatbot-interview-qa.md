# Bộ Q&A thuyết trình dự án Chatbot tuyển sinh HUSC

## Cách sử dụng nhanh
- Mỗi câu trả lời nên đi theo 3 nhịp: **Bài toán → Cách làm → Kết quả/đảm bảo chất lượng**.
- Tránh nói quá rộng về AI; bám vào bằng chứng mã nguồn và kiến trúc thật của dự án.
- Khi bị hỏi sâu, dùng pipeline để giải thích quyết định kỹ thuật.

---

## 1) Elevator pitch (30-45 giây)
"Đây là chatbot tư vấn tuyển sinh HUSC theo hướng evidence-aware RAG. Hệ thống có 2 tuyến xử lý: tuyến nhanh cho câu hỏi đơn giản và tuyến GraphRAG cho câu hỏi đa chiều. Ngoài việc trả lời, chatbot còn trả nguồn, confidence, cảnh báo thiếu dữ liệu và guardrail để chặn truy vấn ngoài phạm vi hoặc chứa PII, nên phù hợp bối cảnh giáo dục cần độ tin cậy cao."

---

## 2) Q&A phỏng vấn thường gặp

## Q1. Bài toán thực tế mà dự án giải quyết là gì?
**Trả lời gợi ý:**
- Người học thường hỏi lặp lại về điểm chuẩn, học phí, ngành, tổ hợp, thủ tục.
- Dữ liệu tuyển sinh thay đổi theo năm, nên cần chatbot vừa trả lời nhanh vừa kiểm soát sai lệch.
- Dự án giải quyết bằng kiến trúc RAG có route thông minh + trust layer.

**Bằng chứng:** `rag2025/src/main.py:486`, `rag2025/src/services/guardrail.py:24`

## Q2. Vì sao không dùng một pipeline RAG duy nhất?
**Trả lời gợi ý:**
- Câu hỏi đơn giản và câu hỏi multi-hop có nhu cầu xử lý khác nhau.
- Dùng một pipeline duy nhất dễ hoặc chậm quá, hoặc thiếu chiều sâu.
- Nên tách `padded_rag` và `graph_rag`, rồi auto-route theo complexity/intent.

**Bằng chứng:** `rag2025/src/services/query_router.py:127`, `rag2025/src/services/graphrag_retriever.py:209`

## Q3. Smart Router hoạt động thế nào?
**Trả lời gợi ý:**
- Router chạy 3 bước: Step-back → HyDE doc → classification.
- Kết quả quyết định route + cung cấp metadata giải thích (`intent`, `complexity`, `reasoning`).
- Thiết kế này tăng khả năng tìm đúng ngữ cảnh trước khi generate.

**Bằng chứng:** `rag2025/src/services/query_router.py:3`, `rag2025/src/services/query_router.py:207`

## Q4. Điểm khác biệt của GraphRAG trong dự án là gì?
**Trả lời gợi ý:**
- Không bỏ retrieval cũ, mà fuse baseline retrieval score với PPR graph score.
- Khi câu hỏi đa thực thể/so sánh, GraphRAG giúp liên kết ngữ nghĩa theo thực thể.
- Đây là cách cân bằng giữa tính thực dụng và chiều sâu suy luận.

**Bằng chứng:** `rag2025/src/services/graphrag_retriever.py:76`, `rag2025/src/services/graphrag_retriever.py:153`

## Q5. Bạn kiểm soát hallucination như thế nào?
**Trả lời gợi ý:**
- Hậu xử lý bằng groundedness score; nếu thấp thì trả cảnh báo rõ cho người dùng.
- Kết hợp guardrail no-result để không trả lời bừa khi thiếu dữ liệu.
- Frontend hiển thị mã trạng thái và gợi ý bổ sung dữ liệu.

**Bằng chứng:** `rag2025/src/main.py:1029`, `rag2025/src/main.py:1033`, `uni-guide-ai/src/components/chat/ChatLayout.tsx:42`

## Q6. Hệ thống có tính an toàn dữ liệu gì?
**Trả lời gợi ý:**
- Có precheck phát hiện PII (email/sđt/CCCD) và chặn theo policy.
- Truy vấn ngoài phạm vi tuyển sinh HUSC bị redirect mềm thay vì trả nội dung sai domain.
- Tách internal/public status theo chế độ exposure.

**Bằng chứng:** `rag2025/src/services/guardrail.py:34`, `rag2025/src/services/guardrail.py:69`, `rag2025/src/services/guardrail.py:164`

## Q7. Retrieval stack hiện tại gồm gì?
**Trả lời gợi ý:**
- Nền tảng chính: LanceDB dense retrieval.
- Tùy chọn tăng chất lượng: Hybrid dense + BM25 + RRF.
- Sau đó có reranker để tối ưu top chunks trước generate.

**Bằng chứng:** `rag2025/src/services/lancedb_retrieval.py:71`, `rag2025/src/services/hybrid_search.py:19`, `rag2025/src/main.py:703`

## Q8. Làm sao dự án scale theo mùa tuyển sinh mới?
**Trả lời gợi ý:**
- Có endpoint admin để incremental update graph bằng chunks mới.
- Không cần rebuild toàn hệ thống graph từ đầu mỗi lần có dữ liệu mới.
- Giảm downtime và phù hợp vận hành theo đợt.

**Bằng chứng:** `rag2025/src/main.py:1075`, `rag2025/src/services/graphrag_retriever.py:323`

## Q9. Frontend phản ánh chất lượng backend thế nào?
**Trả lời gợi ý:**
- UI hiển thị `sources`, `confidence`, `status_code`, `data_gap_hints`.
- Nghĩa là UX minh bạch rủi ro dữ liệu, không che lỗi logic retrieval.
- Đây là điểm quan trọng khi làm AI app cho người dùng phổ thông.

**Bằng chứng:** `uni-guide-ai/src/components/chat/ChatLayout.tsx:36`, `uni-guide-ai/src/components/chat/ChatLayout.tsx:42`

## Q10. Bạn đo chất lượng pipeline bằng gì?
**Trả lời gợi ý:**
- Có utility eval để gọi cả `/query` và `/v2/query`.
- Chuẩn hóa output rồi tính recall proxy, hallucination flag, diagnostic report.
- Dùng để so sánh route và phát hiện regression.

**Bằng chứng:** `packages/rag-chatbot-husc/src/notebooks/eval_core.py:142`, `packages/rag-chatbot-husc/src/notebooks/eval_core.py:62`, `packages/rag-chatbot-husc/src/notebooks/eval_core.py:244`

## Q11. Thách thức lớn nhất khi làm dự án này?
**Trả lời gợi ý:**
- Cân bằng giữa độ đúng của thông tin và trải nghiệm trả lời nhanh.
- Nguồn dữ liệu giáo dục thay đổi định kỳ nên ingestion/normalization là điểm sống còn.
- Cần thiết kế fallback và observability đủ tốt để vận hành ổn định.

## Q12. Nếu có thêm thời gian, bạn nâng cấp gì đầu tiên?
**Trả lời gợi ý:**
- Externalize cache/metrics để production-ready hơn.
- Tăng bộ benchmark theo kịch bản tuyển sinh thực.
- Cải thiện quality gates theo route-specific KPIs.

## Q13. Bạn đóng vai trò gì trong dự án?
**Trả lời gợi ý (tuỳ chỉnh theo thực tế của bạn):**
- Thiết kế kiến trúc pipeline RAG và route strategy.
- Triển khai backend orchestration + trust layer.
- Xây dựng quy trình đánh giá chất lượng để đo accuracy/groundedness.

## Q14. Vì sao giải pháp này có giá trị doanh nghiệp?
**Trả lời gợi ý:**
- Giảm tải tư vấn lặp lại cho bộ phận tuyển sinh.
- Nâng tính nhất quán thông tin công bố đến thí sinh.
- Có khung kiểm soát rủi ro dữ liệu giúp dễ đưa vào vận hành thực tế.

## Q15. Câu nào nên chốt khi kết thúc buổi phỏng vấn?
**Trả lời gợi ý:**
- "Em không chỉ làm chatbot trả lời, mà xây một hệ thống có cơ chế tự kiểm soát độ tin cậy và dễ mở rộng theo dữ liệu tuyển sinh mới." 

---

## 3) Talking points ngắn khi bị hỏi xoáy kỹ thuật

- **Vì sao dùng HyDE + Step-back?**
  → Tăng recall ở lớp retrieval trước khi sinh câu trả lời, đặc biệt với câu hỏi diễn đạt mơ hồ.

- **Vì sao giữ cả PaddedRAG và GraphRAG?**
  → Tối ưu trade-off tốc độ vs chiều sâu theo loại câu hỏi, tránh “one-size-fits-all”.

- **Vì sao hiển thị data-gap-hints cho user?**
  → Minh bạch thiếu hụt dữ liệu, giảm ảo tưởng chính xác và hỗ trợ vòng cải thiện dữ liệu.

---

## 4) Checklist trước buổi trình bày
- Đọc lại file tóm tắt kiến trúc: `docs/husc-chatbot-recruitment-summary.md`
- Chuẩn bị 2 demo câu hỏi:
  - 1 câu đơn giản (padded_rag)
  - 1 câu so sánh/multi-hop (graph_rag)
- Chuẩn bị 1 tình huống out-of-scope hoặc thiếu dữ liệu để thể hiện guardrail.
- Nhớ nhấn mạnh: **độ tin cậy + khả năng vận hành + khả năng mở rộng dữ liệu**.
