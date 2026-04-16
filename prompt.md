# Nhiệm vụ: Cải thiện Spec Design — `colab_eval.ipynb`

## Bối cảnh
Bạn là technical writer kiêm solution architect. Bạn nhận được một spec design 
cho notebook đánh giá RAG chatbot chạy trên Google Colab (Phase-1). 
Spec đã qua review và bị yêu cầu bổ sung trước khi approve để chuyển sang phase plan.

Dưới đây là spec gốc:

<spec_goc>
[DÁN TOÀN BỘ NỘI DUNG SPEC GỐC VÀO ĐÂY]
</spec_goc>

---

## Yêu cầu bổ sung (theo thứ tự ưu tiên)

### 🔴 Blocker — phải có trước khi approve

**1. Bổ sung Appendix A: Pipeline Interface Contract**
- Liệt kê tên module/package cần import từ codebase thật.
- Mô tả function signature chính dùng để query pipeline:
  - Input: kiểu dữ liệu, tên tham số.
  - Output: schema dict/object trả về (các trường bắt buộc như `answer`, 
    `context_chunks`, `source_ids`, v.v.).
- Nếu chưa chắc chắn, ghi rõ là "assumed interface — cần xác nhận với dev".

**2. Bổ sung Appendix B: Metric Implementation Decision**
- Với mỗi metric ở Section 4, chỉ định rõ:
  - Dùng thư viện nào (RAGAS / DeepEval / BERTScore / LLM-as-judge / custom)?
  - Công thức hoặc cách tính cụ thể nếu custom.
  - Lý do chọn (1 câu ngắn).
- Đặc biệt làm rõ "hallucination rate" và "groundedness" được 
  operationalize thế nào.

---

### 🟡 Cần làm rõ — không block nhưng phải có trước khi code

**3. Định nghĩa ngưỡng fail-fast ở Section 6 (Smoke eval)**
- Thay câu mơ hồ "nếu lỗi nhiều hơn ngưỡng" bằng con số cụ thể.
- Ví dụ: "Nếu > 50% câu trong smoke set trả về exception hoặc empty answer, 
  dừng toàn bộ eval."

**4. Pin git reference ở Section 3 — Repo bootstrap**
- Thay "mặc định branch hiện tại" bằng một trong:
  - Commit hash cụ thể, hoặc
  - Tag release, hoặc
  - Tên branch cố định (ví dụ `main`) với ghi chú rõ ai chịu trách nhiệm 
    giữ branch đó stable.

**5. Liệt kê tên biến môi trường ở Section 4 — Config & Secrets**
- Thêm bảng các env vars bắt buộc để notebook chạy được, ví dụ:

  | Tên biến | Mục đích | Bắt buộc? |
  |---|---|---|
  | `OPENAI_API_KEY` | Gọi LLM generation | Có |
  | `WEAVIATE_URL` | Kết nối vector store | Có |
  | ... | ... | ... |

**6. Làm rõ `ground_truth_source_chunks` được dùng trong metric nào**
- Trường này ở schema test set nhưng Section 4 không nhắc đến.
- Hoặc gắn nó vào một metric cụ thể (ví dụ: retrieval recall), 
  hoặc ghi rõ "reserved for future use — không dùng trong phase-1".

**7. Điều kiện trigger fallback data path**
- Section 2.2 có đường dẫn fallback nhưng không nói khi nào dùng.
- Bổ sung điều kiện rõ ràng: "Dùng fallback khi file chính không tồn tại 
  hoặc không parse được JSON hợp lệ."

---

## Format đầu ra mong muốn

- Giữ nguyên cấu trúc và đánh số section của spec gốc.
- Các phần **đã có và đạt yêu cầu** — giữ nguyên, không viết lại.
- Các phần **cần chỉnh sửa nhỏ** — chỉnh tại chỗ và đánh dấu 
  `[UPDATED]` ở đầu dòng tiêu đề.
- Các phần **mới thêm** (Appendix A, B) — đặt cuối document, 
  đánh số tiếp theo (Section 11, 12 hoặc Appendix A, B).
- Không thêm nội dung suy đoán nếu thiếu thông tin — 
  dùng placeholder `[TBD — cần xác nhận với dev]` thay thế.

---

## Tiêu chí spec đạt sau khi sửa

- [ ] Người đọc spec có thể viết được function signature gọi pipeline 
      mà không cần hỏi thêm.
- [ ] Hai người implement metrics độc lập sẽ ra cùng một bộ số 
      (hoặc biết rõ chỗ nào có thể khác nhau).
- [ ] Notebook có thể được reviewer clone và chạy lại chỉ với 
      hướng dẫn trong spec, không cần hỏi tác giả.
      