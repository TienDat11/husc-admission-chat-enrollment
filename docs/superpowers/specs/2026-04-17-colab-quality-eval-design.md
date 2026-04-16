# Thiết kế Notebook Colab đánh giá chất lượng trả lời (phase-1)

## 1) Bối cảnh và mục tiêu

Dự án cần một notebook `.ipynb` chạy trực tiếp trên Google Colab để thay thế chạy local do giới hạn dung lượng máy cá nhân. Mục tiêu không phải demo tối giản mà là đánh giá đủ sâu để tạo báo cáo học thuật/đồ án có thể chỉ ra:

1. Hệ thống đang mạnh/yếu ở đâu.
2. Các lỗi chất lượng trả lời theo nhóm lỗi cụ thể.
3. Thứ tự ưu tiên cải tiến để nâng độ tin cậy.

Phạm vi thiết kế này bám theo lựa chọn của user:
- Phương án B (bám codebase thật, không pipeline giả lập).
- Ưu tiên “chất lượng trả lời” hơn benchmark hạ tầng.

## [UPDATED] 2) Phạm vi và đầu vào

### 2.1 Phạm vi kỹ thuật
- Notebook mới đặt tại: `packages/rag-chatbot-husc/src/notebooks/colab_eval.ipynb`.
- Chạy trên Colab GPU, clone repo thật, cài dependency, import module thật từ codebase.
- Không hardcode secret trong notebook.

### [UPDATED] 2.2 Dữ liệu đánh giá
- Test set chính: `rag2025/results/test_questions.json`.
- Fallback backup: `rag2025/backup_mail_package_2026/python_project/rag2025/results/test_questions.json`.

**Điều kiện trigger fallback path:**
- Dùng fallback khi file chính **không tồn tại** hoặc **không parse được JSON hợp lệ**.
- Nếu cả hai path đều không load được: dừng notebook và báo lỗi cấu hình dữ liệu.

Test set chứa trường chính:
- `question`
- `ground_truth_answer`
- `category` (`simple`, `multihop`)
- metadata liên quan `required_hops`, `ground_truth_source_chunks`

## [UPDATED] 3) Kiến trúc thực thi trong notebook

Notebook gồm các khối cell theo thứ tự:

1. **Runtime guard**
   - In Python version.
   - Chạy `nvidia-smi`.
   - Kiểm tra `torch.cuda.is_available()`.
   - Nếu không có GPU: fail-fast với hướng dẫn đổi runtime Colab sang GPU.

2. **Repo bootstrap**
   - Clone repository.
   - Pin git reference: checkout branch cố định `main`.
   - Ghi rõ trong notebook metadata: branch `main` được giả định là stable bởi maintainer repo. Nếu cần tái lập tuyệt đối theo báo cáo, thêm biến `PINNED_COMMIT` để checkout commit hash cụ thể.
   - Cài dependency bằng `pip` theo đúng file requirements của dự án.

3. **Import/bootstrap codebase**
   - Thiết lập `PYTHONPATH`.
   - Smoke import các module pipeline chính.
   - Fail sớm nếu import lỗi.

4. **Config & secrets**
   - Cell nhận env vars qua Colab secrets/runtime input.
   - Không lưu token API vào notebook output.

5. **Data load & validation**
   - Nạp file `test_questions.json`.
   - Validate schema tối thiểu (`question`, `ground_truth_answer`, `category`).
   - Báo số lượng câu hỏi theo category.

6. **Smoke eval**
   - Chạy tập con nhỏ (ví dụ 10 câu) để xác thực pipeline và thời gian phản hồi.
   - **Fail-fast threshold:** dừng toàn bộ eval nếu >50% câu trong smoke set rơi vào 1 trong 2 điều kiện:
     - exception khi gọi pipeline, hoặc
     - `answer` rỗng/blank sau normalize.

7. **Full eval**
   - Chạy toàn bộ test set.
   - Thu prediction, evidence/context (nếu có), latency per query.

8. **Scoring & diagnostics**
   - Tính các metric chất lượng trả lời.
   - Phân cụm lỗi theo taxonomy.
   - Tạo bảng câu hỏi fail điển hình.

9. **Report export**
   - Ghi file kết quả + báo cáo markdown để dùng cho báo cáo học tập.

## [UPDATED] 4) Bộ chỉ số đánh giá (ưu tiên chất lượng trả lời)

### 4.1 Chỉ số cốt lõi
- **Answer correctness** (exact/normalized/semantic).
- **Groundedness**: mức độ câu trả lời bám vào context truy hồi.
- **Hallucination rate**: phát biểu không có bằng chứng trong context.
- **Category breakdown**: chất lượng theo `simple` vs `multihop`.

### 4.2 Chỉ số phụ trợ
- **Partial-credit score** cho câu đúng một phần.
- **Citation support ratio** (nếu pipeline trả nguồn/chunk-id).
- **Latency median/p95** (để hỗ trợ thảo luận trade-off chất lượng-tốc độ).
- **Retrieval recall proxy dùng `ground_truth_source_chunks`**: đo tỷ lệ câu có ít nhất 1 `ground_truth_source_chunks` xuất hiện trong nguồn truy hồi top-k.

### [UPDATED] 4.3 Bảng env vars bắt buộc/khuyến nghị

| Tên biến | Mục đích | Bắt buộc? |
|---|---|---|
| `RAMCLOUDS_API_KEY` hoặc `OPENAI_API_KEY` | Provider chính cho generation qua ramclouds-compatible endpoint | Có (ít nhất 1) |
| `RAMCLOUDS_BASE_URL` | URL endpoint provider chính | Khuyến nghị |
| `RAMCLOUDS_MODEL` | Model generation chính | Khuyến nghị |
| `GROQ_API_KEY` | Provider fallback cho generation/guardrail/query-enhancer | Khuyến nghị |
| `OPENAI_COMPAT_API_KEY` | Provider fallback openai-compatible | Tuỳ chọn |
| `OPENAI_COMPAT_BASE_URL` | Base URL cho provider compat | Tuỳ chọn |
| `OPENAI_COMPAT_MODEL` | Model cho provider compat | Tuỳ chọn |
| `QDRANT_URL` | Endpoint Qdrant khi dùng retrieval Qdrant | Tuỳ cấu hình |
| `QDRANT_API_KEY` | API key Qdrant | Tuỳ cấu hình |
| `QDRANT_COLLECTION` | Tên collection Qdrant | Tuỳ cấu hình |
| `GROUNDING_THRESHOLD` | Ngưỡng cảnh báo low groundedness | Khuyến nghị |
| `ADMIN_API_TOKEN` | Gọi endpoint admin (`/v2/graph/update`) | Không bắt buộc cho eval thường |

Ghi chú: bộ biến này lấy từ các điểm đọc `os.getenv` trong codebase hiện tại; có thể mở rộng khi pipeline thay đổi.

## 5) Taxonomy lỗi phục vụ khuyến nghị cải tiến

Mỗi câu sai sẽ gán ít nhất một nhãn lỗi:

1. `retrieval_missing_fact` — thiếu fact quan trọng từ tầng truy hồi.
2. `retrieval_wrong_context` — truy hồi lệch ngữ nghĩa câu hỏi.
3. `reasoning_multihop_break` — đứt chuỗi suy luận nhiều bước.
4. `generation_fabrication` — sinh thông tin không có trong bằng chứng.
5. `answer_incomplete` — đúng ý chính nhưng thiếu điều kiện/chi tiết bắt buộc.
6. `format_or_instruction_miss` — đúng nội dung nhưng sai format yêu cầu.

## 6) Đầu ra báo cáo

Notebook phải xuất được tối thiểu:

1. `eval_predictions.jsonl` — câu hỏi + prediction + metadata runtime.
2. `eval_scored.jsonl` — prediction + score + nhãn lỗi.
3. `eval_summary_metrics.csv` — bảng metric tổng hợp.
4. `diagnostic_report.md` gồm:
   - Executive summary.
   - Kết quả theo category.
   - Top lỗi chính + ví dụ thực tế.
   - Danh sách điểm yếu hệ thống.
   - Roadmap cải tiến theo mức ưu tiên.

## 7) Roadmap cải tiến trong báo cáo

Báo cáo phải kết thúc bằng ma trận ưu tiên:

- **Quick wins (1–3 ngày)**: chỉnh prompt/template, reranker threshold, hậu xử lý answer format.
- **Medium (1–2 tuần)**: cải tiến retrieval layering, query rewrite, hybrid retrieval tuning.
- **Long-term (2–6 tuần)**: cải tiến graph reasoning, hard negative mining, benchmark mở rộng domain.

Mỗi đề xuất cần có:
- tác động kỳ vọng lên metric,
- chi phí triển khai,
- mức rủi ro,
- cách đo lại sau cải tiến.

## 8) Tiêu chí chấp nhận

Được coi là hoàn thành setup khi:

1. Notebook chạy trên Colab với GPU được xác nhận.
2. Smoke eval pass.
3. Full eval chạy xong trên test set chính.
4. Xuất đủ 4 artifacts báo cáo.
5. Báo cáo chỉ ra tối thiểu 5 hành động cải tiến có thứ tự ưu tiên.

## 9) Ràng buộc và nguyên tắc

- Bám codebase thật, không thay bằng pipeline giả cho phần đánh giá chính.
- Không để lộ secret/token trong notebook output.
- Giữ notebook tái lập được trên Colab mới.
- Nếu gặp chặn do thiếu dependency ngoài, notebook phải fail rõ nguyên nhân và cách khắc phục.

## 10) Ghi chú thực thi

- Định hướng này ưu tiên độ sâu phân tích chất lượng trả lời.
- Benchmark tài nguyên (CPU/GPU cost sâu) chỉ ở mức bổ trợ, không là trọng tâm chính.
- Auto-decided by V5-R011: chọn phương án xử lý root-cause qua taxonomy lỗi thay vì chỉ báo accuracy tổng.

## [UPDATED] 11) Appendix A: Pipeline Interface Contract

### 11.1 Module/package cần import từ codebase thật
- `rag2025/src/main.py`
  - `SimpleQueryRequest` (schema request v1)
  - `QueryResponse` (schema response v1)
  - `UnifiedQueryRequest` (schema request v2)
  - `UnifiedQueryResponse` (schema response v2)
- Endpoint chính để notebook gọi eval:
  - `POST /query`
  - `POST /v2/query`

### 11.2 Function signature chính dùng để query pipeline

Do notebook chạy Colab và gọi service qua HTTP, contract ở mức client function như sau:

```python
async def query_pipeline(
    query: str,
    mode: str = "v2",           # "v1" -> /query, "v2" -> /v2/query
    top_k: int = 5,
    force_route: str | None = None,
) -> dict:
    ...
```

#### Input contract
- `query: str` — bắt buộc, không rỗng.
- `mode: str` — `v1` hoặc `v2`.
- `top_k: int` — dùng cho `v2`.
- `force_route: Optional[str]` — `padded_rag`/`graph_rag` cho `v2`.

#### Output contract (normalized for evaluator)
Notebook chuẩn hoá response về schema chung:

```json
{
  "answer": "...",                    
  "context_chunks": ["..."],          
  "source_ids": ["..."],              
  "confidence": 0.0,
  "groundedness_score": 0.0,
  "route": "padded_rag|graph_rag|v1",
  "raw": {...}
}
```

Ghi chú:
- Với `/query`, trường `chunks` có thể chứa object chunk; notebook map `chunk.text` -> `context_chunks`.
- Với `/v2/query`, trường `sources` map vào `source_ids`; `context_chunks` có thể cần lấy từ payload mở rộng hoặc để rỗng nếu API không trả text chunk đầy đủ.
- Các trường `context_chunks`, `source_ids` là **required trong normalized output**, có thể rỗng nhưng phải tồn tại key.

### 11.3 Phần cần xác nhận
- [TBD — cần xác nhận với dev] Mapping chuẩn giữa `chunks` (v1) và bằng chứng truy hồi dùng cho metric groundedness/hallucination khi API không trả toàn văn chunk ở v2.
- [TBD — cần xác nhận với dev] Có cần expose endpoint debug nội bộ để trả full retrieved chunk texts cho evaluator hay không.

## [UPDATED] 12) Appendix B: Metric Implementation Decision

| Metric | Thư viện / cách tính | Operationalization | Lý do chọn |
|---|---|---|---|
| Answer correctness (exact) | Custom normalize + string match | Lowercase, strip punctuation/space rồi so khớp với `ground_truth_answer` | Đơn giản, tái lập cao |
| Answer correctness (semantic) | `sentence-transformers` cosine similarity | Encode `answer` và `ground_truth_answer`, pass nếu similarity >= ngưỡng cấu hình | Phù hợp tiếng Việt, chấp nhận diễn đạt khác câu chữ |
| Groundedness | Reuse hàm `faithfulness_score` trong `rag2025/results/metrics.py` (cosine max giữa answer và source chunks) | `groundedness = max cosine(answer, retrieved_chunks_text)`; low-grounded nếu < `GROUNDING_THRESHOLD` | Đồng bộ với codebase hiện tại |
| Hallucination rate | Custom từ groundedness + evidence coverage | Hallucination = 1 nếu groundedness < ngưỡng **hoặc** answer chứa claim chính nhưng không có evidence chunk support; rate = tổng hallucination / tổng câu | Chuyển groundedness thành chỉ số lỗi dễ diễn giải cho báo cáo |
| Category breakdown | Custom grouping (`simple`, `multihop`) | Tính correctness/groundedness/hallucination theo từng category | Bám cấu trúc test set |
| Partial-credit score | Custom rubric | 1.0 đúng đủ; 0.5 đúng ý chính nhưng thiếu điều kiện; 0.0 sai trọng tâm | Hữu ích cho câu trả lời bán đúng |
| Citation support ratio | Custom set overlap | Tỷ lệ nguồn trong `source_ids` giao với `ground_truth_source_chunks` | Đo mức bám nguồn chuẩn |
| Retrieval recall proxy | Custom | Với mỗi câu: 1 nếu `source_ids` chứa ít nhất 1 phần tử của `ground_truth_source_chunks`, ngược lại 0; lấy trung bình | Dễ hiểu, gắn trực tiếp ground-truth chunk |
| Latency median/p95 | Custom timing (`time.perf_counter`) | Đo thời gian mỗi request, tổng hợp median và p95 | Bổ trợ trade-off chất lượng/tốc độ |

### 12.1 Làm rõ hai metric trọng tâm

- **Groundedness** trong phase-1 được operationalize bằng semantic similarity giữa `answer` và tập `retrieved_chunks_text` (max similarity). Đây là metric “bám bằng chứng”.
- **Hallucination rate** được operationalize như tỷ lệ câu có groundedness thấp hoặc không có bằng chứng hỗ trợ cho claim cốt lõi. Đây là metric “rủi ro bịa thông tin”.

### 12.2 Ghi chú về sai khác triển khai
- Nếu nhóm triển khai dùng LLM-as-judge thay cho cosine scoring, phải ghi rõ prompt judge và tiêu chí chấm để đảm bảo tái lập.
- Trong phạm vi phase-1, mặc định ưu tiên cosine/custom để giảm biến thiên giữa lần chạy.
