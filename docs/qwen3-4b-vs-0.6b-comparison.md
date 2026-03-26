# So Sánh Qwen3-Embedding: 4B vs 0.6B

**Ngày:** 2026-03-26
**Dự án:** husc-admission-chat-enrollment
**Mục đích:** Giải thích sự khác biệt giữa hai biến thể Qwen3-Embedding để chọn phù hợp với máy

---

## Tóm Tắt Nhanh

| Tiêu chí | **4B (Máy cô)** | **0.6B (Máy bạn)** |
|----------|-----------------|---------------------|
| **Dung lượng đĩa** | ~8 GB | ~1.2 GB |
| **RAM tối thiểu** | 12 GB | 4 GB |
| **Chiều embedding** | 2560 | 1024 |
| **Chất lượng retrieval** | Cao hơn (~5-8%) | Tốt |
| **Tốc độ CPU** | Trung bình (100-200ms/query) | Nhanh (20-50ms/query) |
| **Điểm MTEB** | 69.45 | 64.33 |
| **Phù hợp cho** | Máy có đủ RAM, cần chất lượng cao | Máy RAM hạn chế, cần tốc độ |

**Khuyến nghị:**
- **Máy cô (16GB RAM, đủ dung lượng):** Dùng **4B** — chất lượng tốt hơn, đáng để đổi lấy tốc độ chậm hơn một chút
- **Máy bạn (16GB RAM, hết dung lượng):** Dùng **0.6B** — nhẹ, nhanh, vẫn đủ tốt cho chatbot tuyển sinh

---

## 1. Sự Khác Biệt Về Kích Thước

### 4B (4 tỷ tham số)
- **Dung lượng đĩa:** ~8 GB (định dạng BF16)
- **RAM khi chạy:** ~12 GB (CPU inference)
- **Chiều embedding:** 2560 dimensions
- **Tải lần đầu:** ~2-3 phút (tải từ HuggingFace)

### 0.6B (600 triệu tham số)
- **Dung lượng đĩa:** ~1.2 GB (định dạng BF16)
- **RAM khi chạy:** ~4 GB (CPU inference)
- **Chiều embedding:** 1024 dimensions
- **Tải lần đầu:** ~30 giây (tải từ HuggingFace)

**Tại sao khác biệt nhiều vậy?**
- Model 4B có nhiều layers hơn, mỗi layer có nhiều neurons hơn
- Embedding dimension lớn hơn → vector biểu diễn chi tiết hơn
- Nhưng cũng nặng hơn, chậm hơn

---

## 2. Sự Khác Biệt Về Chất Lượng

### Điểm MTEB (Massive Text Embedding Benchmark)

| Model | MTEB Score | Xếp hạng | Ý nghĩa |
|-------|------------|----------|---------|
| **4B** | **69.45** | Top 5 | Rất tốt — gần bằng các model lớn nhất |
| **0.6B** | **64.33** | Top 20 | Tốt — vượt nhiều model lớn hơn |
| 8B | 70.58 | #1 | Tốt nhất nhưng quá nặng (15GB) |

**Chênh lệch ~5 điểm có nghĩa gì?**
- Với 100 câu hỏi, 4B trả lời đúng ~69 câu, 0.6B trả lời đúng ~64 câu
- Chênh lệch **~5-8%** về độ chính xác retrieval
- Trong thực tế chatbot tuyển sinh: **không quá lớn** vì:
  - Câu hỏi đơn giản (điểm chuẩn, học phí) → cả hai đều tốt
  - Câu hỏi phức tạp (so sánh ngành) → GraphRAG giúp bù đắp

### Thử nghiệm thực tế (dự đoán)

| Loại câu hỏi | 4B | 0.6B | Chênh lệch |
|--------------|-----|------|------------|
| Factoid đơn giản ("Điểm chuẩn CNTT?") | 95% | 93% | Nhỏ |
| Multi-hop ("Ngành nào có điểm thấp hơn CNTT và cùng khoa?") | 85% | 78% | Trung bình |
| Cross-lingual (tiếng Anh → tiếng Việt) | 80% | 72% | Lớn hơn |

**Kết luận:** 4B tốt hơn rõ ràng, nhưng 0.6B vẫn **đủ tốt** cho use case này.

---

## 3. Sự Khác Biệt Về Tốc Độ

### Tốc độ encoding (CPU, 16GB RAM)

| Thao tác | 4B | 0.6B | Nhanh hơn |
|----------|-----|------|-----------|
| **Query encoding** (1 câu hỏi) | 100-200ms | 20-50ms | **4-5x** |
| **Document encoding** (batch 8 chunks) | 800-1500ms | 200-400ms | **4x** |
| **Indexing 1000 chunks** | ~2 phút | ~30 giây | **4x** |

**Ảnh hưởng đến trải nghiệm:**
- **Query (online):**
  - 4B: 100-200ms → user không cảm nhận được (< 300ms là OK)
  - 0.6B: 20-50ms → rất nhanh
  - **Cả hai đều chấp nhận được** cho chatbot real-time

- **Indexing (offline):**
  - 4B: 2 phút cho 1000 chunks → chạy `setup_data.bat` lâu hơn
  - 0.6B: 30 giây → nhanh hơn nhiều
  - **Chỉ ảnh hưởng khi setup lần đầu hoặc re-index**

---

## 4. Sự Khác Biệt Về Yêu Cầu Hệ Thống

### RAM Usage

| Model | Idle | Encoding Query | Encoding Batch | Indexing |
|-------|------|----------------|----------------|----------|
| **4B** | ~8 GB | ~10 GB | ~12 GB | ~12 GB |
| **0.6B** | ~2 GB | ~3 GB | ~4 GB | ~4 GB |

**Máy 16GB RAM:**
- **4B:** Còn ~4 GB cho OS + browser → **hơi chật** nếu mở nhiều tab
- **0.6B:** Còn ~12 GB cho OS + browser → **thoải mái**

### Disk Space

| Model | Model files | LanceDB index (1000 chunks) | Tổng |
|-------|-------------|----------------------------|------|
| **4B** | ~8 GB | ~10 MB (2560 dims × 1000 × 4 bytes) | ~8.01 GB |
| **0.6B** | ~1.2 GB | ~4 MB (1024 dims × 1000 × 4 bytes) | ~1.204 GB |

**Máy bạn (hết dung lượng):** 0.6B tiết kiệm **~7 GB** so với 4B.

---

## 5. Khi Nào Dùng Model Nào?

### Dùng 4B khi:
✅ Máy có đủ RAM (≥12 GB free)
✅ Máy có đủ dung lượng đĩa (≥10 GB free)
✅ Cần chất lượng retrieval cao nhất có thể
✅ Không ngại chờ indexing lâu hơn (2-3 phút)
✅ Có nhiều câu hỏi cross-lingual hoặc multi-hop

**→ Máy cô phù hợp với 4B**

### Dùng 0.6B khi:
✅ Máy RAM hạn chế (8-16 GB)
✅ Máy hết dung lượng đĩa
✅ Cần tốc độ nhanh (query < 50ms)
✅ Cần indexing nhanh (setup_data.bat < 1 phút)
✅ Câu hỏi chủ yếu đơn giản (factoid)

**→ Máy bạn phù hợp với 0.6B**

---

## 6. Cách Chuyển Đổi Giữa 4B và 0.6B

### Hiện tại (sau commit 33a72b4):
- **Main branch:** Đã config sẵn **4B** (2560 dims)
- **Máy cô chạy ngay:** `setup_data.bat` → tải 4B → index → chạy `run_lab.bat`

### Để chuyển sang 0.6B (trên máy bạn):

**Bước 1:** Sửa `rag2025/config/settings.py`
```python
# Dòng 37-41
EMBEDDING_MODEL: str = Field(
    default="Qwen/Qwen3-Embedding-0.6B",  # Đổi từ 4B → 0.6B
    description="HuggingFace model name for embeddings",
)
EMBEDDING_DIM: int = Field(default=1024, description="Embedding dimension")  # Đổi từ 2560 → 1024

# Dòng 46-50 (tương tự)
QWEN_EMBEDDING_MODEL: str = Field(
    default="Qwen/Qwen3-Embedding-0.6B",  # Đổi từ 4B → 0.6B
    description="Qwen3 embedding model for multilingual retrieval",
)
QWEN_EMBEDDING_DIM: int = Field(default=1024, description="Qwen3 embedding dimension")  # Đổi từ 2560 → 1024
```

**Bước 2:** Chạy lại setup
```bash
setup_data.bat data\raw
```
→ Tải 0.6B (1.2GB), index lại với 1024 dims

**Bước 3:** Chạy server
```bash
run_lab.bat
```

**Lưu ý:**
- **PHẢI re-index** sau khi đổi model (embeddings cũ không tương thích)
- Xóa `data/lancedb/` trước khi chạy `setup_data.bat` nếu muốn chắc chắn

---

## 7. Có Nên Dùng 8B Không?

### 8B (8 tỷ tham số)
- **Dung lượng:** ~15 GB
- **RAM:** ~20 GB
- **MTEB:** 70.58 (#1 trên leaderboard)
- **Tốc độ:** 300-500ms/query (chậm nhất)

**Khuyến nghị:** **KHÔNG** dùng 8B vì:
❌ Quá nặng cho máy 16GB RAM (sẽ swap, chậm)
❌ Chênh lệch chất lượng với 4B chỉ ~1% (70.58 vs 69.45)
❌ Tốc độ chậm gấp 3x so với 4B
❌ Không đáng để đổi lấy 1% cải thiện

**Chỉ dùng 8B nếu:**
- Máy có ≥32 GB RAM
- Có GPU (VRAM ≥16 GB)
- Cần chất lượng tuyệt đối cao nhất

---

## 8. Kết Luận

### Cho Máy Cô (Mai Test)
✅ **Dùng 4B** (đã config sẵn trên main)
✅ Chất lượng cao, đủ nhanh cho chatbot
✅ Chạy `setup_data.bat` → `run_lab.bat` → test ngay

### Cho Máy Bạn (Sau Này)
✅ **Chuyển sang 0.6B** (sửa settings.py)
✅ Tiết kiệm 7GB dung lượng, nhanh hơn 4x
✅ Chất lượng vẫn tốt, đủ cho use case tuyển sinh

### Chênh Lệch Thực Tế
- **Chất lượng:** 4B tốt hơn ~5-8%, nhưng 0.6B vẫn đủ tốt
- **Tốc độ:** 0.6B nhanh hơn 4x, nhưng cả hai đều < 300ms (chấp nhận được)
- **Tài nguyên:** 0.6B nhẹ hơn nhiều (1.2GB vs 8GB)

**Không có lựa chọn "sai"** — cả hai đều hoạt động tốt. Chọn theo điều kiện máy:
- Máy mạnh → 4B
- Máy yếu/hết dung lượng → 0.6B

---

## 9. Câu Hỏi Thường Gặp

**Q: Có thể dùng 4B trên máy 8GB RAM không?**
A: Không nên. Model sẽ chạy nhưng rất chậm (swap memory). Dùng 0.6B thay thế.

**Q: Embeddings từ 4B có tương thích với 0.6B không?**
A: Không. Phải re-index khi đổi model (dimensions khác nhau: 2560 vs 1024).

**Q: Có thể dùng GPU để tăng tốc không?**
A: Có, nhưng cần VRAM ≥8 GB cho 4B, ≥2 GB cho 0.6B. Sửa `device="cuda"` trong `embedding.py`.

**Q: Tại sao không dùng model khác (BGE, E5)?**
A: Qwen3 #1 trên MTEB cho multilingual, đặc biệt tốt cho tiếng Việt. BGE/E5 cũng tốt nhưng không bằng.

**Q: Instruction prompt có bắt buộc không?**
A: **CÓ**. Không dùng `prompt_name="query"` → chất lượng giảm 10-15%. Đã implement đúng trong code.

---

**Tài liệu tham khảo:**
- Integration plan: `docs/plans/qwen3-embedding-lancedb-integration.md`
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Qwen3-Embedding HuggingFace: https://huggingface.co/Qwen/Qwen3-Embedding-4B
