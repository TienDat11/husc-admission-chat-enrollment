# Hướng Dẫn Setup RAG 2025 - Máy Mới

**Ngày:** 2026-03-26
**Dành cho:** Máy cô (16GB RAM, đủ dung lượng) - Qwen3-Embedding-4B

---

## Bước 1: Xóa Cache Model Cũ (Nếu Đã Tải Nhầm 8B)

Nếu bạn đã chạy `setup_data.bat` trước đây và nó tải model 8B (15GB) vào ổ C:, xóa cache cũ:

```bash
# Xóa cache HuggingFace trên ổ C: (nếu có)
rm -rf "C:/Users/ADMIN/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B"

# Xóa cache transformers (nếu có)
rm -rf "C:/Users/ADMIN/.cache/huggingface/transformers"
```

**Lưu ý:** Lệnh trên chỉ xóa model Qwen3-Embedding-8B. Các models khác (nếu có) vẫn giữ nguyên.

---

## Bước 2: Clone Repository

```bash
cd D:/chunking
git clone https://github.com/<your-org>/husc-admission-chat-enrollment.git
cd husc-admission-chat-enrollment
```

Hoặc nếu đã có repo, pull code mới nhất:

```bash
cd D:/chunking/husc-admission-chat-enrollment
git pull origin main
```

---

## Bước 3: Tạo Virtual Environment

```bash
cd rag2025
python -m venv venv
```

**Kích hoạt venv:**

```bash
# Windows (Git Bash)
source venv/Scripts/activate

# Windows (CMD)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

## Bước 4: Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Thời gian:** ~5-10 phút (tùy tốc độ mạng).

---

## Bước 5: Tạo File `.env`

Copy từ template:

```bash
cp .env.example .env
```

**Chỉnh sửa `.env`** (dùng text editor):

```bash
# ─── PRIMARY LLM (ramclouds.me – gemini-2.5-flash) ──────────
RAMCLOUDS_API_KEY=<your-key-here>

# ─── FALLBACK LLM (Groq – free, fast) ───────────────────────
GROQ_API_KEY=<your-key-here>

# ─── EMBEDDING MODEL (4B cho máy 16GB) ──────────────────────
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
QWEN_EMBEDDING_DIM=2560
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DIM=2560

# ─── HUGGINGFACE CACHE (redirect to D: drive) ───────────────
HF_HOME=D:/chunking/.cache/huggingface
TRANSFORMERS_CACHE=D:/chunking/.cache/huggingface/hub
```

**Quan trọng:**
- `HF_HOME` và `TRANSFORMERS_CACHE` trỏ đến ổ D: để tránh làm đầy ổ C:
- Model 4B cần ~8GB disk space

---

## Bước 6: Chuẩn Bị Dữ Liệu

Copy file dữ liệu vào thư mục `data/raw/`:

```bash
mkdir -p data/raw
# Copy file 2.jsonl hoặc các file JSONL khác vào data/raw/
```

---

## Bước 7: Chạy Setup Data Pipeline

```bash
setup_data.bat data/raw
```

**Pipeline sẽ chạy:**
1. Normalize raw data
2. Validate JSONL
3. Chunk documents
4. **Tải Qwen3-Embedding-4B (~8GB)** → lưu vào `D:/chunking/.cache/huggingface/`
5. Build LanceDB vector store
6. Build GraphRAG knowledge graph

**Thời gian:** ~10-15 phút (lần đầu tải model lâu hơn).

**Kiểm tra:** Sau khi xong, bạn sẽ thấy:
- `data/lancedb/` — vector database
- `data/graph/knowledge_graph.graphml` — knowledge graph
- `data/graph/entity_index.json` — entity index

---

## Bước 8: Chạy Server

```bash
run_lab.bat
```

Server sẽ khởi động tại `http://localhost:8000`.
Swagger UI tự động mở tại `http://localhost:8000/docs`.

**Test API:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?"}'
```

---

## Bước 9: Kiểm Tra Model Đã Tải

Xác nhận model 4B đã tải đúng:

```bash
ls -lh "D:/chunking/.cache/huggingface/hub/" | grep Qwen3-Embedding
```

Bạn sẽ thấy thư mục:
```
models--Qwen--Qwen3-Embedding-4B
```

**Dung lượng:** ~8GB.

---

## Troubleshooting

### Lỗi: Model vẫn tải về ổ C:

**Nguyên nhân:** `.env` chưa được load hoặc venv chưa activate.

**Giải pháp:**
1. Đảm bảo venv đã activate (dòng đầu terminal hiện `(venv)`)
2. Kiểm tra `.env` có đúng path không
3. Restart terminal và activate lại venv

### Lỗi: `ModuleNotFoundError: No module named 'lancedb'`

**Giải pháp:**
```bash
pip install lancedb pyarrow
```

### Lỗi: `RAMCLOUDS_API_KEY not found`

**Giải pháp:**
- Kiểm tra `.env` có chứa `RAMCLOUDS_API_KEY=...` không
- Đảm bảo không có khoảng trắng thừa
- Restart terminal sau khi sửa `.env`

---

## Chuyển Sang Model 0.6B (Máy RAM Thấp)

Nếu máy chỉ có 8GB RAM hoặc hết dung lượng, chuyển sang 0.6B:

**Sửa `.env`:**
```bash
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
QWEN_EMBEDDING_DIM=1024
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
```

**Xóa LanceDB cũ và re-index:**
```bash
rm -rf data/lancedb
setup_data.bat data/raw
```

Model 0.6B chỉ cần ~1.2GB disk space và ~4GB RAM.

---

## Tài Liệu Tham Khảo

- **So sánh 4B vs 0.6B:** `docs/qwen3-4b-vs-0.6b-comparison.md`
- **Integration plan:** `docs/plans/qwen3-embedding-lancedb-integration.md`
- **API docs:** `http://localhost:8000/docs` (sau khi chạy server)

---

**Hoàn thành!** Giờ bạn có thể query chatbot tuyển sinh qua API.
