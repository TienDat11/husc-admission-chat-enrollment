# HUSC Admission Chatbot

Chatbot tư vấn tuyển sinh Đại học Khoa học Huế (HUSC). RAG pipeline + React SPA + admin console.

## Cấu trúc

| Thư mục | Vai trò | Entry |
|---|---|---|
| `rag2025/` | Backend FastAPI (RAG, LanceDB, GraphRAG) | `src/main.py` |
| `uni-guide-ai/` | Frontend React + Vite + TS | `src/main.tsx` |
| `docs/` | Spec, memo, evidence | — |

## Chạy local

```bash
# Backend (Windows venv ở root repo)
cp rag2025/.env.example rag2025/.env   # set NS24_API_KEY, ADMIN_*
cd rag2025
../.venv/Scripts/python.exe -m uvicorn src.main:app --reload --port 8000
# Health: curl http://localhost:8000/health

# Frontend
cd uni-guide-ai
npm install
npm run dev
```

## Test

```bash
cd rag2025 && pytest          # ~3.4k test thu thập
cd uni-guide-ai && npm run build   # output: dist/
```

## Lưu ý

- Root `package.json` là stub (`@agentic-gateway/mega-template`), không phải manifest frontend — manifest thật là `uni-guide-ai/package.json`.
- `rag2025/data/` (LanceDB + corpus) là dữ liệu local, không được track trong git.
- Root `spec-index.yaml` là cấu hình gate phase/section cho hooks.
- Tài liệu chi tiết: `docs/rag-chatbot-husc-spec.md`.
