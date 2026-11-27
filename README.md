# HUSC — Trợ lý tuyển sinh (RAG 2025)

Chatbot hỏi đáp tuyển sinh Trường Đại học Khoa học — Đại học Huế (HUSC). Hệ thống gồm backend RAG và frontend SPA, đóng gói qua Docker Compose với Caddy làm reverse proxy/TLS.

## Thành phần

| Thư mục | Vai trò | Công nghệ |
| --- | --- | --- |
| `rag2025/` | Backend API: truy xuất (LanceDB, BM25, graph), sinh câu trả lời, admin, script ingest/đánh giá | Python 3.13, FastAPI, uvicorn, LanceDB |
| `uni-guide-ai/` | Frontend SPA (sinh viên, giáo viên, admin) | React, Vite, TypeScript, Vitest |
| `api/openapi.json` | Hợp đồng API, nguồn sinh type cho frontend | OpenAPI (xuất bởi `rag2025/scripts/export_openapi.py`) |
| `deploy/vps/atlas-lab/` | Caddyfile cho môi trường staging (TLS mount sẵn) | Caddy 2 |
| `docker-compose.yml` | Dựng `app` (backend) và `web` (Caddy + SPA) | Docker Compose |
| `.github/workflows/` | CI: contract API, Lighthouse, cảnh báo độ mới dữ liệu, dry-run ingest năm | GitHub Actions |

## Chạy local

Yêu cầu: Python 3.13, Node 24.

**Backend** (từ thư mục `rag2025/`):

```bash
python -m venv .venv
source .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                # điền khóa API và thông tin admin; không commit
PYTHONPATH=src uvicorn src.main:app --reload --port 8000
```

Windows PowerShell thay dòng cuối bằng `$env:PYTHONPATH="src"; uvicorn src.main:app --reload --port 8000`.

**Frontend** (từ thư mục `uni-guide-ai/`):

```bash
npm ci
npm run dev                         # http://localhost:8080, gọi API tại http://localhost:8000
```

## Docker Compose

```bash
cp rag2025/.env.example rag2025/.env   # file này là tùy chọn với compose, nhưng backend cần nó
docker compose up --build
```

- `app` (backend) chỉ lắng nghe `127.0.0.1:8000` trên máy host, để truy cập gián tiếp qua Caddy.
- `web` (Caddy) phục vụ SPA và proxy các đường dẫn API tới `app`, tại cổng 80/443. Đặt biến `DOMAIN` để Caddy tự cấp TLS qua ACME; mặc định dùng `localhost`.
- Thư mục `rag2025/data/` được bind-mount vào `/data` trong container; không nằm trong image.

## Kiểm thử và CI

| Lệnh / workflow | Kiểm tra |
| --- | --- |
| `cd rag2025 && python -m pytest` | Backend unit/integration tests (cấu hình trong `rag2025/pytest.ini`) |
| `cd uni-guide-ai && npm test` | Vitest |
| `cd uni-guide-ai && npm run typecheck` | TypeScript |
| `cd uni-guide-ai && npm run test:smoke` | CRUD smoke giữa client thật và backend thật (cần backend đang chạy) |
| `api-contract.yml` | Contract test, kiểm tra lệch schema BE/FE, CRUD smoke |
| `lighthouse.yml` | Lighthouse CI cho `uni-guide-ai/` khi có thay đổi |
| `freshness-check.yml` | Cảnh báo Slack hằng ngày nếu dữ liệu quá cũ; bỏ qua có cảnh báo khi chưa có file audit |
| `yearly-reingest-dryrun.yml` | Dry-run quy trình chuyển năm tuyển sinh (không ghi LanceDB) |

## Dữ liệu

- Được commit trong git: một phần corpus `rag2025/data/` (`chunked`, `raw`, `normalized`, `validated`, `eval`, `major_codes`). Một số file LanceDB ở thư mục gốc `data/lancedb/` cũng đang được theo dõi.
- Không commit (bị `.gitignore`): phần còn lại của `rag2025/data/` như LanceDB cục bộ, graph, cache, `hf-cache`, `audit_*.json`; và các PDF thô trong `new_data/`.
- Tài liệu, báo cáo và ghi chú làm việc (`docs/reviews/`, `docs/superpowers/`, `.omp/`, `.scratch/`, `.gitnexus/`, `graphify-out/`) là cục bộ.

## Cấu hình và bí mật

- Cấu hình backend đọc từ `rag2025/.env` (không commit). Mẫu nằm ở `rag2025/.env.example` (được commit).
- Không commit: `.env*` (trừ `.env.example`), `deploy-env.json`, `*.pem`, `*.key`, `id_rsa*`, thư mục `secrets/`. Các file này đã nằm trong `.gitignore`.
- Chứng chỉ TLS của staging được mount từ ngoài vào `/etc/atlas-tls`, không nằm trong repo.
- Caddy không đưa `/metrics` ra ngoài. Mọi phản hồi có header bảo mật (HSTS, CSP, `nosniff`, `frame-ancestors 'none'`).

## Cấu trúc repo (rút gọn)

```
rag2025/          backend (src/, scripts/, tests/, config/, alembic/, Dockerfile, Caddyfile)
uni-guide-ai/     frontend (src/, tests/, scripts/, Dockerfile)
api/              openapi.json
deploy/           cấu hình triển khai (vps/atlas-lab)
docs/             tài liệu được commit
.github/workflows/ CI
docker-compose.yml
```
