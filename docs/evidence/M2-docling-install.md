# Probe 2: Docling Install + Timing on Windows CPU-only

## Commands Executed

```
D:/miniconda3/python.exe -c "import docling; print('docling', docling.__version__)"
→ ModuleNotFoundError: No module named 'docling'

D:/miniconda3/python.exe -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda_build', torch.version.cuda)"
→ torch 2.9.1+cpu cuda_available False cuda_build None

D:/miniconda3/python.exe -m pip show docling docling-core torch 2>&1 | head -40
→ WARNING: Packages not found: docling, docling-core
→ Name: torch Version: 2.9.1

D:/miniconda3/python.exe -c "import shutil; u=shutil.disk_usage('D:/'); print('D: free_GB', round(u.free/1e9,1), 'total_GB', round(u.total/1e9,1))"
→ D: free_GB 41.5 total_GB 241.2
```

## Install Command

```
D:/miniconda3/python.exe -m pip install docling --extra-index-url https://download.pytorch.org/whl/cpu --user
```

First two attempts failed with `[WinError 5] Access is denied` on `torch\_C.cp313-win_amd64.pyd` (locked by another process). Third attempt with `--user` succeeded.

Installed packages:
- docling 2.115.0
- docling-slim 2.115.0
- docling-core 2.87.1
- docling-parse 7.8.1
- docling-ibm-models 3.13.3
- torch 2.13.0+cpu (newer wheel installed via --user alongside existing 2.9.1+cpu)
- torchvision 0.28.0+cpu
- accelerate 1.14.0
- rapidocr 3.9.2
- semchunk 3.2.5
- mpire 2.10.2
- polyfactory 3.3.0
- pydantic-settings 2.14.2
- doclang 0.7.3
- multiprocess 0.70.19
- mail-parser 4.4.0

Verification:
```
D:/miniconda3/python.exe -c "from docling.document_converter import DocumentConverter; print('DOCLING OK')"
→ DOCLING OK
```

## Disk Space Before/After

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| D: free | 41.5 GB | 38.0 GB | ~3.5 GB |
| D: total | 241.2 GB | 241.2 GB | — |

## Dependency Conflict Check

Explicitly checked pinned packages:

| Package | Pinned Version | Post-install Version | Status |
|---------|---------------|---------------------|--------|
| lancedb | 0.29.2 | 0.29.2 | NO CHANGE |
| pyarrow | ==23.0.1 | 23.0.1 | NO CHANGE |
| pydantic | ==2.10.3 | 2.10.3 | NO CHANGE |
| fastapi | 0.121.3 | 0.121.3 | NO CHANGE |
| numpy | 2.1.0 | 2.1.0 | NO CHANGE |

Result: **No dependency conflicts.** Docling's pydantic-settings 2.14.2 was installed independently without bumping pydantic itself.

## Timing Probe Results

Test file: `new_data/THONG_TIN_TUYEN_SINH_202611.pdf` (77 pages, admission info PDF)

```json
{
  "file": "THONG_TIN_TUYEN_SINH_202611.pdf",
  "seconds": 161.6,
  "md_chars": 49815,
  "pages": 77,
  "has_pipe_table": true,
  "sample_head": "## BỘ GIÁO DỤC VÀ ĐÀO TẠO ĐẠI HỌC HUẾ\n\n<!-- image -->\n\n## THÔNG TIN\n\n## TUYỂN SINH ĐẠI HỌC HỆ CHÍNH QUY NĂM 2026\n\n(Ban hành kèm theo Quyết định số        /QĐ-HĐTSĐH ngày      tháng 4 năm 2026 của Chủ tịch Hội đồng tuyển sinh đại học năm 2026)\n\n## MỤC LỤC\n\n| A. THÔNG TIN CHUNG ................................................................................................. 3                                                                                            |\n|---------------------------------------------------------------------------------------------------------------------------------"
}
```

Key metrics:
- **Parse time**: 161.6 seconds for 77-page PDF (~2.1 sec/page)
- **Markdown output**: 49,815 characters
- **Pipe tables**: YES — table structure preserved
- **Layout models**: Downloaded on first run (RapidOCR ONNX models in PP-OCRv6 family), cached on second run
- **Engine**: RapidOCR using onnxruntime (CPU)

## Torch Configuration

| Property | Value |
|----------|-------|
| Version | 2.13.0+cpu (--user install) |
| Original version | 2.9.1+cpu (site-packages, locked) |
| CUDA available | False |
| CUDA build | None |
| CPU-only | Confirmed (+cpu suffix) |

## Verdict

**PASS** — Docling is usable CPU-only on Windows.

- Installs and imports successfully via `--user` flag (global install blocked by locked torch .pyd).
- No dependency conflicts with existing pinned packages (lancedb, pyarrow==23.0.1, pydantic==2.10.3, fastapi, numpy all untouched).
- Parse speed: ~2.1 sec/page on CPU (onnxruntime backend). For a 77-page PDF, total wall time was 161.6 seconds including model warmup.
- Produces pipe-table markdown — compatible with standard Markdown parsers.
- Layout models (~hundreds of MB for PP-OCRv6 ONNX files) downloaded to user cache directory on first run.
- First-run model download adds latency; subsequent runs benefit from caching.

### Risks for Production Use

1. **Speed**: 2.1 sec/page on CPU is slow for batch ingestion. A 200-page document would take ~7 minutes. Consider async/background processing.
2. **Disk**: ~3.5 GB consumed (torch CPU wheel + docling + ONNX models). Monitor D: drive space.
3. **Torch version split**: Two torch versions coexist (2.9.1 site-packages, 2.13.0+user). This may cause subtle import conflicts if other packages resolve torch differently.
4. **Windows permission issues**: Global pip install fails when torch is locked. The `--user` workaround is reliable but installs scripts to `%APPDATA%\Python\Scripts` which is not on PATH.
