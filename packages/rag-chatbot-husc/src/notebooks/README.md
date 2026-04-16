# Colab Evaluation Notebook

Run RAG pipeline evaluation directly on Google Colab with GPU support.

## Quick Start

### 1. Open in Colab

Upload `colab_eval.ipynb` to Google Colab or open from Google Drive.

### 2. Configure Secrets (Recommended)

In Colab, go to **Secrets** (key icon in left sidebar) and add:

| Name | Value |
|------|-------|
| `PIPELINE_BASE_URL` | `http://127.0.0.1:8000` (or your server URL) |
| `RAMCLOUDS_API_KEY` | Your API key |
| `OPENAI_API_KEY` | Alternative API key |

### 3. Update Repository URL

In **Cell 2**, replace `<owner>/<repo>` with actual GitHub path:

```python
REPO_URL = "https://github.com/<owner>/<repo>.git"
```

For local testing without cloning, copy the repo contents to `/content/husc-admission-chat-enrollment/` manually.

### 4. Update Test Data Paths (if needed)

In **Cell 4**, adjust `PRIMARY` and `FALLBACK` paths if your test questions file is stored elsewhere.

### 5. Runtime Setup

Go to **Runtime > Change runtime type > GPU** (or TPU if preferred). Then run cells in order.

## Cell Walkthrough

| Cell | Purpose |
|------|---------|
| 0 | Runtime guard: verify GPU is available |
| 1 | Repo bootstrap + dependency install |
| 2 | Import codebase smoke test |
| 3 | Config & secrets (user fills in) |
| 4 | Data load & validation |
| 5 | Smoke eval (5 questions, fail-fast) |
| 6 | Full eval (all questions) |
| 7 | Scoring & export to `results/colab_eval/` |
| 8 | Diagnostic report generation |

## Output Files

After running all cells, `results/colab_eval/` contains:

- `eval_predictions.jsonl` - Minimal prediction output
- `eval_scored.jsonl` - Full results with scores
- `eval_summary_metrics.csv` - Per-category summary
- `diagnostic_report.md` - Human-readable report

## Troubleshooting

### GPU not detected

Runtime > Change runtime type > GPU. Restart runtime after changing.

### Connection errors

- Verify `PIPELINE_BASE_URL` is correct and the server is running
- Check firewall/proxy settings
- For local testing, use ngrok or cloud server URL

### Import errors

Ensure `sys.path` points to `/content/husc-admission-chat-enrollment`. Adjust if repo folder name differs.

## Local Pipeline Server

For local development, run the pipeline server:

```bash
cd rag2025
uvicorn rag2025.src.main:app --host 0.0.0.0 --port 8000
```

Then set `PIPELINE_BASE_URL=http://127.0.0.1:8000` in Secrets.
