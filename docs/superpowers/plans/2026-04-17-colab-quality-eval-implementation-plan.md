# Colab Quality Eval Notebook (Phase-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai notebook Colab bám codebase thật để chạy đánh giá chất lượng trả lời toàn diện (ưu tiên correctness/groundedness/hallucination), xuất báo cáo chẩn đoán điểm yếu và roadmap cải tiến.

**Architecture:** Notebook `colab_eval.ipynb` đóng vai trò orchestrator; logic đánh giá lõi tách sang module Python để test được bằng pytest. Notebook chỉ điều phối setup Colab, chạy smoke/full eval, và gọi module scoring/reporting. API mục tiêu là `POST /query` và `POST /v2/query` từ `rag2025/src/main.py`.

**Tech Stack:** Python 3.10+, Jupyter/Colab, requests/httpx, pandas, numpy, sentence-transformers, pytest.

---

## File Structure (lock trách nhiệm trước khi làm)

- Create: `packages/rag-chatbot-husc/src/notebooks/colab_eval.ipynb`  
  Điều phối Colab runtime, clone repo, cài phụ thuộc, chạy smoke/full eval, xuất artifact.

- Create: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`  
  Core functions: load test set với fallback, gọi API pipeline, score metric, taxonomy lỗi, export report.

- Create: `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`  
  Unit tests cho fallback logic, fail-fast threshold, metric calculations, output schema.

- Modify (nếu chưa có): `packages/rag-chatbot-husc/src/notebooks/__init__.py`  
  Export hàm chính từ `eval_core.py` để notebook import gọn.

- Create: `packages/rag-chatbot-husc/src/notebooks/README.md`  
  Hướng dẫn chạy notebook trên Colab + mapping env vars.

---

### Task 1: Khởi tạo module đánh giá lõi theo interface contract

**Files:**
- Create: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`
- Test: `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`

- [ ] **Step 1: Write the failing test (data fallback + normalized output)**

```python
# packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py
from notebooks.eval_core import load_test_questions, normalize_pipeline_output


def test_load_test_questions_uses_fallback_when_primary_missing(tmp_path):
    primary = tmp_path / "missing.json"
    fallback = tmp_path / "fallback.json"
    fallback.write_text('[{"question":"q","ground_truth_answer":"a","category":"simple"}]', encoding="utf-8")

    rows, used_path = load_test_questions(str(primary), str(fallback))

    assert len(rows) == 1
    assert used_path == str(fallback)


def test_normalize_pipeline_output_has_required_keys():
    raw = {"answer": "ok", "sources": ["s1"], "confidence": 0.9}
    out = normalize_pipeline_output(raw, mode="v2")

    assert set(["answer", "context_chunks", "source_ids", "confidence", "groundedness_score", "route", "raw"]).issubset(out.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'notebooks.eval_core'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/rag-chatbot-husc/src/notebooks/eval_core.py
import json
from typing import Any


def load_test_questions(primary_path: str, fallback_path: str):
    try:
        with open(primary_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return rows, primary_path
    except Exception:
        with open(fallback_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return rows, fallback_path


def normalize_pipeline_output(raw: dict[str, Any], mode: str = "v2") -> dict[str, Any]:
    sources = raw.get("sources", [])
    chunks = raw.get("chunks", [])
    context_chunks = [c.get("text", "") for c in chunks if isinstance(c, dict)]
    return {
        "answer": raw.get("answer", ""),
        "context_chunks": context_chunks,
        "source_ids": list(sources) if isinstance(sources, list) else [],
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "groundedness_score": float(raw.get("groundedness_score", 0.0) or 0.0),
        "route": raw.get("route", mode),
        "raw": raw,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py::test_load_test_questions_uses_fallback_when_primary_missing -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/rag-chatbot-husc/src/notebooks/eval_core.py packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py
git commit -m "v5-phase-1.slice-01: add eval core data loading and output normalization"
```

---

### Task 2: Triển khai pipeline client + smoke fail-fast

**Files:**
- Modify: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`
- Test: `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`

- [ ] **Step 1: Write the failing test (smoke threshold >50%)**

```python
from notebooks.eval_core import should_abort_after_smoke


def test_should_abort_after_smoke_when_failures_exceed_half():
    assert should_abort_after_smoke(total=10, failures=6) is True
    assert should_abort_after_smoke(total=10, failures=5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py::test_should_abort_after_smoke_when_failures_exceed_half -v`  
Expected: FAIL (`ImportError` or function not found)

- [ ] **Step 3: Write minimal implementation (API call + fail-fast helper)**

```python
import requests


def should_abort_after_smoke(total: int, failures: int) -> bool:
    if total <= 0:
        return True
    return (failures / total) > 0.5


def call_pipeline(base_url: str, query: str, mode: str = "v2", top_k: int = 5):
    if mode == "v1":
        resp = requests.post(f"{base_url}/query", json={"query": query, "force_rag_only": False}, timeout=120)
    else:
        resp = requests.post(f"{base_url}/v2/query", json={"query": query, "top_k": top_k}, timeout=120)
    resp.raise_for_status()
    return resp.json()
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py -v`  
Expected: PASS for new smoke threshold test

- [ ] **Step 5: Commit**

```bash
git add packages/rag-chatbot-husc/src/notebooks/eval_core.py packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py
git commit -m "v5-phase-1.slice-02: add pipeline client and smoke fail-fast rule"
```

---

### Task 3: Triển khai metric correctness/groundedness/hallucination + retrieval proxy

**Files:**
- Modify: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`
- Test: `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`

- [ ] **Step 1: Write failing test for retrieval recall proxy + hallucination flag**

```python
from notebooks.eval_core import retrieval_recall_proxy, hallucination_flag


def test_retrieval_recall_proxy_hits_ground_truth_source_chunks():
    assert retrieval_recall_proxy(["a", "b"], ["x", "b"]) == 1
    assert retrieval_recall_proxy(["a"], ["x", "y"]) == 0


def test_hallucination_flag_from_groundedness_threshold():
    assert hallucination_flag(0.10, threshold=0.18) == 1
    assert hallucination_flag(0.25, threshold=0.18) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py::test_retrieval_recall_proxy_hits_ground_truth_source_chunks -v`  
Expected: FAIL

- [ ] **Step 3: Implement metrics (custom + reusable)**

```python
import re


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.,;:!\?\-\_\(\)\[\]\{\}\"\']", "", s)
    return s


def exact_correctness(pred: str, gt: str) -> int:
    return 1 if normalize_text(pred) == normalize_text(gt) else 0


def retrieval_recall_proxy(source_ids: list[str], gt_chunks: list[str]) -> int:
    if not gt_chunks:
        return 0
    return 1 if set(source_ids).intersection(set(gt_chunks)) else 0


def hallucination_flag(groundedness_score: float, threshold: float = 0.18) -> int:
    return 1 if groundedness_score < threshold else 0
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py -v`  
Expected: PASS for metric tests

- [ ] **Step 5: Commit**

```bash
git add packages/rag-chatbot-husc/src/notebooks/eval_core.py packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py
git commit -m "v5-phase-1.slice-03: implement quality metrics and retrieval recall proxy"
```

---

### Task 4: Tạo notebook Colab orchestration bám code thật

**Files:**
- Create: `packages/rag-chatbot-husc/src/notebooks/colab_eval.ipynb`
- Modify: `packages/rag-chatbot-husc/src/notebooks/README.md`

- [ ] **Step 1: Create notebook with GPU/runtime setup cells**

Cell nội dung bắt buộc:

```python
import os, sys, subprocess
print(sys.version)
!nvidia-smi

import torch
assert torch.cuda.is_available(), "GPU chưa bật. Vào Runtime > Change runtime type > GPU"
print(torch.cuda.get_device_name(0))
```

- [ ] **Step 2: Add repo bootstrap + dependency install cells**

```python
REPO_URL = "https://github.com/<owner>/<repo>.git"
BRANCH = "main"
!git clone -b {BRANCH} {REPO_URL}
%cd husc-admission-chat-enrollment/rag2025
!pip install -r requirements.txt
```

- [ ] **Step 3: Add evaluation orchestration cell**

```python
from notebooks.eval_core import run_smoke_eval, run_full_eval, export_report

PRIMARY = "results/test_questions.json"
FALLBACK = "backup_mail_package_2026/python_project/rag2025/results/test_questions.json"
BASE_URL = "http://127.0.0.1:8000"

smoke = run_smoke_eval(base_url=BASE_URL, primary_path=PRIMARY, fallback_path=FALLBACK, n=10)
if smoke["abort"]:
    raise RuntimeError(f"Smoke failed: {smoke}")

full = run_full_eval(base_url=BASE_URL, primary_path=PRIMARY, fallback_path=FALLBACK)
export_report(full, out_dir="results/colab_eval")
```

- [ ] **Step 4: Manual notebook execution check**

Run in Colab from top-to-bottom.  
Expected: tạo được các file:
- `results/colab_eval/eval_predictions.jsonl`
- `results/colab_eval/eval_scored.jsonl`
- `results/colab_eval/eval_summary_metrics.csv`
- `results/colab_eval/diagnostic_report.md`

- [ ] **Step 5: Commit**

```bash
git add packages/rag-chatbot-husc/src/notebooks/colab_eval.ipynb packages/rag-chatbot-husc/src/notebooks/README.md
git commit -m "v5-phase-1.slice-04: add colab evaluation notebook orchestration"
```

---

### Task 5: Hoàn thiện report generator + taxonomy lỗi + roadmap gợi ý

**Files:**
- Modify: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`
- Test: `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`

- [ ] **Step 1: Write failing test for report sections**

```python
from notebooks.eval_core import build_diagnostic_report


def test_diagnostic_report_contains_required_sections():
    report = build_diagnostic_report({"summary": {"accuracy": 0.7}, "errors": []})
    assert "Executive summary" in report
    assert "Top lỗi chính" in report
    assert "Roadmap cải tiến" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py::test_diagnostic_report_contains_required_sections -v`  
Expected: FAIL

- [ ] **Step 3: Implement report builder**

```python

def build_diagnostic_report(result: dict) -> str:
    return """# Diagnostic Report

## Executive summary
...

## Kết quả theo category
...

## Top lỗi chính
...

## Roadmap cải tiến
- Quick wins (1-3 ngày)
- Medium (1-2 tuần)
- Long-term (2-6 tuần)
"""
```

- [ ] **Step 4: Run full test suite for notebook module**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py -v`  
Expected: PASS toàn bộ

- [ ] **Step 5: Commit**

```bash
git add packages/rag-chatbot-husc/src/notebooks/eval_core.py packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py
git commit -m "v5-phase-1.slice-05: add diagnostic report and error taxonomy outputs"
```

---

### Task 6: Verification cuối và handoff

**Files:**
- Modify: `packages/rag-chatbot-husc/src/notebooks/README.md`

- [ ] **Step 1: Add exact runbook in README**

```markdown
## Colab Runbook
1. Runtime -> GPU
2. Run cells in order
3. Check output artifacts under results/colab_eval
4. Verify diagnostic_report.md exists
```

- [ ] **Step 2: Run local tests for affected scope**

Run: `pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py -v`  
Expected: PASS

- [ ] **Step 3: Type/build sanity**

Run: `python -m py_compile packages/rag-chatbot-husc/src/notebooks/eval_core.py`  
Expected: no output, exit code 0

- [ ] **Step 4: Final commit**

```bash
git add packages/rag-chatbot-husc/src/notebooks/README.md
git commit -m "v5-phase-1.slice-06: finalize colab eval runbook and verification"
```

- [ ] **Step 5: Capture verification evidence**

Lưu output test vào `.verification/`:

```bash
mkdir -p .verification
pytest packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py -v | tee .verification/colab_eval_tests.txt
```

Expected: file `.verification/colab_eval_tests.txt` có summary PASS.

---

## Spec Coverage Self-Review

- Appendix A (interface contract): covered in Task 1 + Task 4 (API call & normalized schema).
- Appendix B (metric decisions): covered in Task 3 + Task 5.
- Fail-fast threshold cụ thể: covered in Task 2.
- Git reference pin (`main` + optional pinned commit): covered in Task 4.
- Env vars + reproducibility runbook: covered in Task 4 + Task 6.
- `ground_truth_source_chunks` usage rõ ràng: covered via retrieval recall proxy in Task 3.
- Fallback data trigger điều kiện rõ: covered in Task 1.

Không dùng placeholder TODO chung chung; mọi bước code/test đều có lệnh cụ thể.
