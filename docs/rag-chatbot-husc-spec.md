# Spec: rag-chatbot-husc — Colab Quality Eval (phase-1)

> Source of truth for phase-1 implementation.
> Authority: `docs/superpowers/specs/2026-04-17-colab quality-eval-design.md`

## Phase: phase-1

### Sections implemented
- S2: Notebook architecture (`colab_eval.ipynb`)
- S3: Evaluation metrics (accuracy, groundedness, hallucination_rate, recall, latency_p95)
- S4: Diagnostic report enrichment
- S5: Route parity (force_route vs auto-route)
- S6: Reproducibility (seeds, bootstrap CI, NI margins)

### Files in scope
- `packages/rag-chatbot-husc/src/notebooks/eval_core.py`
- `packages/rag-chatbot-husc/src/notebooks/colab_eval.ipynb`
- `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`

### Out of scope
- Backend pipeline changes
- E2E tests (notebook runs against live Colab environment)
