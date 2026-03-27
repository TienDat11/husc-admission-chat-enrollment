## Context
RAG backend currently has high-risk surfaces and unstable behavior under load. We need a hardened baseline for `dev` with reproducible verification.

## Problems to track
- [ ] `POST /v2/graph/update` lacked strong protection (auth/validation/rate limiting/concurrency guard)
- [ ] CORS policy too permissive for production
- [ ] Internal error details exposed to clients
- [ ] Async endpoints did blocking embedding work
- [ ] No low-confidence short-circuit (wasted LLM calls)
- [ ] Prompt/context handling vulnerable to instruction injection patterns

## Acceptance criteria
- [ ] Admin token required for graph mutation endpoint
- [ ] Request limits enforced (`MAX_QUERY_LENGTH`, graph chunk limits)
- [ ] Request rate limit active and returns HTTP 429 when exceeded
- [ ] Generic 500 error payloads, detailed errors only in server logs
- [ ] Embedding execution moved off event loop for query paths
- [ ] Low-confidence answers skip expensive generation path
- [ ] Context bounded by max length and wrapped in explicit delimiters
- [ ] No raw context preview in logs

## Verification checklist
- [ ] `python -m py_compile rag2025/src/main.py`
- [ ] `python -m py_compile rag2025/src/services/llm_generator.py`
- [ ] Manual API checks: `/query`, `/v2/query`, `/v2/graph/update`
- [ ] Confirm non-admin call to `/v2/graph/update` returns 403
