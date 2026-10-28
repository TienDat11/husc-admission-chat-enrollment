# RAG Pipeline Assessment Prompt With Workflow Harness

Use this file as the prompt for a new Codex session in repo `D:\chunking\husc-admission-chat-enrollment`.

```text
$openai-model-prompting $cost-optimized-delegation $ramclouds-agent-router $amp-workflow-sop

Use `gpt-5.5` as leader with high reasoning effort.
Use cheap lanes first for routine repo exploration, docs/model research, and writing.
Keep `gpt-5.5` focused on scope, architecture decisions, safety, and final review.

If `$amp-workflow-sop` is available, load it first.
If `$amp-workflow-sop` is not available, apply this embedded Workflow Harness SOP instead:
- Stage 0 triage first.
- FAST path only if all are true:
  1. single-file change
  2. <= 2h estimate
  3. no cross-file or cross-package dependency changes
  4. acceptance criteria are explicit
  5. no missing external constraints
- If any condition is false or uncertain, use FULL path:
  - collect evidence with cheap lanes first
  - synthesize once
  - execute only the bounded plan
- First response must include:
  1. Path selected: FAST or FULL
  2. Rationale: up to 2 bullets
  3. Blocking ambiguity: one line, or `none`
- Do not spend leader effort on routine file discovery.
- Do not use Oracle or paid second-model review unless user explicitly approves.
- Finish with proof of work: changed files, commands run, checks, blockers, risks.

Goal:
Finish the RAG pipeline assessment for `husc-admission-chat-enrollment`.
Update or create one compact memo at:
`docs/rag_pipeline_context_memo.md`

The memo must cover:
- project architecture context
- key RAG pipeline components
- important paths
- old evaluation notebook path
- next-session notes
- BGE recommendation
- graph + PaddedRAG + HYDE assessment
- lightweight upgrade options
- local evaluation plan

Context:
- Workspace: `D:\chunking\husc-admission-chat-enrollment`
- GitNexus project name: `husc-admission-chat-enrollment`
- Main product: HUSC admission/enrollment RAG chatbot.
- Existing assessment prompt: `prompts/rag_pipeline_assessment_session_prompt.md`
- Existing memo may already exist at `docs/rag_pipeline_context_memo.md`; if it exists, refine it instead of creating duplicate docs.
- Legacy eval notebook: `packages\rag-chatbot-husc\src\notebooks\colab_eval_v5_thesis.ipynb`
- Local test set: `rag2025\results\test_questions.json`
- Backup test set: `rag2025\backup_mail_package_2026\python_project\rag2025\results\test_questions.json`
- Current local constraint: low disk. Do not download multiple embedding models or run heavy benchmarks.

Use GitNexus first when available:
- Read `AGENTS.md` first.
- Use `gitnexus_query` for concept/code flow lookup.
- Use `gitnexus_context` for symbol-level context.
- Before editing any function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})`.
- If impact returns HIGH or CRITICAL, warn before editing.
- Before finishing any code modification task, run `gitnexus_detect_changes()`.
- If GitNexus warns index is stale, run `npx gitnexus analyze` first.
- If GitNexus MCP tools are not available, say so and use `npx gitnexus status`, `rg`, and file reads as fallback.

Core files to inspect only as needed:
- `rag2025\src\main.py`
- `rag2025\src\services\query_router.py`
- `rag2025\src\services\query_enhancer.py`
- `rag2025\src\services\graphrag_retriever.py`
- `rag2025\src\services\embedding.py`
- `rag2025\src\services\hybrid_search.py`
- `rag2025\src\services\lancedb_retrieval.py`
- `rag2025\scripts\build_graph.py`
- `rag2025\src\domain\graph.py`
- `packages\rag-chatbot-husc\src\notebooks\eval_core.py`
- `packages\rag-chatbot-husc\src\notebooks\colab_eval_v5_thesis.ipynb`

Model / routing:
- Use `researcher` lane for official BAAI / Hugging Face / release-note research.
- Use `quick` or `explorer` lane for read-only repo mapping.
- Use local leader only for final architecture call and memo review.
- Use `backend` only if implementing code changes becomes necessary.
- Do not use Oracle unless user explicitly approves.

BGE research rules:
- Prefer official sources:
  - BAAI Hugging Face model cards
  - FlagEmbedding / official BAAI repos
  - arXiv paper if useful
- Recommend exactly one default model.
- Prefer `BAAI/bge-m3` unless official evidence proves a better low-disk multilingual BGE fit.
- Mention `BAAI/bge-multilingual-gemma2` only if useful as a larger comparison; reject it for low-disk local use if it is too heavy.
- Explain:
  - exact model name
  - Vietnamese/English retrieval fit
  - disk/runtime tradeoff
  - when to choose smaller/larger variants

Pipeline assessment:
Assess graph + PaddedRAG + HYDE:
- what is sound
- what is risky
- what should stay
- what should change lightly

Local evaluation plan:
Use `rag2025\results\test_questions.json`.
Explain how to compare:
- `padded_rag`
- `graph_rag`
- auto routing

Enough metrics:
- semantic accuracy
- retrieval recall proxy
- hallucination rate
- groundedness
- p95 latency
- route split / route correctness
- zero-error run requirement

Constraints:
- Do not run heavy benchmarks.
- Do not download several large models.
- Do not refactor broadly.
- Do not modify code symbols without GitNexus impact analysis first.
- Do not create duplicate docs if memo already exists.
- Do not revert user changes.
- Use `rg` / `rg --files` for search.
- Use `apply_patch` for manual edits.
- Keep memo compact and actionable.

Verification:
- For docs-only changes:
  - read back changed memo
  - run `git diff -- docs/rag_pipeline_context_memo.md`
  - run `npx gitnexus status`
- For any code change:
  - run targeted tests
  - run `gitnexus_detect_changes()`
  - report changed files and verification

Output:
Return:
1. Path to memo file.
2. BGE recommendation summary.
3. Pipeline assessment summary.
4. Local evaluation path summary.
5. Changed files.
6. Commands/checks run.
7. Any blockers or verification not run.

Stop:
Stop when:
- memo exists or is updated
- BGE recommendation is clear
- pipeline assessment is clear
- local evaluation path is actionable
- verification for touched files is reported

If missing facts block a confident conclusion, stop with:
- missing fact
- exact file/command/source needed next
```

