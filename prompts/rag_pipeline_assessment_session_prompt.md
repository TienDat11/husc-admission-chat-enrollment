# RAG Pipeline Assessment Session Prompt

Use this prompt in a new Codex session for the HUSC admission/enrollment RAG project.

```text
Goal:
- Understand repo `husc-admission-chat-enrollment` enough to write a compact project-context memo in the repo, so future sessions do not need to reread from zero.
- Evaluate the current pipeline pragmatically: graph + PaddleRAG + HYDE at the front. Decide whether it is production-ready enough, whether the current usage is technically sound, and what lightweight upgrades make sense.
- Find the latest suitable BGE embedding model from BAAI for this project. Prioritize local use, low disk footprint, stable production behavior, and good Vietnamese/English retrieval tradeoffs.
- Extract the old evaluation questions from `packages\rag-chatbot-husc\src\notebooks\colab_eval_v5_thesis.ipynb` and move evaluation toward local machine execution with BGE, without relying on Colab.
- Recommend only upgrades that are not too heavy for a low-disk local environment.

Context:
- Repo has GitNexus index under project name `husc-admission-chat-enrollment`.
- Use GitNexus first for code navigation when available:
  - `gitnexus_query` for concepts and flows.
  - `gitnexus_context` for symbol-level context.
  - `gitnexus_impact` before editing any function, class, or method.
  - `gitnexus_detect_changes` before finishing any code modification task.
- If GitNexus warns that the index is stale, run `npx gitnexus analyze` first.
- Main product area is a RAG chatbot for HUSC admission/enrollment.
- Previous work tested multiple embedding types, but current constraint is low local disk space. BGE is acceptable if it is the best practical choice.
- Goal is not a heavy academic benchmark. Goal is a production-practical recommendation and repeatable local evaluation path.

Constraints:
- Cheap-lane-first. Split into low-cost lanes when useful:
  - repo exploration,
  - external docs/model research,
  - memo writing,
  - test harness review.
- Keep the leader model focused on scope, architecture decisions, and final review.
- Do not run heavy benchmarks or download several large embedding models.
- Do not refactor broadly.
- Do not modify code symbols without GitNexus impact analysis first.
- If impact analysis returns HIGH or CRITICAL, warn before editing.
- Prefer official sources for BGE: BAAI model cards, official repos, release notes, or Hugging Face model pages from the model publisher.
- Do not depend on Colab for final evaluation flow.

Tools/Evidence:
- Read `AGENTS.md` first.
- Use GitNexus resources/tools if available.
- Inspect only the minimum needed source files.
- Inspect `packages\rag-chatbot-husc\src\notebooks\colab_eval_v5_thesis.ipynb` for old test questions and evaluation setup.
- Use current official docs/model cards for BGE selection.
- If OpenAI docs/model guidance is needed, use official OpenAI docs only.

Output:
1. Create one compact markdown memo in the repo with:
   - project architecture context,
   - key RAG pipeline components,
   - important paths,
   - old evaluation notebook path,
   - next-session notes.
2. Give one clear BGE recommendation:
   - exact model name,
   - why it fits,
   - disk/runtime tradeoff,
   - when to choose a smaller or larger variant.
3. Assess graph + PaddleRAG + HYDE:
   - what is sound,
   - what is risky,
   - what should stay,
   - what should change lightly.
4. List lightweight upgrade options ranked by priority and cost.
5. Provide local evaluation plan:
   - how to extract/use test questions,
   - how to run against local BGE,
   - what metrics or checks are enough.
6. If code was changed:
   - list changed files,
   - run relevant tests/checks,
   - run `gitnexus_detect_changes`,
   - report any verification that could not be run.

Stop:
- Stop when memo file exists, BGE recommendation is clear, pipeline assessment is clear, and local evaluation path is actionable.
- If missing facts block a confident conclusion, state the missing fact and the exact next command/file to inspect.
```

