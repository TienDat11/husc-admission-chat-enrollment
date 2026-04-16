---
name: spec-status
description: Show current phase, coverage matrix, and orphan sections from spec-index.yaml
---

Run spec-status diagnostic:

1. Read `spec-index.yaml` from repo root
2. Show active phase + its declared sections + paths
3. Run coverage script: `node node_modules/@agentic-gateway/mega-template/scripts/spec-coverage.mjs`
4. Run orphan sweep: `node node_modules/@agentic-gateway/mega-template/hooks/orphan-sweep.mjs < /dev/null`
5. Report to user:
   - Active phase name + status
   - Sections declared (count)
   - Sections with @spec annotation in code (count)
   - Orphan sections list
   - Coverage percentage
   - Files touched in current phase vs out-of-phase

If user asks to dig into a specific section, grep for `@spec(section=X)` in packages/*/src/.
