---
name: spec-init
description: Bootstrap spec-index.yaml from an existing spec document
---

Interactive spec-index.yaml generator.

1. Ask user for spec doc path (default: `docs/spec.md` or first `.md` in `docs/`)
2. Read the spec, extract:
   - Phase headings (look for "Phase N", "### Phase N", "## Phase N")
   - Section numbers (look for "§N", "## N.M", "### N.M.K")
3. Ask user:
   - Which phase is active right now?
   - Which sections belong to each phase?
   - Which path patterns belong to each phase (default: `packages/<pkg>/src/**`)?
4. Generate `spec-index.yaml` from template
5. Show preview, ask user to approve
6. Write file
7. Add to git: `git add spec-index.yaml`

If `spec-index.yaml` already exists, warn and ask confirm before overwriting.

Template structure:

```yaml
active_phase: phase-1
spec: docs/<name>-spec.md

phases:
  phase-1:
    status: in-progress
    sections: [2, 3.1, 3.2, 4, 5, 6, 7]
    paths:
      - packages/<pkg>/src/**
      - tests/unit/**

  phase-2:
    status: planned
    gate: "phase-1.status == verified"
    sections: [8, 9]
    paths: [packages/<pkg>/src/phase2/**]
```
