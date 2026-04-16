---
name: phase-promote
description: Mark current phase as verified and advance to next phase in spec-index.yaml
---

Promote current phase to verified status:

1. Read `spec-index.yaml`
2. Identify `active_phase` and the phase object
3. Pre-check gates:
   - **Independent review (V5-R055, V5-R056)**: `.omc/reviews/<phase>-{reviewer,verifier,critic}.md`
     all exist with accepted verdict (`approved` or `approved-with-notes`).
     If missing or blocking verdict → STOP and instruct user to run `/phase-review`.
   - Run coverage: all declared sections have @spec annotation (orphan count = 0)
   - Run tests: `npm test --workspaces` passes
   - Run build: `npm run build` passes
   - PROGRESS.md is up-to-date (mtime < 24h)
4. If any pre-check fails, STOP and report failures. Do not promote.
5. If all pass:
   - Update phase status to `verified`
   - Find next phase where `gate` condition now satisfies
   - Update `active_phase` to next phase
   - Set next phase status to `in-progress`
6. Update PROGRESS.md with promotion entry (timestamp, from-phase, to-phase)
7. Commit: `phase-promote: {from-phase} verified → {to-phase} active`

If user explicitly requests forced promotion without gate passing, require
them to type `FORCE PROMOTE` literal string first. Document reason in commit.
