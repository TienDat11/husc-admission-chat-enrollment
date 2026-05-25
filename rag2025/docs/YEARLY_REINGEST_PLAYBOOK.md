# Yearly Reingest Playbook (S13.8)

> Operational runbook for rotating the HUSC admissions index from year N to N+1.
> Honors **C13** (lifelong audit retention) and **C14** (auto-trigger when crawler detects new year).

## When to trigger

A rotation must be run when ANY of the following holds:

1. The crawler emits a `new_year_detected` event (auto-trigger via Slack notify).
2. `detect_new_year_signal()` returns a year value greater than the current `CURRENT_ADMISSION_YEAR`.
3. Cô tuyển sinh confirms the new admissions notification is published on tuyensinh.husc.edu.vn.

The freshness alert workflow runs daily and will notify Slack when `freshness_lag_days > 90`.

## Pre-flight checks

Before running the rotation, verify:

- `rag2025/data/audit_pre_reingest.json` contains the latest crawl date.
- `ADMIN_API_TOKEN` env var is set on the running FastAPI host (required for `/admin/reload-table`).
- `SLACK_RAG_OPS_WEBHOOK` repo secret is set so progress notifications fire.
- Disk has at least 5 GB free for the new blue table + chunked outputs.
- Phase 0 baseline snapshot (`rag2025/data/chunked_legacy_2025/`) exists from a prior phase.

## Step-by-step rotation (year = N+1)

1. **Crawl the new year's notifications**
   - Update `crawl_urls.py` ID range to include the new notification ids.
   - Run: `python rag2025/scripts/crawl_urls.py --ids "75,76,77,..."`.

2. **Run the 3-way chunker pipeline**
   - For each new notification HTML:
     `python rag2025/scripts/chunker_3way.py --nid 75 --html-path rag2025/data/raw/75.html`
   - Then run the arbiter for each: `python rag2025/scripts/chunker_arbiter.py --nid 75 --input-dir rag2025/data/chunked_3way/75/ --source-html rag2025/data/raw/75.html`

3. **Bootstrap the blue table**
   - `python rag2025/scripts/bootstrap_lancedb_blue.py --table husc_v${nextYear}_blue`
   - Reuse Qwen3-Embedding-0.6B (per C7) — do NOT switch embedder.

4. **Mark prior year as superseded**
   - In the existing `husc` table, run a LanceDB SQL update setting
     `is_superseded=true, valid_to='{N}-12-31'` on rows where `data_year={N}`.

5. **Atomic flip via `/admin/reload-table`**
   - PowerShell example (Windows):
     ```powershell
     $token = $env:ADMIN_API_TOKEN
     Invoke-RestMethod `
       -Uri "http://localhost:8000/admin/reload-table" `
       -Method Post `
       -Headers @{ "X-Admin-Token" = $token } `
       -ContentType "application/json" `
       -Body (@{ table_name = "husc_v${nextYear}_blue"; drain_timeout_s = 25 } | ConvertTo-Json)
     ```
     where `$nextYear = 2027  # ← Update to the actual year being rotated to`.
   - SLA: total < 30s = drain ≤ 25s + swap ≤ 1s + healthcheck ≤ 4s.

6. **Verify the flip**
   - `Invoke-RestMethod -Uri "http://localhost:8000/api/meta"` — `current_admission_year` should reflect the new year (after env update).
   - Run `python rag2025/scripts/smoke_test_blue.py --table husc_v${nextYear}_blue --gold rag2025/data/eval/86q_gold.json`.
   - Confirm hallucination type counters are within expected envelope.

7. **Post Slack confirmation**
   - The yearly_rotation script emits a Slack info notification on completion.

## Rollback procedure

If the new blue table fails the post-flip smoke test:

1. Re-run `/admin/reload-table` with the previous table name (e.g. `husc`).
   ```powershell
   Invoke-RestMethod `
     -Uri "http://localhost:8000/admin/reload-table" `
     -Method Post `
     -Headers @{ "X-Admin-Token" = $env:ADMIN_API_TOKEN } `
     -ContentType "application/json" `
     -Body '{ "table_name": "husc", "drain_timeout_s": 25 }'
   ```

2. Verify `/api/meta` reflects the rollback.

3. Investigate failures by reading `rag2025/results/regression_blue.json` and the chunker decision log at `rag2025/data/chunker_decision.jsonl`.

## Owner roles

| Role | Owner | Notes |
|---|---|---|
| Trigger detection | dev cron + Slack notify | Per C14 — automatic |
| Pipeline execution | dev | Slack info notifications go to ops channel |
| Sanity verification | cô tuyển sinh | Read-only Slack notification — confirms new dossier matches expectations |
| Rollback decision | dev + ops | Within 30 min if smoke test fails |

## Storage retention policy (C13)

- **All `husc_v{year}_legacy` tables are kept for the lifetime of the cluster.** No automatic drop after 30 days.
- Audit JSON snapshots in `rag2025/data/chunked_legacy_{year}/` are also kept indefinitely.
- Rationale: enables audit of "what answer would 2025 chatbot have given for this question?" — critical for trust + compliance.
- Storage cost is bounded — chunks are JSONL + LanceDB columns, typically a few hundred MB per year.

## Common pitfalls

- **Don't switch the embedding model mid-rotation.** Qwen3-Embedding-0.6B is locked by C7; switching breaks vector compatibility with prior years.
- **Don't run rotation while a long-running query is in flight.** The drain timeout is 25s; a 60s query would still get reset by the OS when the old retriever closes. Run rotations during off-peak windows.
- **Don't auto-drop legacy tables.** If you see disk pressure, capacity-plan first; never silently drop history.
- **Don't skip the smoke test.** A bad flip surfaces hallucinations within minutes — `regression_blue.json` is the gate.

## References

- C7: reuse Qwen3-Embedding-0.6B
- C13: lifelong audit retention (no 30-day legacy drop)
- C14: 2027 lifecycle auto-trigger when crawler detects new year
- Plan: `rag2025/.omc/plans/temporal_reingest_plan.md` §Phase 7
- Spec: `docs/rag-chatbot-husc-spec.md` §S13.8
