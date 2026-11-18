# Runbook: cut over the canonical store from files to PostgreSQL

Scope: U3 (A3). Moves the record store and the audit chain from files to PostgreSQL
behind `STORE_BACKEND=postgres`. Every other legacy store is copied to PostgreSQL but
keeps its file-based runtime adapter until its own unit (see section 7).

Paths below are relative to `rag2025/` unless marked absolute. Variable names and
flags were checked against `rag2025/scripts/migrate_file_stores.py`,
`rag2025/src/app/lifecycle.py`, `rag2025/config/settings.py`, `rag2025/src/main.py`,
`rag2025/src/api/errors.py`, and `rag2025/src/infrastructure/sql_audit_ledger.py`.

---

## 0. Settings used in this runbook

| Name | Where read | Meaning |
| --- | --- | --- |
| `STORE_BACKEND` | `config/settings.py` (`file` default, `postgres`) | Selects the record store and audit ledger. Read at process start only. |
| `WRITE_FREEZE` | `config/settings.py` (bool, default false) | When true at startup, admin writes answer `503 WRITES_FROZEN`. Read at process start only. |
| `DATABASE_URL` | environment | Async SQLAlchemy URL, e.g. `postgresql+asyncpg://rag@localhost:5439/husc`. Required when `STORE_BACKEND=postgres`. |
| `ADMIN_API_TOKEN` | environment | Ops credential, sent as header `X-Admin-Token`. |

Both `STORE_BACKEND` and `WRITE_FREEZE` are read once at startup. Every flip in this
runbook is a process restart, not a live toggle. Run every API instance with the same
values.

---

## 1. Prerequisites

1. PostgreSQL 16 is running and reachable at the host and port in `DATABASE_URL`.
   The repo's `docker-compose.pg.yml` maps 5439 to 5432, with database `husc` and user `rag`.
2. The target database is EMPTY, or at least `audit_events` is empty. Start from a
   fresh database. The migrator's audit precheck (`_write_audit`) raises `MigrationError`
   when `audit_events` holds any row whose `event_id` is not in the file ledger, because
   appending migrated history after live rows would fork the chain.
3. Database schema is at head:
   ```
   cd rag2025
   alembic upgrade head
   ```
   `alembic/env.py` reads `DATABASE_URL` as-is.
4. Databases migrated before commit `ad4565a` must not be used for cutover. Earlier
   versions of the migrator did not persist `request_id` on migrated audit rows, so the
   legacy marker (section 3, step 4) is missing. Drop or recreate the database and migrate again.
5. Set `DATABASE_URL` in the environment that runs the migrator and the API. The migrator's
   real run accepts the asyncpg form and converts it to psycopg (`sync_database_url`).

Windows caveat (environment only, not a code issue): `alembic/env.py` runs
`asyncio.run` with `create_async_engine`. On Windows the default loop is Proactor. With the
documented `postgresql+asyncpg://` URL this is not a problem. If you use an async psycopg
URL instead, psycopg's async driver needs a selector loop, and the repo sets no loop policy.
Do not change code for this; choose the asyncpg URL.

---

## 2. Pre-checks on the real legacy data

Run every check against the legacy root you plan to migrate (default
`rag2025/data/canonical_store`, or pass `--root`). The same root is read by the dry run.

1. **Retraction count must be 0.** `load_retractions` yields `superseded_version = None`
   for every marker, and `_write_retractions` raises `MigrationError` for any `None` version.
   The dry run does NOT catch this; only the real run does.
   ```
   find "<root>/supersession_retractions" -name '*.json' 2>/dev/null | wc -l
   ```
   The result must be `0`. If it is not, stop. The real run cannot pass until the data is
   reviewed; do not edit the files as part of this runbook.

2. **Naive `published_at` values.** The migrator does not reject naive timestamps. `_utc`
   assumes UTC and attaches it. The result reads back as aware, which is a known parity gap
   (section 8). Check the legacy values yourself and record the count before cutover:
   count `published_at` strings in `<root>/records/**/v*.json` that have no `+HH:MM`,
   `-HH:MM`, or `Z` suffix. Non-zero means the parity gap applies to those rows.

3. **Multiple open or published versions per record (Ruling AA).** Legacy data may hold
   several open (`draft`, `review`, `approved`) or several `published` versions for one
   `record_id`. The SQL model allows one open version (`uq_one_open_draft`) and one published
   version per coordinate (`uq_current_coordinate`). The migrator normalises the extra
   versions: older open versions and all but the newest published version are written as
   `archived` / `superseded`. The newest of each group is kept. The count is reported in
   `normalised.records` of the real-run report. Expect that count to equal the number of
   extra versions you found; any mismatch means you need to investigate.
   The original status is kept as `legacy_status` in `structured_payload`, and the affected
   `record_id@version` keys are listed in the migration report.

4. **Published records citing an unknown document.** `source_document_id` values that are
   not in `source_documents` are set to NULL and kept as `legacy_source_document_id`, and
   counted in `dangling_source_document_id`. Exception: a PUBLISHED record whose source is
   not in `source_documents` aborts the real run with `MigrationError`, because the DDL check
   `status <> 'published' OR source_document_id IS NOT NULL` forbids NULL. Fix the registry
   data or the record before cutover, not during it.

5. **Dry run.**
   ```
   cd rag2025
   python scripts/migrate_file_stores.py --dry-run --root <root> --report /tmp/mig-dry.json
   ```
   Exit 0 does NOT prove `mismatch == 0`. The dry-run path does not check mismatch. Open the
   report and confirm `mismatch` is 0 for every entry in `stores`. The dry run never connects
   to a database.

---

## 3. The four-step cutover

### Step 1: freeze the legacy writes

1. Restart the API with `WRITE_FREEZE=true` and `STORE_BACKEND=file`. Startup engages the
   freeze (`main.py`, `WRITE_FREEZE.engage("operator", ...)`).
2. Admin write requests (POST/PUT/PATCH/DELETE under `/admin/`, except login and logout)
   now answer `503` with code `WRITES_FROZEN` and carry the reason, before routing
   (`src/api/errors.py`). Public chat is not blocked by the freeze.
3. Limitation: the freeze covers admin writes only. Public chat still writes to the
   file-backed stores that stay file-based after cutover (unanswered queue, feedback). Rows
   written after the final migrator run are not in PostgreSQL (section 7).

### Step 2: final migration (idempotent) and reconcile

1. Dry run first, then the real run. The real run requires `--confirm-frozen`; the migrator
   cannot check the freeze itself, so this flag is your attestation that step 1 is done.
   ```
   cd rag2025
   python scripts/migrate_file_stores.py --dry-run --root <root> --report /tmp/mig-final-dry.json
   python scripts/migrate_file_stores.py --root <root> --confirm-frozen --database-url "$DATABASE_URL" --report /tmp/mig-final.json
   ```
   The real run prefers the `DATABASE_URL` environment variable. `--database-url` puts the
   password in argv, which other users can see. Use the env var.
2. Exit codes: `0` on success. `2` on: a missing `--confirm-frozen` or `DATABASE_URL`,
   `MigrationError`, a PostgreSQL error, a report-write failure, or a post-commit mismatch.
   On `2` for a post-commit mismatch, the report is written and the message lists the stores.
3. The real run is one transaction across all stores. A failure rolls everything back.
   Inserts use `ON CONFLICT DO NOTHING`, so re-running is safe and existing PostgreSQL rows
   are never updated.
4. Acceptance: `mismatch == 0` for every store in `stores` of the real-run report. Exit `2`
   with an unexplained mismatch means stop, do not switch. `defaulted`, `skipped`,
   `dangling_source_document_id`, `normalised`, and `quarantined_rows` are reported counts
   that must be explained, not treated as errors.

Store names in the report: `records`, `retractions`, `releases`, `outbox`, `conflicts`,
`idempotency`, `unanswered`, `feedback`, `source_documents`, `identities`, `audit`. The
loader list has 11 entries. The plan text says 12 stores; this runbook follows the code.

### Step 3: switch to PostgreSQL and unfreeze

1. Restart the API with `STORE_BACKEND=postgres`, `WRITE_FREEZE` unset or false, and
   `DATABASE_URL` set. Record `:cutover_at` from the database clock right before the restart:
   ```
   SELECT now();
   ```
   Use the database clock, not the application clock.
2. With `STORE_BACKEND=postgres` the API uses `SqlRecordStore` and `SqlAuditLedger`.
   Startup logs a WARNING for each stale file store still under the root (`records`,
   `audit/ledger.jsonl`), because nothing under postgres writes to them any more.

### Step 4: verify

1. Start the API on postgres, then:
   ```
   curl -s -H "X-Admin-Token: $ADMIN_API_TOKEN" localhost:8000/admin/audit/verify
   ```
   Expect `"ok": true`. The response also contains `legacy_checked`, `legacy_last_hash`,
   and `skipped_legacy_rows`.
2. Compare with the real-run report:
   - `legacy_checked` should equal `stores[name=audit].source_rows` in the report. These are
     the migrated legacy rows, identified by `request_id = 'legacy:file-ledger'`.
   - `legacy_last_hash` should equal the `entry_hash` of the last line of
     `<root>/audit/ledger.jsonl`. The report does not carry this value. Read it from the
     file yourself.
   - `skipped_legacy_rows` counts migrated rows whose action or target kind is not a valid
     enum value. It must be explained, not assumed zero.
3. Legacy marker, accepted risk (Ruling Y). The `legacy:file-ledger` request id is not part
   of any hash and can be changed with an UPDATE. An actor who can change the table can
   therefore relabel rows. The verify check accepts a leading run of marked rows on sequence
   and `prev_hash` linkage only. The migration report is the anchor for pre-cutover history.
   Any actor able to disable the append-only trigger could already recompute the SQL chain,
   so the marker adds no new capability for that actor. This is an accepted limitation, not a
   bug to fix in this cutover.

---

## 4. Rollback

Rollback to `STORE_BACKEND=file` is allowed ONLY before the first write to PostgreSQL.

1. Check for any PostgreSQL write since `:cutover_at`:
   ```
   SELECT count(*) FROM audit_events WHERE created_at > :cutover_at;
   ```
   It must return `0`. This check covers audit rows only. A write that appends no audit row is
   not visible to it, so also confirm no admin write went through the postgres backend since
   the cutover.
2. If it returns `0`, set `STORE_BACKEND=file`, restart, and the file stores are the source of
   truth again. Writes to PostgreSQL after `:cutover_at` are not in the file store.
3. If it returns anything other than `0`, there is no rollback. Recovery goes through
   `pg_restore` (section 6), not through the file stores.

---

## 5. `--scrub-identities` (separate, late step)

`--scrub-identities` replaces each plaintext credential in `identity/identities.json` with
`scrubbed:<sha256 hex>`. It runs only after a real run, and only if every identity is in
PostgreSQL (`identities.target_rows == source_rows`).

- Run it only AFTER the rollback window has closed (section 4). Until then the plaintext
  file is the rollback path.
- Mechanics: it is a real run with `--confirm-frozen` and `--scrub-identities`. The scrub
  runs after the commit. If the scrub fails, the message reads "data committed; ... re-run is
  safe".
- Do not rebuild a new PostgreSQL database from a scrubbed file. Identity rows are hashed from
  the credential as read; a scrubbed file would hash the `scrubbed:` string instead. Recovery
  of an existing database is `pg_restore`, not re-migration.
- Known issue: the report is written after the scrub, so if the scrub fails the report is not
  written. Keep the run output.

---

## 6. pg_dump / pg_restore rehearsal

Use the custom format and restore into a scratch database. Do not restore over the live
database in a rehearsal.

```
# backup (custom format)
pg_dump -h localhost -p 5439 -U rag -Fc -f /tmp/husc.dump husc

# scratch restore target
createdb -h localhost -p 5439 -U rag husc_rehearsal
pg_restore -h localhost -p 5439 -U rag -d husc_rehearsal --no-owner /tmp/husc.dump

# row counts, source and restored, must match
psql -h localhost -p 5439 -U rag -d husc -c "SELECT 'audit_events', count(*) FROM audit_events"
psql -h localhost -p 5439 -U rag -d husc_rehearsal -c "SELECT 'audit_events', count(*) FROM audit_events"

# verify chain on the restored copy: start the API with
#   DATABASE_URL=postgresql+asyncpg://rag@localhost:5439/husc_rehearsal STORE_BACKEND=postgres
# then curl /admin/audit/verify as in section 3, step 4
```

The plan's compose form is `docker compose -f rag2025/docker-compose.pg.yml exec -T pg sh -c 'pg_dump -U rag -Fc husc > /tmp/h.dump && pg_restore -U rag --clean -d husc /tmp/h.dump'`.
That form uses `--clean` on the live database. Use it only in an incident with a fresh
dump taken first.

---

## 7. Post-cutover divergence (Ruling W)

Only two stores are SQL-backed in production after cutover: the record store
(`SqlRecordStore`) and the audit chain (`SqlAuditLedger`).

Stay file-backed until their own units: outbox, releases, idempotency, conflicts, the
document registry, unanswered and feedback, identities. Their PostgreSQL copies are snapshots
from the migration and are not kept in sync by the app.

File stores that become read-only and stale: `records/` and `audit/ledger.jsonl` under the root.
Startup logs a WARNING for each one still present.

Limitation: the source-document registry stays file-backed, but `SqlRecordStore` requires a
cited `source_document_id` to exist in the SQL `source_documents` table. Documents migrated by
the migrator are fine. A document registered AFTER cutover is not in SQL, so creating or
updating a record that cites it returns `409` with "source document is not registered in the
SQL store", until the documents unit lands. Mitigation: register all documents you need
before cutover and run the migrator, or hold off registering new documents.

---

## 8. Known limitations (parked minors)

- A deadlock (`TransactionRollbackError`) is reported as "database unavailable" by the record
  and audit adapters, which map errors generically.
- Supersession checks run one retraction query per candidate (N+1 pattern in
  `_supersession_index` / `_retracted_keys`).
- `SqlAuditLedger.get()` and `find()` load the full table.
- Case-insensitive filesystems: two legacy ids that differ only by case may collide on
  Windows. Not covered by a test; not traced to code in this runbook.
- `admission_round` None is written as `all` and is not invertible back to None. `authored_by`
  None is written as the `legacy-migration` sentinel, which is not invertible either.
- Naive `published_at` reads back aware (section 2, check 2).
- Pre-check psycopg errors and some mid-transaction errors are not mapped to typed errors.

---

## 9. Windows environment note

Repeated from section 1, for convenience: with an async psycopg URL on Windows, the default
Proactor loop breaks `alembic/env.py`. Use the documented asyncpg URL. This is an environment
caveat only; no code changes are made by this runbook.
