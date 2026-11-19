# Release and rollback runbook

## Rotating `ADMIN_API_TOKEN`

`ADMIN_API_TOKEN` is the operator header credential (`X-Admin-Token`). The API reads it at process start, so a rotation takes effect on restart.

1. Generate a new token with at least 256 bits of entropy:
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`
2. Store the new value in the deployment secret store as `ADMIN_API_TOKEN`. Keep the previous value at hand for rollback.
3. Restart the API process so it reads the new value.
4. Verify the new token is accepted and the old one is refused:
   - `GET /admin/identity` with `X-Admin-Token: <new>` returns 200.
   - The same request with the old token returns 401 or 403.
5. Record the rotation time and operator in the release notes. Do not paste either token into tickets or chat.

Rollback: restore the previous `ADMIN_API_TOKEN` value in the secret store, restart the API, and repeat step 4 with the old token.

Per-account header tokens (`admin_identities.header_token_hash`) are not rotated by this procedure. They are issued once at account creation (`POST /admin/identities` with `issue_header_token: true`). Replacing one means disabling the account and creating a new one.

## Route parity: `/admin/*` to `/quan-ly/*`

The teacher console replaced the `/admin/*` screens. Use the right-hand paths in bookmarks, runbooks and rehearsal scripts.

| Old (`/admin/*`) | New (`/quan-ly/*`) |
|---|---|
| `/admin/config` | `/quan-ly` |
| `/admin/records` | `/quan-ly/kho` |
| `/admin/records/new` | `/quan-ly/them` |
| `/admin/records/:recordId/edit` | `/quan-ly/them` |
| `/admin/records/:recordId/review` | `/quan-ly/cho-duyet` |
| `/admin/releases` | `/quan-ly/lich-su` |

Table exceptions for an uploaded source document are handled at `/quan-ly/them/ngoai-le` (TableFix). It is linked from the upload result card in the Thêm thông tin flow.

## Golden Gate and release rehearsal

Run the golden gate for a candidate release before activating it. The gate stores its verdict in `golden_report` on the release row. Only a release whose `golden_report.verdict` is `pass` and whose status is `ready` can be activated; a failed gate returns `409 RELEASE_GATE_FAILED`.

1. Gate the candidate (from `rag2025/`):

        DATABASE_URL="postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc" python scripts/run_golden_gate.py --release <release_id> --baseline eval/baseline_A1.json

2. Activate it (needs the `RECORDS_PUBLISH` grant). The previously active release is marked `superseded` in the same transaction:

        curl -sS -X POST "$API/admin/releases/<release_id>/activate" -H "X-Admin-Token: $ADMIN_API_TOKEN"

3. Roll back to a previous release (needs the `RECORDS_ROLLBACK` grant; `reason` is required):

        curl -sS -X POST "$API/admin/releases/<release_id>/rollback" -H "X-Admin-Token: $ADMIN_API_TOKEN" -H "Content-Type: application/json" -d '{"reason": "rehearsal: rollback to <release_id> after golden gate regression"}'

Note: `POST /admin/rollback` (no release id) is the two-level swap that moves the canonical pointer and the serving runtime together. It is still mounted; see the release notes before retiring it.
