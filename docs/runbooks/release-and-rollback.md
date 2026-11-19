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
