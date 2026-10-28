import { describe, it, expect } from 'vitest';
import { AdminApiError, isMissingCredential } from '@/lib/adminApi';

// Pins which failures mean "show the sign-in form" and which stay real errors.
// A bare status check on 403 would turn a permission denial or a disabled
// account into a login prompt, so 403 is matched on the backend's message.
describe('isMissingCredential', () => {
  it('treats 401 as missing credential', () => {
    expect(isMissingCredential(new AdminApiError(401, 'not signed in'))).toBe(true);
  });

  it('treats the 422 no-credential response as missing credential', () => {
    expect(
      isMissingCredential(
        new AdminApiError(422, 'no credential: X-Admin-Token header or session cookie required'),
      ),
    ).toBe(true);
  });

  it('treats an expired session 403 as missing credential', () => {
    expect(isMissingCredential(new AdminApiError(403, 'invalid or expired session'))).toBe(true);
  });

  it('treats a session that no longer maps to an identity as missing credential', () => {
    expect(
      isMissingCredential(new AdminApiError(403, 'session no longer maps to a stored identity')),
    ).toBe(true);
  });

  it('does not treat a permission 403 as missing credential', () => {
    expect(
      isMissingCredential(
        new AdminApiError(403, "role 'viewer' is not permitted to perform 'records.approve'"),
      ),
    ).toBe(false);
  });

  it('does not treat a disabled account as missing credential', () => {
    expect(isMissingCredential(new AdminApiError(403, 'admin identity is disabled'))).toBe(false);
  });
});
