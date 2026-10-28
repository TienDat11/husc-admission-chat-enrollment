import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@/hooks/useTheme';
import App from '@/App';
import { adminConfig, NS24_API_KEY } from '@/constants/adminConfig';

const VIEWER = {
  actor_id: 'cb.chi-xem',
  role: 'viewer',
  actions: ['records.read'],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * Stubs fetch and mounts the real App at `path`.
 *
 * `signedIn` decides what /admin/identity answers: the identity when true, a
 * 401 when false (the anonymous path the sign-in form depends on). Every other
 * request gets an empty success so lazy screens can settle without a backend.
 *
 * The URL is set with pushState before render, so `window.location.pathname`
 * is the path under test on every run, not whatever a previous test left.
 */
function renderAppAt(path: string, signedIn: boolean) {
  window.history.pushState({}, '', path);
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/admin/identity')) {
        return signedIn ? json(VIEWER) : json({ detail: 'not signed in' }, 401);
      }
      return json({ records: [], total: 0 });
    }),
  );
  return render(
    <ThemeProvider>
      <App />
    </ThemeProvider>,
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Admin shell routes', () => {
  it('renders the Config route with the sidebar and config table when signed in', async () => {
    renderAppAt('/admin/config', true);

    const heading = await screen.findByText('Config', { selector: 'h2' });
    expect(heading).toBeInTheDocument();

    // Sidebar nav has 6 items (Dữ liệu + Release + Config/Ingest/Retrieval/Debt).
    const nav = screen.getByRole('navigation');
    expect(nav.querySelectorAll('a')).toHaveLength(6);

    // Header + 8 adminConfig rows + NS24_API_KEY row.
    const table = document.querySelector('table');
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll('tr')).toHaveLength(10);

    // The key row shows the redacted placeholder, and never a real value.
    expect(screen.getByText('[REDACTED]')).toBeInTheDocument();
  });

  it('exposes exactly the ns24 config keys, with no RAM Clouds rows', () => {
    const keys = adminConfig.map((c) => c.key);

    for (const required of [
      'NS24_MODEL',
      'NS24_ROUTER_MODEL',
      'NS24_GEN_MODEL',
      'NS24_BASE_URL',
      'EMBEDDING_MODEL',
      'RERANKER_MODEL',
      'LANCEDB_TABLE',
      'CURRENT_ADMISSION_YEAR',
    ]) {
      expect(keys).toContain(required);
    }
    expect(keys).not.toContain('RAMCLOUDS_MODEL');
    expect(keys).not.toContain('RAMCLOUDS_BASE_URL');
    expect(keys).toHaveLength(8);

    const byKey = (k: string) => adminConfig.find((c) => c.key === k);
    expect(byKey('CURRENT_ADMISSION_YEAR')?.value).toBe('2026');
    expect(byKey('LANCEDB_TABLE')?.value).toBe('husc_chunks');
    expect(byKey('NS24_MODEL')?.value).toBe('gemini-3.5-flash-lite');
    expect(byKey('NS24_GEN_MODEL')?.value).toBe('gpt-6-luna');
    expect(NS24_API_KEY).toBe('[REDACTED]');
  });

  it('shows the sign-in form, not the config table, when no session exists', async () => {
    renderAppAt('/admin/config', false);

    // The gate answers the anonymous probe with its credential form.
    const password = await screen.findByLabelText(/mật khẩu/i, {}, { timeout: 5000 });
    expect(password).toBeInTheDocument();
    expect(password).toHaveAttribute('type', 'password');

    // The config table must not render behind the gate.
    expect(document.querySelector('table')).toBeNull();
    expect(screen.queryByText('[REDACTED]')).toBeNull();
  });
});

// The records screens are reached only through App's route tree. Each
// iteration sets its own URL with pushState, so the assertion checks the
// route under test and not whatever URL an earlier test left behind.
describe('Admissions route wiring', () => {
  it('routes /admin/records and /admin/releases through the credential gate', async () => {
    for (const path of ['/admin/records', '/admin/releases']) {
      const view = renderAppAt(path, false);

      expect(await screen.findByLabelText(/mật khẩu/i, {}, { timeout: 5000 })).toBeInTheDocument();
      expect(window.location.pathname).toBe(path);

      view.unmount();
      vi.unstubAllGlobals();
    }
  });

  it('keeps the create and edit record routes inside the gate', async () => {
    for (const path of ['/admin/records/new', '/admin/records/CHI_TIEU:nganh:7480201:2026:ttts/edit']) {
      const view = renderAppAt(path, false);

      expect(await screen.findByLabelText(/mật khẩu/i, {}, { timeout: 5000 })).toBeInTheDocument();
      expect(window.location.pathname).toBe(path);

      view.unmount();
      vi.unstubAllGlobals();
    }
  });
});
