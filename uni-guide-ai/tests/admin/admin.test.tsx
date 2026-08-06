import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@/hooks/useTheme';
import App from '@/App';
import { adminConfig, RAMCLOUDS_API_KEY } from '@/constants/adminConfig';

const renderApp = () =>
  render(
    <ThemeProvider>
      <App />
    </ThemeProvider>
  );

describe('Admin shell routes', () => {
  it('renders the Config route with the sidebar and config table', async () => {
    window.history.pushState({}, '', '/admin/config');
    renderApp();

    // Lazy Suspense flushes async — wait for the table to appear.
    const heading = await screen.findByText('Config', { selector: 'h2' });
    expect(heading).toBeInTheDocument();

    // Sidebar nav has 4 items.
    const nav = screen.getByRole('navigation');
    expect(nav).toBeInTheDocument();
    expect(nav.querySelectorAll('a')).toHaveLength(4);

    // ConfigEditor table has 8 rows (header + 7 value rows: 6 adminConfig + RAMCLOUDS_API_KEY).
    const table = document.querySelector('table');
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll('tr')).toHaveLength(8);

    // RAMCLOUDS_API_KEY shows [REDACTED].
    expect(screen.getByText('[REDACTED]')).toBeInTheDocument();
  });

  it('exposes the required config keys with expected values', () => {
    expect(adminConfig).toHaveLength(6);

    const byKey = (k: string) => adminConfig.find((c) => c.key === k);
    expect(byKey('CURRENT_ADMISSION_YEAR')?.value).toBe('2026');
    expect(byKey('LANCEDB_TABLE')?.value).toBe('husc_chunks');
    expect(byKey('EMBEDDING_MODEL')).toBeTruthy();
    expect(byKey('RAMCLOUDS_MODEL')).toBeTruthy();
    expect(byKey('RAMCLOUDS_BASE_URL')).toBeTruthy();
    expect(byKey('RERANKER_MODEL')).toBeTruthy();
    expect(RAMCLOUDS_API_KEY).toBe('[REDACTED]');
  });
});
