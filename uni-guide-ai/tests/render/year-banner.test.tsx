// @spec(S13.7) YearBanner — loading, canonical default URL, success
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { YearBanner } from '@/components/chat/YearBanner';

vi.mock('@/lib/api', () => ({
  getApiBaseUrl: vi.fn(() => 'http://localhost:8000'),
}));

const meta = {
  current_admission_year: 2026,
  latest_crawl_date: '2026-06-10T00:00:00Z',
  total_notifications: 3,
  freshness_lag_days: null,
  freshness_alert: false,
};

beforeEach(() => {
  global.fetch = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(meta), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }),
  );
});

afterEach(() => vi.restoreAllMocks());

describe('YearBanner', () => {
  it('uses canonical API URL and renders loading then success', async () => {
    render(<YearBanner />);

    expect(screen.getByTestId('year-banner-loading')).toBeInTheDocument();
    expect(await screen.findByTestId('year-banner')).toHaveTextContent('Tư vấn tuyển sinh năm 2026');
    expect(global.fetch).toHaveBeenCalledWith('http://localhost:8000/api/meta');
  });

  it('keeps apiUrl override', async () => {
    render(<YearBanner apiUrl="/custom/meta" />);

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith('/custom/meta'));
  });
});
