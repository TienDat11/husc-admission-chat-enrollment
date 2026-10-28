/**
 * Ticket 17 criterion 2: the effective admission year comes from the
 * metadata API (/api/meta), never a hardcoded literal.
 *
 * The anti-hardcode proof is behavioral: the mocked API answers with a year
 * that appears NOWHERE in src/ (2029). A hardcoded year could not produce
 * 2029; the hook must echo whatever the API says.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useAdmissionYear } from '@/hooks/useAdmissionYear';

const META_URL = 'http://meta.test/api/meta';

function mockMeta(body: unknown, ok = true, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useAdmissionYear — year comes from /api/meta', () => {
  it('returns the current_admission_year the API reports', async () => {
    const fetchMock = mockMeta({
      current_admission_year: 2029,
      latest_crawl_date: null,
      total_notifications: null,
      freshness_lag_days: null,
      freshness_alert: false,
    });
    const { result } = renderHook(() => useAdmissionYear(META_URL));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.currentAdmissionYear).toBe(2029);
    expect(fetchMock).toHaveBeenCalledWith(META_URL);
  });

  it('follows the API across a season rollover (no cached literal)', async () => {
    // Same hook, second mount, different answer: the UI year is data.
    mockMeta({ current_admission_year: 2031 });
    const { result } = renderHook(() => useAdmissionYear(META_URL));
    await waitFor(() => expect(result.current.currentAdmissionYear).toBe(2031));
  });

  it('fail-open: HTTP error leaves the year null and reports the error', async () => {
    mockMeta({}, false, 503);
    const { result } = renderHook(() => useAdmissionYear(META_URL));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.currentAdmissionYear).toBeNull();
    expect(result.current.error).toContain('503');
  });

  it('fail-open: malformed payload (year missing) yields null, not a default', async () => {
    mockMeta({ latest_crawl_date: null });
    const { result } = renderHook(() => useAdmissionYear(META_URL));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.currentAdmissionYear).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('fail-open: network rejection yields null with the error surfaced', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Failed to fetch')),
    );
    const { result } = renderHook(() => useAdmissionYear(META_URL));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.currentAdmissionYear).toBeNull();
    expect(result.current.error).toBe('Failed to fetch');
  });
});
