// @spec(plan-warmup-ux-20260705-111031 §3.2)
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useBackendReadiness } from '@/hooks/useBackendReadiness';

vi.mock('@/lib/api', () => ({ getReadiness: vi.fn() }));

import { getReadiness } from '@/lib/api';
const mockGetReadiness = vi.mocked(getReadiness);

function setVisibility(value: 'visible' | 'hidden'): void {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => value,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

beforeEach(() => {
  vi.useFakeTimers();
  mockGetReadiness.mockReset();
  setVisibility('visible');
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('useBackendReadiness — polling lifecycle', () => {
  it('delegates polling to canonical readiness API and stops once ready', async () => {
    mockGetReadiness
      .mockResolvedValueOnce({ ready: false, stage: 'warming', services: { embedding: false } })
      .mockResolvedValueOnce({ ready: false, stage: 'warming', services: { embedding: false } })
      .mockResolvedValueOnce({ ready: false, stage: 'warming', services: { embedding: true } })
      .mockResolvedValueOnce({ ready: true, stage: 'ready', services: { embedding: true } });

    const { result } = renderHook(() => useBackendReadiness({ pollMs: 1000 }));
    await act(async () => vi.advanceTimersByTimeAsync(3000));

    expect(mockGetReadiness).toHaveBeenCalledTimes(4);
    expect(result.current.ready).toBe(true);
    expect(result.current.stage).toBe('ready');

    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(mockGetReadiness).toHaveBeenCalledTimes(4);
  });

  it('pauses while hidden and resumes immediately when visible', async () => {
    mockGetReadiness.mockResolvedValue({ ready: false, stage: 'warming', services: {} });
    renderHook(() => useBackendReadiness({ pollMs: 1000 }));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    const initialCalls = mockGetReadiness.mock.calls.length;

    await act(async () => {
      setVisibility('hidden');
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(mockGetReadiness).toHaveBeenCalledTimes(initialCalls);

    await act(async () => {
      setVisibility('visible');
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(mockGetReadiness.mock.calls.length).toBeGreaterThan(initialCalls);
  });

  it('transitions to failed on readiness rejection', async () => {
    mockGetReadiness.mockRejectedValue(new TypeError('Failed to fetch'));
    const { result } = renderHook(() => useBackendReadiness({ pollMs: 1000 }));

    await act(async () => vi.advanceTimersByTimeAsync(0));

    expect(result.current).toMatchObject({ ready: false, stage: 'failed', error: 'Failed to fetch' });
  });

  it('fires onFailed again after a successful warming transition', async () => {
    mockGetReadiness
      .mockRejectedValueOnce(new Error('first failure'))
      .mockResolvedValueOnce({ ready: false, stage: 'warming', services: {} })
      .mockRejectedValueOnce(new Error('second failure'));
    const onFailed = vi.fn();
    renderHook(() => useBackendReadiness({ pollMs: 1000, onFailed }));

    await act(async () => vi.advanceTimersByTimeAsync(2000));

    expect(onFailed).toHaveBeenCalledTimes(2);
    expect(onFailed).toHaveBeenNthCalledWith(1, 'first failure');
    expect(onFailed).toHaveBeenNthCalledWith(2, 'second failure');
  });

  it('fires onReady once and clears polling', async () => {
    mockGetReadiness
      .mockResolvedValueOnce({ ready: false, stage: 'warming', services: {} })
      .mockResolvedValueOnce({ ready: true, stage: 'ready', services: {} });
    const onReady = vi.fn();
    renderHook(() => useBackendReadiness({ pollMs: 1000, onReady }));

    await act(async () => vi.advanceTimersByTimeAsync(1000));
    await act(async () => vi.advanceTimersByTimeAsync(5000));

    expect(onReady).toHaveBeenCalledTimes(1);
    expect(mockGetReadiness).toHaveBeenCalledTimes(2);
  });

  it('stops polling after unmount', async () => {
    mockGetReadiness.mockResolvedValue({ ready: false, stage: 'warming', services: {} });
    const { unmount } = renderHook(() => useBackendReadiness({ pollMs: 1000 }));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    const callsAtUnmount = mockGetReadiness.mock.calls.length;

    unmount();
    await act(async () => vi.advanceTimersByTimeAsync(5000));

    expect(mockGetReadiness).toHaveBeenCalledTimes(callsAtUnmount);
  });
});
