import { describe, it, expect, afterEach, vi } from 'vitest';
import { getApiBaseUrl } from '@/lib/api';

describe('getApiBaseUrl', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('treats an explicitly empty VITE_API_BASE_URL as same-origin in production', () => {
    vi.stubEnv('VITE_API_BASE_URL', '');
    vi.stubEnv('PROD', true);
    expect(getApiBaseUrl()).toBe('');
  });

  it('falls back to the public API in production when VITE_API_BASE_URL is unset', () => {
    vi.stubEnv('VITE_API_BASE_URL', undefined);
    vi.stubEnv('PROD', true);
    const url = getApiBaseUrl();
    expect(url).not.toBe('');
    expect(url).toMatch(/^https:\/\//);
  });

  it('falls back to localhost outside production when unset', () => {
    vi.stubEnv('VITE_API_BASE_URL', undefined);
    vi.stubEnv('PROD', false);
    expect(getApiBaseUrl()).toBe('http://localhost:8000');
  });

  it('returns an explicit URL unchanged', () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test');
    expect(getApiBaseUrl()).toBe('https://api.example.test');
  });
});
