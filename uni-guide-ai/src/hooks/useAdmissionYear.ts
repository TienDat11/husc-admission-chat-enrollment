// @spec(ticket-17) Criterion 2: the effective admission year used by the
// answer surface (stale detection, year labels) comes from the metadata API
// `/api/meta` — never a hardcoded literal. Mirrors the fetch pattern of
// YearBanner.tsx (same endpoint, same shape); the two components fetch
// independently and the browser dedupes per page load.
import { useEffect, useState } from 'react';
import { getApiBaseUrl } from '@/lib/api';
import type { MetaResponse } from '@/components/chat/YearBanner';

export interface AdmissionYearState {
  /** `current_admission_year` from /api/meta; null until loaded or on error. */
  currentAdmissionYear: number | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetch the live admission year once per mount.
 *
 * `apiUrl` is overridable for tests (same convention as YearBanner).
 * A failed fetch leaves `currentAdmissionYear` null, which makes every
 * stale predicate fail open — the surface degrades to "no staleness badge",
 * never to a fabricated year.
 */
export function useAdmissionYear(
  apiUrl: string = `${getApiBaseUrl()}/api/meta`,
): AdmissionYearState {
  const [state, setState] = useState<AdmissionYearState>({
    currentAdmissionYear: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(apiUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const meta = (await res.json()) as MetaResponse;
        if (!cancelled) {
          setState({
            currentAdmissionYear:
              typeof meta.current_admission_year === 'number'
                ? meta.current_admission_year
                : null,
            loading: false,
            error: null,
          });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            currentAdmissionYear: null,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiUrl]);

  return state;
}
