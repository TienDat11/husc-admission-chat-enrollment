// @spec(plan-warmup-ux-20260705-111031 §3.2)
// useBackendReadiness — poll /health every pollMs, stop when ready=true, pause on tab hidden, resume on visible.
import { useEffect, useRef, useState } from 'react';
import { getReadiness } from '@/lib/api';

export type ReadinessStage = 'warming' | 'ready' | 'failed';

export interface UseBackendReadinessOptions {
  /** Poll interval in ms. Default 2000. */
  pollMs?: number;
  /** Fires once per warming/failed -> ready transition. */
  onReady?: () => void;
  /** Fires every time stage transitions TO 'failed'. */
  onFailed?: (err: string) => void;
}

export interface BackendReadinessState {
  ready: boolean;
  stage: ReadinessStage;
  services: Record<string, boolean>;
  lastChecked: number | null;
  error: string | null;
}

const DEFAULT_POLL_MS = 2000;

export function useBackendReadiness(
  opts?: UseBackendReadinessOptions,
): BackendReadinessState {
  const pollMs = opts?.pollMs ?? DEFAULT_POLL_MS;
  const onReadyRef = useRef(opts?.onReady);
  const onFailedRef = useRef(opts?.onFailed);

  // Keep latest callbacks without retriggering the poll effect on identity change.
  useEffect(() => {
    onReadyRef.current = opts?.onReady;
  }, [opts?.onReady]);
  useEffect(() => {
    onFailedRef.current = opts?.onFailed;
  }, [opts?.onFailed]);

  const [state, setState] = useState<BackendReadinessState>({
    ready: false,
    stage: 'warming',
    services: {},
    lastChecked: null,
    error: null,
  });

  // Refs so the polling effect can read latest values without re-subscribing.
  const readyFiredRef = useRef(false);
  const failedFiredRef = useRef(false);
  const stoppedRef = useRef(false);
  const intervalIdRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;

    const clearTimer = () => {
      if (intervalIdRef.current !== null) {
        clearInterval(intervalIdRef.current);
        intervalIdRef.current = null;
      }
    };

    const poll = async () => {
      // Skip polling while tab is hidden. The visibilitychange handler
      // resumes polling when the tab becomes visible again.
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return;
      }
      try {
        const body = await getReadiness();
        if (cancelled) return;

        const isReady = body.ready === true;
        const stage: ReadinessStage = isReady ? 'ready' : 'warming';
        const services = body.services ?? {};

        setState({
          ready: isReady,
          stage,
          services,
          lastChecked: Date.now(),
          error: null,
        });

        if (isReady) {
          if (!readyFiredRef.current) {
            readyFiredRef.current = true;
            failedFiredRef.current = false;
            stoppedRef.current = true;
            clearTimer();
            onReadyRef.current?.();
          }
        } else {
          failedFiredRef.current = false;
          // Back to warming after a previous ready? Reset the guards so we
          // could re-fire onReady if BE later recovers, but ONLY in the
          // explicit warming->ready transition path above.
          // We do NOT clear stoppedRef here because the contract is single-shot
          // ready: once ready, polling stops and stays stopped.
        }
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Network error';
        setState((prev) => ({
          ...prev,
          ready: false,
          stage: 'failed',
          lastChecked: Date.now(),
          error: message,
        }));
        if (!failedFiredRef.current) {
          failedFiredRef.current = true;
          onFailedRef.current?.(message);
        }
        // Ensure polling continues so we can recover once BE comes back.
        stoppedRef.current = false;
      }
    };

    const start = () => {
      if (stoppedRef.current) return;
      if (intervalIdRef.current !== null) return;
      intervalIdRef.current = setInterval(() => {
        void poll();
      }, pollMs);
    };

    const restart = () => {
      clearTimer();
      if (!stoppedRef.current) start();
    };

    // Kick off the first poll immediately so we don't wait `pollMs` for the
    // initial state. Also arm the recurring interval.
    void poll();
    start();

    const onVisibilityChange = () => {
      if (typeof document === 'undefined') return;
      if (document.visibilityState === 'hidden') {
        clearTimer();
      } else if (document.visibilityState === 'visible') {
        restart();
        // Run one immediate poll on becoming visible so the UI catches up.
        if (!stoppedRef.current) void poll();
      }
    };

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange);
    }

    return () => {
      cancelled = true;
      clearTimer();
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisibilityChange);
      }
    };
  }, [pollMs]);

  return state;
}