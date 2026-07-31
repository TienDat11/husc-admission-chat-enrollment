// @spec(plan-warmup-ux-20260705-111031 §3.2)
// api.ts — RAG API client + retry wrapper + error classification.
import type { GraphStats, RouterInfo, SourceChip } from './api-types';

// ─────────────────────────────────────────────────────────────────────────
// QueryError — typed error surface for the FE resilience layer
//
// Plan §3.2 / §4. Mirrors the BE 503 contract: warming is retryable, 5xx
// is not (per Risk 3 — don't mask server bugs). The FE renders a
// Vietnamese message per kind (MessageBubble.tsx). Callers (e.g. retry
// loop) read `retryable` + `retryAfterMs` to decide whether to back off.
// ─────────────────────────────────────────────────────────────────────────

export type QueryErrorKind =
  | 'network'
  | 'warming'
  | 'unauthorized'
  | 'server'
  | 'unknown';

export interface QueryError {
  kind: QueryErrorKind;
  retryable: boolean;
  retryAfterMs?: number;
  message: string;
}

const DEFAULT_WARMING_RETRY_MS = 2000;
const RETRY_BASE_MS = 1000;
const RETRY_MAX_MS = 30_000;
const MAX_ATTEMPTS = 2; // 1 initial + 1 retry (N3: bounded retry)

/**
 * Classify an unknown error (typically thrown by fetch or a wrapped
 * `throw new Error('Stream unavailable: …')` from queryStream) into a
 * QueryError the FE can branch on.
 *
 * Rules (plan §3.2):
 *   - TypeError('Failed to fetch') / AbortError → network, retryable
 *   - HTTP 503 with Retry-After → warming, retryable, retryAfterMs
 *   - HTTP 401/403 → unauthorized, NOT retryable
 *   - HTTP 5xx (not 503) → server, NOT retryable (Risk 3)
 *   - other → unknown, NOT retryable
 *
 * The function inspects:
 *   - err.name === 'AbortError' / DOMException with that name
 *   - err.message for the TypeError('Failed to fetch') shape
 *   - an optional `.status` field (assigned by api.ts before throwing)
 *   - an optional `.headers` field (Response.headers from api.ts)
 */
export function classifyFetchError(err: unknown): QueryError {
  // 1. AbortError — fetch was cancelled. Treated as network/retryable so
  //    that the caller's own retry loop (e.g. queryStreamWithRetry) does
  //    not try to retry; a user-initiated abort is a separate case and
  //    should be handled by the caller before reaching classify.
  if (err instanceof Error) {
    if (err.name === 'AbortError') {
      return {
        kind: 'network',
        retryable: true,
        message: err.message || 'Request aborted',
      };
    }
    // TypeError('Failed to fetch') — the canonical browser signal for
    // network-level failures (offline, CORS, DNS, connection refused).
    if (err.name === 'TypeError' && /Failed to fetch/i.test(err.message)) {
      return {
        kind: 'network',
        retryable: true,
        message: 'Không kết nối được máy chủ',
      };
    }
  }
  if (
    typeof DOMException !== 'undefined' &&
    err instanceof DOMException &&
    err.name === 'AbortError'
  ) {
    return { kind: 'network', retryable: true, message: err.message || 'Request aborted' };
  }

  // 2. Tagged QueryError — pass through (re-throws preserve classification).
  if (isQueryError(err)) {
    return err;
  }

  // 3. Structured status — we tag fetch failures in sendChatMessage/queryStream
  //    with `.status` (number) and `.headers` (Headers | Record) before
  //    throwing, so the classifier can read them.
  const status = (err as { status?: unknown } | null)?.status;
  if (typeof status === 'number') {
    const headers =
      (err as { headers?: Headers | Record<string, string> | null }).headers ?? undefined;
    if (status === 503) {
      return {
        kind: 'warming',
        retryable: true,
        retryAfterMs: parseRetryAfter(headers) ?? DEFAULT_WARMING_RETRY_MS,
        message: 'Máy chủ đang khởi động',
      };
    }
    if (status === 401 || status === 403) {
      return { kind: 'unauthorized', retryable: false, message: `HTTP ${status}` };
    }
    if (status >= 500) {
      return { kind: 'server', retryable: false, message: `HTTP ${status}` };
    }
  }

  // 4. Fallback — opaque error, do not retry.
  const message = err instanceof Error ? err.message : 'Unknown error';
  return { kind: 'unknown', retryable: false, message };
}

function isQueryError(value: unknown): value is QueryError {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.kind === 'string' &&
    typeof v.retryable === 'boolean' &&
    typeof v.message === 'string'
  );
}

function parseRetryAfter(headers?: Headers | Record<string, string>): number | undefined {
  if (!headers) return undefined;
  let raw: string | null = null;
  if (typeof Headers !== 'undefined' && headers instanceof Headers) {
    raw = headers.get('Retry-After');
  } else if (typeof headers === 'object') {
    const v = (headers as Record<string, string>)['Retry-After'];
    raw = v ?? null;
  }
  if (!raw) return undefined;
  const seconds = parseInt(raw, 10);
  if (Number.isFinite(seconds) && seconds > 0) return seconds * 1000;
  return undefined;
}

/**
 * Sleep that aborts when the AbortSignal fires. Returns true if the sleep
 * completed normally; false if it was aborted (caller should stop retrying).
 */
function abortableSleep(ms: number, signal?: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(false);
      return;
    }
    const timer = setTimeout(() => {
      cleanup();
      resolve(true);
    }, ms);
    const onAbort = () => {
      cleanup();
      resolve(false);
    };
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener?.('abort', onAbort);
    };
    signal?.addEventListener?.('abort', onAbort);
  });
}

/**
 * Inner core of queryStream: a single attempt that returns the assembled
 * answer. Throws on transport error (caller may classify + retry).
 */
async function queryStreamOnce(
  query: string,
  handlers: StreamHandlers,
  history: ChatTurn[] | undefined,
  signal: AbortSignal | undefined,
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/v2/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      query,
      ...(history && history.length > 0 ? { history } : {}),
    }),
    signal,
  });
  if (!response.ok || !response.body) {
    // Tag the error with .status + .headers so classifyFetchError can
    // recover the HTTP surface (503 warming vs 5xx server vs 401).
    const tagged = new Error(
      `Stream unavailable: ${response.status} ${response.statusText}`,
    ) as Error & { status: number; headers: Headers };
    tagged.status = response.status;
    tagged.headers = response.headers;
    throw tagged;
  }
  const reader = response.body.getReader();
  let fullAnswer = '';
  for await (const frame of parseSseStream(reader)) {
    if (frame.event === 'meta') {
      try {
        const meta = JSON.parse(frame.data) as StreamMeta;
        handlers.onMeta?.(meta);
      } catch (e) {
        // malformed meta JSON — ignore, continue streaming
      }
    } else if (frame.event === 'stage') {
      try {
        const firstLine = frame.data.split('\n')[0];
        const payload = JSON.parse(firstLine) as { stage?: string };
        if (typeof payload.stage === 'string' && payload.stage.length > 0) {
          handlers.onStage?.(payload.stage);
        }
      } catch (e) {
        // malformed stage JSON — ignore, continue streaming
      }
    } else if (frame.event === 'delta') {
      try {
        const firstLine = frame.data.split('\n')[0];
        const payload = JSON.parse(firstLine) as { text?: string };
        const text = payload.text ?? '';
        fullAnswer += text;
        handlers.onDelta?.(text);
      } catch (e) {
        // malformed delta JSON — skip this chunk
      }
    } else if (frame.event === 'done') {
      try {
        const firstLine = frame.data.split('\n')[0];
        const payload = JSON.parse(firstLine) as { answer?: string };
        if (typeof payload.answer === 'string') fullAnswer = payload.answer;
      } catch (e) {
        // malformed done JSON — fall back to the concatenated deltas
      }
      handlers.onDone?.(fullAnswer);
      return fullAnswer;
    }
  }
  // Stream ended without a `done` frame — still surface what we have.
  handlers.onDone?.(fullAnswer);
  return fullAnswer;
}

/**
 * Bounded retry wrapper for `queryStream` (plan §3.2 G4).
 *  - max 2 attempts (1 initial + 1 retry)
 *  - exponential backoff: 1000ms → 2000ms (cap 30s)
 *  - retry only on `network` and `warming` (per Risk 3)
 *  - on retryable=false OR final attempt: throw classified QueryError
 *  - on abort: stop immediately, do not retry
 */
export async function queryStream(
  query: string,
  handlers: StreamHandlers,
  history?: ChatTurn[],
  signal?: AbortSignal,
): Promise<string> {
  let attempt = 0;
  let backoffMs = RETRY_BASE_MS;
  let lastError: unknown = null;
  while (attempt < MAX_ATTEMPTS) {
    if (signal?.aborted) {
      throw classifyFetchError(new DOMException('aborted', 'AbortError'));
    }
    try {
      return await queryStreamOnce(query, handlers, history, signal);
    } catch (err) {
      lastError = err;
      const classified = classifyFetchError(err);
      if (!classified.retryable) {
        throw classified;
      }
      if (attempt + 1 >= MAX_ATTEMPTS) {
        // Bounded retry exhausted. Surface a final, non-retryable copy so
        // callers (e.g. useChatStream) read a stable shape.
        throw {
          ...classified,
          retryable: false,
          message: `${classified.message} (đã thử lại ${attempt} lần)`,
        } satisfies QueryError;
      }
      // Use Retry-After from BE if present (warming), else our schedule.
      const sleepMs = classified.retryAfterMs ?? backoffMs;
      const slept = await abortableSleep(sleepMs, signal);
      if (!slept) {
        // Aborted during backoff — surface network/aborted.
        throw classifyFetchError(new DOMException('aborted', 'AbortError'));
      }
      backoffMs = Math.min(backoffMs * 2, RETRY_MAX_MS);
      attempt += 1;
    }
  }
  // Unreachable — the loop always returns or throws — but TS requires it.
  throw classifyFetchError(lastError);
}

/**
 * Bounded retry wrapper for `sendChatMessage` (plan §3.2 G4). Returns the
 * `QueryResponse` on success or throws a classified `QueryError` on the
 * final failure.
 */
export async function sendChatMessageWithRetry(
  query: string,
  history?: ChatTurn[],
  signal?: AbortSignal,
): Promise<QueryResponse> {
  let attempt = 0;
  let backoffMs = RETRY_BASE_MS;
  let lastError: unknown = null;
  while (attempt < MAX_ATTEMPTS) {
    if (signal?.aborted) {
      throw classifyFetchError(new DOMException('aborted', 'AbortError'));
    }
    try {
      return await sendChatMessageOnce(query, history, signal);
    } catch (err) {
      lastError = err;
      const classified = classifyFetchError(err);
      if (!classified.retryable) {
        throw classified;
      }
      if (attempt + 1 >= MAX_ATTEMPTS) {
        throw {
          ...classified,
          retryable: false,
          message: `${classified.message} (đã thử lại ${attempt} lần)`,
        } satisfies QueryError;
      }
      const sleepMs = classified.retryAfterMs ?? backoffMs;
      const slept = await abortableSleep(sleepMs, signal);
      if (!slept) {
        throw classifyFetchError(new DOMException('aborted', 'AbortError'));
      }
      backoffMs = Math.min(backoffMs * 2, RETRY_MAX_MS);
      attempt += 1;
    }
  }
  throw classifyFetchError(lastError);
}

async function sendChatMessageOnce(
  query: string,
  history: ChatTurn[] | undefined,
  signal: AbortSignal | undefined,
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/v2/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      ...(history && history.length > 0 ? { history } : {}),
    }),
    signal,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const message = errorData.detail || `API error: ${response.status} ${response.statusText}`;
    const tagged = new Error(message) as Error & {
      status: number;
      headers: Headers;
    };
    tagged.status = response.status;
    tagged.headers = response.headers;
    throw tagged;
  }
  return (await response.json()) as QueryResponse;
}

/**
 * Base URL for the RAG backend API.
 *
 * REQUIRED ENV VAR: VITE_API_BASE_URL
 *   - In development builds (`import.meta.env.DEV`) it falls back to the local
 *     FastAPI server on `http://localhost:8000` so `npm run dev` works out
 *     of the box.
 *   - In production builds (`import.meta.env.PROD`) a missing value falls back
 *     to a same-origin relative base (`""`). This lets a reverse proxy route
 *     requests from the same origin to `/api/*` without baking a hard URL into
 *     the static bundle, and avoids silently shipping localhost.
 */
export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (configured) return configured;
  if (import.meta.env.PROD) {
    return '';
  }
  return 'http://localhost:8000';
}

const API_BASE_URL = getApiBaseUrl();

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * /v2/query request body shape (per BE UnifiedQueryRequest):
 *   { query: string, top_k?: number, force_route?: "padded_rag" | "graph_rag", history?: ChatTurn[] }
 * The legacy `force_rag_only` flag is dropped in favor of `force_route`.
 * `history` is OPTIONAL — clients that omit it keep the original single-turn
 * behavior unchanged. The cap is enforced at the slice site in ChatLayout.
 */
export interface QueryRequest {
  query: string;
  top_k?: number;
  force_route?: 'padded_rag' | 'graph_rag';
  history?: ChatTurn[];
}

/**
 * /v2/query response shape (per BE UnifiedQueryResponse, enriched contract).
 * Carries the SAME status surface as legacy /query so ChatLayout warning-banner
 * / abstain / data-gap-hints UX continues to work.
 */
export interface QueryResponse {
  // v2-native fields
  query?: string;
  route?: string;
  router_info?: RouterInfo;
  graph_stats?: GraphStats;
  latency_ms?: number;
  // answer + retrieval
  answer?: string;
  sources?: SourceChip[]; // enriched object[] on /v2 (was string[] on legacy /query)
  confidence?: number;
  groundedness_score?: number;
  trace_id?: string;
  // status surface (now present on /v2 — used by ChatLayout banner)
  status_code?: string;
  status_reason?: string;
  data_gap_hints?: string[];
  internal_status_code?: string | null;
  pii_detected?: boolean;
}

export interface HealthResponse {
  status: string;
  lancedb_connected: boolean;
  vectors_count: number;
  collection: string;
  embedding_model: string;
  reranker_model: string;
}

/**
 * /health readiness probe — thin wrapper used by useBackendReadiness.
 * The hook itself does its own fetch so this stays free of retry policy.
 */
export async function getReadiness(): Promise<HealthResponse & {
  ready?: boolean;
  stage?: string;
  services?: Record<string, boolean>;
}> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Check backend health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
    }

    const data: HealthResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to check health:', error);
    throw error;
  }
}

/**
 * Fire-and-forget warmup: trigger backend embedding-model preload so the
 * first real /v2/query does not pay the ~15s cold-load. MUST never throw —
 * the chat screen calls this on mount and a backend outage must not break
 * the UI. Awaiters can `await warmup()` for tests, but production call
 * sites should fire-and-forget (no await) so render is never blocked.
 */
export async function warmup(): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/warmup`, { method: 'GET' });
  } catch (error) {
    // Swallow — warmup is best-effort. Log so devs can see it in the console
    // but never propagate to the UI mount path.
    console.warn('Warmup ping failed (non-fatal):', error);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// /v2/query/stream — SLICE-B.LATENCY fake-stream SSE
//
// The backend runs the FULL /v2 pipeline (router → booster → graph →
// LLMGenerator with all post-guards) and emits the post-guard answer as
// a text/event-stream. The first frame is `event: meta` (route + sources
// + status), then one or more `event: delta` frames carrying chunks of
// the answer text, then a final `event: done` carrying the full answer
// (so the caller can reconcile). The concatenation of all `delta.text`
// fields equals the `done.answer` value — guaranteed by the BE tests.
//
// The FE uses this to render a "typing" UX. If parsing/streaming fails
// for any reason, callers fall back to `sendChatMessage()` (non-stream).
// ─────────────────────────────────────────────────────────────────────────

export interface StreamMeta {
  route?: string;
  sources?: SourceChip[];
  status_code?: string;
  data_gap_hints?: string[];
  trace_id?: string;
}

export interface StreamHandlers {
  onMeta?: (meta: StreamMeta) => void;
  onDelta?: (text: string) => void;
  onDone?: (fullAnswer: string) => void;
  onError?: (err: Error) => void;
  // PART 1 LATENCY: BE emits `event: stage` early so the FE can replace
  // the typing indicator with a transient progress label during the
  // ~30s router+retrieval phase. The `stage` payload is a short label
  // string (e.g. "Đang phân tích câu hỏi…"). Stage events always arrive
  // BEFORE the first `delta` event for the same response.
  onStage?: (stage: string) => void;
}

interface SseFrame {
  event: string;
  data: string;
}

/**
 * Parse a `text/event-stream` body incrementally. Yields parsed
 * `{event, data}` frames. Handles:
 *   - CRLF (`\r\n`) and LF (`\n`) line endings
 *   - multi-line `data:` (concatenated with a single newline per spec)
 *   - blank-line frame separator
 */
async function* parseSseStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncGenerator<SseFrame> {
  const decoder = new TextDecoder('utf-8');
  // `pending` holds the unconsumed suffix between reads. We normalize
  // CRLF, find each `\n\n` separator, and emit exactly the bytes
  // between the previous separator (or the start of `pending`) and
  // that `\n\n` — that is one frame. The trailing partial frame (after
  // the last complete separator) stays in `pending` for the next read.
  let pending = '';
  const emit = (raw: string): SseFrame | null => {
    const ev: SseFrame = { event: '', data: '' };
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) {
        ev.event = line.slice('event:'.length).trim();
      } else if (line.startsWith('data:')) {
        const piece = line.slice('data:'.length);
        ev.data = ev.data
          ? ev.data + '\n' + piece.replace(/^ /, '')
          : piece.replace(/^ /, '');
      }
    }
    return ev.event || ev.data ? ev : null;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (value) pending += decoder.decode(value, { stream: true });
    // Normalize CRLF and split into complete frames on `\n\n`. Start
    // scanning at 0 because `pending` already holds only the
    // unconsumed suffix.
    const normalized = pending.replace(/\r\n/g, '\n');
    let start = 0;
    while (true) {
      const sepIdx = normalized.indexOf('\n\n', start);
      if (sepIdx === -1) break;
      const rawFrame = normalized.slice(start, sepIdx);
      const ev = emit(rawFrame);
      if (ev) yield ev;
      start = sepIdx + 2;
    }
    pending = normalized.slice(start);
    if (done) break;
  }
  // Tail: best-effort parse of any final partial frame.
  const tail = pending.trim();
  if (tail) {
    const ev = emit(tail);
    if (ev) yield ev;
  }
}
/**
 * Bounded retry wrapper for `sendChatMessage` (plan §3.2 G4). Returns the
 * `QueryResponse` on success or throws a classified `QueryError` on the
 * final failure. The legacy `sendChatMessage` below is preserved as a
 * thin alias so existing callers / tests keep working — the retry loop is
 * now the canonical entry point.
 */
export async function sendChatMessage(query: string, history?: ChatTurn[]): Promise<QueryResponse> {
  try {
    return await sendChatMessageWithRetry(query, history);
  } catch (error) {
    // Legacy callers do not yet consume QueryError. Log for now and
    // re-throw so the caller's existing error path (useChatStream) gets
    // the typed error and can surface `err.kind` to MessageBubble.
    console.error('Failed to send chat message:', error);
    throw error;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// /v2/query/stream — SLICE-B.LATENCY fake-stream SSE
//
// The backend runs the FULL /v2 pipeline (router → booster → graph →
// LLMGenerator with all post-guards) and emits the post-guard answer as
// a text/event-stream. The first frame is `event: meta` (route + sources
// + status), then one or more `event: delta` frames carrying chunks of
// the answer text, then a final `event: done` carrying the full answer
// (so the caller can reconcile). The concatenation of all `delta.text`
// fields equals the `done.answer` value — guaranteed by the BE tests.
//
// The FE uses this to render a "typing" UX. If parsing/streaming fails
// for any reason, callers fall back to `sendChatMessage()` (non-stream).
// ─────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────
// /v2/recommend + /v2/whatif — deterministic, pure-arithmetic, single-digit ms.
// Both endpoints take student score (0–30), optional tổ hợp, optional priority
// bonus. The FE uses these for the "Gợi ý ngành" demo panel; they are stable,
// test-covered on the BE, and never invoke the LLM. Mirror the existing fetch
// + error pattern from sendChatMessage() above.
// ─────────────────────────────────────────────────────────────────────────

export interface RecommendRequest {
  score: number;
  to_hop?: string;
  uu_tien?: number;
}

export type RecommendLabel = 'an_toan' | 'can_nhac' | 'mao_hiem';

export interface RecommendItem {
  major_code: string;
  major_name: string;
  latest_diem_chuan: number;
  latest_year: number;
  delta: number;
  label: RecommendLabel;
  explanation: string;
  to_hop?: string[];
}

export interface RecommendResponse {
  score: number;
  uu_tien: number;
  to_hop?: string;
  count: number;
  recommendations: RecommendItem[];
  basis: string;
  disclaimer: string;
}

export interface WhatIfRequest {
  score: number;
  major_code: string;
  to_hop?: string;
  uu_tien?: number;
}

export interface WhatIfResponse {
  score: number;
  major_code: string;
  p_pass: string;
  band: string;
  basis: string;
  disclaimer: string;
  latest_diem_chuan?: number;
  latest_year?: number;
  delta?: number;
  n_years: number;
}

export async function getRecommendations(req: RecommendRequest): Promise<RecommendResponse> {
  const response = await fetch(`${API_BASE_URL}/v2/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `API error: ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as RecommendResponse;
}

export async function getWhatIf(req: WhatIfRequest): Promise<WhatIfResponse> {
  const response = await fetch(`${API_BASE_URL}/v2/whatif`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `API error: ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as WhatIfResponse;
}
