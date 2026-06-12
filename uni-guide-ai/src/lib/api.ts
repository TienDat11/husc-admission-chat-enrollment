/**
 * API Service for RAG Backend
 *
 * Connects to FastAPI backend running on port 8000
 * Endpoints:
 * - POST /v2/query - Main RAG query endpoint (enriched contract; carries status_code / data_gap_hints / etc.)
 * - GET /health - Health check
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * /v2/query request body shape (per BE UnifiedQueryRequest):
 *   { query: string, top_k?: number, force_route?: "padded_rag" | "graph_rag" }
 * The legacy `force_rag_only` flag is dropped in favor of `force_route`.
 */
export interface QueryRequest {
  query: string;
  top_k?: number;
  force_route?: 'padded_rag' | 'graph_rag';
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
  router_info?: any;
  graph_stats?: any;
  latency_ms?: number;
  // answer + retrieval
  answer?: string;
  sources?: any[]; // enriched object[] on /v2 (was string[] on legacy /query)
  confidence?: number;
  groundedness_score?: number;
  trace_id?: string;
  // status surface (now present on /v2 — used by ChatLayout banner)
  status_code?: string;
  status_reason?: string;
  data_gap_hints?: string[];
  internal_status_code?: string | null;
  pii_detected?: boolean;
  // legacy /query aliases — kept optional for backward compatibility
  original_query?: string;
  enhanced_query?: string;
  query_type?: string;
  top_k_used?: number;
  chunks_used?: number;
  provider?: string;
  chunks?: any[];
  // camelCase aliases for internal use
  originalQuery?: string;
  enhancedQuery?: string;
  queryType?: string;
  statusCode?: string;
  statusReason?: string;
  dataGapHints?: string[];
  internalStatusCode?: string;
  topKUsed?: number;
  chunksUsed?: number;
  piiDetected?: boolean;
  groundednessScore?: number;
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
 * Send a chat message to the RAG backend
 *
 * Targets POST /v2/query (enriched UnifiedQueryResponse contract).
 * The /v2 payload is `{query, top_k?, force_route?}` — `force_rag_only` is
 * NOT used on /v2; we send only `{query}` and let the BE route by its own rules.
 */
export async function sendChatMessage(query: string): Promise<QueryResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/v2/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `API error: ${response.status} ${response.statusText}`
      );
    }

    const data: QueryResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to send chat message:', error);
    throw error;
  }
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
  sources?: any[];
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
 * Fire a /v2/query/stream request and pipe the SSE frames to the
 * supplied handlers. The promise resolves once the `done` frame is
 * received (with the full answer) or rejects on transport / parse error.
 *
 * Callers should fall back to `sendChatMessage()` on rejection.
 */
export async function queryStream(
  query: string,
  handlers: StreamHandlers,
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/v2/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ query }),
  });
  if (!response.ok || !response.body) {
    throw new Error(
      `Stream unavailable: ${response.status} ${response.statusText}`,
    );
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
      // PART 1 LATENCY: pass through the BE's early progress label.
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
        // SSE allows multi-line `data:` fields concatenated with `\n`.
        // The BE only ever sends a single line, so we take the first.
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
 * Get API base URL
 */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
