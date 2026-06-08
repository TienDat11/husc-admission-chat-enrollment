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
 * Get API base URL
 */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
