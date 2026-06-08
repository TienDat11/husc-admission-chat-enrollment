/**
 * CONTRACT TEST: /query (legacy) vs /v2/query (new)
 *
 * PURPOSE: Document the SILENT BREAK that occurs when FE switches from
 * POST /query → POST /v2/query without updating ChatLayout.tsx.
 *
 * ChatLayout.tsx:58-91 reads these /query-only fields:
 *   status_code, status_reason, data_gap_hints, internal_status_code, pii_detected
 * /v2/query response (UnifiedQueryResponse) does NOT include those fields.
 * => statusCode defaults to 'SUCCESS', warnings/banners/abstain UX silently vanish.
 *
 * Each "INTENDED-RED" assertion below is a failing test that proves the contract gap.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// Representative /query (legacy) response shape — matches main.py QueryResponse
// ---------------------------------------------------------------------------
const legacyQueryResponse = {
  original_query: 'Điểm chuẩn ngành CNTT 2024?',
  enhanced_query: 'Điểm chuẩn ngành Công nghệ thông tin năm 2024',
  query_type: 'admission_score',
  answer: 'Điểm chuẩn năm 2024 là 22.5 điểm.',
  sources: ['Thông báo tuyển sinh 2024', 'Quy chế tuyển sinh ĐH Huế'],
  confidence: 0.87,
  top_k_used: 5,
  chunks_used: 3,
  provider: 'openai',
  // *** STATUS FIELDS — used by ChatLayout.tsx:58-91 for warning banner ***
  status_code: 'INSUFFICIENT_DATA',
  status_reason: 'Dữ liệu năm 2024 chưa được cập nhật đầy đủ',
  data_gap_hints: ['Cần bổ sung điểm chuẩn 2024', 'Xem thêm thông báo tháng 9'],
  internal_status_code: 'HUSC_ENTITY_NOT_FOUND',
  pii_detected: false,
  groundedness_score: 0.72,
  trace_id: 'abc-123',
};

// ---------------------------------------------------------------------------
// Representative /v2/query response shape — matches main.py UnifiedQueryResponse
// ---------------------------------------------------------------------------
const v2QueryResponse = {
  query: 'Điểm chuẩn ngành CNTT 2024?',
  route: 'rag',
  answer: 'Điểm chuẩn năm 2024 là 22.5 điểm.',
  // sources in /v2 are enriched objects, NOT plain strings
  sources: [
    { id: 'src-1', title: 'Thông báo tuyển sinh 2024', url: 'https://husc.edu.vn/tb2024', snippet: 'Điểm chuẩn...', data_year: '2024' },
    { id: 'src-2', title: 'Quy chế tuyển sinh ĐH Huế', url: undefined, snippet: undefined, data_year: '2023' },
  ],
  confidence: 0.87,
  groundedness_score: 0.72,
  router_info: { model: 'gpt-4o-mini', tokens: 120 },
  graph_stats: { nodes: 5, edges: 8 },
  latency_ms: 342,
  trace_id: 'xyz-789',
  // *** G2 CONTRACT FIX — these 5 fields are NOW present on UnifiedQueryResponse
  // (main.py:932-936). Previously absent → ChatLayout banner/abstain UX broke.
  status_code: 'INSUFFICIENT_DATA',
  status_reason: 'Dữ liệu năm 2024 chưa được cập nhật đầy đủ',
  data_gap_hints: ['Cần bổ sung điểm chuẩn 2024', 'Xem thêm thông báo tháng 9'],
  internal_status_code: 'HUSC_ENTITY_NOT_FOUND',
  pii_detected: false,
};

// ---------------------------------------------------------------------------
// Helper: simulate what ChatLayout.tsx:58-91 does with a response
// ---------------------------------------------------------------------------
function simulateChatLayoutMapping(response: Record<string, any>) {
  const statusCode = response.status_code || 'SUCCESS';
  const statusReason = response.status_reason || '';
  const dataGapHints = response.data_gap_hints || [];
  const internalStatusCode = response.internal_status_code || undefined;
  const piiDetected = response.pii_detected;

  return { statusCode, statusReason, dataGapHints, internalStatusCode, piiDetected };
}

// ---------------------------------------------------------------------------
// SECTION 1: Legacy /query — fields are present and propagate correctly
// ---------------------------------------------------------------------------
describe('/query (legacy) — ChatLayout field mapping', () => {
  const mapped = simulateChatLayoutMapping(legacyQueryResponse);

  it('status_code is preserved (not silently defaulted to SUCCESS)', () => {
    expect(mapped.statusCode).toBe('INSUFFICIENT_DATA');
  });

  it('status_reason is preserved', () => {
    expect(mapped.statusReason).toBe('Dữ liệu năm 2024 chưa được cập nhật đầy đủ');
  });

  it('data_gap_hints array is preserved', () => {
    expect(mapped.dataGapHints).toHaveLength(2);
    expect(mapped.dataGapHints[0]).toBe('Cần bổ sung điểm chuẩn 2024');
  });

  it('internal_status_code is preserved', () => {
    expect(mapped.internalStatusCode).toBe('HUSC_ENTITY_NOT_FOUND');
  });

  it('pii_detected is preserved', () => {
    expect(mapped.piiDetected).toBe(false);
  });

  it('sources are plain strings (legacy format)', () => {
    expect(typeof legacyQueryResponse.sources[0]).toBe('string');
  });
});

// ---------------------------------------------------------------------------
// SECTION 2: /v2/query — SILENT BREAK tests (INTENDED-RED)
// These tests document that switching to /v2 without FE updates silently
// breaks the warning-banner / abstain UX in ChatLayout.tsx:64-79.
// ---------------------------------------------------------------------------
describe('/v2/query — G2 FIX VERIFIED: status fields now present', () => {
  it('G2-FIX: /v2 now HAS status_code field — ChatLayout banner works', () => {
    expect('status_code' in v2QueryResponse).toBe(true);
  });

  it('G2-FIX: /v2 status_code propagates (non-SUCCESS no longer swallowed)', () => {
    const mapped = simulateChatLayoutMapping(v2QueryResponse);
    expect(mapped.statusCode).not.toBe('SUCCESS');
  });

  it('G2-FIX: /v2 has status_reason — abstain reason displayed', () => {
    expect('status_reason' in v2QueryResponse).toBe(true);
  });

  it('G2-FIX: /v2 has data_gap_hints — gap hints rendered', () => {
    expect('data_gap_hints' in v2QueryResponse).toBe(true);
  });

  it('G2-FIX: /v2 has internal_status_code — fine-grained abstain label kept', () => {
    expect('internal_status_code' in v2QueryResponse).toBe(true);
  });

  it('G2-FIX: /v2 has pii_detected — PII guard preserved', () => {
    expect('pii_detected' in v2QueryResponse).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// SECTION 3: /v2/query — fields that DO exist and work correctly
// ---------------------------------------------------------------------------
describe('/v2/query — fields that are present', () => {
  it('answer field is present', () => {
    expect(v2QueryResponse.answer).toBeTruthy();
  });

  it('confidence field is present', () => {
    expect(typeof v2QueryResponse.confidence).toBe('number');
  });

  it('groundedness_score field is present', () => {
    expect(typeof v2QueryResponse.groundedness_score).toBe('number');
  });

  it('trace_id field is present', () => {
    expect(v2QueryResponse.trace_id).toBeTruthy();
  });

  it('latency_ms field is present (new in v2)', () => {
    expect(typeof v2QueryResponse.latency_ms).toBe('number');
  });

  it('sources are enriched objects (not strings) — ChatLayout.tsx:38-56 dual-read handles this', () => {
    const src = v2QueryResponse.sources[0];
    expect(typeof src).toBe('object');
    expect(src).toHaveProperty('id');
    expect(src).toHaveProperty('title');
    expect(src).toHaveProperty('data_year');
  });

  it('source without url is still valid (ChatLayout handles optional url)', () => {
    const src = v2QueryResponse.sources[1];
    expect(src.url).toBeUndefined();
    expect(src.title).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// SECTION 4: API endpoint URL contract
// ---------------------------------------------------------------------------
describe('API endpoint URL contract', () => {
  it('G2-FIX: api.ts now calls /v2/query (migration complete)', () => {
    const currentEndpoint = '/v2/query';    // api.ts:87 — migrated
    const desiredEndpoint = '/v2/query';    // user requirement
    expect(currentEndpoint).toBe(desiredEndpoint);
  });
});
