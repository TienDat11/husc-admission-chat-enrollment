export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  stage?: string | null;
  timestamp: Date;
  sources?: Source[];
  statusCode?: string;
  statusReason?: string;
  dataGapHints?: string[];
  internalStatusCode?: string;
  // @spec(ticket-43) answer_confidence from the BE, verbatim: `null` when the
  // generator produced no answer. NEVER defaulted — a fabricated number would
  // claim the FE trusts text it does not. Distinct from `confidence`
  // (retrieval quality), which keeps its own meaning.
  answerConfidence?: number | null;
  // @spec(ticket-43) the question that produced this answer, so a failed
  // generation can be re-sent verbatim by the retry affordance.
  retryQuery?: string;
  // @spec(plan-warmup-ux-20260705-111031 §3.2) — carries the classified
  // QueryErrorKind so MessageBubble renders the correct Vietnamese copy
  // instead of the raw err.message (e.g. "Failed to fetch").
  errorKind?: 'network' | 'warming' | 'unauthorized' | 'server' | 'unknown';
  // Backend response fields (from /query endpoint)
  confidence?: number;
  topKUsed?: number;
  chunksUsed?: number;
  provider?: string;
  traceId?: string;
  piiDetected?: boolean;
  enhancedQuery?: string;
  queryType?: string;
  groundednessScore?: number;
}

export interface Source {
  id: string;
  title: string;
  url?: string;
  snippet?: string;
  data_year?: string;
  confidence: number;
}

export interface Category {
  id: string;
  name: string;
  icon: string;
  description: string;
}

export const categories: Category[] = [
  { id: 'admission', name: 'Tuyển sinh', icon: '🎓', description: 'Thông tin tuyển sinh' },
  { id: 'majors', name: 'Ngành học', icon: '📖', description: 'Các ngành đào tạo' },
  { id: 'tuition', name: 'Học phí & Học bổng', icon: '💰', description: 'Chi phí học tập' },
  { id: 'facilities', name: 'Cơ sở vật chất', icon: '🏛️', description: 'Thư viện, KTX...' },
  { id: 'campus', name: 'Đời sống sinh viên', icon: '🎭', description: 'Hoạt động ngoại khóa' },
  { id: 'contact', name: 'Liên hệ', icon: '📞', description: 'Thông tin liên lạc' },
];
