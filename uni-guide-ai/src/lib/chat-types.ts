export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Source[];
  statusCode?: string;
  statusReason?: string;
  dataGapHints?: string[];
  internalStatusCode?: string;
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

export interface Conversation {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messages: Message[];
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

// ---------------------------------------------------------------------------
// Year helpers — keep the UI in sync with the academic cycle.
// All suggestedQuestions and mockSources that previously hardcoded "2024" /
// "2025-2026" / "2025" now derive from CURRENT_YEAR so the FE rolls forward
// without a code change. Tests assert no literal year in this file.
// ---------------------------------------------------------------------------
const CURRENT_YEAR: number = new Date().getFullYear();
const NEXT_ACADEMIC_YEAR: string = `${CURRENT_YEAR}-${CURRENT_YEAR + 1}`;

export const suggestedQuestions: string[] = [
  `Điểm chuẩn ngành Công nghệ thông tin năm ${CURRENT_YEAR}?`,
  `Học phí năm học ${NEXT_ACADEMIC_YEAR} là bao nhiêu?`,
  "Thư viện trường có những gì?",
  "Quy trình xét tuyển như thế nào?",
  "Hồ sơ đăng ký cần những gì?",
  "Học bổng dành cho sinh viên mới?",
];

export const mockConversations: Conversation[] = [
  {
    id: '1',
    title: 'Hỏi về ngành CNTT',
    preview: 'Điểm chuẩn ngành Công nghệ...',
    timestamp: new Date(Date.now() - 3600000),
    messages: [],
  },
  {
    id: '2',
    title: 'Thông tin học phí',
    preview: 'Học phí các ngành như thế...',
    timestamp: new Date(Date.now() - 86400000),
    messages: [],
  },
];

export const generateId = (): string => {
  return Math.random().toString(36).substring(2, 15);
};

export const generateMockResponse = (userMessage: string): string => {
  const responses: Record<string, string> = {
    default: `Cảm ơn bạn đã hỏi về **"${userMessage}"**. 

Tôi là trợ lý tuyển sinh của Đại học Khoa học Huế. Dựa trên thông tin tôi có, đây là câu trả lời:

### Thông tin chung
- Đại học Khoa học Huế là trường đại học đào tạo đa ngành thuộc Đại học Huế
- Trường có nhiều chương trình đào tạo từ cử nhân đến tiến sĩ
- Cơ sở vật chất hiện đại, đội ngũ giảng viên giàu kinh nghiệm

### Liên hệ thêm
Để biết thêm chi tiết, bạn có thể:
- Gọi đến: **0234.3823.290**
- Email: **tuyensinh@hueuni.edu.vn**
- Website: **husc.hueuni.edu.vn**

Bạn có câu hỏi gì thêm không?`,
  };

  return responses.default;
};

export const generateMockSources = (): Source[] => {
  return [
    { id: '1', title: `Thông báo tuyển sinh ${CURRENT_YEAR}`, confidence: 0.95 },
    { id: '2', title: 'Quy chế đào tạo đại học', confidence: 0.87 },
  ];
};
