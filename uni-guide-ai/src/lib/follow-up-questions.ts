import type { Message } from './chat-types';

/**
 * Ticket 31: follow-up questions are derived from the turn that just
 * happened (the major/year the student asked about) — never from a static
 * list. Suggestions are questions only; they assert no fact.
 *
 * When neither a major nor a year can be read out of the turn, this returns
 * an empty list rather than inventing a subject.
 */
const MAX_QUESTIONS = 4;

const YEAR_RE = /\b(20\d{2})\b/;
const MAJOR_RE = /ngành\s+([^,.;:!?\n()]{2,40})/iu;
const MAJOR_TAIL_RE = /\s+(năm|của|tại|ở|thuộc|như|nào|gì|ra sao|bao nhiêu)(?=\s|$).*$/iu;

function cleanMajor(raw: string): string | null {
  const value = raw
    .replace(/["'“”‘’]/g, '')
    .replace(MAJOR_TAIL_RE, '')
    .replace(/[\s\-–—:]+$/u, '')
    .trim();
  return value.length >= 2 ? value : null;
}

function firstMajor(...texts: string[]): string | null {
  for (const text of texts) {
    const match = text.match(MAJOR_RE);
    if (match) {
      const major = cleanMajor(match[1]);
      if (major) return major;
    }
  }
  return null;
}

/**
 * The year the student is actually asking about. Source `data_year` is the
 * authoritative stamp on the retrieved evidence, so it wins over free text.
 */
function resolveYear(answer: Message, question: string): string | null {
  for (const source of answer.sources ?? []) {
    const match = source.data_year?.match(YEAR_RE);
    if (match) return match[1];
  }
  return (
    answer.content.match(YEAR_RE)?.[1] ??
    question.match(YEAR_RE)?.[1] ??
    null
  );
}

export function deriveFollowUpQuestions(messages: Message[]): string[] {
  let answerIndex = -1;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'assistant') {
      answerIndex = i;
      break;
    }
  }
  if (answerIndex === -1) return [];

  const answer = messages[answerIndex];
  // A failed turn has nothing to follow up on.
  if (answer.errorKind || !answer.content.trim()) return [];

  const previous = messages[answerIndex - 1];
  const question = previous?.role === 'user' ? previous.content : '';

  const year = resolveYear(answer, question);
  const major = firstMajor(question, answer.content);

  let candidates: string[];
  if (major && year) {
    candidates = [
      `Học phí ngành ${major} năm ${year}?`,
      `Tổ hợp xét tuyển ngành ${major}?`,
      `Chỉ tiêu tuyển sinh ngành ${major} năm ${year}?`,
      `Điểm chuẩn ngành ${major} năm ${year}?`,
    ];
  } else if (major) {
    candidates = [
      `Học phí ngành ${major}?`,
      `Tổ hợp xét tuyển ngành ${major}?`,
      `Điểm chuẩn ngành ${major}?`,
    ];
  } else if (year) {
    candidates = [
      `Điểm chuẩn các ngành năm ${year}?`,
      `Học phí năm ${year}?`,
      `Chỉ tiêu tuyển sinh năm ${year}?`,
    ];
  } else {
    return [];
  }

  const asked = question.toLowerCase().replace(/\s+/g, ' ').trim();
  return candidates
    .filter((candidate) => candidate.toLowerCase().replace(/\s+/g, ' ').trim() !== asked)
    .slice(0, MAX_QUESTIONS);
}
