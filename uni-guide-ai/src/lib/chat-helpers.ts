/**
 * Pure helpers shared by the chat UI and the streaming hook.
 */

import type { SourceChip } from './api-types';
import type { Source } from './chat-types';

/**
 * Convert the backend's source shape (enriched SourceChip or legacy plain
 * string) to the FE presentation Source type. Adds a synthetic id and a
 * default confidence when missing.
 */
export function mapSources(
  rawSources: Array<SourceChip | string> | undefined,
  confidence: number,
): Source[] {
  const items = rawSources || [];
  return items.map((source, index) => {
    if (typeof source === 'string') {
      return {
        id: `source-${index}`,
        title: source,
        confidence,
      };
    }
    return {
      id: source.id || `source-${index}`,
      title: source.title || 'Unknown source',
      url: source.url || undefined,
      snippet: source.snippet || undefined,
      data_year: source.data_year || undefined,
      confidence,
    };
  });
}

/**
 * Decorate a non-SUCCESS backend answer with a warning banner, optional
 * reason line, and optional data-gap hints. SUCCESS answers pass through
 * unchanged.
 */
export function decorateAnswer(
  rawAnswer: string,
  statusCode: string,
  statusReason: string,
  dataGapHints: string[],
  internalStatusCode: string | undefined,
): string {
  if (statusCode === 'SUCCESS') return rawAnswer;
  const labels: Record<string, string> = {
    NOT_IN_HUSC_SCOPE: 'Ngoài phạm vi tuyển sinh HUSC',
    HUSC_ENTITY_NOT_FOUND: 'Không thấy ngành/sự việc trong dữ liệu HUSC',
    INSUFFICIENT_DATA: 'Thiếu dữ liệu để trả lời chắc chắn',
  };
  const displayCode = internalStatusCode || statusCode;
  const header =
    labels[displayCode] || labels[statusCode] || 'Cần kiểm tra dữ liệu';
  const hintLines =
    dataGapHints.length > 0
      ? `\n\n**Gợi ý bổ sung dữ liệu**:\n${dataGapHints.map((h) => `- ${h}`).join('\n')}`
      : '';
  const reasonLine = statusReason ? `\n\n*Lý do:* ${statusReason}` : '';
  return `⚠️ **${header}**\n\n${rawAnswer}${reasonLine}${hintLines}`;
}
