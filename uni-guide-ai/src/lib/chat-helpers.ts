/**
 * Pure helpers shared by the chat UI and the streaming hook.
 */

import type { SourceChip } from './api-types';
import type { Message, Source } from './chat-types';

/**
 * Ticket 44 publishes this token on the PUBLIC `status_code` when the
 * generator produced no model answer; `internal_status_code` mirrors it only
 * under dev error-exposure (rag2025 `main.py:1164-1169,1982-1988`), else null.
 * Ticket 43 keys off the status code only — never the `provider` string,
 * which carries season-dependent suffixes ("Fallback(Error) + season").
 */
export const GENERATION_FAILED_STATUS = 'GENERATION_FAILED';

/**
 * True when the PUBLIC status code carries the degraded-generation token.
 * Deliberately NOT `statusCode || internalStatusCode`: at both emit sites the
 * internal mirror is written only inside the branch that also sets the public
 * code, so the OR adds zero coverage; and `decorateAnswer`'s SUCCESS
 * early-return keys off the public code, so an OR gate could pair a retry
 * button with an undecorated, normal-looking answer if the codes ever
 * diverged. `decorateAnswer` keeps `displayCode` for its label lookup; the
 * affordance gate does not.
 */
export function isGenerationFailure(message: { statusCode?: string }): boolean {
  return message.statusCode === GENERATION_FAILED_STATUS;
}

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
    // Ticket 43: a transient infrastructure outage, NOT missing data — the
    // distinct label keeps it out of the 'Can kiem tra du lieu' fallback.
    GENERATION_FAILED: 'Máy chủ gặp sự cố khi tạo câu trả lời. Vui lòng thử lại.',
  };
  const displayCode = internalStatusCode || statusCode;
  const header =
    labels[displayCode] || labels[statusCode] || 'Cần kiểm tra dữ liệu';
  // A failed generation may still carry data_gap_hints from an earlier stage;
  // rendering them would dress an infra outage up as missing data (ticket 43
  // criterion 2), so the hint block is suppressed for GENERATION_FAILED.
  const isGenFailure =
    statusCode === GENERATION_FAILED_STATUS || displayCode === GENERATION_FAILED_STATUS;
  const hintLines =
    !isGenFailure && dataGapHints.length > 0
      ? `\n\n**Gợi ý bổ sung dữ liệu**:\n${dataGapHints.map((h) => `- ${h}`).join('\n')}`
      : '';
  const reasonLine = statusReason ? `\n\n*Lý do:* ${statusReason}` : '';
  return `⚠️ **${header}**\n\n${rawAnswer}${reasonLine}${hintLines}`;
}

/**
 * Localized Vietnamese copy for transport/API failures, keyed by the
 * classified QueryErrorKind. Shared by MessageBubble (renders the body) and
 * deriveAnswerAnnouncement (screen-reader announcement) so the announced
 * text and the visible text can never drift apart.
 */
export const LOCALIZED_ERROR_COPY: Record<string, string> = {
  network: 'Không kết nối được máy chủ. Vui lòng kiểm tra mạng và thử lại.',
  warming: 'Máy chủ đang khởi động, vui lòng đợi trong giây lát...',
  unauthorized: 'Phiên đăng nhập hết hạn, vui lòng tải lại trang.',
  server: 'Máy chủ gặp sự cố. Vui lòng thử lại sau ít phút.',
  unknown: 'Có lỗi xảy ra',
};

/** Vietnamese labels for non-SUCCESS public status codes (abstain/conflict). */
export const STATUS_ANSWER_LABELS: Record<string, string> = {
  NOT_IN_HUSC_SCOPE: 'Ngoài phạm vi tuyển sinh HUSC',
  HUSC_ENTITY_NOT_FOUND: 'Không thấy ngành/sự việc trong dữ liệu HUSC',
  INSUFFICIENT_DATA: 'Thiếu dữ liệu để trả lời chắc chắn',
  GENERATION_FAILED: 'Máy chủ gặp sự cố khi tạo câu trả lời. Vui lòng thử lại.',
};

export interface AnswerAnnouncement {
  /** Content-complete updates (stage transitions, final answer ready). */
  polite: string | null;
  /** Urgent updates (abstain / conflict / generation-failed / transport error). */
  alert: string | null;
}

/**
 * Ticket 23 criterion 2: derive what a screen reader should announce for the
 * current conversation state, at a sensible granularity.
 *
 * The streaming answer text itself is NEVER announced per token (the message
 * container is role="log" aria-live="off"); this helper emits exactly one
 * polite string per stage transition, one polite "answer ready" summary per
 * completed SUCCESS answer, and one alert per abstain/conflict/failed state.
 * Pure function of (messages, isTyping) so tests can pin every branch.
 */
export function deriveAnswerAnnouncement(
  messages: Message[],
  isTyping: boolean,
): AnswerAnnouncement {
  if (messages.length === 0) return { polite: null, alert: null };
  const last = messages[messages.length - 1];

  // The user's own echo is not worth announcing (input is already visible and
  // the assistant placeholder arrives in the same tick).
  if (last.role !== 'assistant') return { polite: null, alert: null };

  // Processing: stage text is stable while it streams, so one polite update.
  if (last.stage) return { polite: last.stage, alert: null };

  // Transport/API error bubble (errorKind set by useChatStream).
  if (last.errorKind) {
    const copy = LOCALIZED_ERROR_COPY[last.errorKind] ?? LOCALIZED_ERROR_COPY.unknown;
    return { polite: null, alert: `Lỗi: ${copy}` };
  }

  // While the first delta is still in flight there is nothing meaningful to
  // announce yet (and announcing partial content would re-fire per token).
  if (isTyping) return { polite: null, alert: null };

  const code = last.statusCode;
  if (code === GENERATION_FAILED_STATUS) {
    const retry = last.retryQuery ? ' Bạn có thể bấm nút thử lại.' : '';
    return {
      polite: null,
      alert: `Không tạo được câu trả lời. ${STATUS_ANSWER_LABELS.GENERATION_FAILED}${retry}`,
    };
  }

  const preview = last.content.slice(0, 120);
  if (code && code !== 'SUCCESS') {
    const label = STATUS_ANSWER_LABELS[code] ?? 'Cần kiểm tra dữ liệu';
    return { polite: null, alert: `${label}: ${preview}` };
  }

  const sourceCount = last.sources?.length ?? 0;
  const sources = sourceCount > 0 ? ` Có ${sourceCount} nguồn tham khảo.` : '';
  return { polite: `Câu trả lời đã sẵn sàng: ${preview}${sources}`, alert: null };
}
