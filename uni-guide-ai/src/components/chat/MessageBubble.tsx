// @spec(plan-warmup-ux-20260705-111031 §3.2)
// MessageBubble — renders one chat bubble. Plan §3.2 last section:
// when an error bubble carries an `errorKind` we render the localized
// Vietnamese message corresponding to that kind and DROP the raw
// `err.message` (especially "Failed to fetch") from the user-visible text.
import { motion } from 'framer-motion';
import { Copy, ThumbsUp, ThumbsDown, Check, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message, Source } from '@/lib/chat-types';
import type { QueryErrorKind } from '@/lib/api';

interface MessageBubbleProps {
  message: Message;
  /** When set, the bubble renders the localized Vietnamese error copy
   *  for this kind instead of any user-provided content. Used by
   *  useChatStream to surface api.ts' classified QueryError. */
  errorKind?: QueryErrorKind;
}

const LOCALIZED_ERROR_COPY: Record<QueryErrorKind, string> = {
  network: 'Không kết nối được máy chủ. Vui lòng kiểm tra mạng và thử lại.',
  warming: 'Máy chủ đang khởi động, vui lòng đợi trong giây lát...',
  unauthorized: 'Phiên đăng nhập hết hạn, vui lòng tải lại trang.',
  server: 'Máy chủ gặp sự cố. Vui lòng thử lại sau ít phút.',
  unknown: '⚠️ Có lỗi xảy ra',
};

function localizedErrorBody(kind: QueryErrorKind): string {
  const copy = LOCALIZED_ERROR_COPY[kind];
  if (kind === 'unknown') return copy;
  return `⚠️ ${copy}`;
}

export function MessageBubble({ message, errorKind }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<'like' | 'dislike' | null>(null);

  // Plan §3.2: when caller passes an `errorKind`, swap the rendered body
  // to the localized Vietnamese copy and DROP the raw `err.message`
  // (e.g. "Failed to fetch"). We never let the raw engine string reach
  // the user.
  const renderedContent = errorKind ? localizedErrorBody(errorKind) : message.content;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(renderedContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isBot = message.role === 'assistant';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, type: 'spring', bounce: 0.4 }}
      className={`flex min-w-0 max-w-full gap-3 ${isBot ? 'justify-start' : 'justify-end'}`}
    >
      {/* Bot Avatar - Locket style */}
      {isBot && (
        <motion.div
          initial={{ scale: 0.5, rotate: -20 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: 'spring', bounce: 0.5 }}
          className="flex-shrink-0 w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center border-3 border-primary-dark shadow-[0_4px_0_hsl(217,91%,50%)]"
        >
          <span className="text-xl">🎓</span>
        </motion.div>
      )}

      <div className={`flex min-w-0 max-w-full flex-col ${isBot ? 'items-start' : 'items-end'}`}>
        {/* Message Bubble */}
        <div className={isBot ? 'chat-bubble-bot' : 'chat-bubble-user'}>
          {/* Status banner */}
          {isBot && message.statusCode && message.statusCode !== 'SUCCESS' && (
            <div className="mb-3 rounded-xl border-2 border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-100">
              <div className="font-bold uppercase tracking-wide">
                {message.internalStatusCode || message.statusCode}
              </div>
              {message.statusReason && (
                <div className="mt-1 opacity-90">{message.statusReason}</div>
              )}
            </div>
          )}

          {/* Stage pill — shown while the assistant is processing */}
          {isBot && message.stage && (
            <div className="flex items-center gap-2 mb-3 px-3 py-2 rounded-xl bg-primary/10 border border-primary/20">
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
              <span className="text-sm font-semibold text-primary">{message.stage}</span>
            </div>
          )}

          {isBot ? (
            <>
              {/* content may be empty when errorKind drives the text */}
              {(message.content || errorKind) && (
                <div className="message-content prose prose-sm min-w-0 max-w-full">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <p className="mb-3 last:mb-0 text-[15px] leading-[1.7] font-medium">
                          {children}
                        </p>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-base font-bold mt-4 mb-2">
                          {children}
                        </h3>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-none pl-0 my-3 space-y-2">
                          {children}
                        </ul>
                      ),
                      li: ({ children }) => (
                        <li className="relative pl-7 text-[15px] leading-[1.7] font-medium">
                          <span className="absolute left-0 top-0 text-primary font-bold">→</span>
                          {children}
                        </li>
                      ),
                      strong: ({ children }) => (
                        <strong className="font-bold text-primary">{children}</strong>
                      ),
                      a: ({ href, children }) => {
                        const safeHref =
                          href && (/^(https?:|mailto:)/i.test(href) || href.startsWith('/'))
                            ? href
                            : '#';
                        return (
                          <a
                            href={safeHref}
                            className="text-primary underline underline-offset-2 hover:text-secondary transition-colors font-semibold"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {children}
                          </a>
                        );
                      },
                      code: ({ children }) => (
                        <code className="px-2 py-1 rounded-lg bg-muted text-sm font-mono font-semibold">
                          {children}
                        </code>
                      ),
                      table: ({ children }) => (
                        <div className="my-3 w-full max-w-full overflow-x-auto">
                          <table className="min-w-max text-sm border-collapse">
                            {children}
                          </table>
                        </div>
                      ),
                      thead: ({ children }) => (
                        <thead className="bg-muted/50">{children}</thead>
                      ),
                      th: ({ children }) => (
                        <th className="border border-border px-2 py-1.5 text-left font-bold text-xs">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="border border-border px-2 py-1.5 text-xs">
                          {children}
                        </td>
                      ),
                    }}
                  >
                    {renderedContent}
                  </ReactMarkdown>
                </div>
              )}
            </>
          ) : (
            <p className="text-[15px] leading-[1.7] font-semibold">{message.content}</p>
          )}

          {/* Source Chips - Locket style */}
          {isBot && message.sources && message.sources.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t-2 border-border">
              {message.sources.map((source) => (
                <SourceChip key={source.id} source={source} />
              ))}
            </div>
          )}
        </div>

        {/* Action Buttons - Locket style */}
        {isBot && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex items-center gap-1 mt-2 px-1"
          >
            <ActionButton
              onClick={handleCopy}
              active={copied}
              icon={copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              label={copied ? 'Đã sao chép!' : 'Sao chép'}
            />
            <ActionButton
              onClick={() => setLiked(liked === 'like' ? null : 'like')}
              active={liked === 'like'}
              icon={<ThumbsUp className="w-4 h-4" />}
              label="Hay lắm! 👍"
            />
            <ActionButton
              onClick={() => setLiked(liked === 'dislike' ? null : 'dislike')}
              active={liked === 'dislike'}
              icon={<ThumbsDown className="w-4 h-4" />}
              label="Chưa ổn 👎"
            />
          </motion.div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-muted-foreground mt-1 px-1 font-medium">
          {message.timestamp.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

      {/* User Avatar - Locket style */}
      {!isBot && (
        <motion.div
          initial={{ scale: 0.5, rotate: 20 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: 'spring', bounce: 0.5 }}
          className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-success flex items-center justify-center text-white font-bold text-sm border-3 border-accent/50 shadow-[0_3px_0_hsl(173,80%,30%)]"
        >
          😊
        </motion.div>
      )}
    </motion.div>
  );
}

function SourceChip({ source }: { source: Source }) {
  const content = (
    <>
      <span>📄</span>
      <span className="max-w-[120px] truncate">{source.title}</span>
      {source.data_year && source.data_year !== 'N/A' && (
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-semibold">
          {source.data_year}
        </span>
      )}
      <ExternalLink className="w-3 h-3 flex-shrink-0" />
    </>
  );

  if (source.url) {
    return (
      <motion.a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        whileHover={{ scale: 1.05, y: -2 }}
        whileTap={{ scale: 0.95 }}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-muted hover:bg-primary hover:text-white text-xs font-bold transition-all border-2 border-border hover:border-primary shadow-[0_2px_0_hsl(var(--border))] hover:shadow-[0_2px_0_hsl(var(--primary-dark))]"
      >
        {content}
      </motion.a>
    );
  }

  return (
    <motion.button
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-muted hover:bg-primary hover:text-white text-xs font-bold transition-all border-2 border-border hover:border-primary shadow-[0_2px_0_hsl(var(--border))] hover:shadow-[0_2px_0_hsl(var(--primary-dark))]"
    >
      {content}
    </motion.button>
  );
}

function ActionButton({
  onClick,
  active,
  icon,
  label,
}: {
  onClick: () => void;
  active?: boolean;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <motion.button
      whileHover={{ scale: 1.1, y: -2 }}
      whileTap={{ scale: 0.9 }}
      onClick={onClick}
      className={`p-2 rounded-xl transition-all border-2 ${
        active
          ? 'bg-primary text-white border-primary-dark shadow-[0_2px_0_hsl(var(--primary-dark))]'
          : 'hover:bg-muted border-transparent hover:border-border text-muted-foreground hover:text-foreground'
      }`}
      title={label}
      aria-label={label}
    >
      {icon}
    </motion.button>
  );
}
