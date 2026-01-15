import { motion } from 'framer-motion';
import { Copy, ThumbsUp, ThumbsDown, Volume2, Check, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Message, Source } from '@/lib/chat-types';

interface MessageBubbleProps {
  message: Message;
  isLast?: boolean;
}

export function MessageBubble({ message, isLast }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<'like' | 'dislike' | null>(null);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isBot = message.role === 'assistant';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, type: 'spring', bounce: 0.4 }}
      className={`flex gap-3 ${isBot ? 'justify-start' : 'justify-end'}`}
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

      <div className={`flex flex-col ${isBot ? 'items-start' : 'items-end'} max-w-[85%]`}>
        {/* Message Bubble */}
        <div className={isBot ? 'chat-bubble-bot' : 'chat-bubble-user'}>
          {isBot ? (
            <div className="message-content prose prose-sm max-w-none">
              <ReactMarkdown
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
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      className="text-primary underline underline-offset-2 hover:text-secondary transition-colors font-semibold"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {children}
                    </a>
                  ),
                  code: ({ children }) => (
                    <code className="px-2 py-1 rounded-lg bg-muted text-sm font-mono font-semibold">
                      {children}
                    </code>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
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
  return (
    <motion.button
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-muted hover:bg-primary hover:text-white text-xs font-bold transition-all border-2 border-border hover:border-primary shadow-[0_2px_0_hsl(var(--border))] hover:shadow-[0_2px_0_hsl(var(--primary-dark))]"
    >
      <span>📄</span>
      <span className="max-w-[120px] truncate">{source.title}</span>
      <ExternalLink className="w-3 h-3" />
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
