import { motion, AnimatePresence } from 'framer-motion';
import { X, MessageSquare, Trash2, Plus, Sparkles } from 'lucide-react';
import { categories } from '@/lib/chat-types';
import type { SessionHistoryEntry } from '@/lib/session-history';

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  /** Questions asked in the current session, newest last. */
  history: SessionHistoryEntry[];
  activeMessageId: string | null;
  onSelectEntry: (messageId: string) => void;
  onDeleteEntry: (messageId: string) => void;
}

const emojiMap: Record<string, string> = {
  admission: '🎓',
  majors: '📚',
  tuition: '💰',
  facilities: '🏛️',
  campus: '🎭',
  contact: '📞',
};

export function ChatSidebar({
  isOpen,
  onClose,
  onNewChat,
  history,
  activeMessageId,
  onSelectEntry,
  onDeleteEntry,
}: ChatSidebarProps) {
  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar - Desktop: always visible, Mobile: animated */}
      <aside
        className={`fixed lg:relative top-0 left-0 z-50 lg:z-0 h-full w-[300px] bg-sidebar-background border-r-4 border-border flex flex-col transition-transform duration-300 ease-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b-4 border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center border-3 border-primary-dark shadow-[0_3px_0_hsl(var(--primary-dark))]">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="font-display font-bold text-lg">Menu</span>
          </div>
          <motion.button
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            onClick={onClose}
            className="lg:hidden chat-focus p-2 rounded-xl hover:bg-muted transition-colors border-2 border-border"
            aria-label="Đóng menu"
          >
            <X className="w-5 h-5" />
          </motion.button>
        </div>

        {/* New Chat Button - Locket style */}
        <div className="p-4">
          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98, y: 2 }}
            onClick={onNewChat}
            className="w-full chat-focus flex items-center justify-center gap-2 py-4 rounded-2xl font-bold text-base text-white bg-gradient-to-r from-[hsl(var(--chat-user-from))] to-[hsl(var(--chat-user-to))] border-3 border-primary-dark shadow-[0_4px_0_hsl(var(--primary-dark))] transition-all duration-200 hover:-translate-y-0.5"
          >
            <Plus className="w-5 h-5" />
            <span>Chat mới 💬</span>
          </motion.button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar px-4 pb-4">
          {/* Categories - Locket card style */}
          <div className="mb-6">
            <h3 className="flex items-center gap-2 text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 px-2">
              📚 Danh mục
            </h3>
            <div className="space-y-2">
              {categories.map((category, index) => (
                <motion.button
                  key={category.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full chat-focus flex items-center gap-3 px-4 py-3 rounded-2xl text-left bg-card hover:bg-muted transition-all border-3 border-border shadow-[0_3px_0_hsl(var(--border))] hover:shadow-[0_4px_0_hsl(var(--border))] hover:-translate-y-0.5"
                >
                  <span className="text-xl">{emojiMap[category.id] || category.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-bold text-sm">{category.name}</div>
                    <div className="text-xs text-muted-foreground truncate">{category.description}</div>
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Conversation History — the current session only. A new chat
              empties this list; nothing is invented to fill it. */}
          <div className="mb-6">
            <h3 className="flex items-center gap-2 text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 px-2">
              💬 Lịch sử chat
            </h3>
            <div className="space-y-2">
              {history.length === 0 ? (
                <p
                  data-testid="session-history-empty"
                  className="px-4 py-3 text-xs text-muted-foreground rounded-2xl border-3 border-dashed border-border"
                >
                  Chưa có câu hỏi nào trong phiên này.
                </p>
              ) : (
                history.map((entry, index) => (
                  <HistoryItem
                    key={entry.messageId}
                    entry={entry}
                    isActive={activeMessageId === entry.messageId}
                    onSelect={() => onSelectEntry(entry.messageId)}
                    onDelete={() => onDeleteEntry(entry.messageId)}
                    index={index}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

function HistoryItem({
  entry,
  isActive,
  onSelect,
  onDelete,
  index,
}: {
  entry: SessionHistoryEntry;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
  index: number;
}) {
  const timeAgo = getTimeAgo(entry.timestamp);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      data-testid="session-history-item"
      // Ticket 23: this row used to be a <button> wrapping the delete
      // <button> — invalid nesting that made "xoá" unreachable by
      // keyboard. It is now a container with two sibling controls, so
      // select and delete are separate tab stops (WCAG 2.1.1).
      className={`w-full flex items-center gap-1 p-1.5 rounded-2xl transition-all group ${
        isActive
          ? 'bg-gradient-to-r from-[hsl(var(--chat-user-from))] to-[hsl(var(--chat-user-to))] text-white border-3 border-primary-dark shadow-[0_4px_0_hsl(var(--primary-dark))]'
          : 'bg-card hover:bg-muted border-3 border-border shadow-[0_3px_0_hsl(var(--border))] hover:shadow-[0_4px_0_hsl(var(--border))] hover:-translate-y-0.5'
      }`}
    >
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onSelect}
        aria-current={isActive ? 'true' : undefined}
        className="chat-focus flex items-center gap-3 flex-1 min-w-0 px-2.5 py-1.5 rounded-xl text-left"
      >
        <div className={`flex items-center justify-center w-9 h-9 rounded-xl flex-shrink-0 ${isActive ? 'bg-white/20' : 'bg-muted'}`}>
          <MessageSquare className={`w-4 h-4 ${isActive ? 'text-white' : 'text-primary'}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-bold text-sm truncate">{entry.question}</div>
          <div className={`text-xs truncate ${isActive ? 'text-white' : 'text-muted-foreground'}`}>{timeAgo}</div>
        </div>
      </motion.button>
      <motion.button
        whileHover={{ scale: 1.2 }}
        whileTap={{ scale: 0.8 }}
        onClick={onDelete}
        // Keyboard users get the same affordance as hover users: the
        // delete control is invisible until the row is hovered or it
        // itself is focused.
        className={`chat-focus flex-shrink-0 p-1.5 rounded-lg transition-all opacity-0 group-hover:opacity-100 focus-visible:opacity-100 focus:opacity-100 ${
          isActive ? 'hover:bg-white/20 text-white' : 'hover:bg-destructive/10 text-muted-foreground hover:text-destructive'
        }`}
        aria-label="Xóa cuộc trò chuyện"
      >
        <Trash2 className="w-4 h-4" />
      </motion.button>
    </motion.div>
  );
}

function getTimeAgo(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(hours / 24);

  if (hours < 1) return 'Vừa xong 🆕';
  if (hours < 24) return `${hours} giờ trước`;
  if (days < 7) return `${days} ngày trước`;
  return date.toLocaleDateString('vi-VN');
}
