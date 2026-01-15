import { motion, AnimatePresence } from 'framer-motion';
import { X, GraduationCap, BookOpen, Wallet, Building2, Theater, Phone, MessageSquare, Trash2, Plus, Sparkles } from 'lucide-react';
import { categories, mockConversations, type Conversation } from '@/lib/chat-types';
import { ConversationSkeleton } from './Skeleton';

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  isLoadingConversations?: boolean;
}

const iconMap: Record<string, React.ReactNode> = {
  admission: <GraduationCap className="w-5 h-5" />,
  majors: <BookOpen className="w-5 h-5" />,
  tuition: <Wallet className="w-5 h-5" />,
  facilities: <Building2 className="w-5 h-5" />,
  campus: <Theater className="w-5 h-5" />,
  contact: <Phone className="w-5 h-5" />,
};

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
  activeConversationId, 
  onSelectConversation,
  isLoadingConversations = false 
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
            className="lg:hidden p-2 rounded-xl hover:bg-muted transition-colors border-2 border-border"
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
            className="w-full locket-btn-primary flex items-center justify-center gap-2 py-4"
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
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-left bg-card hover:bg-muted transition-all border-3 border-border shadow-[0_3px_0_hsl(var(--border))] hover:shadow-[0_4px_0_hsl(var(--border))] hover:-translate-y-0.5"
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

          {/* Conversation History */}
          <div className="mb-6">
            <h3 className="flex items-center gap-2 text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 px-2">
              💬 Lịch sử chat
            </h3>
            <div className="space-y-2">
              {isLoadingConversations ? (
                <>
                  {[1, 2, 3, 4, 5].map((i) => (
                    <ConversationSkeleton key={i} />
                  ))}
                </>
              ) : (
                mockConversations.map((conversation, index) => (
                  <ConversationItem
                    key={conversation.id}
                    conversation={conversation}
                    isActive={activeConversationId === conversation.id}
                    onSelect={() => onSelectConversation(conversation.id)}
                    index={index}
                  />
                ))
              )}
            </div>
          </div>

          {/* Quick Stats - Locket card style */}
          <div className="space-y-3">
            <h3 className="flex items-center gap-2 text-sm font-bold text-muted-foreground uppercase tracking-wider px-2">
              ⭐ Thống kê
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <motion.div 
                whileHover={{ scale: 1.02, y: -2 }}
                className="p-4 rounded-2xl bg-gradient-to-br from-primary/10 to-secondary/10 border-3 border-primary/30 shadow-[0_3px_0_hsl(var(--primary)/0.3)]"
              >
                <div className="text-2xl font-bold text-primary">234</div>
                <div className="text-xs text-muted-foreground font-bold">Hôm nay 🔥</div>
              </motion.div>
              <motion.div 
                whileHover={{ scale: 1.02, y: -2 }}
                className="p-4 rounded-2xl bg-gradient-to-br from-accent/10 to-success/10 border-3 border-accent/30 shadow-[0_3px_0_hsl(var(--accent)/0.3)]"
              >
                <div className="text-2xl font-bold text-accent flex items-center gap-1">
                  4.8 ⭐
                </div>
                <div className="text-xs text-muted-foreground font-bold">Đánh giá</div>
              </motion.div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

function ConversationItem({
  conversation,
  isActive,
  onSelect,
  index,
}: {
  conversation: Conversation;
  isActive: boolean;
  onSelect: () => void;
  index: number;
}) {
  const timeAgo = getTimeAgo(conversation.timestamp);

  return (
    <motion.button
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      whileHover={{ scale: 1.02, x: 4 }}
      whileTap={{ scale: 0.98 }}
      onClick={onSelect}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-left transition-all group ${
        isActive
          ? 'bg-gradient-to-r from-primary to-secondary text-white border-3 border-primary-dark shadow-[0_4px_0_hsl(var(--primary-dark))]'
          : 'bg-card hover:bg-muted border-3 border-border shadow-[0_3px_0_hsl(var(--border))] hover:shadow-[0_4px_0_hsl(var(--border))] hover:-translate-y-0.5'
      }`}
    >
      <div className={`flex items-center justify-center w-9 h-9 rounded-xl ${isActive ? 'bg-white/20' : 'bg-muted'}`}>
        <MessageSquare className={`w-4 h-4 ${isActive ? 'text-white' : 'text-primary'}`} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-bold text-sm truncate">{conversation.title}</div>
        <div className={`text-xs truncate ${isActive ? 'text-white/80' : 'text-muted-foreground'}`}>{timeAgo}</div>
      </div>
      <motion.button
        whileHover={{ scale: 1.2 }}
        whileTap={{ scale: 0.8 }}
        onClick={(e) => {
          e.stopPropagation();
        }}
        className={`opacity-0 group-hover:opacity-100 p-1.5 rounded-lg transition-all ${
          isActive ? 'hover:bg-white/20 text-white' : 'hover:bg-destructive/10 text-muted-foreground hover:text-destructive'
        }`}
        aria-label="Xóa cuộc trò chuyện"
      >
        <Trash2 className="w-4 h-4" />
      </motion.button>
    </motion.button>
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
