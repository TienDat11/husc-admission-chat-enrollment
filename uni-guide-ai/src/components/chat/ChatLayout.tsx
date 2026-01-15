import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Message, generateId, Source } from '@/lib/chat-types';
import { sendChatMessage } from '@/lib/api';
import { ChatHeader } from './ChatHeader';
import { ChatSidebar } from './ChatSidebar';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { SuggestedQuestions } from './SuggestedQuestions';

export function ChatLayout() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSendMessage = useCallback(async (content: string) => {
    // Add user message immediately (optimistic UI)
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setError(null);

    // Show typing indicator
    setIsTyping(true);

    try {
      // Call real RAG API
      const response = await sendChatMessage(content);

      // Convert sources from backend format to frontend format
      const sources: Source[] = response.sources.map((source, index) => ({
        id: `source-${index}`,
        title: source,
        confidence: response.confidence,
      }));

      // Add bot response
      const botMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sources,
      };

      setIsTyping(false);
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setIsTyping(false);
      const errorMessage = err instanceof Error ? err.message : 'Không thể kết nối đến máy chủ';
      setError(errorMessage);

      // Add error message as bot response
      const errorMessageBot: Message = {
        id: generateId(),
        role: 'assistant',
        content: `⚠️ Có lỗi xảy ra: ${errorMessage}\n\nVui lòng kiểm tra kết nối hoặc thử lại sau.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessageBot]);

      console.error('Failed to send message:', err);
    }
  }, []);

  const handleSelectQuestion = useCallback((question: string) => {
    handleSendMessage(question);
  }, [handleSendMessage]);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveConversationId(null);
    setIsSidebarOpen(false);
    setError(null);
  }, []);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background mesh-gradient-bg">
      {/* Sidebar */}
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader onMenuClick={() => setIsSidebarOpen(true)} />

        <ChatMessages messages={messages} isTyping={isTyping} />

        {/* Suggested Questions */}
        <AnimatePresence>
          {messages.length === 0 && !isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              <SuggestedQuestions onSelect={handleSelectQuestion} />
            </motion.div>
          )}
        </AnimatePresence>

        <ChatInput onSend={handleSendMessage} isLoading={isTyping} />
      </div>
    </div>
  );
}