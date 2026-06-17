import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStream } from '@/hooks/useChatStream';
import { ChatHeader } from './ChatHeader';
import { ChatSidebar } from './ChatSidebar';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { SuggestedQuestions } from './SuggestedQuestions';

export function ChatLayout() {
  const [messages, setMessages] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);

  const { send, isTyping } = useChatStream({
    messages,
    onMessagesChange: setMessages,
  });

  const handleSendMessage = useCallback(
    (content) => {
      send(content);
    },
    [send],
  );

  const handleSelectQuestion = useCallback(
    (question) => {
      handleSendMessage(question);
    },
    [handleSendMessage],
  );

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveConversationId(null);
    setIsSidebarOpen(false);
  }, []);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background mesh-gradient-bg">
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader onMenuClick={() => setIsSidebarOpen(true)} />

        <ChatMessages messages={messages} isTyping={isTyping} />

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
