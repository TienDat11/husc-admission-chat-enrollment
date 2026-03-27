import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { HeroSection } from '@/components/chat/HeroSection';
import { ChatLayout } from '@/components/chat/ChatLayout';
import { ThemeProvider } from '@/hooks/useTheme';

const Index = () => {
  const [showChat, setShowChat] = useState(false);

  return (
    <ThemeProvider>
      <AnimatePresence mode="wait">
        {!showChat ? (
          <motion.div
            key="hero"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.4 }}
          >
            <HeroSection onStartChat={() => setShowChat(true)} />
          </motion.div>
        ) : (
          <motion.div
            key="chat"
            initial={{ opacity: 0, scale: 1.02 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            className="h-screen"
          >
            <ChatLayout />
          </motion.div>
        )}
      </AnimatePresence>
    </ThemeProvider>
  );
};

export default Index;
