import { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { HeroSection } from '@/components/chat/HeroSection';
import { ChatLayout } from '@/components/chat/ChatLayout';
import { ThemeProvider } from '@/hooks/useTheme';
import { MajorRecommender } from '@/components/MajorRecommender';
import { warmup } from '@/lib/api';

const Index = () => {
  const [showChat, setShowChat] = useState(false);

  // SLICE-A.LATENCY — fire-and-forget embedding warmup the INSTANT the site
  // lands (Hero), NOT when ChatLayout mounts. The backend Qwen3 embedding
  // model lazy-loads (~8s cold); pinging /warmup here means it preloads while
  // the user reads the hero, so the first real query skips the cold-start.
  // warmup() is internally non-throwing and never blocks render.
  useEffect(() => {
    void warmup();
  }, []);

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
            className="min-h-screen overflow-y-auto"
          >
            <HeroSection onStartChat={() => setShowChat(true)} />
            <motion.section
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative z-10 px-4 pb-20"
            >
              <MajorRecommender />
            </motion.section>
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
