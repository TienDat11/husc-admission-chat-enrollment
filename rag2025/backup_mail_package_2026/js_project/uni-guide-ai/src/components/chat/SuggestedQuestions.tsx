import { useRef } from 'react';
import { motion } from 'framer-motion';
import { suggestedQuestions } from '@/lib/chat-types';

interface SuggestedQuestionsProps {
  onSelect: (question: string) => void;
}

const emojis = ['📚', '💰', '🏛️', '🎓', '📝', '🏆'];

export function SuggestedQuestions({ onSelect }: SuggestedQuestionsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Handle mouse wheel horizontal scroll for desktop
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (scrollRef.current && e.deltaY !== 0) {
      e.preventDefault();
      scrollRef.current.scrollLeft += e.deltaY;
    }
  };

  return (
    <div className="px-4 pb-4">
      <div className="max-w-4xl mx-auto">
        <p className="text-sm font-bold text-muted-foreground mb-3 text-center">
          ✨ Câu hỏi gợi ý
        </p>
        <div 
          ref={scrollRef}
          onWheel={handleWheel}
          className="suggested-questions-scroll flex gap-3 overflow-x-auto pb-2 -mx-4 px-4"
        >
          {suggestedQuestions.map((question, index) => (
            <motion.button
              key={question}
              initial={{ opacity: 0, y: 20, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: index * 0.08, type: 'spring', bounce: 0.4 }}
              whileHover={{ scale: 1.05, y: -4 }}
              whileTap={{ scale: 0.95, y: 2 }}
              onClick={() => onSelect(question)}
              className="flex-shrink-0 flex items-center gap-2 px-5 py-3 rounded-2xl bg-card hover:bg-gradient-to-r hover:from-primary hover:to-secondary hover:text-white border-3 border-border hover:border-primary-dark text-foreground font-bold text-sm transition-all duration-200 shadow-[0_4px_0_hsl(var(--border))] hover:shadow-[0_4px_0_hsl(var(--primary-dark))]"
            >
              <span className="text-lg">{emojis[index % emojis.length]}</span>
              <span className="whitespace-nowrap">{question}</span>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
