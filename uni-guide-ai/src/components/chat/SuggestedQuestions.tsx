import { useRef } from 'react';
import { motion } from 'framer-motion';

interface SuggestedQuestionsProps {
  /** Follow-ups derived from the last answer; empty renders nothing. */
  questions: string[];
  onSelect: (question: string) => void;
}

const emojis = ['📚', '💰', '🏛️', '🎓', '📝', '🏆'];

export function SuggestedQuestions({ questions, onSelect }: SuggestedQuestionsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Handle mouse wheel horizontal scroll for desktop
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (scrollRef.current && e.deltaY !== 0) {
      e.preventDefault();
      scrollRef.current.scrollLeft += e.deltaY;
    }
  };

  if (questions.length === 0) return null;

  return (
    <div className="px-4 pb-4">
      <div className="max-w-4xl mx-auto">
        <p className="text-sm font-bold text-muted-foreground mb-3 text-center">
          ✨ Câu hỏi tiếp theo
        </p>
        <div 
          ref={scrollRef}
          onWheel={handleWheel}
          role="group"
          aria-label="Câu hỏi gợi ý"
          className="suggested-questions-scroll flex gap-3 overflow-x-auto pb-2 -mx-4 px-4"
        >
          {questions.map((question, index) => (
            <motion.button
              key={question}
              initial={{ opacity: 0, y: 20, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: index * 0.08, type: 'spring', bounce: 0.4 }}
              whileHover={{ scale: 1.05, y: -4 }}
              whileTap={{ scale: 0.95, y: 2 }}
              onClick={() => onSelect(question)}
              className="chat-focus flex-shrink-0 flex items-center gap-2 px-5 py-3 rounded-2xl bg-card hover:bg-gradient-to-r hover:from-[hsl(var(--chat-user-from))] hover:to-[hsl(var(--chat-user-to))] hover:text-white border-3 border-border hover:border-primary-dark text-foreground font-bold text-sm transition-all duration-200 shadow-[0_4px_0_hsl(var(--border))] hover:shadow-[0_4px_0_hsl(var(--primary-dark))]"
            >
              <span className="text-lg" aria-hidden="true">{emojis[index % emojis.length]}</span>
              <span className="whitespace-nowrap">{question}</span>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
