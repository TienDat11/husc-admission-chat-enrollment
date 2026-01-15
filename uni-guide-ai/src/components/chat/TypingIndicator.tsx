import { motion } from 'framer-motion';

export function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.9 }}
      transition={{ duration: 0.3, type: 'spring', bounce: 0.4 }}
      className="flex gap-3"
    >
      {/* Bot Avatar - Locket style */}
      <motion.div
        initial={{ scale: 0.5, rotate: -20 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: 'spring', bounce: 0.5 }}
        className="flex-shrink-0 w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center border-3 border-primary-dark shadow-[0_4px_0_hsl(217,91%,50%)]"
      >
        <motion.span 
          className="text-xl"
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          🎓
        </motion.span>
      </motion.div>

      {/* Typing Bubble - Locket style */}
      <div className="chat-bubble-bot px-6 py-4">
        <div className="flex items-center gap-3">
          {/* Animated Dots */}
          <div className="flex gap-2">
            {[0, 1, 2].map((i) => (
              <motion.span
                key={i}
                className="w-3 h-3 rounded-full bg-primary"
                animate={{
                  y: [0, -8, 0],
                  scale: [1, 1.2, 1],
                }}
                transition={{
                  duration: 0.8,
                  repeat: Infinity,
                  ease: 'easeInOut',
                  delay: i * 0.15,
                }}
              />
            ))}
          </div>
          <span className="text-sm text-muted-foreground font-bold">Đang suy nghĩ... 🤔</span>
        </div>
      </div>
    </motion.div>
  );
}
