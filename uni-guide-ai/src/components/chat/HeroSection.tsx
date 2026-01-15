import { motion } from 'framer-motion';
import { ArrowRight, Sparkles, MessageCircle, Zap, Target, Star } from 'lucide-react';
import { FloatingParticles } from './FloatingParticles';
import { MeshGradientOrbs } from './MeshGradientOrbs';

interface HeroSectionProps {
  onStartChat: () => void;
}

export function HeroSection({ onStartChat }: HeroSectionProps) {
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Gradient Background */}
      <div className="absolute inset-0 animated-gradient-bg" />
      
      {/* Mesh Gradient Orbs */}
      <MeshGradientOrbs />
      
      {/* Floating Particles */}
      <FloatingParticles />
      
      {/* Content */}
      <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
        {/* Logo - Locket style blocky */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5, rotate: -10 }}
          animate={{ opacity: 1, scale: 1, rotate: 0 }}
          transition={{ duration: 0.5, type: 'spring', bounce: 0.5 }}
          className="mb-8 inline-block"
        >
          <div className="relative">
            <div className="w-28 h-28 rounded-3xl bg-white flex items-center justify-center border-4 border-white/30 shadow-[0_8px_0_rgba(255,255,255,0.3)]">
              <span className="text-6xl">🎓</span>
            </div>
            {/* Decorative badge */}
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.3, type: 'spring', bounce: 0.6 }}
              className="absolute -top-2 -right-2 w-10 h-10 rounded-xl bg-secondary flex items-center justify-center border-3 border-white/30 shadow-lg"
            >
              <Sparkles className="w-5 h-5 text-white" />
            </motion.div>
          </div>
        </motion.div>

        {/* Main Title - Bold Locket style */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="font-display text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-4 leading-tight tracking-tight"
        >
          Trợ Lý Tuyển Sinh
          <br />
          <span className="inline-flex items-center gap-3">
            Thông Minh
            <motion.span
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              ✨
            </motion.span>
          </span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="text-lg md:text-xl text-white/90 mb-10 max-w-2xl mx-auto font-medium"
        >
          Đại học Khoa học Huế - Giải đáp mọi thắc mắc về tuyển sinh 24/7! 🚀
        </motion.p>

        {/* CTA Button - Locket style */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <motion.button
            onClick={onStartChat}
            whileHover={{ scale: 1.05, y: -4 }}
            whileTap={{ scale: 0.98, y: 2 }}
            className="group inline-flex items-center gap-3 px-10 py-5 rounded-2xl bg-white text-foreground font-bold text-xl border-4 border-white/30 shadow-[0_6px_0_rgba(0,0,0,0.15)] hover:shadow-[0_8px_0_rgba(0,0,0,0.15)] transition-all"
          >
            <MessageCircle className="w-7 h-7 text-primary" />
            <span>Bắt Đầu Chat</span>
            <ArrowRight className="w-6 h-6 text-primary transition-transform group-hover:translate-x-1" />
          </motion.button>
        </motion.div>

        {/* Stats - Locket card style */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="mt-12 flex flex-wrap justify-center gap-4"
        >
          <StatCard icon={<MessageCircle className="w-5 h-5" />} value="10K+" label="Học sinh" emoji="👥" />
          <StatCard icon={<Zap className="w-5 h-5" />} value="24/7" label="Online" emoji="⚡" />
          <StatCard icon={<Target className="w-5 h-5" />} value="95%" label="Chính xác" emoji="🎯" />
        </motion.div>
      </div>
      
      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
          className="text-4xl"
        >
          👇
        </motion.div>
      </motion.div>
    </div>
  );
}

function StatCard({ icon, value, label, emoji }: { icon: React.ReactNode; value: string; label: string; emoji: string }) {
  return (
    <motion.div 
      whileHover={{ scale: 1.05, y: -4 }}
      className="flex items-center gap-3 px-5 py-3 rounded-2xl bg-white/20 backdrop-blur-sm border-3 border-white/30 shadow-[0_4px_0_rgba(255,255,255,0.2)]"
    >
      <span className="text-2xl">{emoji}</span>
      <div className="text-left">
        <div className="font-bold text-white text-lg">{value}</div>
        <div className="text-sm text-white/80 font-medium">{label}</div>
      </div>
    </motion.div>
  );
}
