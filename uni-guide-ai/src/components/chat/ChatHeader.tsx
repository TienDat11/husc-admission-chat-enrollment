import { motion } from 'framer-motion';
import { Menu, Settings, BarChart3 } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

interface ChatHeaderProps {
  onMenuClick: () => void;
}

export function ChatHeader({ onMenuClick }: ChatHeaderProps) {
  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="sticky top-0 z-50 h-[72px] px-4 md:px-6 flex items-center justify-between bg-background border-b-4 border-border"
    >
      {/* Left Section */}
      <div className="flex items-center gap-3">
        <motion.button
          whileHover={{ scale: 1.1, rotate: 5 }}
          whileTap={{ scale: 0.9 }}
          onClick={onMenuClick}
          className="lg:hidden p-2.5 rounded-xl bg-muted hover:bg-muted/80 transition-colors border-3 border-border shadow-[0_3px_0_hsl(var(--border))]"
          aria-label="Mở menu"
        >
          <Menu className="w-5 h-5" />
        </motion.button>
        
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ scale: 0.8, rotate: -10 }}
            animate={{ scale: 1, rotate: 0 }}
            whileHover={{ scale: 1.1, rotate: 5 }}
            className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center border-3 border-primary-dark shadow-[0_4px_0_hsl(var(--primary-dark))] cursor-pointer"
          >
            <span className="text-2xl">🎓</span>
          </motion.div>
          
          <div className="hidden sm:block">
            <h1 className="font-display font-bold text-lg">
              ĐHKH Huế
            </h1>
            <p className="text-xs text-muted-foreground font-bold">
              Trợ Lý Tuyển Sinh AI ✨
            </p>
          </div>
        </div>
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-2">
        <motion.button
          whileHover={{ scale: 1.1, y: -2 }}
          whileTap={{ scale: 0.9 }}
          className="hidden md:flex p-2.5 rounded-xl bg-muted hover:bg-muted/80 transition-colors border-3 border-border shadow-[0_3px_0_hsl(var(--border))]"
          aria-label="Thống kê"
        >
          <BarChart3 className="w-5 h-5 text-muted-foreground" />
        </motion.button>
        
        <ThemeToggle />
        
        <motion.button
          whileHover={{ scale: 1.1, rotate: 45, y: -2 }}
          whileTap={{ scale: 0.9 }}
          transition={{ type: 'spring', stiffness: 400 }}
          className="p-2.5 rounded-xl bg-muted hover:bg-muted/80 transition-colors border-3 border-border shadow-[0_3px_0_hsl(var(--border))]"
          aria-label="Cài đặt"
        >
          <Settings className="w-5 h-5 text-muted-foreground" />
        </motion.button>
      </div>
    </motion.header>
  );
}
