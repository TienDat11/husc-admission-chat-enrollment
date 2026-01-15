import { motion } from 'framer-motion';

export function MeshGradientOrbs() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Primary Blue Orb */}
      <motion.div
        className="absolute w-[800px] h-[800px] rounded-full opacity-40"
        style={{
          background: 'radial-gradient(circle, hsl(210 100% 50% / 0.6) 0%, transparent 70%)',
          left: '-20%',
          top: '-20%',
          filter: 'blur(80px)',
        }}
        animate={{
          x: [0, 100, 50, 0],
          y: [0, 50, 100, 0],
          scale: [1, 1.1, 0.9, 1],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      
      {/* Purple Orb */}
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full opacity-30"
        style={{
          background: 'radial-gradient(circle, hsl(262 83% 58% / 0.6) 0%, transparent 70%)',
          right: '-10%',
          top: '20%',
          filter: 'blur(80px)',
        }}
        animate={{
          x: [0, -80, -40, 0],
          y: [0, 80, 40, 0],
          scale: [1, 0.9, 1.1, 1],
        }}
        transition={{
          duration: 30,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: -5,
        }}
      />
      
      {/* Teal Orb */}
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full opacity-25"
        style={{
          background: 'radial-gradient(circle, hsl(173 80% 40% / 0.6) 0%, transparent 70%)',
          left: '30%',
          bottom: '-10%',
          filter: 'blur(80px)',
        }}
        animate={{
          x: [0, 60, -30, 0],
          y: [0, -60, -30, 0],
          scale: [1, 1.15, 0.95, 1],
        }}
        transition={{
          duration: 22,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: -10,
        }}
      />
    </div>
  );
}
