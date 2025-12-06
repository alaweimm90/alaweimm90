import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface AnimatedSectionProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
  duration?: number;
}

const directionVariants = {
  up: { y: 50, opacity: 0 },
  down: { y: -50, opacity: 0 },
  left: { x: 50, opacity: 0 },
  right: { x: -50, opacity: 0 }
};

export const AnimatedSection: React.FC<AnimatedSectionProps> = ({
  children,
  className,
  delay = 0,
  direction = 'up',
  duration = 0.6
}) => {
  return (
    <motion.div
      className={cn(className)}
      initial={directionVariants[direction]}
      animate={{ x: 0, y: 0, opacity: 1 }}
      transition={{ 
        duration, 
        delay,
        ease: [0.25, 0.46, 0.45, 0.94] // Custom easing
      }}
      whileInView={{ x: 0, y: 0, opacity: 1 }}
      viewport={{ once: true, margin: "-100px" }}
    >
      {children}
    </motion.div>
  );
};

// Micro-interaction button with haptic feedback
export const InteractiveButton: React.FC<{
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  className?: string;
}> = ({ children, onClick, variant = 'primary', className }) => {
  const handleClick = async () => {
    // Add haptic feedback for mobile
    if ('vibrate' in navigator) {
      navigator.vibrate(50);
    }
    onClick?.();
  };

  return (
    <motion.button
      className={cn(
        'px-6 py-3 rounded-lg font-medium transition-all duration-200',
        'focus:outline-none focus:ring-2 focus:ring-offset-2',
        variant === 'primary' && 'bg-primary text-primary-foreground hover:bg-primary/90 focus:ring-primary',
        variant === 'secondary' && 'bg-secondary text-secondary-foreground hover:bg-secondary/80 focus:ring-secondary',
        variant === 'ghost' && 'hover:bg-accent hover:text-accent-foreground focus:ring-accent',
        className
      )}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98, y: 0 }}
      onClick={handleClick}
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
    >
      {children}
    </motion.button>
  );
};

// Glassmorphism card component
export const GlassCard: React.FC<{
  children: React.ReactNode;
  className?: string;
  blur?: boolean;
}> = ({ children, className, blur = true }) => {
  return (
    <motion.div
      className={cn(
        'relative overflow-hidden rounded-xl border border-white/20',
        blur && 'backdrop-blur-md bg-white/10 dark:bg-black/10',
        'shadow-xl shadow-black/5',
        className
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={{ 
        scale: 1.02,
        boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
      }}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent pointer-events-none" />
      {children}
    </motion.div>
  );
};