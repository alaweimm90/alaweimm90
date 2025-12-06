import React from 'react';
import { motion } from 'framer-motion';
import { TierType } from '@/constants/tiers';

interface FloatingParticlesProps {
  count?: number;
  tier: TierType;
  className?: string;
}

// Tier color mapping for particles
const TIER_COLORS = {
  core: '#3B82F6',      // Trust Blue
  adaptive: '#F15B23',   // REPZ Orange (brand color)
  performance: '#A855F7', // Sophistication Purple
  longevity: '#EAB308'   // Luxury Gold
};

export const FloatingParticles: React.FC<FloatingParticlesProps> = ({ 
  count = 5, 
  tier,
  className = ""
}) => {
  return (
    <div className={`absolute inset-0 pointer-events-none overflow-hidden ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 rounded-full"
          style={{ backgroundColor: TIER_COLORS[tier] }}
          animate={{
            x: [0, 30, -30, 0],
            y: [0, -40, -20, 0],
            rotate: [0, 180, 360],
            opacity: [0.3, 0.8, 0.3]
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            delay: i * 0.5,
            ease: "easeInOut"
          }}
          initial={{
            left: `${20 + i * 15}%`,
            top: `${30 + i * 10}%`
          }}
        />
      ))}
    </div>
  );
};

export default FloatingParticles;