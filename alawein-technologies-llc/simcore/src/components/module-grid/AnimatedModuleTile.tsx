import React from 'react';
import { motion } from 'framer-motion';
import { ModuleCard } from '@/components/ModuleCard';
import { PhysicsModule } from '@/data/modules';

interface AnimatedModuleTileProps {
  module: PhysicsModule;
  index: number;
  isVisible: boolean;
  hoveredIndex: number | null;
  setHoveredIndex: React.Dispatch<React.SetStateAction<number | null>>;
  prefersReducedMotion: boolean;
  onExplore: () => void;
  onTheory: () => void;
}

export const AnimatedModuleTile: React.FC<AnimatedModuleTileProps> = ({
  module,
  index,
  isVisible,
  hoveredIndex,
  setHoveredIndex,
  prefersReducedMotion,
  onExplore,
  onTheory
}) => {
  const getMagneticOffset = (idx: number) => {
    if (prefersReducedMotion || hoveredIndex === null) return { x: 0, y: 0 };
    const cols = 3;
    const currentRow = Math.floor(idx / cols);
    const currentCol = idx % cols;
    const hoveredRow = Math.floor(hoveredIndex / cols);
    const hoveredCol = hoveredIndex % cols;
    const dx = currentCol - hoveredCol;
    const dy = currentRow - hoveredRow;
    const distance = Math.hypot(dx, dy);
    if (distance === 0 || distance > 2) return { x: 0, y: 0 };
    const force = Math.max(0, 2 - distance) * 3;
    return {
      x: dx !== 0 ? (dx / Math.abs(dx)) * force : 0,
      y: dy !== 0 ? (dy / Math.abs(dy)) * force : 0,
    };
  };

  const magneticOffset = getMagneticOffset(index);

  const cardStyle: React.CSSProperties = prefersReducedMotion
    ? {}
    : {
        transform: hoveredIndex === index ? 'translateY(-8px) scale(1.02)' : 'translateY(0px) scale(1)',
        opacity: isVisible ? 1 : 0,
        filter: hoveredIndex === index ? 'brightness(1.1) saturate(1.1)' : 'brightness(1) saturate(1)',
        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        animationDelay: `${index * 100}ms`,
      };

  const createRipple = (e: React.MouseEvent) => {
    if (prefersReducedMotion) return;
    const card = e.currentTarget as HTMLElement;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ripple = document.createElement('div');
    ripple.className = 'absolute rounded-full bg-primary/20 pointer-events-none animate-pulse';
    ripple.style.left = `${x}px`;
    ripple.style.top = `${y}px`;
    ripple.style.width = '20px';
    ripple.style.height = '20px';
    ripple.style.transform = 'translate(-50%, -50%)';
    ripple.style.zIndex = '1';
    card.appendChild(ripple);
    setTimeout(() => ripple.remove(), 600);
  };

  return (
    <motion.div
      key={module.id}
      data-index={index}
      role="gridcell"
      className={`relative group ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}
      style={{
        ...cardStyle,
        transform: `${(cardStyle as any).transform || 'translateY(0px) scale(1)'} translate(${magneticOffset.x}px, ${magneticOffset.y}px)`,
      }}
      initial={prefersReducedMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
      whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, amount: 0.2 }}
      whileHover={prefersReducedMotion ? undefined : { scale: 1.02 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      onMouseEnter={(e) => {
        setHoveredIndex(index);
        createRipple(e);
      }}
      onMouseLeave={() => setHoveredIndex(null)}
    >
      {/* Energy field visualization around hovered card */}
      {hoveredIndex === index && !prefersReducedMotion && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 rounded-xl bg-gradient-primary/10 animate-pulse" />
          <div className="absolute -inset-2 rounded-xl border border-primary/20 animate-pulse" />
        </div>
      )}

      {/* Card with enhanced interactions */}
      <div className="relative z-10">
        <ModuleCard
          title={module.title}
          description={module.description}
          category={module.category}
          difficulty={module.difficulty}
          tags={module.tags}
          equation={module.equation}
          isImplemented={module.isImplemented}
          onExplore={onExplore}
          onTheory={onTheory}
        />
      </div>

      {/* Particle trail effect */}
      {hoveredIndex === index && !prefersReducedMotion && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-xl">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-primary/60 rounded-full animate-quantum-drift"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${i * 200}ms`,
                animationDuration: `${2000 + Math.random() * 1000}ms`,
              }}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default AnimatedModuleTile;
