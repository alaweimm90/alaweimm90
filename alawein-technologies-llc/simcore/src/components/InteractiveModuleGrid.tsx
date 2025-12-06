import React, { useState, useRef, useEffect } from 'react';

import { ModuleGrid } from '@/components/EnhancedResponsiveGrid';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { PhysicsModule } from '@/data/modules';
import { useAnalytics } from '@/lib/analytics';
import { motion } from 'framer-motion';
import { AnimatedModuleTile } from '@/components/module-grid/AnimatedModuleTile';

interface InteractiveModuleGridProps {
  modules: PhysicsModule[];
  onModuleExplore: (route?: string) => void;
  onModuleTheory: () => void;
}

export const InteractiveModuleGrid: React.FC<InteractiveModuleGridProps> = ({
  modules,
  onModuleExplore,
  onModuleTheory
}) => {
  const gridRef = useRef<HTMLDivElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [visibleCards, setVisibleCards] = useState<Set<number>>(new Set());
  const prefersReducedMotion = usePrefersReducedMotion();
  const { trackModuleUsage } = useAnalytics();

  // Intersection Observer for card animations
  useEffect(() => {
    if (prefersReducedMotion) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const index = parseInt(entry.target.getAttribute('data-index') || '0');
          if (entry.isIntersecting) {
            setVisibleCards(prev => new Set(prev).add(index));
          }
        });
      },
      {
        threshold: 0.1,
        rootMargin: '50px'
      }
    );

    const cards = gridRef.current?.querySelectorAll('[data-index]');
    cards?.forEach(card => observer.observe(card));

    return () => observer.disconnect();
  }, [modules.length, prefersReducedMotion]);

  // Ripple effect on hover
  const createRipple = (e: React.MouseEvent, index: number) => {
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

    setTimeout(() => {
      ripple.remove();
    }, 600);
  };

  const getCardStyle = (index: number) => {
    if (prefersReducedMotion) return {};

    const isVisible = visibleCards.has(index);
    const isHovered = hoveredIndex === index;
    
    return {
      transform: isHovered ? 'translateY(-8px) scale(1.02)' : 'translateY(0px) scale(1)',
      opacity: isVisible ? 1 : 0,
      filter: isHovered ? 'brightness(1.1) saturate(1.1)' : 'brightness(1) saturate(1)',
      transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      animationDelay: `${index * 100}ms`
    };
  };

  // Create magnetic field effect between cards
  const getMagneticOffset = (index: number) => {
    if (prefersReducedMotion || hoveredIndex === null) return { x: 0, y: 0 };

    const cols = 3;
    const currentRow = Math.floor(index / cols);
    const currentCol = index % cols;
    const hoveredRow = Math.floor(hoveredIndex / cols);
    const hoveredCol = hoveredIndex % cols;

    const dx = currentCol - hoveredCol;
    const dy = currentRow - hoveredRow;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance === 0 || distance > 2) return { x: 0, y: 0 };

    const force = Math.max(0, 2 - distance) * 3;
    return {
      x: dx !== 0 ? (dx / Math.abs(dx)) * force : 0,
      y: dy !== 0 ? (dy / Math.abs(dy)) * force : 0
    };
  };

  return (
    <div 
      ref={gridRef}
      role="grid"
      aria-label={`Scientific modules grid showing ${modules.length} modules`}
    >
      <ModuleGrid 
        className="w-full"
        gap="lg"
        fixedColumns={3}
      >
      {modules.map((module, index) => (
        <AnimatedModuleTile
          key={module.id}
          module={module}
          index={index}
          isVisible={visibleCards.has(index)}
          hoveredIndex={hoveredIndex}
          setHoveredIndex={setHoveredIndex}
          prefersReducedMotion={prefersReducedMotion}
          onExplore={() => {
            trackModuleUsage(module.id, 'enter');
            onModuleExplore(module.route);
          }}
          onTheory={() => {
            trackModuleUsage(module.id, 'enter');
            onModuleTheory();
          }}
        />
      ))}
      </ModuleGrid>
    </div>
  );
};