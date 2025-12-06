import React from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

interface AnimatedSectionProps {
  children: React.ReactNode;
  className?: string;
  animation?: 'fade-in' | 'slide-up' | 'scale-in' | 'slide-in-left' | 'slide-in-right';
  delay?: number;
  threshold?: number;
}

export const AnimatedSection: React.FC<AnimatedSectionProps> = ({
  children,
  className,
  animation = 'fade-in',
  delay = 0,
  threshold = 0.1
}) => {
  const { ref, isVisible } = useScrollAnimation({ threshold });

  const animationClasses = {
    'fade-in': 'translate-y-8 opacity-0',
    'slide-up': 'translate-y-12 opacity-0',
    'scale-in': 'scale-95 opacity-0',
    'slide-in-left': '-translate-x-12 opacity-0',
    'slide-in-right': 'translate-x-12 opacity-0'
  };

  const visibleClasses = {
    'fade-in': 'translate-y-0 opacity-100',
    'slide-up': 'translate-y-0 opacity-100',
    'scale-in': 'scale-100 opacity-100',
    'slide-in-left': 'translate-x-0 opacity-100',
    'slide-in-right': 'translate-x-0 opacity-100'
  };

  return (
    <div
      ref={ref as React.RefObject<HTMLDivElement>}
      className={cn(
        'transition-all duration-700 ease-out',
        isVisible ? visibleClasses[animation] : animationClasses[animation],
        className
      )}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
};