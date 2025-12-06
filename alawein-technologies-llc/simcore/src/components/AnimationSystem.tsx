import React, { memo, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';

// Animation variants for consistent micro-interactions
export const animations = {
  // Page transitions
  pageEnter: 'animate-fade-in',
  pageExit: 'animate-fade-out',
  
  // Card interactions
  cardHover: 'hover:scale-[1.02] hover:shadow-lg transition-all duration-200',
  cardPress: 'active:scale-[0.98] transition-transform duration-100',
  
  // Button interactions
  buttonHover: 'hover:shadow-md hover:-translate-y-0.5 transition-all duration-200',
  buttonPress: 'active:scale-[0.96] transition-transform duration-100',
  
  // Loading states
  pulse: 'animate-pulse',
  bounce: 'animate-bounce',
  spin: 'animate-spin',
  
  // Reveal animations
  slideUp: 'animate-[slideUp_0.3s_ease-out]',
  slideDown: 'animate-[slideDown_0.3s_ease-out]',
  slideLeft: 'animate-[slideLeft_0.3s_ease-out]',
  slideRight: 'animate-[slideRight_0.3s_ease-out]',
  
  // Attention grabbers
  glow: 'animate-[glow_2s_ease-in-out_infinite_alternate]',
  wiggle: 'animate-[wiggle_1s_ease-in-out_infinite]',
};

interface AnimatedContainerProps {
  children: React.ReactNode;
  animation?: keyof typeof animations;
  delay?: number;
  className?: string;
  disabled?: boolean;
}

export const AnimatedContainer = memo<AnimatedContainerProps>(({
  children,
  animation = 'pageEnter',
  delay = 0,
  className,
  disabled = false
}) => {
  const prefersReducedMotion = usePrefersReducedMotion();
  
  const animationClass = (!disabled && !prefersReducedMotion) 
    ? animations[animation] 
    : '';
    
  const style = delay > 0 ? { animationDelay: `${delay}ms` } : undefined;
  
  return (
    <div 
      className={cn(animationClass, className)}
      style={style}
    >
      {children}
    </div>
  );
});

AnimatedContainer.displayName = 'AnimatedContainer';

// Staggered animation for lists
interface StaggeredListProps {
  children: React.ReactNode[];
  staggerDelay?: number;
  animation?: keyof typeof animations;
  className?: string;
}

export const StaggeredList = memo<StaggeredListProps>(({
  children,
  staggerDelay = 100,
  animation = 'slideUp',
  className
}) => {
  const prefersReducedMotion = usePrefersReducedMotion();
  
  if (prefersReducedMotion) {
    return <div className={className}>{children}</div>;
  }
  
  return (
    <div className={className}>
      {children.map((child, index) => (
        <AnimatedContainer
          key={index}
          animation={animation}
          delay={index * staggerDelay}
        >
          {child}
        </AnimatedContainer>
      ))}
    </div>
  );
});

StaggeredList.displayName = 'StaggeredList';

// Interactive element wrapper
interface InteractiveElementProps {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  hoverAnimation?: keyof typeof animations;
  pressAnimation?: keyof typeof animations;
}

export const InteractiveElement = memo<InteractiveElementProps>(({
  children,
  onClick,
  className,
  hoverAnimation = 'cardHover',
  pressAnimation = 'cardPress'
}) => {
  const prefersReducedMotion = usePrefersReducedMotion();
  
  const handleClick = useCallback(() => {
    onClick?.();
  }, [onClick]);
  
  const animationClasses = !prefersReducedMotion 
    ? `${animations[hoverAnimation]} ${animations[pressAnimation]}`
    : '';
  
  return (
    <div
      className={cn(
        'cursor-pointer',
        animationClasses,
        className
      )}
      onClick={handleClick}
    >
      {children}
    </div>
  );
});

InteractiveElement.displayName = 'InteractiveElement';

// Progress indicator with smooth animation
interface AnimatedProgressProps {
  value: number;
  max?: number;
  className?: string;
  showPercentage?: boolean;
}

export const AnimatedProgress = memo<AnimatedProgressProps>(({
  value,
  max = 100,
  className,
  showPercentage = false
}) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  return (
    <div className={cn('relative w-full', className)}>
      <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-gradient-primary transition-all duration-700 ease-out"
          style={{ 
            width: `${percentage}%`,
            transform: 'translateX(0%)'
          }}
        />
      </div>
      {showPercentage && (
        <span className="absolute right-0 top-3 text-sm text-muted-foreground">
          {Math.round(percentage)}%
        </span>
      )}
    </div>
  );
});

AnimatedProgress.displayName = 'AnimatedProgress';