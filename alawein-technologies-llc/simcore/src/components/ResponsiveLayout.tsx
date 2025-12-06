import React from 'react';
import { useResponsiveEnhanced, useResponsiveContainer, useSafeAreaEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';

// Apple-quality responsive container
interface ResponsiveContainerProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'page' | 'section' | 'card' | 'sidebar';
  enableSafeArea?: boolean;
}

export function ResponsiveContainer({ 
  children, 
  className = '',
  variant = 'section',
  enableSafeArea = false
}: ResponsiveContainerProps) {
  const { viewport, getSpacing } = useResponsiveEnhanced();
  const { className: containerClass } = useResponsiveContainer();
  const safeArea = useSafeAreaEnhanced();
  const spacing = getSpacing();

  const variantClasses = {
    page: 'min-h-screen',
    section: 'w-full',
    card: 'rounded-lg border bg-card',
    sidebar: 'h-full'
  };

  const baseClasses = cn(
    variantClasses[variant],
    containerClass,
    enableSafeArea && 'supports-[padding:env(safe-area-inset-top)]:pt-[env(safe-area-inset-top)]',
    enableSafeArea && 'supports-[padding:env(safe-area-inset-bottom)]:pb-[env(safe-area-inset-bottom)]',
    className
  );

  const style = enableSafeArea ? safeArea.cssVars as React.CSSProperties : undefined;

  return (
    <div className={baseClasses} style={style}>
      {children}
    </div>
  );
}

// Fluid grid system with CSS Grid fallback
interface FluidGridProps {
  children: React.ReactNode;
  minItemWidth?: string;
  gap?: 'sm' | 'md' | 'lg';
  className?: string;
  fallbackColumns?: { mobile: number; tablet: number; desktop: number };
}

export function FluidGrid({ 
  children, 
  minItemWidth = '280px',
  gap = 'md',
  className = '',
  fallbackColumns = { mobile: 1, tablet: 2, desktop: 3 }
}: FluidGridProps) {
  const { viewport } = useResponsiveEnhanced();

  const gapClasses = {
    sm: 'gap-[var(--semantic-spacing-inline)]',
    md: 'gap-[var(--semantic-spacing-card)]',
    lg: 'gap-[var(--semantic-spacing-module)]'
  };

  // CSS Grid with auto-fit for modern browsers
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${minItemWidth}), 1fr))`,
  };

  // Fallback classes for older browsers
  const fallbackClasses = `grid-cols-${fallbackColumns.mobile} md:grid-cols-${fallbackColumns.tablet} lg:grid-cols-${fallbackColumns.desktop}`;

  return (
    <div
      className={cn(
        'grid w-full',
        gapClasses[gap],
        // Fallback for browsers without CSS Grid support
        fallbackClasses,
        className
      )}
      style={gridStyle}
    >
      {children}
    </div>
  );
}

// Responsive stack with progressive enhancement
interface ResponsiveStackProps {
  children: React.ReactNode;
  direction?: { mobile: 'column' | 'row'; tablet?: 'column' | 'row'; desktop?: 'column' | 'row' };
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  gap?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function ResponsiveStack({
  children,
  direction = { mobile: 'column', tablet: 'row', desktop: 'row' },
  align = 'start',
  justify = 'start',
  gap = 'md',
  className = ''
}: ResponsiveStackProps) {
  const gapClasses = {
    sm: 'gap-[var(--semantic-spacing-inline)]',
    md: 'gap-[var(--semantic-spacing-card)]',
    lg: 'gap-[var(--semantic-spacing-module)]'
  };

  const alignClasses = {
    start: 'items-start',
    center: 'items-center',
    end: 'items-end',
    stretch: 'items-stretch'
  };

  const justifyClasses = {
    start: 'justify-start',
    center: 'justify-center',
    end: 'justify-end',
    between: 'justify-between',
    around: 'justify-around'
  };

  const directionClasses = [
    `flex-${direction.mobile}`,
    direction.tablet && `md:flex-${direction.tablet}`,
    direction.desktop && `lg:flex-${direction.desktop}`
  ].filter(Boolean).join(' ');

  return (
    <div className={cn(
      'flex',
      directionClasses,
      alignClasses[align],
      justifyClasses[justify],
      gapClasses[gap],
      className
    )}>
      {children}
    </div>
  );
}

// Touch-optimized button wrapper
interface TouchButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function TouchButton({
  children,
  onClick,
  disabled = false,
  variant = 'primary',
  size = 'md',
  className = ''
}: TouchButtonProps) {
  const { isTouch } = useResponsiveEnhanced();

  const baseClasses = cn(
    'inline-flex items-center justify-center rounded-md font-medium transition-colors',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    'disabled:opacity-50 disabled:pointer-events-none',
    // Touch optimization
    isTouch && 'min-h-[var(--touch-target-min)] min-w-[var(--touch-target-min)] touch-manipulation',
    // Prevent 300ms tap delay on mobile
    'cursor-pointer active:scale-95',
    className
  );

  const sizeClasses = {
    sm: isTouch ? 'h-11 px-4 py-2' : 'h-9 px-3 py-1',
    md: isTouch ? 'h-12 px-6 py-3' : 'h-10 px-4 py-2',
    lg: isTouch ? 'h-14 px-8 py-4' : 'h-11 px-8 py-2'
  };

  const variantClasses = {
    primary: 'bg-primary text-primary-foreground hover:bg-primary/90',
    secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    ghost: 'hover:bg-accent hover:text-accent-foreground'
  };

  return (
    <button
      className={cn(baseClasses, sizeClasses[size], variantClasses[variant])}
      onClick={onClick}
      disabled={disabled}
      type="button"
    >
      {children}
    </button>
  );
}

// Responsive text with fluid typography
interface ResponsiveTextProps {
  children: React.ReactNode;
  as?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'p' | 'span';
  variant?: 'hero' | 'title' | 'subtitle' | 'body' | 'caption' | 'overline';
  className?: string;
}

export function ResponsiveText({
  children,
  as: Component = 'p',
  variant = 'body',
  className = ''
}: ResponsiveTextProps) {
  const { viewport } = useResponsiveEnhanced();

  const variantClasses = {
    hero: 'text-3xl md:text-5xl lg:text-7xl font-bold tracking-tight',
    title: 'text-2xl md:text-3xl lg:text-4xl font-semibold tracking-tight',
    subtitle: 'text-lg md:text-xl lg:text-2xl font-medium',
    body: 'text-sm md:text-base lg:text-lg',
    caption: 'text-xs md:text-sm',
    overline: 'text-xs md:text-sm font-medium uppercase tracking-wide'
  };

  return (
    <Component className={cn(variantClasses[variant], 'leading-relaxed', className)}>
      {children}
    </Component>
  );
}