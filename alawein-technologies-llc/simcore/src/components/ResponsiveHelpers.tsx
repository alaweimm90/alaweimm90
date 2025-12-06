import React from 'react';
import { useResponsive } from '@/hooks/use-responsive';
import { cn } from '@/lib/utils';

interface ResponsiveContainerProps {
  children: React.ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const maxWidthClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md', 
  lg: 'max-w-4xl',
  xl: 'max-w-6xl',
  '2xl': 'max-w-7xl',
  full: 'max-w-full'
};

const paddingClasses = {
  none: '',
  sm: 'px-4 sm:px-6',
  md: 'px-4 sm:px-6 lg:px-8',
  lg: 'px-6 sm:px-8 lg:px-12'
};

export function ResponsiveContainer({ 
  children, 
  className = '',
  maxWidth = 'xl',
  padding = 'md'
}: ResponsiveContainerProps) {
  return (
    <div className={cn(
      'mx-auto w-full',
      maxWidthClasses[maxWidth],
      paddingClasses[padding],
      className
    )}>
      {children}
    </div>
  );
}

interface ResponsiveStackProps {
  children: React.ReactNode;
  direction?: 'column' | 'row';
  spacing?: 'sm' | 'md' | 'lg';
  align?: 'start' | 'center' | 'end';
  justify?: 'start' | 'center' | 'end' | 'between';
  className?: string;
}

const spacingClasses = {
  sm: 'gap-2 sm:gap-3',
  md: 'gap-3 sm:gap-4 lg:gap-6', 
  lg: 'gap-4 sm:gap-6 lg:gap-8'
};

const alignClasses = {
  start: 'items-start',
  center: 'items-center',
  end: 'items-end'
};

const justifyClasses = {
  start: 'justify-start',
  center: 'justify-center', 
  end: 'justify-end',
  between: 'justify-between'
};

export function ResponsiveStack({
  children,
  direction = 'column',
  spacing = 'md',
  align = 'start',
  justify = 'start',
  className = ''
}: ResponsiveStackProps) {
  const directionClass = direction === 'column' ? 'flex-col' : 'flex-col sm:flex-row';
  
  return (
    <div className={cn(
      'flex',
      directionClass,
      spacingClasses[spacing],
      alignClasses[align],
      justifyClasses[justify],
      className
    )}>
      {children}
    </div>
  );
}

interface ResponsiveTextProps {
  children: React.ReactNode;
  variant?: 'hero' | 'title' | 'subtitle' | 'body' | 'caption';
  align?: 'left' | 'center' | 'right';
  className?: string;
}

const textVariants = {
  hero: 'text-4xl sm:text-6xl lg:text-8xl font-bold',
  title: 'text-2xl sm:text-3xl lg:text-4xl font-semibold',
  subtitle: 'text-lg sm:text-xl lg:text-2xl font-medium',
  body: 'text-sm sm:text-base lg:text-lg',
  caption: 'text-xs sm:text-sm'
};

const alignClasses2 = {
  left: 'text-left',
  center: 'text-center', 
  right: 'text-right'
};

export function ResponsiveText({
  children,
  variant = 'body',
  align = 'left',
  className = ''
}: ResponsiveTextProps) {
  return (
    <div className={cn(
      textVariants[variant],
      alignClasses2[align],
      'leading-tight',
      className
    )}>
      {children}
    </div>
  );
}

interface ResponsiveGridProps {
  children: React.ReactNode;
  columns?: { mobile: number; tablet: number; desktop: number };
  gap?: 'sm' | 'md' | 'lg';
  className?: string;
}

const gapClasses = {
  sm: 'gap-2 sm:gap-3',
  md: 'gap-4 sm:gap-6',
  lg: 'gap-6 sm:gap-8'
};

export function ResponsiveGrid({ 
  children,
  columns = { mobile: 1, tablet: 2, desktop: 3 },
  gap = 'md',
  className = ''
}: ResponsiveGridProps) {
  const gridCols = `grid-cols-${columns.mobile} sm:grid-cols-${columns.tablet} lg:grid-cols-${columns.desktop}`;
  
  return (
    <div className={cn(
      'grid',
      gridCols,
      gapClasses[gap],
      className
    )}>
      {children}
    </div>
  );
}