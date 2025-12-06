import React from 'react';
import { cn } from '@/lib/utils';

interface ResponsiveGridProps {
  children: React.ReactNode;
  className?: string;
  cols?: {
    default?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
    '2xl'?: number;
  };
  gap?: number;
  autoFit?: boolean;
  minItemWidth?: string;
}

export function ResponsiveGrid({
  children,
  className,
  cols = { default: 1, md: 2, lg: 3 },
  gap = 6,
  autoFit = false,
  minItemWidth = '280px',
}: ResponsiveGridProps) {
  const gridClasses = React.useMemo(() => {
    if (autoFit) {
      return `grid-cols-[repeat(auto-fit,minmax(${minItemWidth},1fr))]`;
    }

    const colClasses = [];
    if (cols.default) colClasses.push(`grid-cols-${cols.default}`);
    if (cols.sm) colClasses.push(`sm:grid-cols-${cols.sm}`);
    if (cols.md) colClasses.push(`md:grid-cols-${cols.md}`);
    if (cols.lg) colClasses.push(`lg:grid-cols-${cols.lg}`);
    if (cols.xl) colClasses.push(`xl:grid-cols-${cols.xl}`);
    if (cols['2xl']) colClasses.push(`2xl:grid-cols-${cols['2xl']}`);

    return colClasses.join(' ');
  }, [cols, autoFit, minItemWidth]);

  const gapClass = `gap-${gap}`;

  return (
    <div 
      className={cn(
        'grid',
        gridClasses,
        gapClass,
        className
      )}
    >
      {children}
    </div>
  );
}

// Responsive container with proper breakpoints
interface ResponsiveContainerProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  padding?: boolean;
}

export function ResponsiveContainer({
  children,
  className,
  size = 'xl',
  padding = true,
}: ResponsiveContainerProps) {
  const sizeClasses = {
    sm: 'max-w-2xl',
    md: 'max-w-4xl',
    lg: 'max-w-6xl',
    xl: 'max-w-7xl',
    full: 'max-w-full',
  };

  return (
    <div 
      className={cn(
        'mx-auto w-full',
        sizeClasses[size],
        padding && 'px-4 sm:px-6 lg:px-8',
        className
      )}
    >
      {children}
    </div>
  );
}

// Responsive spacing utilities
export const responsiveSpacing = {
  section: 'py-12 sm:py-16 lg:py-20',
  container: 'px-4 sm:px-6 lg:px-8',
  content: 'space-y-6 sm:space-y-8 lg:space-y-12',
  grid: 'gap-6 sm:gap-8 lg:gap-12',
  card: 'p-4 sm:p-6 lg:p-8',
};