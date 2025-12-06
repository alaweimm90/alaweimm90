import React from 'react';
import { useResponsive } from '@/hooks/use-responsive';
import { cn } from '@/lib/utils';

interface ResponsiveGridProps {
  children: React.ReactNode;
  variant?: 'modules' | 'simulation' | 'compact' | 'wide';
  className?: string;
  itemMinWidth?: string;
  gap?: 'sm' | 'md' | 'lg' | 'xl';
  fixedColumns?: number; // when provided, enforce fixed columns (e.g., 3 on desktop)
}

export const EnhancedResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  children,
  variant = 'modules',
  className = '',
  itemMinWidth,
  gap = 'md',
  fixedColumns
}) => {
  const { isMobile, isTablet } = useResponsive();

  // Grid configuration based on variant
  const gridConfigs = {
    modules: {
      mobile: 'grid-cols-1',
      tablet: 'sm:grid-cols-2',
      desktop: 'lg:grid-cols-3',
      minWidth: itemMinWidth || '320px'
    },
    simulation: {
      mobile: 'grid-cols-1',
      tablet: 'md:grid-cols-2',
      desktop: 'lg:grid-cols-2',
      minWidth: itemMinWidth || '320px'
    },
    compact: {
      mobile: 'grid-cols-2',
      tablet: 'sm:grid-cols-3',
      desktop: 'lg:grid-cols-4 xl:grid-cols-6',
      minWidth: itemMinWidth || '200px'
    },
    wide: {
      mobile: 'grid-cols-1',
      tablet: 'sm:grid-cols-1',
      desktop: 'lg:grid-cols-2',
      minWidth: itemMinWidth || '400px'
    }
  };

  // Gap configurations
  const gapConfigs = {
    sm: 'gap-2 sm:gap-3',
    md: 'gap-4 sm:gap-6',
    lg: 'gap-6 sm:gap-8',
    xl: 'gap-8 sm:gap-10'
  };

  const config = gridConfigs[variant];
  const gapClasses = gapConfigs[gap];

  // Use CSS Grid with auto-fit unless fixedColumns is requested
  const gridStyle = fixedColumns
    ? undefined
    : {
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${config.minWidth}), 1fr))`,
      };

  const fixedColsClasses = fixedColumns
    ? (fixedColumns === 3
        ? 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'
        : `grid-cols-${fixedColumns}`)
    : '';

  return (
    <div
      className={cn(
        'grid w-full',
        // Use traditional responsive classes as fallback
        config.mobile,
        config.tablet,
        config.desktop,
        fixedColsClasses,
        gapClasses,
        className
      )}
      style={gridStyle}
    >
      {children}
    </div>
  );
};

// Specialized grids for different use cases
export const ModuleGrid: React.FC<Omit<ResponsiveGridProps, 'variant'>> = (props) => (
  <EnhancedResponsiveGrid {...props} variant="modules" />
);

export const SimulationGrid: React.FC<Omit<ResponsiveGridProps, 'variant'>> = (props) => (
  <EnhancedResponsiveGrid {...props} variant="simulation" />
);

export const CompactGrid: React.FC<Omit<ResponsiveGridProps, 'variant'>> = (props) => (
  <EnhancedResponsiveGrid {...props} variant="compact" />
);

// Responsive card container with aspect ratio preservation
interface ResponsiveCardContainerProps {
  children: React.ReactNode;
  aspectRatio?: 'square' | 'video' | 'golden' | 'auto';
  minHeight?: string;
  className?: string;
}

export const ResponsiveCardContainer: React.FC<ResponsiveCardContainerProps> = ({
  children,
  aspectRatio = 'auto',
  minHeight,
  className = ''
}) => {
  const { isMobile } = useResponsive();

  const aspectRatioClasses = {
    square: 'aspect-square',
    video: 'aspect-video',
    golden: 'aspect-[1.618/1]',
    auto: ''
  };

  const mobileMinHeight = isMobile ? 'min-h-[240px]' : 'min-h-[280px]';
  const customMinHeight = minHeight ? `min-h-[${minHeight}]` : mobileMinHeight;

  return (
    <div
      className={cn(
        'w-full',
        aspectRatio !== 'auto' ? aspectRatioClasses[aspectRatio] : customMinHeight,
        'rounded-lg overflow-hidden',
        className
      )}
    >
      {children}
    </div>
  );
};

// Layout wrapper for responsive simulation pages
interface ResponsiveSimulationLayoutProps {
  sidebar: React.ReactNode;
  main: React.ReactNode;
  sidebarWidth?: string;
  className?: string;
}

export const ResponsiveSimulationLayout: React.FC<ResponsiveSimulationLayoutProps> = ({
  sidebar,
  main,
  sidebarWidth = 'lg:w-80 xl:w-96',
  className = ''
}) => {
  const { isMobile } = useResponsive();

  if (isMobile) {
    return (
      <div className={cn('flex flex-col gap-4', className)}>
        <div className="w-full">{sidebar}</div>
        <div className="w-full">{main}</div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col lg:flex-row gap-4 sm:gap-6', className)}>
      <div className={cn('w-full', sidebarWidth, 'flex-shrink-0')}>
        {sidebar}
      </div>
      <div className="flex-1 min-w-0">
        {main}
      </div>
    </div>
  );
};