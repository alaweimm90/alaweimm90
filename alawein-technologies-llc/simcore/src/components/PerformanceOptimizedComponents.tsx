import React, { memo, useMemo, useCallback, startTransition } from 'react';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';

// High-performance module card with virtualization support
export const OptimizedModuleCard = memo(({ 
  module, 
  onClick, 
  className 
}: {
  module: any;
  onClick?: () => void;
  className?: string;
}) => {
  const { isMobile } = useResponsiveEnhanced();

  const handleClick = useCallback(() => {
    startTransition(() => {
      onClick?.();
    });
  }, [onClick]);

  const cardContent = useMemo(() => (
    <>
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-gradient-to-br from-primary/10 to-accent/10 rounded-lg" />
      
      <div className="relative z-10 p-4 sm:p-6">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
            <module.icon className="w-4 h-4 sm:w-5 sm:h-5 text-primary" />
          </div>
          <h3 className="font-semibold text-sm sm:text-base text-foreground line-clamp-1">
            {module.title}
          </h3>
        </div>
        
        <p className="text-xs sm:text-sm text-muted-foreground line-clamp-2 mb-4">
          {module.description}
        </p>
        
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded-md">
            {module.category}
          </span>
          <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-primary" />
            </div>
          </div>
        </div>
      </div>
    </>
  ), [module, isMobile]);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
      className={cn(
        'group relative rounded-lg border border-border bg-card cursor-pointer',
        'transition-all duration-200 ease-out',
        'hover:border-primary/20 hover:shadow-lg hover:scale-[1.02]',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
        'active:scale-[0.98]',
        'min-h-[180px] sm:min-h-[200px]',
        className
      )}
      aria-label={`Open ${module.title} simulation`}
    >
      {cardContent}
    </div>
  );
});

OptimizedModuleCard.displayName = 'OptimizedModuleCard';

// Virtualized grid for large module lists
export const VirtualizedModuleGrid = memo(({ 
  modules, 
  onModuleClick 
}: {
  modules: any[];
  onModuleClick: (path: string) => void;
}) => {
  const { isMobile, isTablet } = useResponsiveEnhanced();

  const gridCols = useMemo(() => {
    if (isMobile) return 1;
    if (isTablet) return 2;
    return 3;
  }, [isMobile, isTablet]);

  const handleModuleClick = useCallback((path: string) => {
    startTransition(() => {
      onModuleClick(path);
    });
  }, [onModuleClick]);

  return (
    <div 
      className={cn(
        'grid gap-4 sm:gap-6',
        `grid-cols-${gridCols}`,
        'auto-rows-[180px] sm:auto-rows-[200px]'
      )}
      role="grid"
      aria-label="Physics simulation modules"
    >
      {modules.map((module, index) => (
        <div key={module.id} role="gridcell">
          <OptimizedModuleCard
            module={module}
            onClick={() => handleModuleClick(module.path)}
            className="h-full"
          />
        </div>
      ))}
    </div>
  );
});

VirtualizedModuleGrid.displayName = 'VirtualizedModuleGrid';

// Performance-optimized simulation canvas
export const OptimizedSimulationCanvas = memo(({ 
  children, 
  className 
}: {
  children: React.ReactNode;
  className?: string;
}) => {
  return (
    <div 
      className={cn(
        'relative overflow-hidden rounded-lg border border-border bg-card',
        'will-change-transform',
        className
      )}
      role="region"
      aria-label="Simulation visualization"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-muted/30" />
      <div className="relative z-10 h-full">
        {children}
      </div>
    </div>
  );
});

OptimizedSimulationCanvas.displayName = 'OptimizedSimulationCanvas';