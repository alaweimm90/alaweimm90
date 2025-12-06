/**
 * Responsive Physics Layout Components
 * 
 * Mobile-first responsive layout components designed for scientific simulations
 * with proper semantic tokens and accessibility features.
 */

import { ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useResponsive } from '@/hooks/use-responsive';
import { semanticColors, semanticSpacing, semanticShadows } from '@/theme/tokens';

interface ResponsiveControlPanelProps {
  children: ReactNode;
  title?: string;
  description?: string;
  className?: string;
  sticky?: boolean;
}

export function ResponsiveControlPanel({ 
  children, 
  title, 
  description, 
  className,
  sticky = true 
}: ResponsiveControlPanelProps) {
  const { isMobile, isTablet } = useResponsive();
  
  return (
    <Card 
      className={cn(
        "bg-surface/90 backdrop-blur-sm border-muted/30 shadow-elevation1",
        sticky && "lg:sticky lg:top-4",
        isMobile && "mb-4",
        className
      )}
    >
      {(title || description) && (
        <CardHeader className="pb-4">
          {title && (
            <CardTitle className="text-textPrimary text-lg sm:text-xl">
              {title}
            </CardTitle>
          )}
          {description && (
            <CardDescription className="text-textSecondary text-sm">
              {description}
            </CardDescription>
          )}
        </CardHeader>
      )}
      <CardContent className={cn(
        "space-y-4",
        isMobile && "px-4 py-3",
        isTablet && "px-5 py-4"
      )}>
        {children}
      </CardContent>
    </Card>
  );
}

interface ResponsiveVisualizationPanelProps {
  children: ReactNode;
  title?: string;
  description?: string;
  className?: string;
  height?: number | string;
}

export function ResponsiveVisualizationPanel({ 
  children, 
  title, 
  description, 
  className,
  height = "400px"
}: ResponsiveVisualizationPanelProps) {
  const { isMobile, isTablet } = useResponsive();
  
  return (
    <Card className={cn(
      "overflow-hidden bg-surface/90 backdrop-blur-sm border-muted/30 shadow-elevation1",
      className
    )}>
      {(title || description) && (
        <CardHeader className="pb-4">
          {title && (
            <CardTitle className="text-textPrimary text-lg sm:text-xl">
              {title}
            </CardTitle>
          )}
          {description && (
            <CardDescription className="text-textSecondary text-sm">
              {description}
            </CardDescription>
          )}
        </CardHeader>
      )}
      <CardContent className={cn(
        "p-2 sm:p-4 lg:p-6"
      )}>
        <div 
          className="w-full rounded-lg overflow-hidden"
          style={{ 
            height: isMobile ? "300px" : isTablet ? "350px" : height 
          }}
        >
          {children}
        </div>
      </CardContent>
    </Card>
  );
}

interface ResponsiveSimulationLayoutProps {
  children: ReactNode;
  controlPanel: ReactNode;
  visualization: ReactNode;
  plots?: ReactNode;
  className?: string;
}

export function ResponsiveSimulationLayout({ 
  children,
  controlPanel,
  visualization,
  plots,
  className 
}: ResponsiveSimulationLayoutProps) {
  const { isMobile, isTablet } = useResponsive();
  
  return (
    <div className={cn(
      "space-y-4 sm:space-y-6",
      className
    )}>
      {children}
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6">
        {/* Control Panel - Mobile First */}
        <div className="lg:col-span-1 order-1 lg:order-first">
          {controlPanel}
        </div>

        {/* Visualization Area - Mobile Responsive */}
        <div className="lg:col-span-3 space-y-4 sm:space-y-6">
          {visualization}
          {plots}
        </div>
      </div>
    </div>
  );
}

interface ResponsiveGridProps {
  children: ReactNode;
  columns?: {
    mobile?: number;
    tablet?: number;
    desktop?: number;
  };
  gap?: {
    mobile?: string;
    tablet?: string;
    desktop?: string;
  };
  className?: string;
}

export function ResponsiveGrid({ 
  children, 
  columns = { mobile: 1, tablet: 2, desktop: 3 },
  gap = { mobile: "1rem", tablet: "1.5rem", desktop: "2rem" },
  className 
}: ResponsiveGridProps) {
  const { isMobile, isTablet } = useResponsive();
  
  const gridCols = isMobile 
    ? `grid-cols-${columns.mobile}` 
    : isTablet 
    ? `grid-cols-${columns.tablet}` 
    : `grid-cols-${columns.desktop}`;
    
  const gridGap = isMobile 
    ? gap.mobile 
    : isTablet 
    ? gap.tablet 
    : gap.desktop;
  
  return (
    <div 
      className={cn("grid", gridCols, className)}
      style={{ gap: gridGap }}
    >
      {children}
    </div>
  );
}

interface ResponsivePlotContainerProps {
  children: ReactNode;
  title?: string;
  height?: {
    mobile?: number;
    tablet?: number;
    desktop?: number;
  };
  className?: string;
}

export function ResponsivePlotContainer({ 
  children, 
  title,
  height = { mobile: 300, tablet: 350, desktop: 400 },
  className 
}: ResponsivePlotContainerProps) {
  const { isMobile, isTablet } = useResponsive();
  
  const plotHeight = isMobile 
    ? height.mobile 
    : isTablet 
    ? height.tablet 
    : height.desktop;
  
  return (
    <Card className={cn(
      "bg-surface/90 backdrop-blur-sm border-muted/30 shadow-elevation1",
      className
    )}>
      {title && (
        <CardHeader className="pb-4">
          <CardTitle className="text-textPrimary text-lg sm:text-xl">
            {title}
          </CardTitle>
        </CardHeader>
      )}
      <CardContent className="p-2 sm:p-4">
        <div 
          className="w-full overflow-hidden"
          style={{ height: `${plotHeight}px` }}
        >
          {children}
        </div>
      </CardContent>
    </Card>
  );
}