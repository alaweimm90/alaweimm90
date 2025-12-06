import React, { memo, useMemo } from 'react';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";

// Memoized components for better performance
export const MemoizedButton = memo(Button);
export const MemoizedCard = memo(Card);
export const MemoizedBadge = memo(Badge);

// Performance-optimized list renderer
interface OptimizedListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T, index: number) => string | number;
  className?: string;
}

export const OptimizedList = memo(<T,>({ 
  items, 
  renderItem, 
  keyExtractor, 
  className = "" 
}: OptimizedListProps<T>) => {
  const renderedItems = useMemo(() => 
    items.map((item, index) => (
      <div key={keyExtractor(item, index)}>
        {renderItem(item, index)}
      </div>
    )), 
    [items, renderItem, keyExtractor]
  );

  return <div className={className}>{renderedItems}</div>;
});

// Optimized grid component
interface OptimizedGridProps {
  children: React.ReactNode;
  cols?: {
    mobile?: number;
    tablet?: number;
    desktop?: number;
  };
  gap?: string;
  className?: string;
}

export const OptimizedGrid = memo(({ 
  children, 
  cols = { mobile: 1, tablet: 2, desktop: 3 },
  gap = "gap-6",
  className = ""
}: OptimizedGridProps) => {
  const gridClasses = useMemo(() => 
    `grid grid-cols-${cols.mobile} sm:grid-cols-${cols.tablet} lg:grid-cols-${cols.desktop} ${gap} ${className}`,
    [cols, gap, className]
  );

  return <div className={gridClasses}>{children}</div>;
});

// Performance-optimized section wrapper
interface OptimizedSectionProps {
  children: React.ReactNode;
  className?: string;
  background?: string;
  padding?: string;
}

export const OptimizedSection = memo(({ 
  children, 
  className = "",
  background = "bg-background",
  padding = "py-12"
}: OptimizedSectionProps) => {
  const sectionClasses = useMemo(() => 
    `${background} ${padding} ${className}`,
    [background, padding, className]
  );

  return <section className={sectionClasses}>{children}</section>;
});

OptimizedList.displayName = 'OptimizedList';
OptimizedGrid.displayName = 'OptimizedGrid'; 
OptimizedSection.displayName = 'OptimizedSection';