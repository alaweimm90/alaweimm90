import React, { memo, useMemo } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';

interface OptimizedCardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'premium' | 'elegant' | 'glass';
  interactive?: boolean;
  loading?: boolean;
  onClick?: () => void;
}

const OptimizedCardComponent: React.FC<OptimizedCardProps> = ({
  title,
  subtitle,
  children,
  className,
  variant = 'default',
  interactive = false,
  loading = false,
  onClick
}) => {
  // Memoize the computed class names to prevent recalculation
  const cardClasses = useMemo(() => {
    const baseClasses = 'transition-all duration-300';
    const variantClasses = {
      default: 'bg-white border border-gray-200 shadow-sm hover:shadow-md',
      premium: 'bg-gradient-to-br from-white via-gray-50 to-white border border-gray-200 shadow-lg hover:shadow-xl',
      elegant: 'bg-white/95 backdrop-blur-sm border border-white/20 shadow-elegant hover:shadow-strong',
      glass: 'bg-white/10 backdrop-blur-xl border border-white/20 shadow-lg hover:shadow-xl'
    };
    
    const interactiveClasses = interactive ? 'cursor-pointer hover:scale-[1.02] hover:border-repz-primary/30' : '';
    
    return cn(
      baseClasses,
      variantClasses[variant],
      interactiveClasses,
      className
    );
  }, [variant, interactive, className]);

  // Memoize the header content to prevent unnecessary renders
  const headerContent = useMemo(() => {
    if (!title && !subtitle) return null;
    
    return (
      <CardHeader className="pb-4">
        {title && (
          <CardTitle className="text-lg font-semibold text-gray-900">
            {title}
          </CardTitle>
        )}
        {subtitle && (
          <p className="text-sm text-gray-600 mt-1">
            {subtitle}
          </p>
        )}
      </CardHeader>
    );
  }, [title, subtitle]);

  // Memoize loading content
  const loadingContent = useMemo(() => {
    if (!loading) return null;
    
    return (
      <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center rounded-lg">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-repz-primary border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600">Loading...</span>
        </div>
      </div>
    );
  }, [loading]);

  // Handle click events with performance optimization
  const handleClick = useMemo(() => {
    if (!onClick || !interactive) return undefined;
    
    // Throttle click events to prevent rapid-fire clicks
    let lastClick = 0;
    return () => {
      const now = Date.now();
      if (now - lastClick > 200) { // 200ms throttle
        lastClick = now;
        onClick();
      }
    };
  }, [onClick, interactive]);

  return (
    <Card 
      className={cardClasses}
      onClick={handleClick}
    >
      <div className="relative">
        {headerContent}
        <CardContent className="pt-0">
          {children}
        </CardContent>
        {loadingContent}
      </div>
    </Card>
  );
};

// Memoize the component to prevent unnecessary re-renders
export const OptimizedCard = memo(OptimizedCardComponent, (prevProps, nextProps) => {
  // Custom comparison function for better performance
  return (
    prevProps.title === nextProps.title &&
    prevProps.subtitle === nextProps.subtitle &&
    prevProps.className === nextProps.className &&
    prevProps.variant === nextProps.variant &&
    prevProps.interactive === nextProps.interactive &&
    prevProps.loading === nextProps.loading &&
    prevProps.onClick === nextProps.onClick &&
    // For children, we do a shallow comparison
    React.isValidElement(prevProps.children) &&
    React.isValidElement(nextProps.children) &&
    prevProps.children.key === nextProps.children.key
  );
});

OptimizedCard.displayName = 'OptimizedCard';