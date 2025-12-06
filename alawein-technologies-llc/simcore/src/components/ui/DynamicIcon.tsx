import React, { lazy, Suspense } from 'react';
import type { LucideProps } from 'lucide-react';
import dynamicIconImports from 'lucide-react/dynamicIconImports';

// Dynamic Lucide Icon loader
// Usage: <DynamicIcon name="home" size={20} className="text-muted-foreground" />
// Loads icons on demand to keep initial bundle small.
export interface DynamicIconProps extends Omit<LucideProps, 'ref'> {
  name: keyof typeof dynamicIconImports;
  fallbackClassName?: string;
}

export const DynamicIcon: React.FC<DynamicIconProps> = ({ name, size = 24, className, fallbackClassName, ...rest }) => {
  const LucideIcon = lazy(dynamicIconImports[name]);
  const fallback = (
    <div
      className={fallbackClassName ?? 'bg-muted rounded'}
      style={{ width: size, height: size }}
      aria-hidden
    />
  );
  return (
    <Suspense fallback={fallback}>
      <LucideIcon size={size} className={className} {...rest} />
    </Suspense>
  );
};

export default DynamicIcon;
