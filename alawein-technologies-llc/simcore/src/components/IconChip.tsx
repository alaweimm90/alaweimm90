import React from 'react';
import { cn } from '@/lib/utils';

export interface IconChipProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'quantum' | 'field' | 'energy' | 'neutral';
  'aria-label'?: string;
}

export const IconChip: React.FC<IconChipProps> = ({ children, className, variant = 'quantum', ...props }) => {
  const variantRing =
    variant === 'quantum'
      ? 'ring-[hsl(var(--semantic-domain-quantum)/0.35)]'
      : variant === 'field'
      ? 'ring-[hsl(var(--semantic-domain-field)/0.35)]'
      : variant === 'energy'
      ? 'ring-[hsl(var(--semantic-domain-energy)/0.35)]'
      : 'ring-muted';

  return (
    <div
      className={cn(
        'icon-chip-quantum inline-flex items-center justify-center rounded-xl border border-muted bg-surfaceMuted/60 p-2.5 shadow-sm backdrop-blur-sm transition-all duration-300 hover:shadow-glow focus-visible:ring-2',
        variantRing,
        className
      )}
      aria-hidden={props['aria-label'] ? undefined : true}
      {...props}
    >
      {children}
    </div>
  );
};

export default IconChip;
