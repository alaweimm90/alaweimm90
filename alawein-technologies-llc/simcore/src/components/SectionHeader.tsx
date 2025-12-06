import React from 'react';
import { cn } from '@/lib/utils';

interface SectionHeaderProps {
  id?: string;
  title: string;
  subtitle?: string;
  className?: string;
  gradient?: {
    start?: string; // legacy support
    mid?: string;
    end?: string;
  };
  variant?: 'quantum' | 'field' | 'energy' | 'materials';
  styleType?: 'panel' | 'subsection';
  eyebrow?: string;
  eyebrowClassName?: string;
  as?: 'h1' | 'h2' | 'h3';
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  id,
  title,
  subtitle,
  className = '',
  gradient = { start: 'blue-400', mid: 'purple-400', end: 'blue-500' },
  variant,
  styleType = 'panel',
  eyebrow,
  eyebrowClassName,
  as,
}) => {
  const HeadingTag: React.ElementType = as ?? (styleType === 'panel' ? 'h2' : 'h3');
  return (
    <div className={cn('mb-16', className)}>
      {styleType === 'panel' ? (
        <div className="relative card-surface-glass rounded-3xl shadow-2xl p-8 md:p-12 text-center overflow-hidden">
          <div
            aria-hidden
            className={cn('pointer-events-none absolute inset-0 opacity-30',
              variant === 'field' ? 'glow-overlay-field' : 'glow-overlay-quantum'
            )}
          />
          <div className="relative text-center">
            {eyebrow && (
              <div className={cn(
                'mb-3 inline-block px-3 py-1 rounded-full text-[11px] tracking-wide uppercase border border-border/60 bg-card/60 backdrop-blur-sm text-muted-foreground transition-colors duration-300',
                eyebrowClassName
              )}>
                {eyebrow}
              </div>
            )}
            <HeadingTag
              id={id}
              className={cn(
                'text-[clamp(28px,4vw,48px)] font-bold mb-4 bg-clip-text text-transparent transition-transform duration-300 hover:-translate-y-0.5',
                (variant === 'field' || variant === 'energy')
                  ? 'gradient-text-field'
                  : 'gradient-text-quantum',
              )}
            >
              {title}
            </HeadingTag>
            {subtitle && (
              <p
                className={cn(
                  'text-lg font-medium bg-clip-text text-transparent transition-opacity duration-300 hover:opacity-95',
                  (variant === 'field' || variant === 'energy') ? 'gradient-text-field' : 'gradient-text-quantum'
                )}
              >
                {subtitle}
              </p>
            )}
          </div>
        </div>
      ) : (
        <div className="text-center">
          {eyebrow && (
            <div className={cn(
              'mb-2 inline-block px-3 py-1 rounded-full text-[11px] tracking-wide uppercase border border-border/60 bg-card/60 backdrop-blur-sm text-muted-foreground transition-colors duration-300',
              eyebrowClassName
            )}>
              {eyebrow}
            </div>
          )}
          <h3
            id={id}
            className={cn(
              'text-[clamp(22px,3vw,36px)] font-bold mb-2 bg-clip-text text-transparent transition-transform duration-300 hover:-translate-y-0.5',
              (variant === 'field' || variant === 'energy') ? 'gradient-text-field' : 'gradient-text-quantum'
            )}
          >
            {title}
          </h3>
          {subtitle && (
            <p className={cn(
              'text-base md:text-lg font-medium bg-clip-text text-transparent transition-opacity duration-300 hover:opacity-95',
              (variant === 'field' || variant === 'energy') ? 'gradient-text-field' : 'gradient-text-quantum'
            )}>
              {subtitle}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default SectionHeader;
