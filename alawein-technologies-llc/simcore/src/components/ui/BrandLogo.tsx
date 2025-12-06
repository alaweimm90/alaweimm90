import React from 'react';
import { Atom } from 'lucide-react';
import { cn } from '@/lib/utils';

export type BrandLogoProps = {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  withIcon?: boolean;
  inline?: boolean; // if true, renders inline-friendly span
  className?: string;
  ariaLabel?: string;
  wordmarkOnly?: boolean; // alias for withIcon={false}
  effect?: 'sheen' | 'rgb' | 'bands' | 'interference' | 'glitch' | 'crt' | 'holo'; // wordmark effect variant
  gridDensity?: 'tight' | 'normal' | 'loose'; // pixel grid density
  intensity?: 'subtle' | 'medium' | 'vivid'; // effect intensity
};

const sizeMap: Record<NonNullable<BrandLogoProps['size']>, string> = {
  xs: 'text-sm',
  sm: 'text-base',
  md: 'text-xl',
  lg: 'text-3xl',
  xl: 'text-6xl',
};

const chipSize: Record<NonNullable<BrandLogoProps['size']>, string> = {
  xs: 'w-4 h-4 rounded-md',
  sm: 'w-5 h-5 rounded-lg',
  md: 'w-6 h-6 rounded-lg',
  lg: 'w-8 h-8 rounded-xl',
  xl: 'w-12 h-12 rounded-2xl',
};

export const BrandLogo: React.FC<BrandLogoProps> = ({
  size = 'md',
  withIcon = true,
  inline = true,
  className,
  ariaLabel = 'Simulation Core',
  wordmarkOnly,
  effect = 'interference',
  gridDensity = 'normal',
  intensity = 'medium',
}) => {
  const showIcon = wordmarkOnly ? false : withIcon;
  const Container: any = inline ? 'span' : 'div';
  const effectClass =
    effect === 'rgb' ? 'brand-tech-text-rgb' :
    effect === 'bands' || effect === 'interference' ? 'brand-tech-text-bands' :
    effect === 'glitch' ? 'brand-tech-text-glitch' :
    effect === 'crt' ? 'brand-tech-text-crt' :
    effect === 'holo' ? 'brand-tech-text-holo' :
    'brand-tech-text';
  const densityClass = `brand-grid-${gridDensity}`;
  const intensityClass = `brand-intensity-${intensity}`;
  return (
    <Container
      className={cn('brand-logo flex items-center gap-2 select-none', inline && 'inline-flex', className)}
      aria-label={ariaLabel}
      title={ariaLabel}
    >
      {showIcon && (
        <span
          className={cn('shrink-0 relative overflow-hidden flex items-center justify-center border shadow-sm brand-sheen icon-chip-brand', chipSize[size])}
          style={{
            background: 'var(--brand-gradient)',
            boxShadow: '0 12px 26px hsl(var(--brand-gradient-end) / 0.35), inset 0 1px 0 hsla(0,0%,100%,0.12)',
            borderColor: 'hsl(var(--brand-ink) / 0.18)'
          }}
          aria-hidden
        >
          <span
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(120% 100% at -20% -20%, hsla(0,0%,100%,0.5) 0%, hsla(0,0%,100%,0) 55%)'
            }}
          />
          <Atom className={cn({
            'w-3.5 h-3.5': size === 'xs',
            'w-4 h-4': size === 'sm',
            'w-5 h-5': size === 'md',
            'w-6 h-6': size === 'lg',
            'w-8 h-8': size === 'xl',
          })} color="hsl(var(--brand-ink))" strokeWidth={2.4} />
        </span>
      )}
      <span className="relative inline-block">
        <span
          className="absolute inset-[-8%] -z-10 pointer-events-none"
          style={{
            background: 'radial-gradient(60% 60% at 50% 45%, hsl(var(--brand-primary-glow) / 0.35) 0%, hsla(0,0%,100%,0.05) 15%, transparent 60%)',
            filter: 'blur(6px)'
          }}
        />
        <span
          className={cn('brand-logo-text font-semibold tracking-tight bg-clip-text text-transparent brand-sheen', effectClass, densityClass, intensityClass, sizeMap[size])}
          style={{
            filter: 'drop-shadow(0 3px 18px hsl(var(--brand-primary-glow) / 0.45))',
            fontFamily: 'var(--font-heading)'
          }}
        >
          SimCore
        </span>
      </span>
    </Container>
  );
};

export default BrandLogo;
