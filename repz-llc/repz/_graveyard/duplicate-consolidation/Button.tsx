import React from 'react';
import { cn } from '@/lib/utils';

/**
 * Atomic Design - Button Atom
 * Consumes design tokens exclusively for consistent styling
 * No hard-coded values - all styling derived from tokens
 */

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'tertiary' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({
  className,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  children,
  ...props
}, ref) => {
  return (
    <button
      className={cn(
        // Base styles using design tokens
        'repz-button',
        'repz-button--' + variant,
        'repz-button--' + size,
        loading && 'repz-button--loading',
        disabled && 'repz-button--disabled',
        className
      )}
      disabled={disabled || loading}
      ref={ref}
      {...props}
    >
      {loading && (
        <span className="repz-button__spinner" aria-hidden="true">
          <svg className="repz-button__spinner-icon" viewBox="0 0 24 24">
            <circle
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
              strokeLinecap="round"
              strokeDasharray="32"
              strokeDashoffset="32"
            >
              <animate
                attributeName="stroke-dasharray"
                dur="2s"
                values="0 32;16 16;0 32;0 32"
                repeatCount="indefinite"
              />
              <animate
                attributeName="stroke-dashoffset"
                dur="2s"
                values="0;-16;-32;-32"
                repeatCount="indefinite"
              />
            </circle>
          </svg>
        </span>
      )}
      <span className={cn('repz-button__content', loading && 'repz-button__content--loading')}>
        {children}
      </span>
    </button>
  );
});

Button.displayName = 'Button';