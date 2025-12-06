import React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../lib/utils';

/**
 * Button component variants using CSS custom properties from design tokens
 * Follows REPZ design system specifications with enterprise-grade accessibility
 */
const buttonVariants = cva(
  [
    // Base styles using design tokens
    'inline-flex items-center justify-center gap-2',
    'whitespace-nowrap rounded-md text-sm font-medium',
    'transition-colors duration-200 ease-in-out',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
    'disabled:pointer-events-none disabled:opacity-50',
    'active:scale-[0.98] transform transition-transform',
    
    // Design token integration
    'font-family: var(--font-family-base)',
    'border-radius: var(--radius-md)',
    'transition: var(--transition-default)'
  ],
  {
    variants: {
      variant: {
        primary: [
          'bg-[var(--color-brand-primary)] text-[var(--color-text-on-primary)]',
          'hover:bg-[var(--color-brand-primary-hover)]',
          'focus-visible:ring-[var(--color-brand-primary)]',
          'shadow-[var(--shadow-sm)]',
          'hover:shadow-[var(--shadow-md)]'
        ],
        secondary: [
          'bg-[var(--color-background-secondary)] text-[var(--color-text-primary)]',
          'border border-[var(--color-border-primary)]',
          'hover:bg-[var(--color-background-tertiary)]',
          'focus-visible:ring-[var(--color-brand-primary)]'
        ],
        tertiary: [
          'bg-transparent text-[var(--color-text-primary)]',
          'hover:bg-[var(--color-background-secondary)]',
          'focus-visible:ring-[var(--color-brand-primary)]'
        ],
        destructive: [
          'bg-[var(--color-status-error)] text-[var(--color-text-on-error)]',
          'hover:bg-[var(--color-status-error-hover)]',
          'focus-visible:ring-[var(--color-status-error)]',
          'shadow-[var(--shadow-sm)]'
        ],
        outline: [
          'border border-[var(--color-border-primary)] bg-transparent',
          'text-[var(--color-text-primary)]',
          'hover:bg-[var(--color-background-secondary)]',
          'focus-visible:ring-[var(--color-brand-primary)]'
        ],
        ghost: [
          'bg-transparent text-[var(--color-text-primary)]',
          'hover:bg-[var(--color-background-secondary)]',
          'focus-visible:ring-[var(--color-brand-primary)]'
        ],
        link: [
          'bg-transparent text-[var(--color-brand-primary)]',
          'underline-offset-4 hover:underline',
          'focus-visible:ring-[var(--color-brand-primary)]'
        ]
      },
      size: {
        sm: [
          'h-9 px-3 text-xs',
          'min-width: var(--size-touch-target-sm)',
          'gap: var(--spacing-xs)'
        ],
        md: [
          'h-10 px-4 py-2',
          'min-width: var(--size-touch-target-md)',
          'gap: var(--spacing-sm)'
        ],
        lg: [
          'h-11 px-8 text-base',
          'min-width: var(--size-touch-target-lg)',
          'gap: var(--spacing-md)'
        ],
        xl: [
          'h-12 px-10 text-lg',
          'min-width: var(--size-touch-target-xl)',
          'gap: var(--spacing-md)'
        ],
        icon: [
          'h-10 w-10',
          'min-width: var(--size-touch-target-md)'
        ]
      },
      loading: {
        true: 'cursor-wait',
        false: ''
      }
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
      loading: false
    }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /**
   * Render as a different element while preserving button behavior
   */
  asChild?: boolean;
  
  /**
   * Loading state - shows spinner and disables interaction
   */
  loading?: boolean;
  
  /**
   * Icon to display before the button text
   */
  leftIcon?: React.ReactNode;
  
  /**
   * Icon to display after the button text
   */
  rightIcon?: React.ReactNode;
  
  /**
   * Full width button
   */
  fullWidth?: boolean;
}

/**
 * Button Component
 * 
 * Enterprise-grade button component with comprehensive variant support,
 * accessibility features, and integration with REPZ design tokens.
 * 
 * @example
 * ```tsx
 * // Primary button
 * <Button>Get Started</Button>
 * 
 * // Loading state
 * <Button loading>Processing...</Button>
 * 
 * // With icons
 * <Button leftIcon={<PlusIcon />}>Add Item</Button>
 * 
 * // Different variants and sizes
 * <Button variant="secondary" size="lg">Large Secondary</Button>
 * ```
 */
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      loading,
      leftIcon,
      rightIcon,
      fullWidth,
      asChild = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : 'button';
    
    // Handle loading state
    const isDisabled = disabled || loading;
    
    return (
      <Comp
        className={cn(
          buttonVariants({ variant, size, loading, className }),
          fullWidth && 'w-full'
        )}
        ref={ref}
        disabled={isDisabled}
        aria-busy={loading}
        {...props}
      >
        {loading && (
          <div className="mr-2">
            <svg
              className="animate-spin h-4 w-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          </div>
        )}
        
        {!loading && leftIcon && (
          <span className="flex-shrink-0" aria-hidden="true">
            {leftIcon}
          </span>
        )}
        
        {children}
        
        {!loading && rightIcon && (
          <span className="flex-shrink-0" aria-hidden="true">
            {rightIcon}
          </span>
        )}
      </Comp>
    );
  }
);

Button.displayName = 'Button';