import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

/* WCAG 2.1 AA Compliant Button System with Design Tokens */
const buttonVariants = cva(
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "font-medium transition-all duration-[350ms] relative overflow-hidden",
    "focus-visible:outline-none focus-visible:ring-[var(--focus-ring)] focus-visible:ring-offset-[var(--focus-ring-offset)]",
    "disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed",
    "active:scale-[0.98] transform-gpu",
    "shadow-[var(--button-shadow)] hover:shadow-[var(--button-shadow-hover)] active:shadow-[var(--button-shadow-active)]",
    "[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 [&_svg]:aria-hidden",
    "before:absolute before:inset-0 before:bg-gradient-to-r before:from-transparent before:via-white/10 before:to-transparent",
    "before:translate-x-[-100%] hover:before:translate-x-[100%] before:transition-transform before:duration-700"
  ],
  {
    variants: {
      variant: {
        primary: [
          "bg-[var(--color-primary)] text-[var(--text-primary)]",
          "hover:bg-[var(--color-primary-hover)]",
          "active:bg-[var(--color-primary-active)]",
          "disabled:bg-[var(--color-primary-disabled)]",
          "rounded-[var(--radius-lg)]"
        ],
        secondary: [
          "bg-[var(--color-secondary)] text-[var(--text-primary)]",
          "hover:bg-[var(--color-secondary-hover)]", 
          "active:bg-[var(--color-secondary-active)]",
          "rounded-[var(--radius-lg)]"
        ],
        ghost: [
          "bg-transparent text-[var(--text-secondary)]",
          "border border-[var(--border-muted)]",
          "hover:bg-[var(--surface-2)] hover:text-[var(--text-primary)]",
          "hover:border-[var(--border-secondary)]",
          "rounded-[var(--radius-md)]"
        ],
        outline: [
          "bg-transparent text-[var(--color-primary)]",
          "border-2 border-[var(--color-primary)]",
          "hover:bg-[var(--color-primary)] hover:text-[var(--text-primary)]",
          "rounded-[var(--radius-md)]"
        ],
        destructive: [
          "bg-[var(--color-danger)] text-[var(--text-primary)]",
          "hover:bg-[var(--color-danger-hover)]",
          "active:bg-[var(--color-danger-active)]",
          "focus-visible:ring-[var(--focus-ring-danger)]",
          "rounded-[var(--radius-lg)]"
        ],
        quantum: [
          "bg-[var(--glass-bg)] text-[var(--text-primary)] glass-panel",
          "border border-[var(--glass-border)]",
          "backdrop-filter:var(--glass-backdrop-filter)",
          "hover:bg-[var(--glass-bg-subtle)] hover:border-[var(--glass-border-strong)]",
          "hover:shadow-[var(--shadow-glass)] hover:-translate-y-0.5",
          "active:translate-y-0 active:shadow-[var(--shadow-elevated)]",
          "rounded-[var(--radius-xl)] glass-interactive"
        ],
        link: [
          "bg-transparent text-[var(--color-primary)] underline-offset-4",
          "hover:underline hover:text-[var(--color-primary-hover)]",
          "rounded-[var(--radius-sm)]"
        ],
        success: [
          "bg-[var(--color-success)] text-[var(--text-primary)]",
          "hover:bg-[var(--color-success-hover)]",
          "active:bg-[var(--color-success-active)]",
          "rounded-[var(--radius-lg)]"
        ],
        warning: [
          "bg-[var(--color-warning)] text-[var(--text-inverse)]",
          "hover:bg-[var(--color-warning-hover)]", 
          "active:bg-[var(--color-warning-active)]",
          "rounded-[var(--radius-lg)]"
        ]
      },
      size: {
        sm: "h-[var(--button-height-sm)] px-[var(--button-padding-x-sm)] text-[var(--font-size-sm)] min-h-[36px]",
        md: "h-[var(--button-height-md)] px-[var(--button-padding-x-md)] text-[var(--font-size-base)] min-h-[44px] min-w-[44px]",
        lg: "h-[var(--button-height-lg)] px-[var(--button-padding-x-lg)] text-[var(--font-size-lg)] min-h-[48px] min-w-[48px]",
        icon: "h-[var(--button-height-md)] w-[var(--button-height-md)] p-0 min-h-[44px] min-w-[44px]",
        touch: "h-12 w-12 p-0 min-h-[48px] min-w-[48px]" // 48px minimum touch target (WCAG 2.1 AA)
      }
    },
    defaultVariants: {
      variant: "primary",
      size: "md"
    }
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  /**
   * Loading state - shows spinner and disables interaction
   */
  loading?: boolean
  /**
   * Accessible label - required for icon-only buttons
   */
  "aria-label"?: string
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, loading, children, disabled, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    
    return (
      <Comp
        className={cn(buttonVariants({ variant, size }), className)}
        ref={ref}
        disabled={disabled || loading}
        aria-disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg
            className="mr-2 h-4 w-4 animate-spin"
            aria-hidden="true"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="2"
              className="opacity-25"
            />
            <path
              fill="currentColor"
              d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              className="opacity-75"
            />
          </svg>
        )}
        {children}
      </Comp>
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
