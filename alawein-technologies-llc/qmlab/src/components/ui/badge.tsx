import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

/* WCAG 2.1 AA Compliant Badge System */
const badgeVariants = cva(
  [
    "inline-flex items-center justify-center gap-1",
    "px-[var(--space-2)] py-[var(--space-1)]",
    "text-[var(--font-size-xs)] font-[var(--font-weight-medium)]",
    "rounded-[var(--radius-sm)]",
    "border border-transparent",
    "transition-all duration-[var(--transition-normal)]",
    "whitespace-nowrap"
  ],
  {
    variants: {
      variant: {
        info: [
          "bg-[var(--color-info-subtle)] text-[var(--color-info)]",
          "border-[var(--color-info)]/20"
        ],
        success: [
          "bg-[var(--color-success-subtle)] text-[var(--color-success)]",
          "border-[var(--color-success)]/20"
        ],
        warning: [
          "bg-[var(--color-warning-subtle)] text-[var(--color-warning)]",
          "border-[var(--color-warning)]/20"
        ],
        danger: [
          "bg-[var(--color-danger-subtle)] text-[var(--color-danger)]",
          "border-[var(--color-danger)]/20"
        ],
        neutral: [
          "bg-[var(--surface-2)] text-[var(--text-secondary)]",
          "border-[var(--border-muted)]"
        ],
        primary: [
          "bg-[var(--color-primary-subtle)] text-[var(--color-primary)]",
          "border-[var(--color-primary)]/20"
        ],
        secondary: [
          "bg-[var(--color-secondary-subtle)] text-[var(--color-secondary)]",
          "border-[var(--color-secondary)]/20"
        ],
        quantum: [
          "bg-[var(--glass-bg-subtle)] text-[var(--text-primary)]",
          "border-[var(--glass-border-strong)]",
          "backdrop-blur-[var(--glass-blur-subtle)]"
        ],
        default: [
          "bg-[var(--color-primary)] text-[var(--text-primary)]",
          "border-[var(--color-primary)]/20"
        ],
        destructive: [
          "bg-[var(--color-danger)] text-[var(--text-primary)]",
          "border-[var(--color-danger)]/20"
        ],
        outline: [
          "bg-transparent text-[var(--text-primary)]",
          "border-[var(--border-primary)]"
        ]
      },
      size: {
        sm: "px-[var(--space-1)] py-0.5 text-[10px]",
        md: "px-[var(--space-2)] py-[var(--space-1)] text-[var(--font-size-xs)]",
        lg: "px-[var(--space-3)] py-[var(--space-2)] text-[var(--font-size-sm)]"
      }
    },
    defaultVariants: {
      variant: "neutral",
      size: "md"
    }
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  /**
   * Optional icon before the text
   */
  icon?: React.ReactNode
  /**
   * Optional close button
   */
  onClose?: () => void
}

const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant, size, icon, onClose, children, ...props }, ref) => {
    return (
      <div
        className={cn(badgeVariants({ variant, size }), className)}
        ref={ref}
        {...props}
      >
        {icon && (
          <span className="shrink-0" aria-hidden="true">
            {icon}
          </span>
        )}
        <span>{children}</span>
        {onClose && (
          <button
            onClick={onClose}
            className="ml-1 shrink-0 rounded-sm opacity-70 hover:opacity-100 focus:outline-none focus:ring-1 focus:ring-current"
            aria-label="Remove badge"
          >
            <svg
              className="h-3 w-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}
      </div>
    )
  }
)
Badge.displayName = "Badge"

export { Badge, badgeVariants }
