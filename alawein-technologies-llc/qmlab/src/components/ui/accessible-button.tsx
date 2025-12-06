import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

/* Enhanced Accessible Button Component with WCAG 2.1 AA Compliance */
const accessibleButtonVariants = cva(
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "font-medium transition-all duration-300 relative overflow-hidden",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2",
    "disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed",
    "active:scale-[0.98] transform-gpu",
    "min-h-[44px] min-w-[44px]", // WCAG 2.1 AA minimum touch target
    "[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 [&_svg]:aria-hidden",
    "before:absolute before:inset-0 before:bg-gradient-to-r before:from-transparent before:via-white/10 before:to-transparent",
    "before:translate-x-[-100%] hover:before:translate-x-[100%] before:transition-transform before:duration-700"
  ],
  {
    variants: {
      variant: {
        primary: [
          "bg-blue-600 text-white",
          "hover:bg-blue-700",
          "active:bg-blue-800",
          "disabled:bg-blue-300",
          "shadow-md hover:shadow-lg active:shadow-sm",
          "rounded-lg"
        ],
        secondary: [
          "bg-slate-600 text-white",
          "hover:bg-slate-700", 
          "active:bg-slate-800",
          "shadow-md hover:shadow-lg active:shadow-sm",
          "rounded-lg"
        ],
        outline: [
          "bg-transparent text-blue-600",
          "border-2 border-blue-600",
          "hover:bg-blue-600 hover:text-white",
          "active:bg-blue-700 active:border-blue-700",
          "rounded-lg"
        ],
        ghost: [
          "bg-transparent text-slate-700",
          "border border-slate-300",
          "hover:bg-slate-100 hover:text-slate-900",
          "hover:border-slate-400",
          "active:bg-slate-200",
          "rounded-md"
        ],
        destructive: [
          "bg-red-600 text-white",
          "hover:bg-red-700",
          "active:bg-red-800",
          "focus-visible:ring-red-400",
          "shadow-md hover:shadow-lg active:shadow-sm",
          "rounded-lg"
        ],
        success: [
          "bg-green-600 text-white",
          "hover:bg-green-700",
          "active:bg-green-800",
          "focus-visible:ring-green-400",
          "shadow-md hover:shadow-lg active:shadow-sm",
          "rounded-lg"
        ],
        warning: [
          "bg-yellow-600 text-white",
          "hover:bg-yellow-700", 
          "active:bg-yellow-800",
          "focus-visible:ring-yellow-400",
          "shadow-md hover:shadow-lg active:shadow-sm",
          "rounded-lg"
        ],
        quantum: [
          "bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 text-white",
          "hover:from-blue-700 hover:via-purple-700 hover:to-indigo-700",
          "active:from-blue-800 active:via-purple-800 active:to-indigo-800",
          "focus-visible:ring-purple-400",
          "shadow-lg hover:shadow-xl active:shadow-md",
          "rounded-xl"
        ]
      },
      size: {
        sm: "h-9 px-3 text-sm",
        md: "h-11 px-4 text-base",
        lg: "h-12 px-6 text-lg",
        icon: "h-11 w-11 p-0",
        touch: "h-12 w-12 p-0" // Enhanced touch target
      }
    },
    defaultVariants: {
      variant: "primary",
      size: "md"
    }
  }
)

export interface AccessibleButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof accessibleButtonVariants> {
  asChild?: boolean
  /**
   * Loading state - shows spinner and disables interaction
   */
  loading?: boolean
  /**
   * Required accessible label for icon-only buttons
   */
  "aria-label"?: string
  /**
   * Pressed state for toggle buttons
   */
  "aria-pressed"?: boolean
  /**
   * Expanded state for collapsible triggers
   */
  "aria-expanded"?: boolean
  /**
   * Controls relationship for ARIA
   */
  "aria-controls"?: string
}

const AccessibleButton = React.forwardRef<HTMLButtonElement, AccessibleButtonProps>(
  ({ className, variant, size, asChild = false, loading, children, disabled, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    
    return (
      <Comp
        className={cn(accessibleButtonVariants({ variant, size }), className)}
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
AccessibleButton.displayName = "AccessibleButton"

export { AccessibleButton, accessibleButtonVariants }