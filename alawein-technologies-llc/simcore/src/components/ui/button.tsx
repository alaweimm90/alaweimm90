// Enhanced accessible button component with WCAG 2.1 AA compliance
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 touch-manipulation select-none cursor-pointer",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 focus:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 focus:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 focus:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline focus:underline",
        // Unified aliases (non-breaking) for consistent usage across the app
        primary: "bg-interactive text-white shadow-glow hover:brightness-110 focus:brightness-110", // alias of cta
        neutral: "text-textPrimary border border-muted hover:bg-surfaceMuted focus:bg-surfaceMuted backdrop-blur-sm transition-all duration-300", // alias of physics
        // Physics-themed variants using semantic tokens
        physics: "text-textPrimary border border-muted hover:bg-surfaceMuted focus:bg-surfaceMuted backdrop-blur-sm transition-all duration-300",
        quantum: "bg-accentQuantum text-white hover:bg-accentQuantum/90 shadow-glow",
        statistical: "bg-transparent text-accentStatistical border border-accentStatistical hover:bg-accentStatistical/10",
        energy: "bg-transparent text-accentEnergy border border-accentEnergy hover:bg-accentEnergy/10",
        fields: "bg-transparent text-accentField border border-accentField hover:bg-accentField/10",
        // CTA variants with semantic gradients
        cta: "bg-interactive text-white shadow-glow hover:brightness-110 focus:brightness-110",
        ctaField: "bg-accentField text-white shadow-elegant hover:brightness-110 focus:brightness-110",
        ctaQuiet: "bg-transparent text-textPrimary border border-muted hover:bg-surfaceMuted/50",
        tertiary: "bg-transparent text-textPrimary border border-muted hover:bg-surfaceMuted/50",
        // Unified CTA variants for site-wide usage
        hero: "bg-interactive text-white shadow-glow hover:brightness-110 focus:brightness-110",
        nav: "bg-transparent text-textPrimary border border-muted hover:bg-surfaceMuted/50",
        pill: "bg-transparent text-textPrimary border border-muted hover:bg-surfaceMuted/50 rounded-full"
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  isLoading?: boolean
  loadingText?: string
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant, 
    size, 
    asChild = false, 
    isLoading = false,
    loadingText,
    children,
    disabled,
    'aria-label': ariaLabel,
    ...props 
  }, ref) => {
    const Comp = asChild ? Slot : "button"
    
    // Enhanced accessibility attributes
    const accessibilityProps = {
      'aria-disabled': disabled || isLoading,
      'aria-busy': isLoading,
      'aria-label': ariaLabel || (typeof children === 'string' ? undefined : ariaLabel),
      'aria-describedby': props['aria-describedby'],
      role: asChild ? undefined : 'button',
      tabIndex: disabled ? -1 : props.tabIndex
    }

    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || isLoading}
        style={{ WebkitTapHighlightColor: 'transparent', touchAction: 'manipulation' }}
        {...accessibilityProps}
        {...props}
      >
        {isLoading && (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current" />
            <span className="sr-only">Loading...</span>
          </>
        )}
        {isLoading && loadingText ? loadingText : children}
      </Comp>
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }