import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-montserrat font-semibold ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 shadow-moderate hover:shadow-elegant hover:scale-105",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-strong hover:shadow-elegant",
        outline: "border-2 border-primary bg-card text-primary hover:bg-primary hover:text-primary-foreground hover:scale-105",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 shadow-moderate hover:shadow-elegant",
        ghost: "bg-transparent text-primary hover:bg-muted hover:text-primary",
        link: "text-primary underline-offset-4 hover:underline p-0 h-auto",
        premium: "bg-[var(--bg-primary)] text-foreground shadow-elegant hover:shadow-glow-orange hover:scale-105 border border-primary/20",
        hero: "bg-card/10 backdrop-blur-sm text-foreground border border-border/20 hover:bg-card/20 hover:scale-105 shadow-moderate",
        coaching: "bg-[var(--bg-accent)] text-foreground shadow-moderate hover:shadow-elegant hover:scale-105",
        elegant: "bg-[var(--surface-frost)] text-foreground border border-border/50 hover:shadow-elegant hover:scale-105",
        admin: "bg-destructive hover:bg-destructive/90 text-destructive-foreground shadow-strong hover:shadow-elegant hover:scale-105 font-bold",
        inverse: "bg-foreground text-background hover:bg-foreground/90 shadow-moderate hover:shadow-elegant hover:scale-105",
        accent: "bg-accent text-accent-foreground hover:bg-accent/90 shadow-moderate hover:shadow-glow-orange hover:scale-105"
      },
      size: {
        default: "h-11 sm:h-12 px-4 sm:px-6 py-2 sm:py-3 text-sm",
        sm: "h-9 sm:h-10 rounded-md px-3 sm:px-4 text-xs sm:text-sm",
        lg: "h-12 sm:h-14 rounded-xl px-6 sm:px-8 text-sm sm:text-base",
        xl: "h-14 sm:h-16 rounded-xl px-8 sm:px-10 text-base sm:text-lg",
        icon: "h-11 w-11 sm:h-12 sm:w-12",
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
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
