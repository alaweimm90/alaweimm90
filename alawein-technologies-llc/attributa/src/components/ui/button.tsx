import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-200",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-all duration-200",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground transition-all duration-200",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-all duration-200",
        ghost: "hover:bg-accent hover:text-accent-foreground transition-all duration-200",
        link: "text-primary underline-offset-4 hover:underline transition-all duration-200",
        hero: "bg-gradient-to-r from-primary to-accent text-primary-foreground hover:from-primary/90 hover:to-accent/90 shadow-lg hover:shadow-xl hover:scale-[1.02] transition-all duration-300 font-medium",
        "secondary-quiet": "bg-background/50 border border-border text-foreground hover:bg-background/80 hover:border-primary/50 transition-all duration-200",
      },
      size: {
        default: "h-10 px-4 py-2 min-w-[44px] md:min-w-0", /* Mobile touch target */
        sm: "h-9 rounded-md px-3 min-w-[44px] md:min-w-0",
        lg: "h-11 rounded-md px-8 min-w-[44px] md:min-w-0",
        icon: "h-10 w-10 min-w-[44px] min-h-[44px] md:min-w-[32px] md:min-h-[32px]",
        touch: "h-[44px] px-4 py-2 min-w-[44px]", /* Explicit touch-friendly size */
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
