import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-medium text-foreground transition-all duration-150 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:border-primary/55 focus-visible:shadow-[0_0_0_3px_color-mix(in_oklch,hsl(var(--focus-ring))_35%,transparent)] disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-danger",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
