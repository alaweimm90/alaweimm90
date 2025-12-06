import * as React from "react"

import { cn } from "@/lib/utils"

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  domain?: 'quantum' | 'statistical' | 'energy' | 'fields';
  variant?: 'default' | 'physics' | 'enhanced';
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, domain, variant = 'default', style, ...props }, ref) => {
    const cardStyle = {
      ...(variant === 'physics' && {
        background: 'var(--component-card-background)',
        border: '1px solid var(--component-card-border)',
        backdropFilter: 'blur(20px)',
        ...(domain && {
          '--component-card-border': `hsla(var(--semantic-domain-${domain}), 0.2)`,
          '--component-card-shadow': domain === 'quantum' ? 'var(--primitive-shadow-quantum)' : `0 0 30px hsl(var(--semantic-domain-${domain}) / 0.25)`
        })
      }),
      ...style
    };

    return (
      <div
        ref={ref}
        className={cn(
          "rounded-lg border bg-card text-card-foreground shadow-sm",
          variant === 'physics' && "transition-all duration-300 hover:shadow-[var(--component-card-hover-glow)]",
          className
        )}
        style={cardStyle}
        data-domain={domain}
        {...props}
      />
    );
  }
)
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }
