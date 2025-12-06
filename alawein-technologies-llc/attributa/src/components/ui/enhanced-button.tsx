import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const enhancedButtonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground shadow hover:bg-primary/90 hover:shadow-elegant active:scale-[0.98]",
        destructive: "bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90 hover:shadow-elegant active:scale-[0.98]",
        outline: "border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground hover:border-accent/50 active:scale-[0.98]",
        secondary: "bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80 hover:shadow-elegant active:scale-[0.98]",
        ghost: "hover:bg-accent hover:text-accent-foreground active:scale-[0.98]",
        link: "text-primary underline-offset-4 hover:underline active:scale-[0.98]",
        premium: "bg-gradient-to-r from-primary to-accent text-foreground shadow-lg hover:shadow-glow hover:scale-[1.02] active:scale-[0.98] relative overflow-hidden",
        hero: "bg-foreground/10 text-foreground border border-foreground/20 hover:bg-foreground/20 hover:shadow-glow backdrop-blur-sm active:scale-[0.98]",
        floating: "bg-background/90 border border-border/30 shadow-lg backdrop-blur-sm hover:shadow-elegant hover:-translate-y-1 active:scale-[0.98]",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        xl: "h-12 rounded-lg px-10 text-base",
        touch: "h-12 px-6 min-w-[48px]", // WCAG touch target
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface EnhancedButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof enhancedButtonVariants> {
  asChild?: boolean;
  loading?: boolean;
  ripple?: boolean;
}

const EnhancedButton = React.forwardRef<HTMLButtonElement, EnhancedButtonProps>(
  ({ className, variant, size, asChild = false, loading, ripple = true, children, onClick, ...props }, ref) => {
    const [ripples, setRipples] = React.useState<Array<{ x: number; y: number; id: number }>>([]);
    const rippleId = React.useRef(0);

    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
      if (ripple && !loading) {
        const rect = event.currentTarget.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        const newRipple = { x, y, id: rippleId.current++ };
        setRipples(prev => [...prev, newRipple]);
        
        setTimeout(() => {
          setRipples(prev => prev.filter(r => r.id !== newRipple.id));
        }, 600);
      }
      
      onClick?.(event);
    };

    const Comp = asChild ? Slot : "button";
    
    return (
      <Comp
        className={cn(enhancedButtonVariants({ variant, size, className }), "relative overflow-hidden")}
        ref={ref}
        onClick={handleClick}
        disabled={loading}
        {...props}
      >
        {ripples.map((ripple) => (
          <span
            key={ripple.id}
            className="absolute rounded-full bg-foreground/30 animate-ping"
            style={{
              left: ripple.x - 10,
              top: ripple.y - 10,
              width: 20,
              height: 20,
            }}
          />
        ))}
        
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80">
            <div className="w-4 h-4 border-2 border-foreground/20 border-t-foreground rounded-full animate-spin" />
          </div>
        )}
        
        {children}
        
        {variant === "premium" && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-foreground/10 to-transparent translate-x-[-100%] hover:translate-x-[100%] transition-transform duration-700" />
        )}
      </Comp>
    );
  }
);

EnhancedButton.displayName = "EnhancedButton";

export { EnhancedButton, enhancedButtonVariants };