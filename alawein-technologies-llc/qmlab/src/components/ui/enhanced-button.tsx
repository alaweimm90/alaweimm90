import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const enhancedButtonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap font-medium transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 relative overflow-hidden group",
  {
    variants: {
      variant: {
        // Primary CTAs - Maximum visual impact
        "cta-primary": [
          "bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600",
          "hover:from-blue-500 hover:via-purple-500 hover:to-indigo-500",
          "text-white font-semibold tracking-wide",
          "shadow-xl hover:shadow-2xl",
          "hover:shadow-blue-500/25",
          "transform hover:scale-105 hover:-translate-y-0.5",
          "before:absolute before:inset-0 before:bg-gradient-to-r before:from-blue-400/20 before:via-purple-400/20 before:to-indigo-400/20",
          "before:opacity-0 before:transition-opacity before:duration-300 group-hover:before:opacity-100",
          "animate-quantum-glow-subtle",
          "border-0 rounded-xl"
        ],
        // Secondary CTAs - Strong but not overwhelming
        "cta-secondary": [
          "bg-gradient-to-r from-slate-800 to-slate-700",
          "hover:from-slate-700 hover:to-slate-600",
          "text-slate-200 hover:text-white",
          "border border-slate-600 hover:border-slate-500",
          "shadow-lg hover:shadow-xl",
          "hover:shadow-slate-500/10",
          "transform hover:scale-102",
          "rounded-xl font-medium"
        ],
        // Learning-focused CTAs
        "cta-learn": [
          "bg-gradient-to-r from-green-600/20 to-emerald-600/20",
          "hover:from-green-600/30 hover:to-emerald-600/30",
          "border border-green-500/40 hover:border-green-500/60",
          "text-green-300 hover:text-green-200",
          "shadow-lg hover:shadow-xl",
          "hover:shadow-green-500/10",
          "transform hover:scale-102",
          "rounded-xl font-medium"
        ],
        // Interactive/Playground CTAs
        "cta-interactive": [
          "bg-gradient-to-r from-purple-600/20 to-pink-600/20",
          "hover:from-purple-600/30 hover:to-pink-600/30",
          "border border-purple-500/40 hover:border-purple-500/60",
          "text-purple-300 hover:text-purple-200",
          "shadow-lg hover:shadow-xl",
          "hover:shadow-purple-500/10",
          "transform hover:scale-102",
          "rounded-xl font-medium"
        ],
        // Quantum-themed floating CTA
        "cta-floating": [
          "bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500",
          "hover:from-blue-400 hover:via-purple-400 hover:to-indigo-400",
          "text-white font-semibold",
          "shadow-2xl hover:shadow-3xl",
          "hover:shadow-blue-500/30",
          "transform hover:scale-110 hover:-translate-y-1",
          "animate-float",
          "rounded-full",
          "before:absolute before:inset-0 before:bg-gradient-to-r before:from-blue-300/30 before:via-purple-300/30 before:to-indigo-300/30",
          "before:opacity-0 before:transition-opacity before:duration-300 group-hover:before:opacity-100 before:blur-sm before:-z-10",
          "after:absolute after:inset-0 after:bg-gradient-to-r after:from-white/20 after:via-transparent after:to-transparent",
          "after:opacity-0 after:transition-opacity after:duration-300 group-hover:after:opacity-100 after:rounded-full"
        ]
      },
      size: {
        sm: "h-9 px-4 text-sm",
        default: "h-11 px-6 text-base",
        lg: "h-14 px-8 text-lg",
        xl: "h-16 px-10 text-xl",
        icon: "h-11 w-11",
        floating: "h-12 w-12 p-0"
      },
    },
    defaultVariants: {
      variant: "cta-primary",
      size: "default",
    },
  }
);

export interface EnhancedButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof enhancedButtonVariants> {
  asChild?: boolean;
  glowEffect?: boolean;
  pulseOnMount?: boolean;
}

const EnhancedButton = React.forwardRef<HTMLButtonElement, EnhancedButtonProps>(
  ({ className, variant, size, asChild = false, glowEffect = false, pulseOnMount = false, children, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    
    return (
      <Comp
        className={cn(
          enhancedButtonVariants({ variant, size }),
          glowEffect && "hover:animate-quantum-glow",
          pulseOnMount && "animate-quantum-pulse",
          className
        )}
        ref={ref}
        {...props}
      >
        {children}
        {/* Quantum shimmer effect for primary CTAs */}
        {variant === "cta-primary" && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 skew-x-12" />
        )}
      </Comp>
    );
  }
);

EnhancedButton.displayName = "EnhancedButton";

export { EnhancedButton, enhancedButtonVariants };