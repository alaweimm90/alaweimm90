import React from 'react';
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

/* =====================================================================
   REPZ UNIFIED BUTTON SYSTEM - ENTERPRISE GRADE
   Mathematical color relationships + semantic usage rules
   ===================================================================== */

const buttonVariants = cva(
  // Base classes with performance optimizations
  "inline-flex items-center justify-center rounded-xl text-sm font-semibold transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 transform hover:scale-[1.02] active:scale-[0.98]",
  {
    variants: {
      variant: {
        // PRIMARY: ONLY for main conversion actions (Choose Plan, Book Session)
        primary: [
          "bg-repz-primary text-white shadow-lg",
          "hover:bg-repz-primary/90 hover:shadow-xl",
          "focus-visible:ring-repz-primary",
          "shadow-[0_8px_25px_hsla(var(--repz-primary-rgb),0.4)]",
          "hover:shadow-[0_12px_35px_hsla(var(--repz-primary-rgb),0.5)]"
        ].join(" "),
        
        // SECONDARY: Supporting actions that don't compete with primary
        secondary: [
          "bg-repz-complement text-white shadow-lg",
          "hover:bg-repz-complement/90 hover:shadow-xl", 
          "focus-visible:ring-repz-complement",
          "shadow-[0_8px_25px_hsla(var(--repz-complement-rgb),0.3)]",
          "hover:shadow-[0_12px_35px_hsla(var(--repz-complement-rgb),0.4)]"
        ].join(" "),
        
        // ENTERPRISE CORRECTION: ALL CTA buttons use REPZ orange
        // Tier colors are for visual identity only (backgrounds, borders, icons)
        // ALL primary actions use unified brand color for clear action hierarchy
        
        // GHOST: Subtle actions, filters, toggles
        ghost: [
          "border-2 border-repz-primary text-repz-primary bg-transparent",
          "hover:bg-repz-primary/10 hover:border-repz-primary/80",
          "focus-visible:ring-repz-primary"
        ].join(" "),
        
        // DESTRUCTIVE: For dangerous actions
        destructive: [
          "bg-red-600 text-white shadow-lg",
          "hover:bg-red-700 hover:shadow-xl",
          "focus-visible:ring-red-600"
        ].join(" "),
      },
      size: {
        sm: "h-9 px-4 text-xs",
        default: "h-11 px-6 py-2",
        lg: "h-13 px-8 py-3 text-base",
        xl: "h-16 px-10 py-4 text-lg",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };

/* =====================================================================
   USAGE EXAMPLES & RULES
   ===================================================================== */

/*
CORRECT USAGE:

// PRIMARY: Only for main conversion actions
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Choose Plan</Button>
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Book Session</Button>  
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Pre-order Access</Button>

// SECONDARY: Supporting actions
<Button variant="secondary" onClick={() => console.log("unified-button button clicked")}>View Details</Button>
<Button variant="secondary" onClick={() => console.log("unified-button button clicked")}>Learn More</Button>
<Button variant="secondary" onClick={() => console.log("unified-button button clicked")}>Contact Coach</Button>

// ENTERPRISE SYSTEM: ALL CTAs use primary (orange) for consistent action hierarchy
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Choose Core Program</Button>
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Choose Adaptive Engine</Button>
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Choose Prime Suite</Button>
<Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Choose Elite Concierge</Button>

// GHOST: Filters, toggles, less important actions
<Button variant="ghost" onClick={() => console.log("unified-button button clicked")}>Cards View</Button>
<Button variant="ghost" onClick={() => console.log("unified-button button clicked")}>Compare All</Button>
<Button variant="ghost" onClick={() => console.log("unified-button button clicked")}>Filter</Button>

WRONG USAGE:
❌ <Button variant="tier-core" onClick={() => console.log("unified-button button clicked")}>Book Session</Button>
❌ <Button variant="primary" onClick={() => console.log("unified-button button clicked")}>Cards View</Button>
❌ Random colors not in this system

ENTERPRISE PSYCHOLOGY:
- Primary (Orange): ALL conversion actions - creates clear action hierarchy
- Secondary (Blue): Supporting actions - builds trust without competing
- Tier Visual Identity: Colors used for backgrounds/borders/icons ONLY
- One Color One Meaning: Eliminates confusion, increases conversions
*/