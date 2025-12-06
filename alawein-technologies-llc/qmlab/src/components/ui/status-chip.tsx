import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const statusChipVariants = cva(
  "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors min-h-[28px] min-w-[28px] md:min-h-[44px] md:min-w-[44px]",
  {
    variants: {
      variant: {
        // Training states with better contrast
        running: "bg-primary/60 text-white border border-primary",
        converged: "bg-green-600/80 text-white border border-green-500",
        failed: "bg-red-600/80 text-white border border-red-500",
        idle: "bg-slate-600/80 text-white border border-slate-500",
        
        // Quantum states with improved contrast
        pure: "bg-blue-600/80 text-white border border-blue-500",
        mixed: "bg-purple-600/80 text-white border border-purple-500",
        entangled: "bg-pink-600/80 text-white border border-pink-500",
        
        // General status with better contrast
        success: "bg-green-600/80 text-white border border-green-500",
        warning: "bg-yellow-600/80 text-white border border-yellow-500",
        error: "bg-red-600/80 text-white border border-red-500",
        info: "bg-blue-600/80 text-white border border-blue-500"
      },
      size: {
        sm: "h-7 px-2 text-xs min-h-[28px] min-w-[28px]",
        md: "h-8 px-3 text-xs min-h-[32px] min-w-[32px]", 
        lg: "h-9 px-4 text-sm min-h-[36px] min-w-[36px]"
      }
    },
    defaultVariants: {
      variant: "info",
      size: "md"
    }
  }
);

export interface StatusChipProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof statusChipVariants> {
  icon?: React.ReactNode;
}

const StatusChip = React.forwardRef<HTMLDivElement, StatusChipProps>(
  ({ className, variant, size, icon, children, ...props }, ref) => {
    return (
      <div
        className={cn(statusChipVariants({ variant, size }), className)}
        ref={ref}
        {...props}
      >
        {icon}
        {children}
      </div>
    );
  }
);
StatusChip.displayName = "StatusChip";

export { StatusChip, statusChipVariants };