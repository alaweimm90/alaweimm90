import * as React from "react";
import { cn } from "@/lib/utils";

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * Required accessible label for the button
   */
  label: string;
  /**
   * Optional visual variant
   */
  variant?: "default" | "ghost" | "quantum";
  /**
   * Optional size
   */
  size?: "sm" | "md" | "lg";
  /**
   * Icon content - should be an icon component
   */
  children: React.ReactNode;
}

const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ label, variant = "default", size = "md", children, className, ...props }, ref) => {
    const sizeClasses = {
      sm: "h-[var(--button-height-sm)] w-[var(--button-height-sm)] p-1",
      md: "h-[var(--button-height-md)] w-[var(--button-height-md)] p-2", 
      lg: "h-[var(--button-height-lg)] w-[var(--button-height-lg)] p-3"
    };

    const variantClasses = {
      default: "bg-[var(--surface-2)] border border-[var(--border-muted)] hover:bg-[var(--surface-3)]",
      ghost: "bg-transparent hover:bg-[var(--surface-2)]",
      quantum: "bg-[var(--glass-bg)] border border-[var(--glass-border)] backdrop-blur-[var(--glass-blur)] hover:bg-[var(--glass-bg-subtle)]"
    };

    return (
      <button
        ref={ref}
        aria-label={label}
        className={cn(
          // Base styles - WCAG 2.1 AA Compliant
          "inline-flex items-center justify-center rounded-[var(--radius-md)]",
          "text-[var(--text-secondary)] hover:text-[var(--text-primary)]",
          "transition-all duration-[var(--transition-normal)]",
          "focus-visible:outline-0 focus-visible:ring-[var(--focus-ring)] focus-visible:ring-offset-[var(--focus-ring-offset)]",
          "disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed",
          "active:scale-95",
          // Size variants
          sizeClasses[size],
          // Style variants
          variantClasses[variant],
          className
        )}
        {...props}
      >
        {React.cloneElement(children as React.ReactElement, {
          "aria-hidden": true,
          focusable: false
        })}
        <span className="sr-only">{label}</span>
      </button>
    );
  }
);

IconButton.displayName = "IconButton";

export { IconButton };