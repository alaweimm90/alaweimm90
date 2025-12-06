import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const actionButtonVariants = cva([
  // Base styles - accessible foundation
  "inline-flex items-center justify-center gap-2 whitespace-nowrap",
  "font-medium transition-all duration-[350ms] relative overflow-hidden",
  "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent/50",
  "disabled:pointer-events-none disabled:opacity-50",
  // Shimmer effect for premium feel
  "before:absolute before:inset-0 before:bg-gradient-to-r before:from-transparent before:via-white/10 before:to-transparent",
  "before:translate-x-[-100%] hover:before:translate-x-[100%] before:transition-transform before:duration-700",
  // Hardware acceleration
  "transform-gpu will-change-transform",
], {
  variants: {
    variant: {
      primary: [
        "bg-accent text-black shadow-lg",
        "hover:brightness-95 hover:shadow-xl",
        "active:scale-[0.98] active:shadow-md",
      ],
      secondary: [
        "bg-[rgba(255,255,255,0.03)] text-muted border border-[rgba(255,255,255,0.1)]",
        "hover:bg-[rgba(255,255,255,0.06)] hover:border-[rgba(255,255,255,0.2)]",
        "active:scale-[0.98]",
      ],
      danger: [
        "bg-red-600 text-white shadow-lg",
        "hover:bg-red-700 hover:shadow-xl",
        "active:scale-[0.98] active:shadow-md",
      ],
      ghost: [
        "text-muted hover:text-foreground",
        "hover:bg-[rgba(255,255,255,0.03)]",
      ],
    },
    size: {
      normal: "px-4 py-2 min-h-[44px] min-w-[44px] text-sm",
      compact: "px-3 py-1.5 min-h-[36px] text-xs",
      large: "px-6 py-3 min-h-[48px] text-base",
      icon: "p-3 min-h-[44px] min-w-[44px]",
    }
  },
  defaultVariants: {
    variant: "primary",
    size: "normal",
  }
});

export interface ActionButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof actionButtonVariants> {
  /**
   * Icon component to display - should be descriptive with text
   */
  icon?: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean }>;
  /**
   * Descriptive action label for screen readers (REQUIRED for accessibility)
   * IMPORTANT: Should be verb-first and describe the action clearly
   * Examples:
   * - "Start training quantum machine learning model"
   * - "Export circuit as Qiskit Python code"
   * - "Reset workspace — clears all gates and qubits"
   */
  ariaLabel: string;
  /**
   * Loading state with accessible announcement
   */
  loading?: boolean;
  /**
   * Loading text announced to screen readers
   */
  loadingText?: string;
  /**
   * File metadata for download actions (e.g., "ONNX, 3.2 MB")
   */
  fileMeta?: string;
  /**
   * Destructive action - adds screen reader warning
   */
  destructive?: boolean;
  /**
   * Quantum energy field effect on hover
   */
  quantumEffect?: boolean;
}

/**
 * Accessible action button with descriptive labels and proper sizing
 *
 * Guidelines:
 * - Always use verb-first descriptive labels ("Start training", not "Train")
 * - Include file metadata for downloads ("Export model (ONNX)")
 * - Minimum 44x44px touch targets (WCAG 2.1 AA requirement)
 * - Clear consequence for destructive actions ("Reset circuit — clears workspace")
 * - Loading states announced to screen readers with aria-live
 *
 * @example
 * <ActionButton
 *   ariaLabel="Start training quantum machine learning model"
 *   icon={Play}
 *   quantumEffect
 *   onClick={handleStart}
 * >
 *   Start Training
 * </ActionButton>
 *
 * @example
 * <ActionButton
 *   variant="danger"
 *   ariaLabel="Reset workspace — clears all gates and qubits"
 *   destructive
 *   icon={RotateCcw}
 *   onClick={handleReset}
 * >
 *   Reset
 * </ActionButton>
 */
export const ActionButton = React.forwardRef<HTMLButtonElement, ActionButtonProps>(
  ({
    className,
    variant,
    size,
    icon: Icon,
    ariaLabel,
    loading,
    loadingText = "Loading",
    fileMeta,
    destructive = false,
    quantumEffect = false,
    children,
    ...props
  }, ref) => {
    // Build visible text with file metadata if provided
    const visibleText = fileMeta ? `${children} (${fileMeta})` : children;

    // Build accessible label (use loading text when loading)
    const accessibleLabel = loading ? loadingText : ariaLabel;

    return (
      <button
        className={cn(
          actionButtonVariants({ variant, size }),
          quantumEffect && "quantum-energy-field",
          className
        )}
        ref={ref}
        aria-label={accessibleLabel}
        aria-busy={loading}
        aria-live={loading ? "polite" : undefined}
        disabled={loading || props.disabled}
        role="button"
        tabIndex={0}
        {...props}
      >
        {loading ? (
          <>
            <div
              className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"
              aria-hidden="true"
            />
            <span className="sr-only">{loadingText}...</span>
          </>
        ) : Icon ? (
          <Icon aria-hidden={true} className="w-4 h-4 flex-shrink-0" />
        ) : null}

        {size !== "icon" && (
          <span className="flex-1 text-left">{visibleText}</span>
        )}

        {/* Screen reader hint for destructive actions */}
        {destructive && !loading && (
          <span className="sr-only"> — this action cannot be undone</span>
        )}
      </button>
    );
  }
);

ActionButton.displayName = "ActionButton";

/**
 * Icon-only button with mandatory accessible label
 */
export interface IconButtonProps extends Omit<ActionButtonProps, 'children' | 'size'> {
  /**
   * Required accessible label describing the action
   */
  label: string;
  /**
   * Required icon component
   */
  icon: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean }>;
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ label, icon: Icon, className, variant = "ghost", ...props }, ref) => {
    return (
      <ActionButton
        ref={ref}
        variant={variant}
        size="icon"
        icon={Icon}
        ariaLabel={label}
        className={className}
        {...props}
      >
        <span className="sr-only">{label}</span>
      </ActionButton>
    );
  }
);

IconButton.displayName = "IconButton";