import * as React from "react"
import { cn } from "@/lib/utils"

interface QuantumGateButtonProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onDragStart' | 'onDragEnd' | 'onSelect'> {
  gate: {
    name: string;
    symbol: string;
    label: string;
    description: string;
  };
  isSelected?: boolean;
  isDragging?: boolean;
  onSelect?: (gateName: string) => void;
  onGateDragStart?: (gateName: string) => void;
  onGateDragEnd?: () => void;
}

export const QuantumGateButton = React.forwardRef<HTMLButtonElement, QuantumGateButtonProps>(
  ({ gate, isSelected, isDragging, onSelect, onGateDragStart, onGateDragEnd, disabled, className, ...props }, ref) => {
    const handleClick = () => {
      onSelect?.(gate.name);
    };

    const handleDragStart = () => {
      onGateDragStart?.(gate.name);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleClick();
      }
    };

    return (
      <button
        {...props}
        ref={ref}
        draggable={!disabled}
        onDragStart={handleDragStart}
        onDragEnd={onGateDragEnd}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={cn(
          "relative group p-3 rounded-lg border transition-all duration-200",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2",
          "min-h-[44px] min-w-[44px]", // WCAG touch target
          isSelected 
            ? "border-blue-400/60 bg-blue-500/10 shadow-lg" 
            : "border-slate-600/30 hover:border-blue-400/40 hover:bg-blue-500/5",
          disabled && "opacity-50 cursor-not-allowed",
          isDragging && "opacity-50",
          !disabled && "cursor-pointer",
          className
        )}
        aria-label={`${gate.label} gate - ${gate.description}`}
        aria-pressed={isSelected}
        title={`${gate.label}: ${gate.description}`}
        role="button"
        tabIndex={0}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="text-2xl font-mono font-bold text-blue-300 group-hover:text-blue-200">
            {gate.symbol}
          </div>
          {isSelected && (
            <div 
              className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" 
              aria-hidden="true"
            />
          )}
        </div>
        
        <div className="text-left">
          <div className="text-sm font-medium text-slate-300 mb-1">
            {gate.label}
          </div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            {gate.description}
          </div>
        </div>
        
        {/* Visual selection indicator */}
        {isSelected && (
          <div 
            className="absolute inset-0 border-2 border-blue-400/60 rounded-lg pointer-events-none" 
            aria-hidden="true"
          />
        )}
      </button>
    );
  }
);

QuantumGateButton.displayName = "QuantumGateButton";