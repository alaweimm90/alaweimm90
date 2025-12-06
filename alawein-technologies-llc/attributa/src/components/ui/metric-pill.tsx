import React from "react";
import { cn } from "@/lib/utils";
import { useCountUp } from "@/hooks/useCountUp";

interface MetricPillProps {
  label: string;
  value: number;
  format?: "percentage" | "decimal" | "integer";
  confidence?: "high" | "medium" | "low";
  animate?: boolean;
  className?: string;
}

export function MetricPill({
  label,
  value,
  format = "decimal",
  confidence,
  animate = true,
  className
}: MetricPillProps) {
  const { ref, display } = useCountUp(value, {
    startOnView: animate,
    durationMs: 800,
    formatter: (val) => {
      switch (format) {
        case "percentage":
          return `${(val * 100).toFixed(1)}%`;
        case "integer":
          return Math.round(val).toString();
        case "decimal":
        default:
          return val.toFixed(3);
      }
    }
  });

  const confidenceStyles = {
    high: "bg-confidence-high/15 text-confidence-high border-confidence-high/30",
    medium: "bg-confidence-medium/15 text-confidence-medium border-confidence-medium/30",
    low: "bg-confidence-low/15 text-confidence-low border-confidence-low/30"
  };

  return (
    <div
      ref={ref}
      className={cn(
        "inline-flex flex-col items-center px-3 py-2 rounded-lg border bg-background/50 backdrop-blur-sm transition-all duration-200 min-h-[44px] md:min-h-0", /* Touch-friendly on mobile */
        confidence && confidenceStyles[confidence],
        !confidence && "border-border/50 text-foreground",
        className
      )}
    >
      <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
        {label}
      </span>
      <span className="text-lg font-mono font-semibold tabular-nums">
        {display}
      </span>
    </div>
  );
}